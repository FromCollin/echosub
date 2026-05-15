# server.py
import os
import threading
import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, WebSocket
from fastapi.responses import JSONResponse
import uvicorn
from datetime import datetime
import re
import asyncio

import json
import random
from pathlib import Path

from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor
from vibevoice.modular.streamer import AudioStreamer


class VibeVoiceDemo:
    def __init__(self, model_path: str, device: str = "cuda", inference_steps: int = 5):
        self.model_path = model_path
        self.device = device
        self.inference_steps = inference_steps
        self.stop_generation = False
        self.current_streamer = None
        self.available_voices = {}
        self.load_model()
        print("✅ VibeVoice initialized")

    def load_model(self):
        if self.device == "mps" and not torch.backends.mps.is_available():
            print("⚠️ MPS not available, falling back to CPU")
            self.device = "cpu"

        self.processor = VibeVoiceProcessor.from_pretrained(self.model_path)

        if self.device == "cuda":
            dtype = torch.bfloat16
            attn = "flash_attention_2"
        else:
            dtype = torch.float32
            attn = "sdpa"

        try:
            print(f"⏳ Loading model with attn={attn}")
            self.model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                self.model_path,
                torch_dtype=dtype,
                device_map=self.device if self.device in ("cuda", "cpu") else None,
                attn_implementation=attn,
            )
        except Exception as e:
            print(f"[WARN] Failed with {attn}: {e}")
            print("➡️ Falling back to sdpa")
            self.model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                self.model_path,
                torch_dtype=dtype,
                device_map=self.device if self.device in ("cuda", "cpu") else None,
                attn_implementation="sdpa",
            )

        self.model.eval()

        self.model.model.noise_scheduler = self.model.model.noise_scheduler.from_config(
            self.model.model.noise_scheduler.config,
            algorithm_type="sde-dpmsolver++",
            beta_schedule="squaredcos_cap_v2",
        )
        self.model.set_ddpm_inference_steps(num_steps=self.inference_steps)

    def read_audio(self, audio_path: str, target_sr: int = 24000) -> np.ndarray:
        try:
            wav, sr = sf.read(audio_path)
            if len(wav.shape) > 1:
                wav = np.mean(wav, axis=1)
            if sr != target_sr:
                import librosa
                wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
            return wav
        except Exception as e:
            print(f"Error reading audio {audio_path}: {e}")
            return np.array([])

    def generate_podcast_streaming(self, num_speakers: int, script: str, cfg_scale: float = 1.3, **speakers):
        voice_samples = []
        for name in speakers.values():
            audio_path = self.available_voices[name]
            audio_data = self.read_audio(audio_path)
            if len(audio_data) == 0:
                raise RuntimeError(f"Failed to load audio for {name}")
            voice_samples.append(audio_data)

        inputs = self.processor(
            text=[script],
            voice_samples=[voice_samples],
            padding=True,
            return_tensors="pt",
            return_attention_mask=True,
        )
        device = self.device if self.device in ("cuda", "mps") else "cpu"
        for k, v in inputs.items():
            if torch.is_tensor(v):
                inputs[k] = v.to(device)

        audio_streamer = AudioStreamer(batch_size=1, stop_signal=None, timeout=None)
        self.current_streamer = audio_streamer

        def run_model():
            self.model.generate(
                **inputs,
                cfg_scale=cfg_scale,
                tokenizer=self.processor.tokenizer,
                audio_streamer=audio_streamer,
                verbose=False,
                refresh_negative=True,
            )

        threading.Thread(target=run_model, daemon=True).start()

        sample_rate = 24000
        all_chunks = []
        for chunk in audio_streamer.get_stream(0):
            if torch.is_tensor(chunk):
                if chunk.dtype == torch.bfloat16:
                    chunk = chunk.float()
                chunk = chunk.cpu().numpy().astype(np.float32)
            all_chunks.append(chunk)
            yield (sample_rate, chunk), None, None, None

        if all_chunks:
            audio = np.concatenate(all_chunks)
            yield None, (sample_rate, audio), None, None
        else:
            raise RuntimeError("No audio chunks were produced")


def normalize_speaker_lines(script: str) -> str:
    """Normalize to 'Speaker 1:', 'Speaker 2:' etc."""
    script = re.sub(r"Speaker\s*0\s*:", "Speaker 1:", script)
    script = re.sub(r"Speaker\s*(\d+)\s*:", lambda m: f"Speaker {int(m.group(1))}:", script)
    return script


# --- Backwards-compatible voice resolution ---
def _normalize_voice_token(v: str) -> str:
    """
    Accept:
      - 'JUNA' or '893807921'                   -> 'JUNA_VOICE.wav' / '893807921_VOICE.wav'
      - 'JUNA_VOICE.wav' / '893..._VOICE.wav'   -> unchanged
      - 'juna_voice.wav'                        -> unchanged (case preserved)
      - 'JUNA_VOICE' (no .wav)                  -> add .wav
    """
    v = (v or "").strip()
    if not v:
        return v
    if v.lower().endswith("_voice.wav"):
        return v
    if v.lower().endswith("_voice"):
        return v + ".wav"
    return f"{v}_VOICE.wav"


def resolve_voice_path(v: str, voice_dirs: list[str]) -> str | None:
    filename = _normalize_voice_token(v)
    if not filename:
        return None
    for d in voice_dirs:
        path = os.path.join(d, filename)
        if os.path.exists(path):
            return path
    return None


app = FastAPI()

BASE_DIR = os.path.dirname(__file__)

# Legacy voices directory (your old setup)
VOICE_DIR_LEGACY = os.getenv("VOICE_DIR_LEGACY", os.path.join(BASE_DIR, "voices"))
# New processed UUID voices (your ingestion pipeline output)
VOICE_DIR_PROCESSED = os.getenv("VOICE_DIR_PROCESSED", "/mnt/data/voiceclone/processed")

VOICE_DIRS = [VOICE_DIR_LEGACY, VOICE_DIR_PROCESSED]
print(f"VOICE_DIRS={VOICE_DIRS}")

OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Greetings metadata pulled from VPS via rsync:
# /mnt/data/voiceclone/meta/{UUID}/greetings.json
META_DIR = Path(os.getenv("VOICECLONE_META_DIR", "/mnt/data/voiceclone/meta"))

DEFAULT_GREETINGS = [
    "Hello, chat!",
    "Hey everyone!",
    "What’s up, chat!",
    "Good to see you all!",
    "Yo chat!",
]


def _load_user_greetings(uuid: str) -> list[str]:
    """
    Reads /mnt/data/voiceclone/meta/{uuid}/greetings.json
    Expected JSON shape (example):
      {"viewer_id":"123", "greetings":["...","..."], "updated_at":1234567890}
    """
    p = META_DIR / uuid / "greetings.json"
    if not p.exists():
        return []

    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return []

    greetings = data.get("greetings", [])
    if not isinstance(greetings, list):
        return []

    cleaned: list[str] = []
    for g in greetings:
        if isinstance(g, str):
            s = g.strip()
            if s:
                cleaned.append(s)

    return cleaned[:3]


@app.get("/api/random_greeting/{uuid}")
async def api_random_greeting(uuid: str):
    user_greetings = _load_user_greetings(uuid)
    pool = user_greetings if user_greetings else DEFAULT_GREETINGS
    text = random.choice(pool) if pool else "Hello, chat!"
    return JSONResponse({"uuid": uuid, "text": text, "custom": bool(user_greetings)})


vv = VibeVoiceDemo(
    model_path="../VibeVoice-1.5B",
    device="cuda",
    inference_steps=10,
)


@app.websocket("/stream")
async def stream_audio(ws: WebSocket):
    await ws.accept()
    data = await ws.receive_json()
    script = data["script"]
    voices = data["voices"]
    cfg_scale = float(data.get("cfg_scale", 1.3))

    script = normalize_speaker_lines(script)

    if not (1 <= len(voices) <= 4):
        await ws.send_json({"event": "error", "message": "voices must contain 1 to 4 items"})
        await ws.close()
        return

    resolved = []
    for v in voices:
        path = resolve_voice_path(v, VOICE_DIRS)
        if not path:
            tried = [os.path.join(d, _normalize_voice_token(v)) for d in VOICE_DIRS]
            await ws.send_json(
                {"event": "error", "message": f"Voice file not found for '{v}'. Tried: {tried}"}
            )
            await ws.close()
            return
        resolved.append(path)

    vv.available_voices = {f"voice_{i}": path for i, path in enumerate(resolved)}
    speaker_args = {f"speaker_{i+1}": f"voice_{i}" for i in range(len(resolved))}

    try:
        complete_audio = None
        chunk_count = 0

        for streaming_audio, complete_audio, _, _ in vv.generate_podcast_streaming(
            num_speakers=len(resolved),
            script=script,
            cfg_scale=cfg_scale,
            **speaker_args,
        ):
            if streaming_audio:
                sr, audio_np = streaming_audio
                await ws.send_bytes(audio_np.astype(np.float32).tobytes())
                chunk_count += 1

                if chunk_count % 10 == 0:
                    await ws.send_json({"event": "progress", "chunks": chunk_count})

        await ws.send_json({"event": "done"})

        if complete_audio:
            sr, full_audio = complete_audio
            audio = np.nan_to_num(full_audio).astype(np.float32).flatten()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_path = os.path.join(OUTPUT_DIR, f"output_{timestamp}.wav")

            def _save():
                try:
                    sf.write(out_path, audio, sr)
                    print(f"💾 Saved complete audio ({len(audio)/sr:.2f}s) to {out_path}")
                except Exception as e:
                    print(f"[SAVE ERROR] {e}")

            asyncio.get_running_loop().run_in_executor(None, _save)

    except Exception as e:
        print(f"[ERROR] {e}")
        try:
            await ws.send_json({"event": "error", "message": str(e)})
        except Exception:
            pass
    finally:
        try:
            await ws.close()
        except Exception:
            pass


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9000)

