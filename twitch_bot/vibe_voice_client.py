import asyncio
import websockets
import sounddevice as sd
import json
import numpy as np
import re
import httpx


class TTSClient:
    def __init__(
        self,
        ws_url="ws://IP:PORT/stream",
        http_base="http://IP:PORT",
        timeout=15.0,
        samplerate=24000,
        channels=1,
        dtype="float32",
        output_device="Chat",  # int index or str name; None = system default
    ):
        self.ws_url = ws_url
        self.http_base = http_base.rstrip("/")
        self.timeout = timeout
        self.samplerate = samplerate
        self.channels = channels
        self.dtype = dtype
        self.output_device = output_device

    @staticmethod
    def list_audio_devices():
        devices = sd.query_devices()
        hostapis = sd.query_hostapis()
        for i, d in enumerate(devices):
            api = hostapis[d["hostapi"]]["name"]
            direction = []
            if d["max_input_channels"] > 0:
                direction.append(f"in:{d['max_input_channels']}")
            if d["max_output_channels"] > 0:
                direction.append(f"out:{d['max_output_channels']}")
            # print(f"[{i:2d}] {d['name']}  ({api})  {' '.join(direction)}")

        default_out = sd.default.device[1] if isinstance(sd.default.device, (list, tuple)) else sd.default.device
        # print("\nDefault output device index:", default_out)

    @staticmethod
    def _resolve_output_device(output_device):
        """
        Accepts:
          - None (use system default)
          - int (device index)
          - str (partial/full device name; first match with output channels)
        Returns: int|None
        """
        if output_device is None:
            return None
        if isinstance(output_device, int):
            return output_device
        if isinstance(output_device, str):
            name_q = output_device.lower()
            devices = sd.query_devices()
            for i, d in enumerate(devices):
                if d["max_output_channels"] > 0 and name_q in d["name"].lower():
                    return i
            raise ValueError(f"No output device matched name contains: {output_device!r}")
        raise TypeError("output_device must be None, int, or str")

    async def _fetch_random_greeting(self, uuid: str) -> tuple[bool, str | None, str | None]:
        """
        Returns (ok, text, err)
        """
        url = f"{self.http_base}/api/random_greeting/{uuid}"
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                r = await client.get(url)
            if r.status_code != 200:
                return False, None, f"greeting http {r.status_code}: {r.text}"
            j = r.json()
            text = (j.get("text") or "").strip()
            if not text:
                return False, None, "empty greeting returned"
            return True, text, None
        except Exception as e:
            return False, None, f"greeting fetch failed: {e}"

    async def synthesize(self, text_or_script: str, speakers: list[str], output_device=None):
        """
        Backwards compatible with your existing !say path:
          speakers=["JUNA"] -> server resolves to JUNA_VOICE.wav

        output_device:
          - None => use self.output_device (or system default if that is None)
          - int => device index
          - str => partial/full name match (first output-capable match)

        Returns (ok: bool, err: str|None)
        """
        request = {
            "script": text_or_script if len(speakers) > 1 else f"Speaker 0: {text_or_script}",
            "voices": speakers,  # IMPORTANT: send raw tokens; server normalizes to *_VOICE.wav
            "cfg_scale": 1.3,
        }

        chosen = self.output_device if output_device is None else output_device
        device_idx = self._resolve_output_device(chosen)

        try:
            async with websockets.connect(
                self.ws_url,
                max_size=None,
                ping_interval=60,
                ping_timeout=600,
                close_timeout=60,
            ) as ws:
                await ws.send(json.dumps(request))
                # print(f"🎙️ Request sent for {speakers}: {text_or_script[:60]}...")

                stream = sd.OutputStream(
                    samplerate=self.samplerate,
                    channels=self.channels,
                    dtype=self.dtype,
                    device=device_idx,  # <-- forces output device
                )
                stream.start()

                try:
                    async for message in ws:
                        if isinstance(message, str):
                            msg = json.loads(message)
                            ev = msg.get("event")
                            if ev == "done":
                                print("✅ Generation complete.")
                                return True, None
                            if ev == "error":
                                err = msg.get("message") or "unknown error"
                                print("❌ Error:", err)
                                return False, err
                            continue

                        audio_np = np.frombuffer(message, dtype=np.float32)
                        if audio_np.size > 0:
                            stream.write(audio_np)

                finally:
                    stream.stop()
                    stream.close()

            return False, "websocket closed unexpectedly"
        except Exception as e:
            return False, str(e)

    async def synthesize_greet(self, uuid: str, output_device=None):
        """
        Used by !greet:
          - fetch a random greeting text for the UUID
          - synthesize it using voices=[uuid]
        Returns (ok: bool, err: str|None)
        """
        ok, text, err = await self._fetch_random_greeting(uuid)
        if not ok:
            return False, err

        return await self.synthesize(text, [uuid], output_device=output_device)

    async def synthesize_chunked(self, text_or_script: str, speakers: list[str], max_chars: int = 800, output_device=None):
        if len(text_or_script) <= max_chars:
            return await self.synthesize(text_or_script, speakers, output_device=output_device)

        print(f"📝 Script is {len(text_or_script)} chars, splitting into chunks...")

        if len(speakers) > 1:
            lines = text_or_script.split("\n")
            chunks = []
            current_chunk = []
            current_length = 0

            for line in lines:
                line_length = len(line)
                if current_length + line_length > max_chars and current_chunk:
                    chunks.append("\n".join(current_chunk))
                    current_chunk = [line]
                    current_length = line_length
                else:
                    current_chunk.append(line)
                    current_length += line_length

            if current_chunk:
                chunks.append("\n".join(current_chunk))
        else:
            sentences = re.split(r"([.!?]+\s+)", text_or_script)
            chunks = []
            current_chunk = ""

            for i in range(0, len(sentences), 2):
                sentence = sentences[i]
                punctuation = sentences[i + 1] if i + 1 < len(sentences) else ""
                full_sentence = sentence + punctuation

                if len(current_chunk) + len(full_sentence) > max_chars and current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = full_sentence
                else:
                    current_chunk += full_sentence

            if current_chunk:
                chunks.append(current_chunk.strip())

        print(f"🎙️ Split into {len(chunks)} chunks")
        for i, chunk in enumerate(chunks):
            print(f"\n📢 Chunk {i + 1}/{len(chunks)} ({len(chunk)} chars)")
            ok, err = await self.synthesize(chunk, speakers, output_device=output_device)
            if not ok:
                return False, err
            await asyncio.sleep(0.5)

        print("\n✅ All chunks complete!")
        return True, None



