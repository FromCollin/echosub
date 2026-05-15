import os
import json
import time
import re
import unicodedata
from pathlib import Path

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, Request, HTTPException, UploadFile, File
from fastapi.responses import RedirectResponse, HTMLResponse, JSONResponse
from starlette.middleware.sessions import SessionMiddleware
from pydantic import BaseModel

load_dotenv()

TWITCH_CLIENT_ID = os.getenv("TWITCH_CLIENT_ID")
TWITCH_CLIENT_SECRET = os.getenv("TWITCH_CLIENT_SECRET")

VIEWER_REDIRECT_URI = os.getenv("TWITCH_VIEWER_REDIRECT_URI", "http://localhost:3000/auth/twitch/callback")
ADMIN_REDIRECT_URI = os.getenv("TWITCH_ADMIN_REDIRECT_URI", "http://localhost:3000/auth/twitch/admin/callback")

# SECURITY: fail loudly if SESSION_SECRET is not set
SESSION_SECRET = os.getenv("SESSION_SECRET")
if not SESSION_SECRET:
    raise RuntimeError("SESSION_SECRET must be set in .env")

ADMIN_KEY = os.getenv("ADMIN_KEY", "")
TOKEN_STORE_PATH = Path(os.getenv("TWITCH_TOKEN_STORE_PATH", "./twitch_admin_token.json"))

UPLOAD_ROOT = Path(os.getenv("UPLOAD_ROOT", "./uploads/raw"))
META_ROOT = Path(os.getenv("META_ROOT", "./uploads/meta"))
CONSENT_ROOT = Path(os.getenv("CONSENT_ROOT", "./uploads/consent"))
EXTRA_BANNED_WORDS_FILE = Path(os.getenv("EXTRA_BANNED_WORDS_FILE", "./banned_words_extra.txt"))

if not all([TWITCH_CLIENT_ID, TWITCH_CLIENT_SECRET]):
    raise RuntimeError("Missing TWITCH_CLIENT_ID / TWITCH_CLIENT_SECRET")

app = FastAPI()
app.add_middleware(SessionMiddleware, secret_key=SESSION_SECRET, same_site="lax")

TWITCH_AUTHORIZE_URL = "https://id.twitch.tv/oauth2/authorize"
TWITCH_TOKEN_URL = "https://id.twitch.tv/oauth2/token"
TWITCH_HELIX_USERS = "https://api.twitch.tv/helix/users"
TWITCH_HELIX_SUBSCRIPTIONS = "https://api.twitch.tv/helix/subscriptions"


# ------------------------
# Greeting validation/filtering
# ------------------------
GREETING_MIN_CHARS = 5
GREETING_MAX_CHARS = 200
GREETING_MAX_COUNT = 3

# Base banned list: keep this conservative in code, and expand via EXTRA_BANNED_WORDS_FILE
# as needed (one term per line).
BANNED_WORDS = {


    # violence / threats
    "kill",
    "murder",
    "die",

}


# Phrase patterns: expand as needed
BANNED_PHRASES = [

    # TTS baiting
    r"say\s+the\s+word\s+\w+",
    r"repeat\s+after\s+me",
    r"spell\s+out\s+\w+",
    r"say\s+i\s+hate\s+\w+",


]

_LEET_MAP = str.maketrans({
    "0": "o", "1": "i", "2": "z", "3": "e", "4": "a",
    "5": "s", "6": "g", "7": "t", "8": "b", "9": "g",
    "@": "a", "$": "s", "!": "i",
})

def normalize_for_filtering(text: str) -> str:
    if not text:
        return ""
    t = unicodedata.normalize("NFKD", text)
    t = "".join(ch for ch in t if not unicodedata.combining(ch))
    t = t.lower()
    t = t.translate(_LEET_MAP)
    t = re.sub(r"(.)\1{2,}", r"\1\1", t)
    t = re.sub(r"[^a-z0-9]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def _load_extra_banned_words() -> set[str]:
    try:
        if EXTRA_BANNED_WORDS_FILE.exists():
            lines = EXTRA_BANNED_WORDS_FILE.read_text(encoding="utf-8", errors="ignore").splitlines()
            return {ln.strip().lower() for ln in lines if ln.strip() and not ln.strip().startswith("#")}
    except Exception:
        pass
    return set()

def build_filter_sets() -> tuple[set[str], re.Pattern, list[re.Pattern]]:
    all_words = set(BANNED_WORDS) | _load_extra_banned_words()
    word_re = re.compile(r"\b(" + "|".join(map(re.escape, sorted(all_words))) + r")\b", re.IGNORECASE)
    phrase_res = [re.compile(p, re.IGNORECASE) for p in BANNED_PHRASES]
    return all_words, word_re, phrase_res

_ALL_BANNED_WORDS, _WORD_RE, _PHRASE_RES = build_filter_sets()

def is_allowed_greeting_text(raw_text: str) -> bool:
    t = normalize_for_filtering(raw_text)
    if _WORD_RE.search(t):
        return False
    for rx in _PHRASE_RES:
        if rx.search(t):
            return False
    return True

def atomic_write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


class GreetingsPayload(BaseModel):
    greetings: list[str] | None = None


# ------------------------
# Token helpers
# ------------------------
def _load_admin_token() -> dict | None:
    if TOKEN_STORE_PATH.exists():
        return json.loads(TOKEN_STORE_PATH.read_text(encoding="utf-8"))
    return None

def _save_admin_token(data: dict) -> None:
    TOKEN_STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    TOKEN_STORE_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")
    try:
        os.chmod(TOKEN_STORE_PATH, 0o600)
    except Exception:
        pass

def _token_is_expired(tok: dict) -> bool:
    return int(tok.get("expires_at", 0)) <= int(time.time()) + 30

async def _refresh_admin_token(tok: dict) -> dict:
    refresh_token = tok.get("refresh_token")
    if not refresh_token:
        raise HTTPException(status_code=500, detail="Admin token missing refresh_token; re-connect admin.")

    async with httpx.AsyncClient(timeout=20) as client:
        resp = await client.post(
            TWITCH_TOKEN_URL,
            data={
                "grant_type": "refresh_token",
                "refresh_token": refresh_token,
                "client_id": TWITCH_CLIENT_ID,
                "client_secret": TWITCH_CLIENT_SECRET,
            },
        )

    if resp.status_code != 200:
        raise HTTPException(status_code=500, detail=f"Failed to refresh admin token: {resp.status_code} {resp.text}")

    j = resp.json()
    access_token = j.get("access_token")
    new_refresh = j.get("refresh_token", refresh_token)
    expires_in = int(j.get("expires_in", 0))
    if not access_token or not expires_in:
        raise HTTPException(status_code=500, detail=f"Unexpected refresh response: {j}")

    tok["access_token"] = access_token
    tok["refresh_token"] = new_refresh
    tok["expires_at"] = int(time.time()) + expires_in
    _save_admin_token(tok)
    return tok

async def get_valid_admin_token() -> dict:
    tok = _load_admin_token()
    if not tok:
        raise HTTPException(status_code=503, detail="Admin not connected. Visit /admin/connect first.")
    if _token_is_expired(tok):
        tok = await _refresh_admin_token(tok)
    return tok

async def exchange_code_for_token(code: str, redirect_uri: str) -> dict:
    async with httpx.AsyncClient(timeout=20) as client:
        r = await client.post(
            TWITCH_TOKEN_URL,
            data={
                "client_id": TWITCH_CLIENT_ID,
                "client_secret": TWITCH_CLIENT_SECRET,
                "code": code,
                "grant_type": "authorization_code",
                "redirect_uri": redirect_uri,
            },
        )
    if r.status_code != 200:
        raise HTTPException(status_code=500, detail=f"Token exchange failed: {r.status_code} {r.text}")
    return r.json()

async def get_user_by_token(access_token: str) -> dict:
    async with httpx.AsyncClient(timeout=20) as client:
        r = await client.get(
            TWITCH_HELIX_USERS,
            headers={
                "Authorization": f"Bearer {access_token}",
                "Client-ID": TWITCH_CLIENT_ID,
            },
        )
    if r.status_code != 200:
        raise HTTPException(status_code=500, detail=f"Helix users failed: {r.status_code} {r.text}")
    data = r.json().get("data", [])
    if not data:
        raise HTTPException(status_code=500, detail="Helix users returned empty data")
    return data[0]

async def is_subscriber(viewer_id: str) -> bool:
    admin_tok = await get_valid_admin_token()
    broadcaster_id = admin_tok.get("broadcaster_id")
    if not broadcaster_id:
        raise HTTPException(status_code=503, detail="Admin token missing broadcaster_id; re-connect admin.")

    async with httpx.AsyncClient(timeout=20) as client:
        r = await client.get(
            TWITCH_HELIX_SUBSCRIPTIONS,
            params={"broadcaster_id": broadcaster_id, "user_id": viewer_id},
            headers={
                "Authorization": f"Bearer {admin_tok['access_token']}",
                "Client-ID": TWITCH_CLIENT_ID,
            },
        )

    if r.status_code == 200:
        return len(r.json().get("data", [])) > 0

    if r.status_code == 401:
        admin_tok = await _refresh_admin_token(admin_tok)
        async with httpx.AsyncClient(timeout=20) as client:
            r2 = await client.get(
                TWITCH_HELIX_SUBSCRIPTIONS,
                params={"broadcaster_id": broadcaster_id, "user_id": viewer_id},
                headers={
                    "Authorization": f"Bearer {admin_tok['access_token']}",
                    "Client-ID": TWITCH_CLIENT_ID,
                },
            )
        if r2.status_code == 200:
            return len(r2.json().get("data", [])) > 0
        raise HTTPException(status_code=500, detail=f"Subscriptions check failed after refresh: {r2.status_code} {r2.text}")

    raise HTTPException(status_code=500, detail=f"Subscriptions check failed: {r.status_code} {r.text}")


# ------------------------
# Pages / OAuth
# ------------------------
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    viewer = request.session.get("viewer")
    admin = _load_admin_token()
    admin_status = "connected" if admin else "not connected"
    if viewer:
        viewer_id = viewer['id']
        return f"""
        <h2>Logged in</h2>
        <p><b>Twitch viewer_id:</b> {viewer_id}</p>
        <p><b>Admin status:</b> {admin_status}</p>
        <p><a href="/voiceclone">Go to /voiceclone</a></p>
        <p><a href="/logout">Logout</a></p>
        """
    return f"""
    <h2>VoiceClone (VPS)</h2>
    <p><b>Admin status:</b> {admin_status}</p>
    <p><a href="/auth/twitch/login">Viewer Login with Twitch</a></p>
    <p>Admin connect: <code>/admin/connect?key=YOUR_ADMIN_KEY</code></p>
    """

@app.get("/auth/twitch/login")
async def viewer_login():
    scope = "user:read:email"
    url = (
        f"{TWITCH_AUTHORIZE_URL}"
        f"?client_id={TWITCH_CLIENT_ID}"
        f"&redirect_uri={VIEWER_REDIRECT_URI}"
        f"&response_type=code"
        f"&scope={scope}"
    )
    return RedirectResponse(url)

@app.get("/auth/twitch/callback")
async def viewer_callback(request: Request):
    code = request.query_params.get("code")
    if not code:
        raise HTTPException(status_code=400, detail="Missing code")

    token = await exchange_code_for_token(code, VIEWER_REDIRECT_URI)
    access_token = token.get("access_token")
    if not access_token:
        raise HTTPException(status_code=500, detail=f"No access_token in viewer token response: {token}")

    user = await get_user_by_token(access_token)
    request.session["viewer"] = {
        "id": user["id"],
        "login": user.get("login"),
        "display_name": user.get("display_name"),
    }
    return RedirectResponse("/")

@app.get("/admin/connect")
async def admin_connect(request: Request):
    if not ADMIN_KEY:
        raise HTTPException(status_code=500, detail="Set ADMIN_KEY in .env first")
    if request.query_params.get("key") != ADMIN_KEY:
        raise HTTPException(status_code=403, detail="Forbidden")

    scope = "channel:read:subscriptions"
    url = (
        f"{TWITCH_AUTHORIZE_URL}"
        f"?client_id={TWITCH_CLIENT_ID}"
        f"&redirect_uri={ADMIN_REDIRECT_URI}"
        f"&response_type=code"
        f"&scope={scope}"
    )
    return RedirectResponse(url)

@app.get("/auth/twitch/admin/callback")
async def admin_callback(request: Request):
    code = request.query_params.get("code")
    if not code:
        raise HTTPException(status_code=400, detail="Missing code")

    token = await exchange_code_for_token(code, ADMIN_REDIRECT_URI)
    access_token = token.get("access_token")
    refresh_token = token.get("refresh_token")
    expires_in = int(token.get("expires_in", 0))
    if not access_token or not refresh_token or not expires_in:
        raise HTTPException(status_code=500, detail=f"Unexpected admin token response: {token}")

    user = await get_user_by_token(access_token)
    store = {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "expires_at": int(time.time()) + expires_in,
        "broadcaster_id": user["id"],
        "broadcaster_login": user.get("login"),
        "broadcaster_name": user.get("display_name"),
    }
    _save_admin_token(store)

    broadcaster_id = user['id']
    return HTMLResponse(
        f"<h3>Admin connected</h3>"
        f"<p>broadcaster_id: <b>{broadcaster_id}</b></p>"
        f"<p><a href='/'>Back home</a></p>"
    )


# ------------------------
# Terms of Service
# ------------------------
_TERMS_FORM_HTML = """
  <form action="/terms/accept" method="post">
    <div class="checkbox-row">
      <input type="checkbox" id="c1" name="c1" onchange="checkboxes()">
      <label for="c1">I confirm that the voice recording I will submit is <strong>my own voice</strong>, and I am not impersonating or submitting another person's voice without their permission.</label>
    </div>
    <div class="checkbox-row">
      <input type="checkbox" id="c2" name="c2" onchange="checkboxes()">
      <label for="c2">I agree that my voice data and greeting phrases may be stored and used by FromCollin to generate audio greetings on this stream, as described above.</label>
    </div>
    <div class="checkbox-row">
      <input type="checkbox" id="c3" name="c3" onchange="checkboxes()">
      <label for="c3">I have read and agree to the EchoSub Terms of Service in full.</label>
    </div>
    <button id="agreeBtn" type="submit" disabled>I Agree - Take Me to Voice Setup</button>
  </form>
  <script>
    function checkboxes() {
      const all = ["c1","c2","c3"].every(function(id) { return document.getElementById(id).checked; });
      document.getElementById("agreeBtn").disabled = !all;
    }
  </script>
"""

_TERMS_AGREED_HTML = "<p class=\"agreed\">You have already agreed. <a href=\"/voiceclone\">Go to voice setup</a></p>"


@app.get("/terms", response_class=HTMLResponse)
async def terms_page(request: Request):
    viewer = request.session.get("viewer")
    consent_path = CONSENT_ROOT / (viewer["id"] + ".json") if viewer else None
    already_agreed = consent_path is not None and consent_path.exists()
    name = viewer["display_name"] if viewer else "there"
    consent_block = _TERMS_AGREED_HTML if already_agreed else _TERMS_FORM_HTML

    return (
        "<!DOCTYPE html>"
        "<html lang=\"en\">"
        "<head>"
        "<meta charset=\"UTF-8\">"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">"
        "<title>EchoSub - Terms of Service</title>"
        "<style>"
        "*, *::before, *::after { box-sizing: border-box; }"
        "body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #0e0e10; color: #efeff1; max-width: 760px; margin: 0 auto; padding: 2rem 1.5rem 4rem; line-height: 1.7; }"
        "h1 { color: #9147ff; font-size: 1.8rem; margin-bottom: 0.25rem; }"
        "h2 { color: #bf94ff; font-size: 1.1rem; margin-top: 2rem; margin-bottom: 0.4rem; border-bottom: 1px solid #2a2a3a; padding-bottom: 0.3rem; }"
        "p, li { color: #c8c8d0; font-size: 0.97rem; }"
        "ul { padding-left: 1.4rem; }"
        "li { margin-bottom: 0.4rem; }"
        ".meta { color: #777; font-size: 0.85rem; margin-bottom: 2rem; }"
        ".card { background: #18181b; border: 1px solid #2a2a3a; border-radius: 10px; padding: 2rem; margin-top: 2.5rem; }"
        ".checkbox-row { display: flex; align-items: flex-start; gap: 0.75rem; margin-bottom: 1.25rem; }"
        ".checkbox-row input[type=checkbox] { margin-top: 4px; width: 18px; height: 18px; accent-color: #9147ff; flex-shrink: 0; cursor: pointer; }"
        ".checkbox-row label { cursor: pointer; font-size: 0.95rem; color: #efeff1; }"
        "button { background: #9147ff; color: white; border: none; padding: 0.75rem 2rem; border-radius: 6px; font-size: 1rem; font-weight: 600; cursor: pointer; width: 100%; margin-top: 0.5rem; transition: background 0.15s; }"
        "button:hover:not(:disabled) { background: #7c2ff0; }"
        "button:disabled { background: #444; cursor: not-allowed; }"
        ".agreed { color: #2ecc71; font-weight: bold; margin-bottom: 1rem; }"
        "a { color: #9147ff; }"
        "</style>"
        "</head>"
        "<body>"
        "<h1>EchoSub Terms of Service</h1>"
        "<p class=\"meta\">Operated by FromCollin &nbsp;&middot;&nbsp; Last updated: June 2025</p>"
        "<h2>1. What This Service Does</h2>"
        "<p>EchoSub is a voice cloning feature available exclusively to active subscribers of this Twitch channel. It records a short sample of your voice and uses it to generate personalized audio greetings that play in the stream when you use the <code>!greet</code> command in chat.</p>"
        "<h2>2. Your Voice Data</h2>"
        "<ul>"
        "<li>By submitting a voice recording, you confirm that it is <strong>your own voice</strong> and that you have the legal right to use and submit it.</li>"
        "<li>You must not submit a recording of another person's voice without their explicit written consent.</li>"
        "<li>Submitting a voice recording that you do not own or do not have permission to use is a violation of these terms and may expose you to legal liability.</li>"
        "</ul>"
        "<h2>3. How Your Data Is Used</h2>"
        "<ul>"
        "<li>Your voice recording and greeting phrases are stored on FromCollin's servers and used solely to generate your personalized <code>!greet</code> audio on this stream.</li>"
        "<li>FromCollin may use anonymized, aggregated data about service usage (not your recordings) to improve EchoSub.</li>"
        "<li>Your voice data will never be sold to third parties.</li>"
        "<li>Generated greetings may be audible to anyone watching the stream at the time they play.</li>"
        "</ul>"
        "<h2>4. Data Retention &amp; Deletion</h2>"
        "<p>Your voice profile and greeting phrases are retained for as long as your subscription is active, or until you request deletion. To request removal of your data, contact FromCollin directly through the Twitch channel.</p>"
        "<h2>5. Subscriber-Only Access</h2>"
        "<p>This service is available only to active subscribers of this Twitch channel. If your subscription lapses, access to the voice portal will be revoked. Your stored voice profile may be retained, in case you resubscribe.</p>"
        "<h2>6. Acceptable Use</h2>"
        "<ul>"
        "<li>You may not submit greeting phrases that contain hate speech, threats, harassment, or content that violates Twitch's Terms of Service.</li>"
        "<li>You may not attempt to circumvent the content filters built into this service.</li>"
        "<li>FromCollin reserves the right to remove your voice profile and revoke access at any time for violations of these terms.</li>"
        "</ul>"
        "<h2>7. No Warranty &amp; Limitation of Liability</h2>"
        "<p>EchoSub is provided as-is. FromCollin makes no guarantees about uptime, audio quality, or continued availability of the service. To the extent permitted by law, FromCollin is not liable for any damages arising from your use of this service.</p>"
        "<h2>8. Changes to These Terms</h2>"
        "<p>These terms may be updated at any time. Continued use of the service after changes constitutes acceptance of the updated terms.</p>"
        "<h2>9. Governing Law</h2>"
        "<p>These terms are governed by the laws of the State of Missouri, United States.</p>"
        "<div class=\"card\">"
        f"<p style=\"margin-top:0;color:#efeff1;font-weight:600;\">Ready to set up your voice greeting, {name}?</p>"
        + consent_block +
        "</div>"
        "</body>"
        "</html>"
    )


@app.post("/terms/accept")
async def terms_accept(request: Request):
    viewer = request.session.get("viewer")
    if not viewer:
        return RedirectResponse("/auth/twitch/login", status_code=303)

    form = await request.form()
    if not all(form.get("c" + str(i)) for i in range(1, 4)):
        raise HTTPException(status_code=400, detail="All checkboxes must be checked.")

    consent_record = {
        "viewer_id": viewer["id"],
        "viewer_login": viewer.get("login"),
        "accepted_at": int(time.time()),
        "ip": request.headers.get("x-forwarded-for", request.client.host if request.client else "unknown"),
        "version": "2025-06",
    }
    consent_path = CONSENT_ROOT / (viewer["id"] + ".json")
    atomic_write_json(consent_path, consent_record)
    request.session["terms_accepted"] = True

    return RedirectResponse("/voiceclone", status_code=303)


# ------------------------
# Voice capture page (includes greetings form)
# ------------------------
@app.get("/voiceclone", response_class=HTMLResponse)
async def voiceclone_page(request: Request):
    viewer = request.session.get("viewer")
    if not viewer:
        return RedirectResponse("/auth/twitch/login")

    # Terms gate — must have agreed to ToS before recording
    consent_path = CONSENT_ROOT / (viewer["id"] + ".json")
    if not consent_path.exists():
        return RedirectResponse("/terms")

    ok = await is_subscriber(viewer["id"])
    if not ok:
        return HTMLResponse("<h3>Subscribers only.</h3>", status_code=403)

    gmin = GREETING_MIN_CHARS
    gmax = GREETING_MAX_CHARS

    return f"""
    <h2>Voice Capture</h2>

    <h3>Optional greetings (0-3)</h3>
    <p>Each greeting must be {gmin}-{gmax} characters. Use good punctuation for best results. Defaults will be used if none provided.</p>

    <div style="max-width:800px;">
      <input id="g1" style="width:100%;" placeholder="Greeting 1 (optional)" maxlength="{gmax}">
      <br><br>
      <input id="g2" style="width:100%;" placeholder="Greeting 2 (optional)" maxlength="{gmax}">
      <br><br>
      <input id="g3" style="width:100%;" placeholder="Greeting 3 (optional)" maxlength="{gmax}">
      <br><br>
      <button id="saveGreetings">Save Greetings</button>
      <p id="greetStatus"></p>
    </div>

    <hr>

    <h3>Record your voice (10s)</h3>
    <p>Read this (or not): <i>"The quick brown fox jumps over the lazy dog. FromCollin voice clone test. This is my normal voice."</i></p>
    <button id="btn">Start Recording (10s)</button>
    <p id="status"></p>
    <audio id="playback" controls></audio>

    <script>
      const g1 = document.getElementById('g1');
      const g2 = document.getElementById('g2');
      const g3 = document.getElementById('g3');
      const saveBtn = document.getElementById('saveGreetings');
      const greetStatus = document.getElementById('greetStatus');

      function clean(s) {{
        return (s || '').trim();
      }}

      saveBtn.onclick = async () => {{
        greetStatus.textContent = "Saving...";
        const arr = [clean(g1.value), clean(g2.value), clean(g3.value)].filter(x => x.length > 0);

        for (const t of arr) {{
          if (t.length < {gmin}) {{
            greetStatus.textContent = "A greeting is too short (min {gmin}).";
            return;
          }}
          if (t.length > {gmax}) {{
            greetStatus.textContent = "A greeting is too long (max {gmax}).";
            return;
          }}
        }}

        const resp = await fetch('/api/greetings', {{
          method: 'POST',
          headers: {{ 'Content-Type': 'application/json' }},
          body: JSON.stringify({{ greetings: arr }})
        }});

        const txt = await resp.text();
        greetStatus.textContent = resp.ok ? "Saved." : ("Save failed: " + txt);
      }};

      const btn = document.getElementById('btn');
      const statusEl = document.getElementById('status');
      const playback = document.getElementById('playback');
      let mediaRecorder;
      let chunks = [];

      btn.onclick = async () => {{
        btn.disabled = true;
        chunks = [];
        statusEl.textContent = 'Requesting microphone...';

        const stream = await navigator.mediaDevices.getUserMedia({{ audio: true }});
        mediaRecorder = new MediaRecorder(stream);

        mediaRecorder.ondataavailable = (e) => {{ if (e.data && e.data.size > 0) chunks.push(e.data); }};

        mediaRecorder.onstop = async () => {{
          stream.getTracks().forEach(t => t.stop());
          const blob = new Blob(chunks, {{ type: mediaRecorder.mimeType }});
          playback.src = URL.createObjectURL(blob);

          statusEl.textContent = 'Uploading...';
          const fd = new FormData();
          fd.append('audio', blob, 'recording.webm');

          const resp = await fetch('/api/recordings', {{ method: 'POST', body: fd }});
          const txt = await resp.text();
          statusEl.textContent = resp.ok ? 'Uploaded OK' : ('Upload failed: ' + txt);
          btn.disabled = false;
        }};

        statusEl.textContent = 'Recording...';
        mediaRecorder.start();
        setTimeout(() => {{ try {{ mediaRecorder.stop(); }} catch (e) {{}} }}, 10000);
      }};
    </script>
    """


# ------------------------
# Raw audio upload
# ------------------------
ALLOWED_AUDIO_TYPES = {"audio/webm", "audio/ogg", "audio/wav", "audio/mpeg"}

@app.post("/api/recordings")
async def upload_recording(request: Request, audio: UploadFile = File(...)):
    viewer = request.session.get("viewer")
    if not viewer:
        raise HTTPException(status_code=401, detail="Not logged in")

    ok = await is_subscriber(viewer["id"])
    if not ok:
        raise HTTPException(status_code=403, detail="Subscribers only")

    # SECURITY: reject non-audio uploads
    if not any(audio.content_type.startswith(t) for t in ALLOWED_AUDIO_TYPES):
        raise HTTPException(status_code=415, detail="Audio files only (webm, ogg, wav, mp3)")

    user_dir = UPLOAD_ROOT / viewer["id"]
    user_dir.mkdir(parents=True, exist_ok=True)
    out_path = user_dir / "latest.webm"

    data = await audio.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty upload")
    if len(data) > 15 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File too large")

    out_path.write_bytes(data)
    return JSONResponse({"ok": True, "viewer_id": viewer["id"], "raw_path": str(out_path)})


# ------------------------
# Greetings save endpoint
# ------------------------
@app.post("/api/greetings")
async def save_greetings(request: Request, payload: GreetingsPayload):
    viewer = request.session.get("viewer")
    if not viewer:
        raise HTTPException(status_code=401, detail="Not logged in")

    ok = await is_subscriber(viewer["id"])
    if not ok:
        raise HTTPException(status_code=403, detail="Subscribers only")

    greetings = payload.greetings or []
    if len(greetings) > GREETING_MAX_COUNT:
        raise HTTPException(status_code=400, detail=f"Too many greetings (max {GREETING_MAX_COUNT})")

    cleaned: list[str] = []
    for g in greetings:
        t = (g or "").strip()
        if not t:
            continue
        if len(t) < GREETING_MIN_CHARS:
            raise HTTPException(status_code=400, detail=f"Greeting too short (min {GREETING_MIN_CHARS})")
        if len(t) > GREETING_MAX_CHARS:
            raise HTTPException(status_code=400, detail=f"Greeting too long (max {GREETING_MAX_CHARS})")
        if not is_allowed_greeting_text(t):
            raise HTTPException(status_code=400, detail="Greeting contains disallowed language")
        cleaned.append(t)

    user_meta_dir = META_ROOT / viewer["id"]
    out_path = user_meta_dir / "greetings.json"

    if len(cleaned) == 0:
        try:
            if out_path.exists():
                out_path.unlink()
        except Exception:
            pass
        return JSONResponse({"ok": True, "viewer_id": viewer["id"], "greetings": [], "note": "Using defaults"})

    atomic_write_json(out_path, {"uuid": viewer["id"], "greetings": cleaned, "updated_at": int(time.time())})
    return JSONResponse({"ok": True, "viewer_id": viewer["id"], "greetings": cleaned})


@app.get("/logout")
async def logout(request: Request):
    request.session.clear()
    return RedirectResponse("/")