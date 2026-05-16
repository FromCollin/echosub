# EchoSub

> Twitch subscribers greet chat in their own cloned voice — live, on stream.

When a subscriber types `!greet`, EchoSub synthesizes a greeting in their voice and plays it on stream in real time. Not a soundboard. Not a pre-recorded clip. Their actual voice, saying whatever they wrote, generated on the fly.

---

## What Makes It Interesting

Most voice cloning demos are offline — you upload audio, wait, download a file. EchoSub does it live, triggered by a chat command, in front of an audience. That required solving a few non-trivial problems at once:

- **Streaming inference** — audio chunks are sent over WebSocket as they're generated so playback starts before synthesis is finished
- **Split infrastructure** — a cheap VPS handles the public-facing portal; a local RTX 3090 runs the model and pulls what it needs via rsync on a timer. The GPU never touches the public internet
- **Subscriber onboarding in-browser** — voice is recorded directly in the browser via MediaRecorder, no app install required, gated behind Twitch OAuth + subscription check
- **Content moderation** — greeting text is filtered with leetspeak normalization before saving, so `h3ll0` doesn't slip through a naive word list

---

## How It Works

```
Subscriber types !greet in chat
        ↓
TwitchIO bot receives the command
        ↓
Has a voice profile?
   ├── No  → sends portal link to chat
   └── Yes → picks a random greeting phrase
                ↓
             WebSocket to voice API on 3090
                ↓
             VibeVoice 1.5B generates audio (streamed in chunks)
                ↓
             Plays live on stream
```

---

## Stack

| Layer | Tech |
|---|---|
| Voice model | Microsoft VibeVoice 1.5B on RTX 3090 |
| Voice API | FastAPI + WebSocket streaming |
| Twitch bot | TwitchIO |
| Subscriber portal | FastAPI + Twitch OAuth |
| Infrastructure | VPS (web) + local GPU machine (inference) |
| Sync | rsync over SSH on a timer |

---

## Project Structure

```
echosub/
├── api/              # Voice synthesis service — VibeVoice + WebSocket streaming
├── twitch_bot/       # TwitchIO bot — watches chat, handles !greet
└── web/              # Subscriber portal — Twitch OAuth, voice recording, greetings
```

---

## License

MIT
