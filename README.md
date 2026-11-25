# WikiBot — Real-time Voice Agent

A real-time voice assistant built with [Pipecat](https://pipecat.ai) that answers factual questions by searching Wikipedia live during conversation.

## Demo

> Ask *"Who is Ada Lovelace?"* or *"What is quantum entanglement?"* — WikiBot searches Wikipedia and speaks back a summary in real time.

## Pipeline

```
Mic → Whisper STT → Gemini 2.0 Flash (+ Wikipedia tool) → Kokoro TTS → Speaker
```

| Layer | Technology | Notes |
|---|---|---|
| STT | [Faster Whisper](https://github.com/SYSTRAN/faster-whisper) `base` | Runs locally, no API key |
| LLM | Google Gemini 2.0 Flash | Function calling for Wikipedia lookup |
| TTS | [Kokoro ONNX](https://github.com/thewh1teagle/kokoro-onnx) | Runs locally, no API key |
| Tool | [Wikipedia API](https://pypi.org/project/Wikipedia-API/) | Live lookup mid-conversation |
| Transport | Daily WebRTC / SmallWebRTC | Low-latency audio streaming |
| Framework | [Pipecat](https://pipecat.ai) 1.16 | Streaming pipeline + VAD |

## Features

- **Voice activity detection** — knows when you start and stop talking, no button to hold
- **Function calling** — Gemini calls the Wikipedia tool mid-conversation and speaks the result
- **Streaming pipeline** — TTS starts speaking before the LLM finishes generating
- **Guided onboarding** — tutorial page + 6-step interactive overlay explaining the UI
- **Local STT + TTS** — Whisper and Kokoro run on-device; only the Gemini call hits the network

## Project Structure

```
pipecat-quickstart/
├── server/
│   ├── bot.py           # Pipeline, Wikipedia tool, tutorial + guide UI
│   ├── pyproject.toml   # Dependencies (uv)
│   ├── .env.example     # Required environment variables
│   ├── Dockerfile       # Container for deployment
│   └── render.yaml      # Render free-tier deployment config
└── README.md
```

## Running Locally

**1. Install dependencies**
```bash
cd server
uv sync
```

**2. Set environment variables**
```bash
cp .env.example .env
# Add your keys:
# GOOGLE_API_KEY = your Gemini key (https://aistudio.google.com/apikey)
# DAILY_API_KEY  = your Daily key  (https://dashboard.daily.co) — only needed for hosted deploy
```

**3. Run**
```bash
uv run bot.py -t webrtc
```

**4. Open** `http://localhost:7860` — the tutorial page loads first, then guides you into the voice UI.

> First run downloads Whisper base (~145 MB) and Kokoro (~80 MB) to `~/.cache/`. Subsequent starts are instant.

## Deployment

### Render (free tier)
1. Connect `chandra23225/wikibot-pipecat` to a Render Web Service
2. Root directory: `server`
3. Build: `pip install uv && uv pip install --system "pipecat-ai[daily,google,kokoro,runner,silero,whisper]>=1.12.0" "wikipedia-api>=0.16.0" "python-dotenv"`
4. Start: `python bot.py -t daily --host 0.0.0.0 --port $PORT`
5. Set env vars: `GOOGLE_API_KEY`, `DAILY_API_KEY`

### Hugging Face Spaces (landing page)
See `hf-space/` — a Gradio Space that creates a Daily room and links users to the bot on Render.

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `GOOGLE_API_KEY` | Yes | Gemini API key — [get free](https://aistudio.google.com/apikey) |
| `DAILY_API_KEY` | For hosted deploy | Daily WebRTC API key — [get free](https://dashboard.daily.co/register) |
| `GOOGLE_MODEL` | No | Defaults to `gemini-2.0-flash-001` |

## What This Demonstrates

- **Pipecat framework** — cascade pipeline (STT → LLM → TTS), VAD, streaming audio over WebRTC
- **LLM function calling** — Gemini tool use wired into a live voice conversation
- **Local model inference** — Whisper and Kokoro running on CPU with no external dependencies
- **Full-stack AI deployment** — bot server, WebRTC transport, custom onboarding UI

## Tech Stack

`Python` `Pipecat` `Google Gemini` `Whisper` `Kokoro` `Daily WebRTC` `FastAPI` `Wikipedia API`
