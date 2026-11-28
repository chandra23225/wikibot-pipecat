# Real-time Conversational Voice Agent

A low-latency conversational voice agent with real-time audio streaming, turn detection, and live external knowledge retrieval — built with [Pipecat](https://pipecat.ai).

The agent listens continuously, detects when you stop speaking, transcribes, queries an LLM with tool calling, synthesises speech, and responds — all in a streaming pipeline with sub-second component latency.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Client (Browser)                         │
│                    WebRTC audio stream                          │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│                   Pipecat Pipeline (server)                     │
│                                                                 │
│  ┌─────────┐   ┌──────────┐   ┌──────────────────────────────┐ │
│  │Transport│──▶│  Silero  │──▶│    Whisper STT (base)        │ │
│  │  Input  │   │  VAD     │   │    local · CPU · ~145MB      │ │
│  └─────────┘   └──────────┘   └──────────────┬───────────────┘ │
│                                               │ transcript      │
│                               ┌───────────────▼───────────────┐ │
│                               │   Gemini 2.0 Flash (LLM)      │ │
│                               │   function calling enabled    │ │
│                               └───────────────┬───────────────┘ │
│                                               │                 │
│                     ┌─────────────────────────▼──────────────┐ │
│                     │        Wikipedia Tool Call              │ │
│                     │   wikipedia-api · live lookup          │ │
│                     └─────────────────────────┬──────────────┘ │
│                                               │ result injected │
│                               ┌───────────────▼───────────────┐ │
│                               │   Gemini (continues response) │ │
│                               └───────────────┬───────────────┘ │
│                                               │ text stream     │
│                               ┌───────────────▼───────────────┐ │
│                               │   Kokoro TTS (af_heart voice) │ │
│                               │   local · CPU · ~80MB ONNX    │ │
│                               └───────────────┬───────────────┘ │
│                                               │ audio stream    │
│  ┌─────────┐                 ┌───────────────▼───────────────┐ │
│  │Transport│◀────────────────│   Transport Output            │ │
│  │ Output  │                 └───────────────────────────────┘ │
│  └─────────┘                                                   │
└─────────────────────────────────────────────────────────────────┘
```

**Key pipeline properties:**
- Frames flow downstream continuously — TTS starts speaking before the LLM finishes generating
- VAD (Silero) detects speech boundaries — no push-to-talk required
- Interruption handling — user speech during bot response cancels in-flight TTS frames
- Latency is logged per turn: VAD endpointing → STT → LLM TTFB → TTS TTFB → first audio

## Performance

Latency is measured end-to-end (user stops speaking → bot starts speaking) using Pipecat's `UserBotLatencyObserver`. Each turn logs a full breakdown:

```
[LATENCY BREAKDOWN]  0.3s  VAD endpointing
[LATENCY BREAKDOWN]  0.4s  Whisper STT
[LATENCY BREAKDOWN]  0.5s  Gemini LLM (TTFB)
[LATENCY BREAKDOWN]  0.2s  Kokoro TTS (TTFB)
[LATENCY BREAKDOWN]  1.4s  TOTAL (voice-to-voice)
[LATENCY] Wikipedia tool call: ~200ms avg
```

> Numbers measured locally on CPU. Wikipedia tool calls add ~150–300ms on top of base LLM latency.

## Running Locally

**1. Install dependencies**
```bash
cd server
uv sync
```

**2. Set environment variables**
```bash
cp .env.example .env
# GOOGLE_API_KEY = your Gemini key (https://aistudio.google.com/apikey)
```

**3. Run**
```bash
uv run bot.py -t webrtc
```

**4. Open** `http://localhost:7860`

The tutorial page loads first with a 6-step interactive guide overlay explaining the UI. On first run, Whisper (~145 MB) and Kokoro (~80 MB) download automatically to `~/.cache/`.

## Tech Stack

| Layer | Technology |
|---|---|
| Framework | [Pipecat](https://pipecat.ai) 1.16 — streaming pipeline, VAD, WebRTC |
| STT | [Faster Whisper](https://github.com/SYSTRAN/faster-whisper) `base` — local, CPU |
| LLM | Google Gemini 2.0 Flash — function calling |
| TTS | [Kokoro ONNX](https://github.com/thewh1teagle/kokoro-onnx) `af_heart` — local, CPU |
| Tool | [Wikipedia API](https://pypi.org/project/Wikipedia-API/) — live retrieval |
| Transport | Daily WebRTC / SmallWebRTC |
| Observability | `UserBotLatencyObserver` — per-turn latency breakdown |

## Project Structure

```
pipecat-quickstart/
├── server/
│   ├── bot.py           # Pipeline, Wikipedia tool, latency observer, onboarding UI
│   ├── pyproject.toml   # uv dependencies
│   ├── .env.example     # Required environment variables
│   ├── Dockerfile       # Container for deployment
│   ├── .dockerignore
│   └── render.yaml      # Render free-tier deployment config
├── hf-space/
│   ├── app.py           # Gradio landing page — creates Daily room, links to bot
│   ├── requirements.txt
│   └── README.md        # HF Space metadata
└── README.md
```

## Deployment

The bot runs fully locally out of the box. Cloud deployment infrastructure is configured and ready:

- `render.yaml` + `server/Dockerfile` — Docker deployment on Render free tier
- `hf-space/` — Gradio landing page on Hugging Face Spaces (creates Daily room → links user to bot on Render)
- Requires: `GOOGLE_API_KEY` and `DAILY_API_KEY` as environment secrets

**To deploy on Render:**
1. Connect `chandra23225/wikibot-pipecat` as a Docker Web Service
2. Dockerfile path: `./server/Dockerfile`, build context: `./server`
3. Add env vars: `GOOGLE_API_KEY`, `DAILY_API_KEY`, `GOOGLE_MODEL=gemini-2.0-flash-001`

## What This Demonstrates

- **Agentic framework usage** — Pipecat pipeline with VAD, streaming STT→LLM→TTS, interruption handling
- **LLM function calling in a live voice loop** — Gemini tool use wired into a real-time conversation
- **Local model inference** — Whisper and Kokoro on CPU, no external STT/TTS API needed
- **Observability** — per-turn latency breakdown logged via `UserBotLatencyObserver`
- **Reproducible environments** — `uv`, `pyproject.toml`, `uv.lock`, `Dockerfile`, `.env.example`
- **End-to-end deployment thinking** — Render + HF Spaces + Daily WebRTC
