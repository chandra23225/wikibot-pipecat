---
title: WikiBot
emoji: 🎙️
colorFrom: purple
colorTo: indigo
sdk: docker
pinned: false
app_port: 7860
---

# WikiBot — Real-time Voice Agent

A real-time voice assistant powered by [Pipecat](https://pipecat.ai) that answers
factual questions by searching Wikipedia live during your conversation.

## Pipeline

```
Your mic → Whisper STT → Gemini 2.0 Flash → Kokoro TTS → Your speaker
                               ↕
                        Wikipedia API (tool call)
```

## How to use

1. Click **Connect** in the top-right corner
2. Allow microphone access when prompted
3. Wait for the greeting (~10 seconds while models load)
4. Speak naturally — no button to hold
5. Try: *"Who is Ada Lovelace?"* or *"What is quantum entanglement?"*

## Stack

| Layer | Service |
|---|---|
| STT | Whisper `base` (local, CPU) |
| LLM | Google Gemini 2.0 Flash |
| TTS | Kokoro ONNX (local, CPU) |
| Tool | Wikipedia API |
| Transport | Daily WebRTC |
| Framework | Pipecat 1.16 |

## Environment variables (set as Space secrets)

| Variable | Required |
|---|---|
| `GOOGLE_API_KEY` | Yes — [get one free](https://aistudio.google.com/apikey) |
| `DAILY_API_KEY` | Yes — [get one free](https://dashboard.daily.co/register) |
