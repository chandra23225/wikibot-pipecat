#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""WikiBot - A real-time voice agent powered by Pipecat

Pipeline: Speech-to-Text → LLM (with Wikipedia tool) → Text-to-Speech

Services:
- Whisper  (STT, local — no API key needed)
- Google Gemini (LLM)
- Kokoro   (TTS, local — no API key needed)

Run with::

    uv run bot.py -t webrtc
"""

import os

import wikipediaapi
from dotenv import load_dotenv
from loguru import logger
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.observers.user_bot_latency_observer import UserBotLatencyObserver
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.google.llm import GoogleLLMService
from pipecat.services.kokoro.tts import KokoroTTSService
from pipecat.services.llm_service import FunctionCallParams
from pipecat.services.whisper.stt import Model, WhisperSTTService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams
from pipecat.workers.runner import WorkerRunner

load_dotenv(override=True)

# ---------------------------------------------------------------------------
# Wikipedia tool
# ---------------------------------------------------------------------------

WIKI_TOOL = {
    "function_declarations": [
        {
            "name": "search_wikipedia",
            "description": (
                "Look up a topic on Wikipedia and return a short summary. "
                "Use this whenever the user asks 'what is', 'who is', 'tell me about', "
                "or any factual question about a person, place, event, or concept."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The topic to search for on Wikipedia.",
                    }
                },
                "required": ["query"],
            },
        }
    ]
}

wiki = wikipediaapi.Wikipedia(user_agent="WikiBot/1.0 (pipecat voice agent)", language="en")


async def search_wikipedia(params: FunctionCallParams) -> None:
    query: str = params.arguments.get("query", "")
    logger.info(f"Wikipedia lookup: {query!r}")

    page = wiki.page(query)

    if page.exists():
        # First 3 sentences is enough for voice — keep it short
        summary = ". ".join(page.summary.split(". ")[:3]) + "."
        result = summary
    else:
        result = f"Sorry, I couldn't find a Wikipedia article for '{query}'."

    logger.info(f"Wikipedia result: {result[:120]}...")
    await params.result_callback(result)


# ---------------------------------------------------------------------------
# Bot
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are WikiBot, a friendly and knowledgeable voice assistant.
Your specialty is answering factual questions by looking things up on Wikipedia.
When a user asks about a person, place, event, or concept, use the search_wikipedia tool.
Your responses will be spoken aloud, so keep them concise — two to three sentences at most.
Never use bullet points, markdown, or emojis.
"""


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments) -> None:
    logger.info("Starting WikiBot")

    # Speech-to-Text (local — no API key needed)
    stt = WhisperSTTService(
        settings=WhisperSTTService.Settings(model=Model.BASE),
        device="cpu",
    )

    # Text-to-Speech (local — no API key needed)
    tts = KokoroTTSService(
        settings=KokoroTTSService.Settings(voice="af_heart"),
    )

    # LLM with Wikipedia tool
    llm = GoogleLLMService(
        api_key=os.getenv("GOOGLE_API_KEY"),
        model=os.getenv("GOOGLE_MODEL", "gemini-2.0-flash-001"),
        system_instruction=SYSTEM_PROMPT,
        tools=[WIKI_TOOL],
    )

    # Register the Wikipedia function handler
    llm.register_function("search_wikipedia", search_wikipedia)

    context = LLMContext()
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer()),
    )

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            user_aggregator,
            llm,
            tts,
            transport.output(),
            assistant_aggregator,
        ]
    )

    latency_observer = UserBotLatencyObserver()

    @latency_observer.event_handler("on_latency_measured")
    async def on_latency_measured(observer, latency_seconds):
        logger.info(f"[LATENCY] Voice-to-voice: {latency_seconds*1000:.0f}ms")

    @latency_observer.event_handler("on_latency_breakdown")
    async def on_latency_breakdown(observer, breakdown):
        for line in breakdown.turn_contribution_lines():
            logger.info(f"[LATENCY BREAKDOWN] {line}")

    @latency_observer.event_handler("on_first_bot_speech_latency")
    async def on_first_bot_speech(observer, latency_seconds):
        logger.info(f"[LATENCY] First bot speech (connect→voice): {latency_seconds*1000:.0f}ms")

    worker = PipelineWorker(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        observers=[latency_observer],
    )

    runner = WorkerRunner(handle_sigint=runner_args.handle_sigint)
    await runner.add_workers(worker)

    @worker.rtvi.event_handler("on_client_ready")
    async def on_client_ready(rtvi):
        context.add_message(
            {
                "role": "developer",
                "content": (
                    "Greet the user briefly and let them know they can ask you "
                    "about any topic and you'll look it up on Wikipedia."
                ),
            }
        )
        await worker.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Client connected")

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected")
        await runner.cancel()

    await runner.run()


async def bot(runner_args: RunnerArguments):
    transport_params = {
        "daily": lambda: DailyParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
        ),
        "webrtc": lambda: TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
        ),
    }

    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from fastapi.responses import HTMLResponse
    from pipecat.runner.run import app, main

    TUTORIAL_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>WikiBot</title>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      background: #0f0f13;
      color: #e8e8f0;
      min-height: 100vh;
      display: flex;
      align-items: center;
      justify-content: center;
      padding: 1rem;
    }

    .card {
      background: #1a1a24;
      border: 1px solid #2e2e42;
      border-radius: 16px;
      max-width: 560px;
      width: 100%;
      padding: 2.5rem 2.5rem 2rem;
      box-shadow: 0 24px 60px rgba(0,0,0,0.5);
    }

    .badge {
      display: inline-block;
      background: #6c47ff22;
      color: #9d7bff;
      border: 1px solid #6c47ff55;
      border-radius: 999px;
      font-size: 0.72rem;
      font-weight: 600;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      padding: 0.25rem 0.75rem;
      margin-bottom: 1.2rem;
    }

    h1 {
      font-size: 1.9rem;
      font-weight: 700;
      line-height: 1.2;
      margin-bottom: 0.6rem;
      color: #fff;
    }

    .subtitle {
      color: #8888aa;
      font-size: 0.95rem;
      line-height: 1.6;
      margin-bottom: 2rem;
    }

    .steps { list-style: none; display: flex; flex-direction: column; gap: 1rem; margin-bottom: 2rem; }

    .step {
      display: flex;
      gap: 1rem;
      align-items: flex-start;
    }

    .step-num {
      flex-shrink: 0;
      width: 2rem;
      height: 2rem;
      border-radius: 50%;
      background: #6c47ff22;
      border: 1px solid #6c47ff55;
      color: #9d7bff;
      font-size: 0.85rem;
      font-weight: 700;
      display: flex;
      align-items: center;
      justify-content: center;
    }

    .step-body { padding-top: 0.2rem; }
    .step-title { font-weight: 600; font-size: 0.95rem; color: #ddddf0; margin-bottom: 0.2rem; }
    .step-desc { font-size: 0.875rem; color: #777799; line-height: 1.5; }

    .examples {
      background: #12121a;
      border: 1px solid #2e2e42;
      border-radius: 10px;
      padding: 1rem 1.2rem;
      margin-bottom: 2rem;
    }

    .examples-label {
      font-size: 0.72rem;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: #55556a;
      margin-bottom: 0.65rem;
    }

    .examples ul { list-style: none; display: flex; flex-direction: column; gap: 0.4rem; }

    .examples li {
      font-size: 0.875rem;
      color: #8888aa;
      padding-left: 1rem;
      position: relative;
    }

    .examples li::before {
      content: "→";
      position: absolute;
      left: 0;
      color: #6c47ff88;
    }

    .stack {
      display: flex;
      gap: 0.5rem;
      flex-wrap: wrap;
      margin-bottom: 2rem;
    }

    .chip {
      background: #1e1e2e;
      border: 1px solid #2e2e42;
      border-radius: 999px;
      font-size: 0.75rem;
      color: #7777aa;
      padding: 0.25rem 0.75rem;
    }

    .chip span { color: #9d7bff; font-weight: 600; }

    .btn {
      display: block;
      width: 100%;
      padding: 0.85rem;
      border-radius: 10px;
      border: none;
      background: #6c47ff;
      color: #fff;
      font-size: 1rem;
      font-weight: 600;
      cursor: pointer;
      transition: background 0.15s;
      text-align: center;
      text-decoration: none;
    }

    .btn:hover { background: #7d5bff; }

    .note {
      text-align: center;
      font-size: 0.78rem;
      color: #44445a;
      margin-top: 1rem;
    }
  </style>
</head>
<body>
  <div class="card">
    <div class="badge">Voice Agent Demo</div>
    <h1>WikiBot</h1>
    <p class="subtitle">
      A real-time voice assistant that answers factual questions by looking
      them up on Wikipedia — live, during your conversation.
    </p>

    <ol class="steps">
      <li class="step">
        <div class="step-num">1</div>
        <div class="step-body">
          <div class="step-title">Click Start, then Allow Microphone</div>
          <div class="step-desc">Your browser will ask for mic permission. WikiBot needs it to hear you.</div>
        </div>
      </li>
      <li class="step">
        <div class="step-num">2</div>
        <div class="step-body">
          <div class="step-title">Wait for the greeting</div>
          <div class="step-desc">The first connection takes a few seconds while local AI models load. You'll hear a welcome message when it's ready.</div>
        </div>
      </li>
      <li class="step">
        <div class="step-num">3</div>
        <div class="step-body">
          <div class="step-title">Ask anything</div>
          <div class="step-desc">Speak naturally. WikiBot will search Wikipedia and read back a summary.</div>
        </div>
      </li>
    </ol>

    <div class="examples">
      <div class="examples-label">Try asking</div>
      <ul>
        <li>"Who is Marie Curie?"</li>
        <li>"What is the Turing machine?"</li>
        <li>"Tell me about the Apollo 11 mission."</li>
        <li>"What is quantum entanglement?"</li>
      </ul>
    </div>

    <div class="stack">
      <div class="chip"><span>STT</span> Whisper (local)</div>
      <div class="chip"><span>LLM</span> Gemini 2.0 Flash</div>
      <div class="chip"><span>TTS</span> Kokoro (local)</div>
      <div class="chip"><span>Tool</span> Wikipedia API</div>
      <div class="chip"><span>Framework</span> Pipecat</div>
    </div>

    <a class="btn" href="/client">Start talking →</a>
    <p class="note">Audio never leaves your machine except for the Gemini API call.</p>
  </div>
</body>
</html>"""

    @app.get("/", response_class=HTMLResponse)
    async def tutorial():
        return TUTORIAL_HTML

    main()
