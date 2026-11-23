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

    worker = PipelineWorker(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        observers=[],
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
      max-width: 600px;
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

    h1 { font-size: 1.9rem; font-weight: 700; line-height: 1.2; margin-bottom: 0.5rem; color: #fff; }

    .subtitle { color: #8888aa; font-size: 0.95rem; line-height: 1.6; margin-bottom: 2rem; }

    h2 { font-size: 0.72rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.08em; color: #55556a; margin-bottom: 0.9rem; }

    /* ── Steps ── */
    .steps { list-style: none; display: flex; flex-direction: column; gap: 0.75rem; margin-bottom: 2rem; }
    .step { display: flex; gap: 1rem; align-items: flex-start; }
    .step-num {
      flex-shrink: 0; width: 2rem; height: 2rem; border-radius: 50%;
      background: #6c47ff22; border: 1px solid #6c47ff55; color: #9d7bff;
      font-size: 0.85rem; font-weight: 700; display: flex; align-items: center; justify-content: center;
    }
    .step-body { padding-top: 0.25rem; }
    .step-title { font-weight: 600; font-size: 0.95rem; color: #ddddf0; margin-bottom: 0.2rem; }
    .step-desc { font-size: 0.875rem; color: #777799; line-height: 1.5; }

    /* ── UI guide ── */
    .ui-guide {
      background: #12121a;
      border: 1px solid #2e2e42;
      border-radius: 12px;
      padding: 1.2rem 1.4rem;
      margin-bottom: 2rem;
    }
    .ui-mock {
      background: #0d0d14;
      border: 1px solid #2a2a3e;
      border-radius: 10px;
      padding: 1.2rem;
      margin: 0.75rem 0 1rem;
      display: flex;
      flex-direction: column;
      align-items: center;
      gap: 0.9rem;
    }
    .mock-circle {
      width: 72px; height: 72px; border-radius: 50%;
      background: #1e1e30; border: 2px solid #6c47ff55;
      display: flex; align-items: center; justify-content: center;
      font-size: 1.8rem;
    }
    .mock-status { font-size: 0.8rem; color: #55556a; }
    .mock-btn {
      background: #6c47ff; color: #fff; border: none; border-radius: 8px;
      padding: 0.55rem 1.4rem; font-size: 0.875rem; font-weight: 600; cursor: default;
    }
    .mock-controls { display: flex; gap: 0.6rem; }
    .mock-ctrl {
      background: #1e1e30; border: 1px solid #2e2e42; border-radius: 8px;
      padding: 0.4rem 0.9rem; font-size: 0.75rem; color: #7777aa; cursor: default;
    }
    .annotation {
      font-size: 0.8rem; color: #9d7bff; display: flex; align-items: center; gap: 0.4rem;
    }
    .annotation::before { content: "↑"; }

    /* ── Examples ── */
    .examples { background: #12121a; border: 1px solid #2e2e42; border-radius: 10px; padding: 1rem 1.2rem; margin-bottom: 2rem; }
    .examples ul { list-style: none; display: flex; flex-direction: column; gap: 0.4rem; margin-top: 0.65rem; }
    .examples li { font-size: 0.875rem; color: #8888aa; padding-left: 1rem; position: relative; }
    .examples li::before { content: "→"; position: absolute; left: 0; color: #6c47ff88; }

    /* ── Stack chips ── */
    .stack { display: flex; gap: 0.5rem; flex-wrap: wrap; margin-bottom: 2rem; }
    .chip { background: #1e1e2e; border: 1px solid #2e2e42; border-radius: 999px; font-size: 0.75rem; color: #7777aa; padding: 0.25rem 0.75rem; }
    .chip span { color: #9d7bff; font-weight: 600; }

    /* ── CTA ── */
    .btn {
      display: block; width: 100%; padding: 0.85rem; border-radius: 10px; border: none;
      background: #6c47ff; color: #fff; font-size: 1rem; font-weight: 600;
      cursor: pointer; transition: background 0.15s; text-align: center; text-decoration: none;
    }
    .btn:hover { background: #7d5bff; }
    .note { text-align: center; font-size: 0.78rem; color: #44445a; margin-top: 1rem; }
  </style>
</head>
<body>
  <div class="card">
    <div class="badge">Voice Agent Demo</div>
    <h1>WikiBot</h1>
    <p class="subtitle">A real-time voice assistant that answers factual questions by searching Wikipedia live during your conversation.</p>

    <!-- How to use -->
    <h2>How to use</h2>
    <ol class="steps">
      <li class="step">
        <div class="step-num">1</div>
        <div class="step-body">
          <div class="step-title">Click "Start talking" below</div>
          <div class="step-desc">This opens the Pipecat voice interface.</div>
        </div>
      </li>
      <li class="step">
        <div class="step-num">2</div>
        <div class="step-body">
          <div class="step-title">Press the purple Connect button</div>
          <div class="step-desc">It's in the centre of the screen. Your browser will ask for microphone permission — click Allow.</div>
        </div>
      </li>
      <li class="step">
        <div class="step-num">3</div>
        <div class="step-body">
          <div class="step-title">Wait for the greeting (~10 sec first time)</div>
          <div class="step-desc">AI models load on first use. The bot will speak once it's ready.</div>
        </div>
      </li>
      <li class="step">
        <div class="step-num">4</div>
        <div class="step-body">
          <div class="step-title">Just speak — no button to hold</div>
          <div class="step-desc">WikiBot detects when you start and stop talking automatically. Ask any question and it will search Wikipedia and read back the answer.</div>
        </div>
      </li>
      <li class="step">
        <div class="step-num">5</div>
        <div class="step-body">
          <div class="step-title">To end, press Disconnect</div>
          <div class="step-desc">The small button at the bottom of the voice screen.</div>
        </div>
      </li>
    </ol>

    <!-- Visual guide -->
    <h2>What the voice screen looks like</h2>
    <div class="ui-guide">
      <div class="ui-mock">
        <div class="mock-circle">🎙️</div>
        <div class="mock-status">Ready to connect</div>
        <div class="mock-btn">Connect</div>
      </div>
      <div class="annotation">Press Connect to start</div>

      <div class="ui-mock" style="margin-top:1rem">
        <div class="mock-circle" style="border-color:#9d7bff; animation: pulse 1.5s infinite;">🤖</div>
        <div class="mock-status" style="color:#9d7bff;">Bot is speaking…</div>
        <div class="mock-controls">
          <div class="mock-ctrl">🔇 Mute</div>
          <div class="mock-ctrl">✖ Disconnect</div>
        </div>
      </div>
      <div class="annotation">While connected — just speak, no button needed</div>
    </div>

    <!-- Examples -->
    <div class="examples">
      <h2>Try asking</h2>
      <ul>
        <li>"Who is Marie Curie?"</li>
        <li>"What is the Turing machine?"</li>
        <li>"Tell me about the Apollo 11 mission."</li>
        <li>"What is quantum entanglement?"</li>
        <li>"Who invented the telephone?"</li>
      </ul>
    </div>

    <!-- Stack -->
    <div class="stack">
      <div class="chip"><span>STT</span> Whisper (local)</div>
      <div class="chip"><span>LLM</span> Gemini 2.0 Flash</div>
      <div class="chip"><span>TTS</span> Kokoro (local)</div>
      <div class="chip"><span>Tool</span> Wikipedia API</div>
      <div class="chip"><span>Framework</span> Pipecat</div>
    </div>

    <a class="btn" href="/guide">Start talking →</a>
    <p class="note">Audio never leaves your machine except for the Gemini API call.</p>
  </div>

  <style>
    @keyframes pulse {
      0%, 100% { box-shadow: 0 0 0 0 #6c47ff44; }
      50% { box-shadow: 0 0 0 8px #6c47ff00; }
    }
  </style>
</body>
</html>"""

    GUIDE_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>WikiBot</title>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    html, body { width: 100%; height: 100%; overflow: hidden; background: #0f0f13; }

    #pipecat-frame {
      position: fixed; inset: 0; width: 100%; height: 100%;
      border: none; display: block;
    }

    /* full-screen overlay */
    #overlay {
      position: fixed; inset: 0; z-index: 100; pointer-events: none;
    }

    /* SVG mask punches a rectangular hole over the target element */
    #mask-svg {
      position: absolute; inset: 0; width: 100%; height: 100%;
      pointer-events: none;
    }

    /* glowing border around the highlighted element */
    #highlight-box {
      position: absolute;
      border: 2px solid #6c47ff;
      border-radius: 8px;
      box-shadow: 0 0 0 3px #6c47ff44, 0 0 20px #6c47ff66;
      pointer-events: none;
      transition: all 0.35s cubic-bezier(.4,0,.2,1);
      display: none;
    }

    /* tooltip card */
    #tooltip {
      position: absolute;
      background: #1a1a26;
      border: 1px solid #6c47ff88;
      border-radius: 14px;
      padding: 1.2rem 1.4rem 1rem;
      width: 300px;
      box-shadow: 0 12px 40px rgba(0,0,0,0.7);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      color: #e8e8f0;
      pointer-events: all;
      transition: top 0.35s cubic-bezier(.4,0,.2,1), left 0.35s cubic-bezier(.4,0,.2,1);
    }

    /* arrow nub */
    #tooltip::before {
      content: "";
      position: absolute;
      width: 11px; height: 11px;
      background: #1a1a26;
      border-left: 1px solid #6c47ff88;
      border-top:  1px solid #6c47ff88;
      display: none;
    }
    #tooltip.arrow-up::before    { display:block; top:-6px;    left:50%; transform:translateX(-50%) rotate(45deg); }
    #tooltip.arrow-down::before  { display:block; bottom:-6px; left:50%; transform:translateX(-50%) rotate(225deg); }
    #tooltip.arrow-right::before { display:block; right:-6px;  top:50%;  transform:translateY(-50%) rotate(135deg); }
    #tooltip.arrow-left::before  { display:block; left:-6px;   top:50%;  transform:translateY(-50%) rotate(-45deg); }

    .tip-badge {
      display:inline-block; background:#6c47ff22; color:#9d7bff;
      border:1px solid #6c47ff55; border-radius:999px;
      font-size:0.68rem; font-weight:700; letter-spacing:0.08em;
      text-transform:uppercase; padding:0.18rem 0.6rem; margin-bottom:0.6rem;
    }
    .tip-title { font-size:1rem; font-weight:700; color:#fff; margin-bottom:0.4rem; line-height:1.3; }
    .tip-body  { font-size:0.875rem; color:#9999bb; line-height:1.55; margin-bottom:1rem; }
    .tip-actions { display:flex; justify-content:space-between; align-items:center; }
    .tip-dots { display:flex; gap:5px; }
    .tip-dot { width:7px; height:7px; border-radius:50%; background:#2e2e42; }
    .tip-dot.active { background:#6c47ff; }
    .tip-btn {
      background:#6c47ff; color:#fff; border:none; border-radius:8px;
      padding:0.5rem 1.1rem; font-size:0.875rem; font-weight:600; cursor:pointer;
    }
    .tip-btn:hover { background:#7d5bff; }
    .tip-skip { background:transparent; color:#55556a; border:none; cursor:pointer; font-size:0.875rem; padding:0.5rem 0.4rem; }
    .tip-skip:hover { color:#9999bb; }
  </style>
</head>
<body>
  <iframe id="pipecat-frame" src="/client" allow="microphone; camera"></iframe>

  <div id="overlay">
    <!-- SVG darkens everything outside the cutout -->
    <svg id="mask-svg" xmlns="http://www.w3.org/2000/svg">
      <defs>
        <mask id="cutout">
          <!-- white = visible (darkened), black = cut out (clear) -->
          <rect width="100%" height="100%" fill="white"/>
          <rect id="cutout-rect" x="0" y="0" width="0" height="0" rx="8" fill="black"/>
        </mask>
      </defs>
      <rect width="100%" height="100%" fill="rgba(0,0,0,0.70)" mask="url(#cutout)"/>
    </svg>

    <div id="highlight-box"></div>
    <div id="tooltip"></div>
  </div>

  <script>
  (function(){
    const PAD = 6; // px padding around highlighted element

    // Layout constants measured from the actual Pipecat Playground screenshot at 1280x800.
    // All values are fractions of viewport so they scale to any window size.
    // Each region: { x, y, w, h } as fractions of vw/vh.
    const regions = {
      header:       { xf:0,     yf:0,      wf:1,    hf:0.055 },
      connect_btn:  { xf:0.868, yf:0.004,  wf:0.12, hf:0.048 },
      bot_audio:    { xf:0.008, yf:0.057,  wf:0.248,hf:0.355 },
      conversation: { xf:0.265, yf:0.057,  wf:0.465,hf:0.690 },
      info_panel:   { xf:0.734, yf:0.057,  wf:0.258,hf:0.690 },
      events:       { xf:0.008, yf:0.718,  wf:0.984,hf:0.272 },
    };

    const steps = [
      {
        badge: "1 of 6",
        title: "Welcome to WikiBot",
        body: "This is the Pipecat Playground — a developer voice interface. This quick tour shows exactly what each part does and what to click.",
        region: null,
        arrow: null,
        tipAnchor: "center",
      },
      {
        badge: "2 of 6",
        title: "The Connect button",
        body: "Click this green button in the top-right corner to start the voice session. Your browser will then ask for microphone permission — click Allow.",
        region: "connect_btn",
        arrow: "arrow-up",
        tipAnchor: "below",
      },
      {
        badge: "3 of 6",
        title: "Bot Audio panel",
        body: "Once connected, this panel shows the bot's audio waveform so you can see it is speaking. It will animate whenever WikiBot responds.",
        region: "bot_audio",
        arrow: "arrow-right",
        tipAnchor: "right",
      },
      {
        badge: "4 of 6",
        title: "Conversation panel",
        body: "Live transcript appears here as you talk. You will see your words and the bot's reply scroll in real-time. Just speak naturally — no button to hold.",
        region: "conversation",
        arrow: "arrow-right",
        tipAnchor: "right",
      },
      {
        badge: "5 of 6",
        title: "Status & Session info",
        body: "This panel shows the connection state, your microphone, and session details. When connected, Client and Agent will both show 'ready'.",
        region: "info_panel",
        arrow: "arrow-left",
        tipAnchor: "left",
      },
      {
        badge: "6 of 6",
        title: "Events log",
        body: "Raw pipeline events appear here — useful for debugging. To end the session, click the Connect button again (it becomes Disconnect once connected). You are all set!",
        region: "events",
        arrow: "arrow-up",
        tipAnchor: "above",
      },
    ];

    let current = 0;
    const overlay      = document.getElementById("overlay");
    const cutoutRect   = document.getElementById("cutout-rect");
    const highlightBox = document.getElementById("highlight-box");
    const tooltip      = document.getElementById("tooltip");

    function vw(){ return window.innerWidth; }
    function vh(){ return window.innerHeight; }

    function getRect(key){
      const r = regions[key];
      return {
        x: r.xf * vw(),
        y: r.yf * vh(),
        w: r.wf * vw(),
        h: r.hf * vh(),
      };
    }

    function render(idx){
      if(idx >= steps.length){ dismiss(); return; }
      const s = steps[idx];

      // --- cutout + highlight ---
      if(s.region){
        const r = getRect(s.region);
        const x = r.x - PAD, y = r.y - PAD, w = r.w + PAD*2, h = r.h + PAD*2;
        cutoutRect.setAttribute("x", x);
        cutoutRect.setAttribute("y", y);
        cutoutRect.setAttribute("width", w);
        cutoutRect.setAttribute("height", h);
        highlightBox.style.cssText = `display:block; left:${x}px; top:${y}px; width:${w}px; height:${h}px;`;
      } else {
        cutoutRect.setAttribute("width", 0);
        cutoutRect.setAttribute("height", 0);
        highlightBox.style.display = "none";
      }

      // --- tooltip position ---
      const TW = 300, margin = 12;
      let tx = 0, ty = 0;

      if(!s.region || s.tipAnchor === "center"){
        tx = vw()/2 - TW/2;
        ty = vh()/2 - 100;
      } else {
        const r = getRect(s.region);
        if(s.tipAnchor === "below"){
          tx = r.x + r.w/2 - TW/2;
          ty = r.y + r.h + PAD + margin;
        } else if(s.tipAnchor === "above"){
          tx = r.x + r.w/2 - TW/2;
          ty = r.y - PAD - margin - 180; // approx card height
        } else if(s.tipAnchor === "right"){
          tx = r.x + r.w + PAD + margin;
          ty = r.y + r.h/2 - 90;
        } else if(s.tipAnchor === "left"){
          tx = r.x - PAD - margin - TW;
          ty = r.y + r.h/2 - 90;
        }
      }

      // clamp inside viewport
      tx = Math.max(margin, Math.min(tx, vw() - TW - margin));
      ty = Math.max(margin, Math.min(ty, vh() - 220));

      tooltip.style.left = tx + "px";
      tooltip.style.top  = ty + "px";
      tooltip.className  = s.arrow || "";

      // --- dots ---
      const dots = steps.map((_,i) =>
        `<div class="tip-dot ${i===idx?"active":""}"></div>`
      ).join("");

      const isLast = idx === steps.length - 1;
      tooltip.innerHTML = `
        <div class="tip-badge">Step ${s.badge}</div>
        <div class="tip-title">${s.title}</div>
        <div class="tip-body">${s.body}</div>
        <div class="tip-actions">
          <div class="tip-dots">${dots}</div>
          <div style="display:flex;gap:6px;align-items:center">
            ${!isLast ? '<button class="tip-skip" id="skip-btn">Skip tour</button>' : ""}
            <button class="tip-btn" id="next-btn">${isLast ? "Done ✓" : "Next →"}</button>
          </div>
        </div>`;

      document.getElementById("next-btn").onclick = () => { current++; render(current); };
      const sb = document.getElementById("skip-btn");
      if(sb) sb.onclick = dismiss;
    }

    function dismiss(){
      overlay.style.transition = "opacity 0.4s";
      overlay.style.opacity = "0";
      setTimeout(() => overlay.style.display = "none", 400);
    }

    setTimeout(() => render(0), 900);
    window.addEventListener("resize", () => render(current));
  })();
  </script>
</body>
</html>"""

    @app.get("/", response_class=HTMLResponse)
    async def tutorial():
        return TUTORIAL_HTML

    @app.get("/guide", response_class=HTMLResponse)
    async def guide():
        return GUIDE_HTML

    main()
