"""
WikiBot — Hugging Face Space landing page

Creates a Daily room via the Daily API, then sends the user to the
Pipecat Playground pointed at that room on the Render bot server.
"""

import os
import requests
import gradio as gr

DAILY_API_KEY = os.environ.get("DAILY_API_KEY", "")
BOT_SERVER_URL = os.environ.get("BOT_SERVER_URL", "")  # your Render URL e.g. https://wikibot.onrender.com


def create_room_and_launch():
    """Create a Daily room and return the join URL."""
    if not DAILY_API_KEY:
        return None, "❌ DAILY_API_KEY secret is not set in this Space."
    if not BOT_SERVER_URL:
        return None, "❌ BOT_SERVER_URL secret is not set in this Space."

    # Create a Daily room
    try:
        resp = requests.post(
            "https://api.daily.co/v1/rooms",
            headers={"Authorization": f"Bearer {DAILY_API_KEY}"},
            json={"properties": {"exp": 3600, "enable_chat": False}},
            timeout=10,
        )
        resp.raise_for_status()
        room = resp.json()
        room_url = room["url"]
    except Exception as e:
        return None, f"❌ Failed to create Daily room: {e}"

    # Kick off the bot on Render by calling /start
    try:
        requests.post(
            f"{BOT_SERVER_URL}/start",
            json={"transport": "daily", "room_url": room_url},
            timeout=15,
        )
    except Exception as e:
        return None, f"❌ Could not reach bot server: {e}. Make sure it is deployed on Render."

    return room_url, None


EXAMPLES = [
    "Who is Ada Lovelace?",
    "What is the Turing machine?",
    "Tell me about the Apollo 11 mission.",
    "What is quantum entanglement?",
    "Who invented the telephone?",
]

STACK_HTML = """
<div style="display:flex;gap:8px;flex-wrap:wrap;margin-top:8px">
  <span style="background:#1e1e2e;border:1px solid #2e2e42;border-radius:999px;font-size:0.75rem;color:#9d7bff;padding:3px 12px">STT: Whisper (local)</span>
  <span style="background:#1e1e2e;border:1px solid #2e2e42;border-radius:999px;font-size:0.75rem;color:#9d7bff;padding:3px 12px">LLM: Gemini 2.0 Flash</span>
  <span style="background:#1e1e2e;border:1px solid #2e2e42;border-radius:999px;font-size:0.75rem;color:#9d7bff;padding:3px 12px">TTS: Kokoro (local)</span>
  <span style="background:#1e1e2e;border:1px solid #2e2e42;border-radius:999px;font-size:0.75rem;color:#9d7bff;padding:3px 12px">Tool: Wikipedia</span>
  <span style="background:#1e1e2e;border:1px solid #2e2e42;border-radius:999px;font-size:0.75rem;color:#9d7bff;padding:3px 12px">Framework: Pipecat</span>
</div>
"""

with gr.Blocks(
    title="WikiBot",
    theme=gr.themes.Base(primary_hue="purple"),
    css="""
    .launch-btn { font-size: 1.1rem !important; padding: 12px 32px !important; }
    .room-link a { font-size: 1rem; color: #9d7bff; font-weight: 600; }
    """,
) as demo:

    gr.Markdown("""
# 🎙️ WikiBot
### Real-time voice agent powered by Pipecat

Ask me anything — I'll search Wikipedia and speak the answer back to you in real time.
""")

    gr.HTML(STACK_HTML)

    gr.Markdown("""
---
### How to use
1. Click **Launch WikiBot** below
2. A voice room link will appear — click it to open
3. Press **Connect** in the top-right corner of the room
4. Allow microphone access when prompted
5. Wait for the greeting, then just speak

### Try asking
""" + "\n".join(f"- *{e}*" for e in EXAMPLES))

    launch_btn = gr.Button("🚀 Launch WikiBot", variant="primary", elem_classes=["launch-btn"])
    status_box = gr.Markdown(visible=False)
    room_link  = gr.Markdown(visible=False, elem_classes=["room-link"])

    def on_launch():
        url, err = create_room_and_launch()
        if err:
            return (
                gr.update(value=err, visible=True),
                gr.update(visible=False),
            )
        return (
            gr.update(value="✅ Room created! Click the link below to join:", visible=True),
            gr.update(value=f"👉 [Open WikiBot voice room]({url})", visible=True),
        )

    launch_btn.click(on_launch, outputs=[status_box, room_link])

    gr.Markdown("""
---
*Audio is processed on the bot server (Render). Only the Gemini API call leaves the server.*
""")

demo.launch()
