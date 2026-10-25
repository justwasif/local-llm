"""Shaktimaan.ai — Premium multilingual assistant powered by an SGLang endpoint."""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import gradio as gr
from openai import APIConnectionError, APIError, OpenAI

SGLANG_URL = os.getenv("SGLANG_BASE_URL", "http://localhost:30000/v1")
MODEL_NAME = os.getenv("SGLANG_MODEL", "local-llm")
LANGUAGES = [
    "Hindi",
    "Bengali",
    "Tamil",
    "Telugu",
    "Marathi",
    "Gujarati",
    "Kannada",
    "Malayalam",
    "Punjabi",
    "Odia",
    "Hinglish",
    "Banglish",
]
DOMAINS = ["Education", "Healthcare", "Agriculture", "Government Services"]
EXAMPLES = [
    [
        "Fasal ke patton par daag aa rahe hain; pehla safe step kya hona chahiye?",
        "Hinglish",
        "Agriculture",
    ],
    ["ডেঙ্গুর সতর্ক লক্ষণগুলি কী?", "Bengali", "Healthcare"],
    ["प्रकाश संश्लेषण को सरल उदाहरण के साथ समझाइए।", "Hindi", "Education"],
    [
        "ரேஷன் கார்டுக்கு விண்ணப்பிக்கும் முன் எந்த அதிகாரப்பூர்வ தகவலை சரிபார்க்க வேண்டும்?",
        "Tamil",
        "Government Services",
    ],
]
CSS = Path(__file__).with_name("ui.css").read_text(encoding="utf-8")

# Sidebar HTML
SIDEBAR_HTML = """
<div class="brand-lockup">
  <div class="brand-orb">✦</div>
  <div class="brand-text">
    <span class="brand-name">Shaktimaan.ai</span>
    <span class="brand-tag">Local AI Assistant</span>
  </div>
</div>
<button class="new-chat-btn" onclick="location.reload()">
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><line x1="12" y1="5" x2="12" y2="19"/><line x1="5" y1="12" x2="19" y2="12"/></svg>
  New Chat
</button>
"""

SIDEBAR_NAV_HTML = """
<div class="sidebar-nav">
  <a class="nav-item" href="#">
    <svg class="nav-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="8"/><path d="m21 21-4.35-4.35"/></svg>
    Explore
  </a>
  <a class="nav-item" href="#">
    <svg class="nav-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"/><path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"/></svg>
    Library
  </a>
</div>
"""

RECENT_CHATS_HTML = """
<div class="recent-chats-header">
  <span>Recent Chats</span>
  <a class="see-all-link" href="#">See all</a>
</div>
<div class="chat-item active">
  <svg class="chat-item-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>
  <div class="chat-item-body">
    <div class="chat-item-title">Explain LCM in maths</div>
    <div class="chat-item-time">Just now</div>
  </div>
</div>
<div class="chat-item">
  <svg class="chat-item-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>
  <div class="chat-item-body">
    <div class="chat-item-title">Crop irrigation methods</div>
    <div class="chat-item-time">1 hour ago</div>
  </div>
</div>
<div class="chat-item">
  <svg class="chat-item-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>
  <div class="chat-item-body">
    <div class="chat-item-title">Diabetes diet in Hindi</div>
    <div class="chat-item-time">3 hours ago</div>
  </div>
</div>
<div class="chat-item">
  <svg class="chat-item-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>
  <div class="chat-item-body">
    <div class="chat-item-title">Travel plan Uttarakhand</div>
    <div class="chat-item-time">1 day ago</div>
  </div>
</div>
"""

RUNNING_BADGE_HTML = """
<div class="running-badge">
  <div class="running-icon">🛡️</div>
  <div class="running-text">
    <div class="running-title">
      Running locally
      <span class="running-dot"></span>
    </div>
    <div class="running-sub">Completely private.<br>No internet required.</div>
  </div>
</div>
"""

RIGHT_PANEL_HTML = """
<div class="right-topbar">
  <div class="model-status">
    <span class="status-dot"></span>
    <span>Model Ready</span>
    <span class="status-chevron">▾</span>
  </div>
  <div class="right-actions">
    <button class="icon-btn" title="Options">⋮</button>
    <button class="icon-btn" title="Profile">
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/></svg>
    </button>
  </div>
</div>

<div class="rpanel-section">
  <div class="rpanel-label">
    <svg class="rpanel-label-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="2" y1="12" x2="22" y2="12"/><path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"/></svg>
    Response language
  </div>
  <div class="lang-select-wrap">
    <select class="lang-select" id="right-lang-select" aria-label="Response language">
      <option>🌐  Hinglish</option>
      <option>Hindi</option>
      <option>Bengali</option>
      <option>Tamil</option>
      <option>Telugu</option>
      <option>Marathi</option>
      <option>Gujarati</option>
      <option>Kannada</option>
      <option>Malayalam</option>
      <option>Punjabi</option>
      <option>Odia</option>
      <option>Banglish</option>
    </select>
  </div>
  <div style="margin-top:14px">
    <div class="other-langs-label">Other languages</div>
    <div class="lang-chips">
      <span class="lang-chip" lang="hi">हिन्दी</span>
      <span class="lang-chip">English</span>
      <span class="lang-chip" lang="gu">ગુજરાતી</span>
      <span class="lang-chip" lang="bn">বাংলা</span>
      <span class="lang-chip" lang="te">తెలుగు</span>
      <span class="lang-chip" lang="ta">தமிழ்</span>
      <span class="lang-chip" lang="mr">मराठी</span>
      <span class="lang-chip more">+ 6 more</span>
    </div>
  </div>
</div>

<div class="rpanel-section">
  <div class="rpanel-label">
    <svg class="rpanel-label-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"/></svg>
    Model
  </div>
  <div class="model-badge">
    <div class="model-badge-left">
      <svg class="model-badge-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="2" y1="12" x2="22" y2="12"/><path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"/></svg>
      <div>
        <div class="model-badge-name">Shaktimaan.ai (Default)</div>
        <div class="model-badge-sub">Fast, private and offline</div>
      </div>
    </div>
    <span class="model-chevron">▾</span>
  </div>
</div>

<div class="rpanel-section">
  <div class="rpanel-label">
    <svg class="rpanel-label-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg>
    Example prompts
  </div>
  <div>
    <div class="example-prompt">
      <span class="ep-text">Explain LCM in simple words</span>
      <span class="ep-arrow">›</span>
    </div>
    <div class="example-prompt">
      <span class="ep-text" lang="hi">भारत सरकार की छात्रवृति योजनाएँ</span>
      <span class="ep-arrow">›</span>
    </div>
    <div class="example-prompt">
      <span class="ep-text">Plan a trip to Mussoorie</span>
      <span class="ep-arrow">›</span>
    </div>
    <div class="example-prompt">
      <span class="ep-text">How to start kitchen gardening?</span>
      <span class="ep-arrow">›</span>
    </div>
  </div>
</div>
"""


def system_prompt(language: str, domain: str) -> str:
    return f"""You are Shaktimaan.ai, a helpful multilingual assistant for Indian citizens.
Reply in {language}, unless the user explicitly requests another language. Your domain is {domain}.
Be clear, practical, and culturally appropriate. Do not invent official procedures, current eligibility, or medical diagnoses. For urgent health concerns, tell the user to seek emergency care; for government services, direct them to the relevant official portal or office."""


def to_messages(
    history: list[dict] | list[tuple[str, str]] | None,
    message: str,
    language: str,
    domain: str,
) -> list[dict[str, str]]:
    messages = [{"role": "system", "content": system_prompt(language, domain)}]
    for turn in history or []:
        if isinstance(turn, dict):
            role, content = turn.get("role"), turn.get("content")
            if isinstance(content, list):
                content = "\n".join(
                    block["text"]
                    for block in content
                    if isinstance(block, dict)
                    and block.get("type") == "text"
                    and block.get("text")
                )
            if role in {"user", "assistant"} and content:
                messages.append({"role": role, "content": str(content)})
        elif len(turn) == 2:
            if turn[0]:
                messages.append({"role": "user", "content": turn[0]})
            if turn[1]:
                messages.append({"role": "assistant", "content": turn[1]})
    messages.append({"role": "user", "content": message})
    return messages


def chat(
    message: str, history: list[dict] | None, language: str, domain: str
) -> Iterator[list[dict[str, str]]]:
    if not message or not message.strip():
        yield history or []
        return
    client = OpenAI(
        base_url=SGLANG_URL,
        api_key=os.getenv("SGLANG_API_KEY", "EMPTY"),
        timeout=30,
        max_retries=0,
    )
    try:
        stream = client.chat.completions.create(
            model=MODEL_NAME,
            messages=to_messages(history, message, language, domain),
            temperature=0.35,
            top_p=0.9,
            max_tokens=700,
            stream=True,
        )
        response = ""
        for chunk in stream:
            token = chunk.choices[0].delta.content if chunk.choices else None
            if token:
                response += token
                yield list(history or []) + [
                    {"role": "user", "content": message},
                    {"role": "assistant", "content": response},
                ]
    except (APIConnectionError, APIError) as error:
        yield list(history or []) + [
            {"role": "user", "content": message},
            {
                "role": "assistant",
                "content": f"**Model server unavailable.** Start SGLang and try again.\n\nDetails: `{error}`",
            },
        ]
    except Exception as error:  # noqa: BLE001 - keep unexpected streaming errors visible in the UI
        yield list(history or []) + [
            {"role": "user", "content": message},
            {
                "role": "assistant",
                "content": f"**Unable to complete the request.** `{error}`",
            },
        ]


def stream_chat_ui(
    message: str, history: list[dict] | None, language: str, domain: str
) -> Iterator[tuple[list[dict[str, str]], str, dict, dict, dict]]:
    current_history = list(history or [])
    if message and message.strip():
        current_history.append({"role": "user", "content": message})
        yield (
            current_history,
            "",
            gr.update(visible=False),
            gr.update(visible=True),
            gr.update(visible=True),
        )
    for updated_history in chat(message, history, language, domain):
        has_messages = bool(updated_history)
        yield (
            updated_history,
            "",
            gr.update(visible=not has_messages),
            gr.update(visible=has_messages),
            gr.update(visible=has_messages),
        )


def clear_chat() -> tuple[None, str, dict, dict, dict]:
    return (
        None,
        "",
        gr.update(visible=True),
        gr.update(visible=False),
        gr.update(visible=False),
    )


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Shaktimaan.ai — Indic Assistant", elem_id="local-app") as app:
        prompt_buttons = []
        with gr.Row(elem_id="app-shell"):

            # ── Left Sidebar ──────────────────────────────────────────
            with gr.Column(elem_id="sidebar", scale=0, min_width=256):
                gr.HTML(SIDEBAR_HTML, elem_id="brand-panel")
                gr.HTML(SIDEBAR_NAV_HTML, elem_id="sidebar-nav")
                gr.HTML(RECENT_CHATS_HTML, elem_id="recent-chats")
                gr.HTML(RUNNING_BADGE_HTML, elem_id="running-badge")

            # ── Main Chat Column ──────────────────────────────────────
            with gr.Column(elem_id="main-column", scale=1, min_width=0):
                with gr.Row(elem_id="topbar"):
                    gr.HTML(
                        '<span class="crumb-muted">Shaktimaan.ai</span>'
                        '<span class="crumb-slash">/</span>'
                        '<span>Chat</span>',
                        elem_id="breadcrumb",
                    )
                    clear = gr.Button(
                        "Clear conversation",
                        elem_id="clear-chat",
                        variant="secondary",
                        visible=False,
                    )

                with gr.Column(elem_id="sheet", scale=1, min_width=0):
                    gr.HTML(
                        """<header class="page-heading">
                          <h1>Ask in your language</h1>
                          <p>Pick a topic, then write your question naturally.</p>
                        </header>""",
                        elem_id="page-heading",
                    )

                    with gr.Row(elem_id="response-settings"):
                        language = gr.Dropdown(
                            LANGUAGES,
                            value="Hinglish",
                            label="Response language",
                            elem_id="language-select",
                            scale=1,
                        )
                        domain = gr.Dropdown(
                            DOMAINS,
                            value="Education",
                            label="Topic",
                            elem_id="topic-select",
                            scale=1,
                        )

                    welcome = gr.Group(elem_id="welcome")
                    with welcome:
                        gr.HTML(
                            '<div class="suggestion-heading">'
                            '<span>Start with a question</span>'
                            '<span>Choose one to fill the message box</span>'
                            '</div>',
                            elem_id="suggestion-heading",
                        )
                        with gr.Row(elem_id="suggestions"):
                            for index, (question, _, _) in enumerate(EXAMPLES):
                                prompt_buttons.append(
                                    gr.Button(
                                        question,
                                        variant="secondary",
                                        elem_id=f"prompt-{index + 1}",
                                        elem_classes=["suggestion-button"],
                                    )
                                )

                    with gr.Group(visible=False, elem_id="conversation-panel") as conversation_panel:
                        chatbot = gr.Chatbot(
                            label="Conversation",
                            show_label=False,
                            height=240,
                            placeholder="Your conversation will appear here.",
                            elem_id="conversation",
                            autoscroll=True,
                            buttons=["copy_all"],
                        )

                    with gr.Row(elem_id="composer-row"):
                        textbox = gr.Textbox(
                            label="Your message",
                            placeholder="Ask in Hinglish, Hindi, English or any Indian language…",
                            lines=2,
                            max_lines=5,
                            elem_id="message-input",
                            scale=1,
                        )
                        send = gr.Button(
                            "Send",
                            variant="primary",
                            elem_id="send-message",
                            scale=0,
                            min_width=100,
                        )

                    gr.HTML(
                        '<div class="composer-meta">'
                        '<span>Enter to send&nbsp; · &nbsp;Shift + Enter for a new line</span>'
                        '<span class="local-mark">Local model</span>'
                        '</div>',
                        elem_id="composer-meta",
                    )
                    gr.HTML(
                        '<p class="trust-note">Model replies can be wrong. '
                        'Confirm medical advice with a qualified professional and '
                        'government details with an official source.</p>',
                        elem_id="trust-note",
                    )

            # ── Right Panel ───────────────────────────────────────────
            with gr.Column(elem_id="right-panel", scale=0, min_width=300):
                gr.HTML(RIGHT_PANEL_HTML, elem_id="right-header")

        # ── Events ───────────────────────────────────────────────────
        for button, (question, prompt_language, prompt_domain) in zip(
            prompt_buttons, EXAMPLES, strict=True
        ):
            button.click(
                lambda q=question, lang=prompt_language, topic=prompt_domain: (
                    q,
                    lang,
                    topic,
                ),
                outputs=[textbox, language, domain],
                show_progress="hidden",
            )

        outputs = [chatbot, textbox, welcome, conversation_panel, clear]
        inputs = [textbox, chatbot, language, domain]
        textbox.submit(stream_chat_ui, inputs, outputs, show_progress="hidden")
        send.click(stream_chat_ui, inputs, outputs, show_progress="hidden")
        clear.click(clear_chat, outputs=outputs, show_progress="hidden")
    return app


if __name__ == "__main__":
    build_app().launch(
        server_name=os.getenv("GRADIO_SERVER_NAME", "127.0.0.1"),
        server_port=int(os.getenv("GRADIO_SERVER_PORT", "7860")),
        theme=gr.themes.Base(
            primary_hue=gr.themes.Color(
                c50="#f0edfb",
                c100="#e2dcf7",
                c200="#cdc5ee",
                c300="#b0a5e4",
                c400="#8a7fd6",
                c500="#5547e0",
                c600="#4035cc",
                c700="#3028b8",
                c800="#251da0",
                c900="#1a1540",
                c950="#110e2a",
            ),
            neutral_hue=gr.themes.Color(
                c50="#f5f3fc",
                c100="#ece9fb",
                c200="#dcd7f5",
                c300="#c5beed",
                c400="#9993c0",
                c500="#6b65a0",
                c600="#524d80",
                c700="#3d3960",
                c800="#2d2860",
                c900="#1e1850",
                c950="#13103a",
            ),
        ),
        css=CSS,
    )
