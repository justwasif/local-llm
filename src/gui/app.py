"""Public-facing Shaktimaan.ai demo powered by an SGLang endpoint."""

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
            with gr.Column(elem_id="sidebar", scale=0, min_width=220):
                gr.HTML(
                    """<div class="brand-lockup">
                      <span class="brand-scripts" aria-hidden="true">अ&nbsp;অ&nbsp;அ</span>
                      <span class="brand-name">Shaktimaan.ai</span>
                    </div>
                    <p class="rail-note">A local assistant for questions about learning, health, farming and public services.</p>
                    <div class="rail-rule"></div>
                    <p class="rail-detail">Choose a response language and topic. Ask in the words you use every day.</p>""",
                    elem_id="brand-panel",
                )
                gr.HTML(
                    '<div class="rail-caption">12 language options</div>',
                    elem_id="rail-languages",
                )

            with gr.Column(elem_id="main-column", scale=1, min_width=0):
                with gr.Row(elem_id="topbar"):
                    gr.HTML(
                        '<span class="crumb-muted">Shaktimaan.ai</span><span class="crumb-slash">/</span><span>Conversation</span>',
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
                            '<div class="suggestion-heading"><span>Start with a question</span><span>Choose one to fill the message box</span></div>',
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
                            height=220,
                            placeholder="Your conversation will appear here.",
                            elem_id="conversation",
                            autoscroll=True,
                            buttons=["copy_all"],
                        )

                    with gr.Row(elem_id="composer-row"):
                        textbox = gr.Textbox(
                            label="Your message",
                            placeholder="Write your question…",
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
                            min_width=92,
                        )

                    gr.HTML(
                        '<div class="composer-meta"><span>Enter to send&nbsp; · &nbsp;Shift + Enter for a new line</span><span class="local-mark">Local model</span></div>',
                        elem_id="composer-meta",
                    )
                    gr.HTML(
                        '<p class="trust-note">Model replies can be wrong. Confirm medical advice with a qualified professional and government details with an official source.</p>',
                        elem_id="trust-note",
                    )

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

        def set_welcome_visibility(history: list[dict] | None) -> dict:
            return gr.update(visible=not bool(history))

        outputs = [chatbot, textbox, welcome, conversation_panel, clear]
        inputs = [textbox, chatbot, language, domain]
        textbox.submit(
            stream_chat_ui, inputs, outputs, show_progress="hidden"
        )
        send.click(stream_chat_ui, inputs, outputs, show_progress="hidden")
        clear.click(clear_chat, outputs=outputs, show_progress="hidden")
    return app


if __name__ == "__main__":
    build_app().launch(
        server_name=os.getenv("GRADIO_SERVER_NAME", "127.0.0.1"),
        server_port=int(os.getenv("GRADIO_SERVER_PORT", "7860")),
        theme=gr.themes.Base(
            primary_hue=gr.themes.Color(
                c50="#eff5f1",
                c100="#dce9df",
                c200="#bfd5c5",
                c300="#9ebba7",
                c400="#71957f",
                c500="#416e53",
                c600="#315b43",
                c700="#264a36",
                c800="#203d2d",
                c900="#193124",
                c950="#102118",
            ),
            neutral_hue=gr.themes.Color(
                c50="#f8f9f7",
                c100="#f0f2ee",
                c200="#e2e5df",
                c300="#d0d4cc",
                c400="#9da39a",
                c500="#6e756b",
                c600="#555b53",
                c700="#424740",
                c800="#30342f",
                c900="#20231f",
                c950="#131512",
            ),
        ),
        css=CSS,
    )
