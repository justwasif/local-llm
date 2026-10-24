"""Public-facing multilingual LocalLLM demo powered by an SGLang endpoint."""
from __future__ import annotations

import os
from collections.abc import Iterator

import gradio as gr
from openai import OpenAI, APIError, APIConnectionError

SGLANG_URL = os.getenv("SGLANG_BASE_URL", "http://localhost:30000/v1")
MODEL_NAME = os.getenv("SGLANG_MODEL", "local-llm")
LANGUAGES = ["Hindi", "Bengali", "Tamil", "Telugu", "Marathi", "Gujarati", "Kannada", "Malayalam", "Punjabi", "Odia", "Hinglish", "Banglish"]
DOMAINS = ["Education", "Healthcare", "Agriculture", "Government Services"]
EXAMPLES = [
    ["Fasal ke patton par daag aa rahe hain; pehla safe step kya hona chahiye?", "Hinglish", "Agriculture"],
    ["ডেঙ্গুর সতর্ক লক্ষণগুলি কী?", "Bengali", "Healthcare"],
    ["प्रकाश संश्लेषण को सरल उदाहरण के साथ समझाइए।", "Hindi", "Education"],
    ["ரேஷன் கார்டுக்கு விண்ணப்பிக்கும் முன் எந்த அதிகாரப்பூர்வ தகவலை சரிபார்க்க வேண்டும்?", "Tamil", "Government Services"],
]
CSS = """
.gradio-container { max-width: 1050px !important; background: #f8fafc; }
#title { text-align: center; margin-bottom: 0; }
#subtitle { text-align: center; color: #475569; }
.disclaimer { color: #7c2d12; background: #fff7ed; border-radius: 8px; padding: .7rem; }
"""


def system_prompt(language: str, domain: str) -> str:
    return f"""You are LocalLLM, a helpful multilingual assistant for Indian citizens.
Reply in {language}, unless the user explicitly requests another language. Your domain is {domain}.
Be clear, practical, and culturally appropriate. Do not invent official procedures, current eligibility, or medical diagnoses. For urgent health concerns, tell the user to seek emergency care; for government services, direct them to the relevant official portal or office."""


def to_messages(history: list[dict] | list[tuple[str, str]] | None, message: str, language: str, domain: str) -> list[dict[str, str]]:
    messages = [{"role": "system", "content": system_prompt(language, domain)}]
    for turn in history or []:
        if isinstance(turn, dict):
            role, content = turn.get("role"), turn.get("content")
            if role in {"user", "assistant"} and content: messages.append({"role": role, "content": str(content)})
        elif len(turn) == 2:
            if turn[0]: messages.append({"role": "user", "content": turn[0]})
            if turn[1]: messages.append({"role": "assistant", "content": turn[1]})
    messages.append({"role": "user", "content": message})
    return messages


def chat(message: str, history: list[dict] | None, language: str, domain: str) -> Iterator[list[dict[str, str]]]:
    if not message or not message.strip():
        yield history or []
        return
    client = OpenAI(base_url=SGLANG_URL, api_key=os.getenv("SGLANG_API_KEY", "EMPTY"))
    try:
        stream = client.chat.completions.create(model=MODEL_NAME, messages=to_messages(history, message, language, domain), temperature=0.35, top_p=0.9, max_tokens=700, stream=True)
        response = ""
        for chunk in stream:
            token = chunk.choices[0].delta.content if chunk.choices else None
            if token:
                response += token
                yield list(history or []) + [{"role": "user", "content": message}, {"role": "assistant", "content": response}]
    except (APIConnectionError, APIError) as error:
        yield list(history or []) + [{"role": "user", "content": message}, {"role": "assistant", "content": f"**Model server unavailable.** Start SGLang and try again.\n\nDetails: `{error}`"}]
    except Exception as error:
        yield list(history or []) + [{"role": "user", "content": message}, {"role": "assistant", "content": f"**Unable to complete the request.** `{error}`"}]


def build_app() -> gr.Blocks:
    with gr.Blocks(title="LocalLLM — Indic Assistant") as app:
        gr.Markdown("# LocalLLM", elem_id="title")
        gr.Markdown("### A local, multilingual assistant for education, healthcare, agriculture, and government services", elem_id="subtitle")
        with gr.Row():
            language = gr.Dropdown(LANGUAGES, value="Hinglish", label="Response language")
            domain = gr.Dropdown(DOMAINS, value="Education", label="Domain")
        chatbot = gr.Chatbot(label="Conversation", height=430)
        textbox = gr.Textbox(label="Your message", placeholder="Ask in the language you are most comfortable with…", lines=2)
        with gr.Row():
            send = gr.Button("Send", variant="primary")
            clear = gr.Button("Clear")
        gr.Examples(EXAMPLES, inputs=[textbox, language, domain], label="Try an example")
        gr.Markdown("Medical information is educational, not a diagnosis. Verify government guidance through official channels.", elem_classes="disclaimer")
        textbox.submit(chat, [textbox, chatbot, language, domain], chatbot).then(lambda: "", None, textbox)
        send.click(chat, [textbox, chatbot, language, domain], chatbot).then(lambda: "", None, textbox)
        clear.click(lambda: ([], ""), outputs=[chatbot, textbox])
    return app


if __name__ == "__main__":
    build_app().launch(
        server_name=os.getenv("GRADIO_SERVER_NAME", "127.0.0.1"),
        server_port=int(os.getenv("GRADIO_SERVER_PORT", "7860")),
        theme=gr.themes.Soft(),
        css=CSS,
    )
