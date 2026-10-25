from types import SimpleNamespace

import httpx
from openai import APIConnectionError

from src.gui import app


def test_build_gui():
    assert app.build_app().title.startswith("Shaktimaan.ai")


def test_gradio_six_history_text_blocks_are_sent_as_plain_text():
    history = [
        {"role": "user", "content": [{"type": "text", "text": "नमस्ते"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "उत्तर"}]},
    ]
    messages = app.to_messages(history, "फिर से", "Hindi", "Education")
    assert messages[1:] == [
        {"role": "user", "content": "नमस्ते"},
        {"role": "assistant", "content": "उत्तर"},
        {"role": "user", "content": "फिर से"},
    ]
    assert isinstance(history[0]["content"], list)


def test_chat_streaming_keeps_history(monkeypatch):
    chunks = [
        SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=token))])
        for token in ["नम", "स्ते"]
    ]
    monkeypatch.setattr(
        app,
        "OpenAI",
        lambda **kwargs: SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=lambda **kwargs: iter(chunks))
            )
        ),
    )
    history = [
        {"role": "user", "content": "पहले"},
        {"role": "assistant", "content": "उत्तर"},
    ]
    outputs = list(app.chat("नमस्ते", history, "Hindi", "Education"))
    assert outputs[-1][-1]["content"] == "नमस्ते"
    assert outputs[-1][:2] == history
    assert len(history) == 2


def test_unavailable_server_is_visible(monkeypatch):
    def unavailable(**kwargs):
        raise APIConnectionError(request=httpx.Request("POST", "http://127.0.0.1:1"))

    monkeypatch.setattr(
        app,
        "OpenAI",
        lambda **kwargs: SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=unavailable))
        ),
    )
    output = list(app.chat("Hello", [], "Hinglish", "Education"))
    assert "Model server unavailable" in output[-1][-1]["content"]


def test_empty_chat_does_not_contact_model(monkeypatch):
    def unexpected(**kwargs):
        raise AssertionError("No model call expected")

    monkeypatch.setattr(app, "OpenAI", unexpected)
    assert list(app.chat("  ", None, "Hindi", "Education")) == [[]]


def test_ui_stream_clears_input_and_hides_examples(monkeypatch):
    response = [
        {"role": "user", "content": "Explain photosynthesis."},
        {"role": "assistant", "content": "Plants make food from sunlight."},
    ]
    monkeypatch.setattr(app, "chat", lambda *args: iter([response]))

    updates = list(app.stream_chat_ui("Explain photosynthesis.", [], "English", "Education"))

    assert len(updates) == 2
    assert updates[0][0] == [{"role": "user", "content": "Explain photosynthesis."}]
    assert all(update[1] == "" for update in updates)
    assert all(update[2]["visible"] is False for update in updates)
    assert all(update[3]["visible"] is True for update in updates)
    assert all(update[4]["visible"] is True for update in updates)


def test_clear_chat_restores_the_empty_state():
    history, message, welcome, conversation, clear = app.clear_chat()

    assert history is None
    assert message == ""
    assert welcome["visible"] is True
    assert conversation["visible"] is False
    assert clear["visible"] is False
