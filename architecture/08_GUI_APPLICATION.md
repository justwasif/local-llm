# 💻 Stage 3.3: GUI Application — Multilingual User Interface

This stage focuses on building the final user-facing interface that allows users to interact with the model in their native languages.

## 🎯 Goal
Develop a professional, interactive Gradio application that demonstrates the model's capabilities in low-resource Indic languages for public-facing domains.

---

## 🛠️ Implementation Steps

### 1. GUI Architecture
Create the application in `src/gui/app.py`.

**Key Features to Implement:**
- **Language Selector**: A dropdown menu for the 10+ target Indic languages + Hinglish.
- **Domain Selector**: Dropdown for "Education", "Healthcare", "Agriculture", "Government Services".
- **Chat Interface**: A clean chat window with support for non-Latin scripts (Unicode).
- **SGLang API Integration**: Connect the GUI to the SGLang server running on port 30000 using the OpenAI-compatible client.
- **Side-by-Side Comparison**: (Optional) Allow users to compare the base model vs. the fine-tuned model.

### 2. Implementation Logic
```python
import gradio as gr
from openai import OpenAI

client = OpenAI(base_url="http://localhost:30000/v1", api_key="None")

def chat_fn(message, history, language, domain):
    system_prompt = f"You are a helpful assistant specializing in {domain} for users speaking {language}."
    # Construct messages for SGLang API...
    # return response
```

### 3. UI Polish
- Custom CSS for a professional look.
- Example prompts for each domain/language to help users start.
- Proper handling of streaming responses for a better UX.

---

## ✅ Verification Checklist
- [ ] GUI launches successfully via `python src/gui/app.py`.
- [ ] All target languages are selectable and output is rendered correctly.
- [ ] API calls to the SGLang server are fast and stable.
- [ ] The UI is intuitive and demonstrates the "Public-Facing" requirement of the PS.

## 🧠 Memory Note
After completing this stage, create `memory/08_memory_gui.md` documenting:
- The GUI's structure and key functions.
- How to launch the GUI and connect it to the server.
- List of example prompts used for demonstration.
