# LocalLLM Landing Page Agent Brief

**Purpose:** Use this brief to design and build a landing page for LocalLLM. The page must explain the language-access problem this project addresses and present the current product honestly. This brief is the deliverable for the current task; it does not authorize changes to application code.

## 1. Project in one paragraph

Shaktimaan.ai is the current public-facing name for the LocalLLM project and its research prototype. It explores how a small language model can make public-facing information easier to access across Indic languages and code-mixed text. A person should be able to ask an everyday question in the language they use, select the language they want a reply in, and explore a response in areas such as education, farming, health, and government services. The intent is to reduce the language barrier between people and useful information. The prototype is not yet validated as a reliable source of advice.

## 2. Problem statement alignment

The selected Local LLM Track problem is **PS4: low-resource and multilingual languages**. The statement asks teams to improve a model for underserved languages, dialects, or code-mixed text in a public-facing setting such as education, public services, agriculture, or healthcare.

The landing page should make this link clear: the central problem is that people may be less able to ask questions or use information when the available interface or answer is in a language they do not use comfortably. LocalLLM explores a language-first way to ask and receive responses across those everyday topics.

The track also requires SGLang as the core inference and serving engine, and calls for CuratorKIT, AlignTune, and AuditKIT across curation, fine-tuning, and evaluation. These are useful technical details for a short “How it is built” section; they are not the main user benefit. The problem statement gives no numeric accuracy or benchmark thresholds.

## 3. Intended audience

Design for people who want to ask practical questions in an Indic language or code-mixed text, including:

- Students and learners seeking explanations.
- Farmers and agricultural users seeking general information.
- Citizens looking for understandable starting points for public services.
- People seeking general health information, with clear direction to qualified professionals for medical concerns.

Treat these as intended audiences, not as claims that the model has been validated for each group. Avoid portraying users as deficient, less capable, or defined by poverty or lack of education. The barrier belongs to systems that do not offer usable language access.

## 4. Product facts to use

- **Public-facing name:** Shaktimaan.ai. **Project/repository and track context:** LocalLLM.
- **Current model:** Qwen2.5-1.5B-Instruct, fine-tuned locally with SFT and DPO. Do not call it a 7B model.
- **Serving:** SGLang. The current development server binds to loopback; it is not a public demo URL.
- **Current interface language choices:** Hindi, Bengali, Tamil, Telugu, Marathi, Gujarati, Kannada, Malayalam, Punjabi, Odia, Hinglish, and Banglish.
- **Current topic choices:** Education, Healthcare, Agriculture, and Government Services.
- **Current interaction:** The person selects a requested response language and topic, writes naturally or chooses a sample prompt, then sees a streamed response. The prompt asks the model to use the selected language unless the user asks otherwise.
- **Data and training scale:** 220 SFT conversations and 7 reviewed preference pairs. The available Aya training sample contains 216 records across 9 Indic languages; it is a small convenience sample, not representative coverage.
- **Tools used:** CuratorKIT for data curation, AlignTune/TRL for SFT and DPO, SGLang for inference, and AuditKIT for evaluation.

The dropdown is a set of requested output-language labels, not evidence that the model is competent in all 12. The current training sample does not cover Odia, Hinglish, or Banglish. Do not imply automatic language detection; the user chooses the response language.

## 5. Claims and evidence guardrails

Present LocalLLM as a **prototype/research effort** and its multilingual performance as a goal under investigation. The current evaluation does not establish that training improved the model:

- The FLORES-200 comparison is a 50-example pilot: five fixed sentences in each of ten languages, not a full benchmark.
- Mean pilot chrF++ was 9.8761 for the baseline, 8.7548 after SFT, and 8.9250 after DPO. Both fine-tuned checkpoints were below the baseline on this pilot.
- Six lexical safety probes are only a smoke screen. They are not an independent multilingual safety review.
- Independent native-speaker review, representative benchmarks, and evidence of reliable public-facing quality remain incomplete.

Do not claim that LocalLLM is fluent, accurate, safe, unbiased, production-ready, or better than the baseline. Do not promise correct medical diagnoses, official government procedures, eligibility guidance, or agricultural recommendations. Do not claim that user data never leaves a device or that the product is private/offline unless the actual deployed architecture has been verified. Do not use fabricated testimonials, user counts, accuracy percentages, awards, partner endorsements, or benchmark badges.

If the page includes evaluation results, label them as a small pilot, show baseline/SFT/DPO together, include sample size and languages, and state the negative comparison plainly. Prefer an honest prototype-status note over using these results as a marketing proof point. A suitable disclosure is:

> LocalLLM is a research prototype. Response quality and safety have not been independently validated across the available language choices. Confirm medical and government information with qualified professionals or official sources.

## 6. Message and copy direction

Keep the primary idea concrete and language-first. Suggested headline:

> **Ask in the language you use. Choose the language you want to hear back in.**

Suggested supporting copy:

> LocalLLM is a research prototype exploring small multilingual AI for everyday questions in education, farming, health, and public services.

Suggested primary CTA: **Try the demo**. Link it to a real, configured demo URL supplied for the landing page. The current `http://127.0.0.1:7860` address is development-only and must not be shipped as a visitor-facing link. Suggested secondary CTA: **Read the evaluation** or **How it works**.

Use plain, respectful language. Prefer “language access,” “ask in the language you use,” and “choose a response language.” Avoid claims such as “break every language barrier,” “answers for everyone,” “instant expert,” or “fluent in 12 languages.” Do not use a language as decoration: any non-English copy must be accurate, legible, and reviewed by a fluent speaker before publication.

## 7. Recommended page narrative

1. **Hero:** State the language-access purpose and show the product interaction clearly. Use one strong headline, concise supporting copy, a primary demo CTA, and a visible “research prototype” qualifier.
2. **The barrier:** Explain that useful information is harder to reach when people cannot comfortably ask in their language. Keep this human and specific; do not exaggerate or blame users.
3. **How it works:** Three steps: choose a response language, choose a topic and ask naturally, review the streamed response. Say that the user controls the response-language choice.
4. **Everyday topics:** Education, agriculture, health, and public services. Describe these as areas the prototype is exploring, not validated expert capabilities.
5. **Language access:** Show the language choices as interface options. Distinguish the 12 UI labels from the smaller training sample and avoid implying equal quality in every language.
6. **How it is built:** Briefly explain curation, fine-tuning, serving, and evaluation, naming CuratorKIT, AlignTune/TRL, SGLang, and AuditKIT accurately. Keep this secondary to the human problem.
7. **Status and trust:** State that this is a prototype, show the limits above, and provide a link to the full benchmark report if the deployment has a stable public URL for it.
8. **Closing CTA:** Invite visitors to try the configured demo or follow the research. Repeat the prototype and advice-verification note in the footer where appropriate.

## 8. Visual and interaction direction

Create a distinctive, editorial landing page rather than a generic AI SaaS template. Let real language scripts and a clear prompt/response interaction carry the idea. Use a calm, readable palette, confident typography, generous but purposeful spacing, and a restrained accent. Avoid neon gradients, generic robot imagery, fake translation symbols, stock-photo stereotypes, and decorative language fragments that cannot be read.

An interface preview may use the current LocalLLM GUI as a reference, especially the multilingual prompt examples, but should be composed for a landing page rather than pasted in as a dashboard screenshot. Never show an unreviewed model answer as proof of correctness. If a static example is not connected to the live model, label it as an illustrative preview. If a demo action is shown as functional, it must link to the configured demo or actually work; do not simulate a response with a fake animation.

The PDF includes event and toolkit branding. Do not copy event graphics or imply sponsorship/endorsement from Syntax Error, RadixArk, Lexsi Labs, or toolkit maintainers. Mention toolkit names as implementation credits only, unless separate permission for logos or partnership claims exists.

## 9. Accessibility, localization, and implementation checks

- Make the page responsive from phone to wide desktop, with no horizontal overflow or clipped Indic text.
- Use semantic headings, keyboard-accessible navigation and controls, visible focus states, adequate contrast, and descriptive link text.
- Check Indic fonts and fallback behavior on real browsers; do not rely on missing-glyph boxes or images of text.
- Respect reduced-motion preferences. Do not require animation, color, hover, or audio to understand the product.
- If the page itself is translated, use human-reviewed translations and correct page language metadata. A language selector must not imply that translations exist when they do not.
- Keep the CTA destination configurable. Never expose API keys or call a private inference endpoint directly from browser code.
- Add honest page title, description, and social metadata. Do not use “best,” “most accurate,” or unsupported performance claims in SEO copy.

## 10. Definition of done

- A first-time visitor can tell within a few seconds that LocalLLM is about asking in a familiar language and choosing a response language.
- The page ties the project to PS4 and presents the four intended public-facing topic areas.
- The prototype status and language-quality limits are visible, not hidden in legal copy.
- Every factual metric is traceable to the benchmark report and labeled with its pilot scope; no unsupported outcome claims appear.
- The demo CTA points to a real configured destination, never localhost.
- Indic script examples render correctly and the page works with keyboard, screen reader structure, and narrow screens.
- The result feels specific to multilingual public-facing assistance, not a reskinned generic chatbot landing page.

## 11. Project sources of truth

- `LocalLLM PS.pdf` - track requirements and selected problem options.
- `README.md` - project architecture, commands, and current scope.
- `results/reports/FINAL_BENCHMARK_REPORT.md` - data provenance, training results, pilot measurements, and limitations.
- `memory/10_remaining_work_handoff.md` - current machine and remaining evidence gaps.
- `src/gui/app.py` and `src/gui/ui.css` - current product flow, language labels, topic labels, sample prompts, and user-facing trust note.
- `.taste-skill/after-final/desktop-fold.jpg` and `.taste-skill/after-final/mobile-fold.jpg` - current UI references, not landing-page templates.
