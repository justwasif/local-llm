# Stage 08 - GUI

Updated: 2026-10-25. Current Windows/Ubuntu WSL2 machine; earlier Nix-host observations are historical and do not establish artifacts in this checkout.

## Implemented and executed

Gradio build, Unicode streaming, history preservation, empty input and unavailable-server handling are covered by tests. Live verification: {"status": "passed", "url": "http://127.0.0.1:7860", "loaded_model_path": "/mnt/c/Users/Krish/local-llm/outputs/local_1_5b_dpo_merged", "checks": ["HTTP page", "real queued streaming", "Unicode input preserved", "Hindi-script response", "follow-up history", "empty message", "clear"], "stream_updates": 32, "prompt": "नमस्ते! दो और दो कितने होते हैं? केवल एक छोटा वाक्य लिखें।", "response": "दो और दो होते हैं, तीन कितने होते हैं?", "followup_response": "2 और 2 होते हैं, तीन कितने होते हैं?", "arithmetic_answer_marker_present": false, "quality_limit": "Arithmetic marker presence is a limited diagnostic, not a factual correctness score. Functional success must not be read as model-quality success. The initial pre-fix GUI responses are preserved in gui_verification_before_history_fix.json.", "visual_browser_check": "blocked: no browser connector; computer-use tool rejected Chrome access as not approved", "limitations": "Functional API smoke test; not independent language-quality or visual-layout verification."}

## File manifest

src/gui/app.py; tests/test_gui.py

## How it works

Use the current WSL wrapper and record artifacts from successful real executions. Preserve failures and negative deltas.

## Configuration

Local endpoint 127.0.0.1:30000, GUI 127.0.0.1:7860. Base and candidate evaluation must use the same frozen prompts and generation settings.

## Dependencies

Depends on completed preceding stages and the current machine's artifacts.

## Troubleshooting

SGLang RoPE kernels require the full matching nvcc archive, runtime headers and CCCL; the pip nvcc package alone only provides ptxas. Original handoff's AMX error does not describe this rebuilt environment.

## Verification

Inspect actual reports, artifacts and live API/UI behavior before marking the stage finished.

## UI refresh (2026-10-25)

- Applied the [Taste Skill](https://github.com/Leonxlnx/taste-skill) framed-sheet direction. The welcome view keeps the transcript and clear action hidden until the first question; sending shows the stream, clears the composer and hides examples, and clearing restores the compact welcome state.
- Added responsive styling in `src/gui/ui.css`, a language/topic picker, Unicode prompt starters and a concise safety note. Visual review at 1440/768/390 px is in `.taste-skill/after-final/`; inspector reports 0 failures and 0 warnings. All six viewport screenshots were inspected.
- Live UI smoke: Hindi prompt example populated Hindi/Education; streaming rendered in Hindi; composer cleared; examples hid; clear restored the welcome view. The smoke's simple arithmetic response happened to be correct; this single response does not certify model quality or benchmark performance.
- File manifest: `src/gui/app.py`, `src/gui/ui.css`, `tests/test_gui.py`, `README.md`, `.taste-skill/`.
