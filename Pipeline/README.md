# Pipeline

This folder contains a clean annotation pipeline using either **OpenAI** or **Gemini**.

## Structure
- `run_annotation_pipeline.py` → main runner.
- `templates/guidelines.md` → annotation rules (single source of truth).
- `templates/prompt.md` → task prompt.
- `templates/json_template.json` → output schema to fill.
- `chatgpt_manual_test_pack.md` → copy/paste pack for manual testing in chat.

## Setup
1. Install dependencies (already mostly present):
   - `requests`
2. Set API key(s):
   - `OPENAI_API_KEY` for OpenAI
   - `GEMINI_API_KEY` for Gemini
3. Optional for private Google Docs export:
   - `GOOGLE_DOCS_BEARER`

## Usage
### OpenAI
```bash
python Pipeline/run_annotation_pipeline.py \
  --provider openai \
  --model gpt-5-mini \
  --source "https://docs.google.com/document/d/<DOC_ID>/edit" \
  --output Pipeline/outputs/openai_annotation.json
```

### Gemini
```bash
python Pipeline/run_annotation_pipeline.py \
  --provider gemini \
  --model gemini-2.0-flash \
  --source ./my_transcript.txt \
  --output Pipeline/outputs/gemini_annotation.json
```

## Notes
- Replace template files with your finalized versions.
- The script enforces JSON output parsing and writes normalized JSON files.
