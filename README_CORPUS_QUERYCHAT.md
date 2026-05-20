# L2 French Agreement Corpus Explorer

This Shiny + QueryChat app explores verb-level annotations for Japanese learners of French. It flattens the final structured-output annotation JSON shape into tables for agreement accuracy, cue strength, attraction effects, structural complexity, participants, files, and evidence examples.

For a detailed explanation of the full annotation-to-dashboard pipeline, see `PIPELINE_DETAILED_GUIDE.md`.

## Prepare Data

From this folder:

```powershell
python corpus_querychat_explorer/prepare_corpus_data.py TS_FA_session01_page01_annotation.json
```

For a full folder of final annotation files:

```powershell
python corpus_querychat_explorer/prepare_corpus_data.py C:\path\to\annotation_folder
```

Generated files are written to `outputs/corpus_querychat`:

- `verbs.csv`
- `utterances.csv`
- `files.csv`
- `learners.csv` (participant summary)
- `cue_summary.csv`
- `attraction_summary.csv`
- `metadata.json`
- `research_summary.json`
- `corpus_analysis.duckdb` when `duckdb` is installed

The current final JSON schema is expected to have top-level `file_id`, `utterance_unit`, `speaker_map`, and `utterances`. Each utterance can use `participant_id`; the prep step also fills the older `learner_id` alias so existing QueryChat prompts continue to work.

## Run

Create a `.env` file from `.env.example` and add `GEMINI_API_KEY`.

On Linux, from a fresh clone:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
cp .env.example .env
# edit .env and set GEMINI_API_KEY if you want QueryChat
shiny run --reload corpus_querychat_explorer/app.py
```

The prepared sample outputs are already committed, so the dashboard can open immediately after installing dependencies. To regenerate them from the included final annotation JSON:

```bash
python corpus_querychat_explorer/prepare_corpus_data.py TS_FA_session01_page01_annotation.json
```

On Windows:

```powershell
shiny run --reload corpus_querychat_explorer/app.py
```

The dashboard still opens without a Gemini key, but the QueryChat sidebar is enabled only when `GEMINI_API_KEY` is set.

QueryChat uses `gemini-2.5-flash-lite` by default. If the Gemini API returns a temporary `503 UNAVAILABLE` high-demand error, wait a moment and retry, or set another available Gemini text model in `.env`:

```powershell
QUERYCHAT_GEMINI_MODEL=gemini-2.5-flash-lite
```
