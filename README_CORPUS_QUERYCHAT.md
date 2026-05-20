# L2 French Agreement Corpus Explorer

This Shiny + QueryChat app explores verb-level annotations for Japanese learners of French. It flattens nested structured-output JSON files into tables for agreement accuracy, cue strength, attraction effects, structural complexity, learners, files, and evidence examples.

## Prepare Data

From this folder:

```powershell
python corpus_querychat_explorer/prepare_corpus_data.py Annotation.py
```

For a full folder of annotation files:

```powershell
python corpus_querychat_explorer/prepare_corpus_data.py C:\path\to\annotation_folder
```

Generated files are written to `outputs/corpus_querychat`:

- `verbs.csv`
- `utterances.csv`
- `files.csv`
- `learners.csv`
- `cue_summary.csv`
- `attraction_summary.csv`
- `metadata.json`
- `research_summary.json`
- `corpus_analysis.duckdb` when `duckdb` is installed

## Run

Create a `.env` file from `.env.example` and add `GEMINI_API_KEY`.

```powershell
shiny run --reload corpus_querychat_explorer/app.py
```

The dashboard still opens without a Gemini key, but the QueryChat sidebar is enabled only when `GEMINI_API_KEY` is set.
