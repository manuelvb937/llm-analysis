# End-to-End Annotation and QueryChat Pipeline Guide

This guide explains the complete workflow in this repository:

1. How the annotation prompt is assembled.
2. How the model is called.
3. What the final annotation JSON is expected to contain.
4. How the JSON is flattened into CSV and DuckDB tables.
5. How the Shiny dashboard and QueryChat use those tables.
6. Where to look if you want to improve quality, reliability, or analysis.

The short version:

```text
transcript text
  -> Pipeline/templates/prompt.md
  -> Pipeline/templates/guidelines.md
  -> Pipeline/templates/json_template.json
  -> Pipeline/run_annotation_pipeline.py
  -> *_annotation.json
  -> corpus_querychat_explorer/prepare_corpus_data.py
  -> outputs/corpus_querychat/*.csv
  -> corpus_querychat_explorer/app.py
  -> Shiny dashboard + QueryChat
```

## 1. Repository Map

### Annotation generation

`Pipeline/run_annotation_pipeline.py`

This is the runner that:

- reads a transcript from a local file or Google Doc export;
- reads the prompt, guidelines, and JSON template;
- combines them into one model payload;
- calls OpenAI or Gemini;
- extracts JSON from the model response;
- writes a final `*_annotation.json` file.

`Pipeline/templates/prompt.md`

This is the high-level task instruction. It tells the model what kind of job it is doing.

`Pipeline/templates/guidelines.md`

This is the annotation rulebook. This is where the linguistic definitions live:

- what counts as `target`, `non-target`, or `unexpected`;
- how to assign `participant_id`;
- how to annotate cue strength;
- how to annotate attraction;
- how to use confidence and comments.

`Pipeline/templates/json_template.json`

This is the expected final output shape. The current final schema has top-level keys:

```json
{
  "file_id": "TS_FA_session01_page01",
  "utterance_unit": "line_break",
  "speaker_map": {},
  "utterances": []
}
```

Each utterance contains speaker information and a list of verbs. Each verb contains nested annotation objects:

```json
{
  "subject": {},
  "cue_strength": {},
  "attraction": {},
  "annotation_metadata": {}
}
```

Important: the final template does not contain `intervener_type`. The dashboard should not use it.

### Data preparation

`corpus_querychat_explorer/prepare_corpus_data.py`

This script converts the final nested JSON into flat analysis tables. It writes:

- `outputs/corpus_querychat/verbs.csv`
- `outputs/corpus_querychat/utterances.csv`
- `outputs/corpus_querychat/files.csv`
- `outputs/corpus_querychat/learners.csv`
- `outputs/corpus_querychat/cue_summary.csv`
- `outputs/corpus_querychat/attraction_summary.csv`
- `outputs/corpus_querychat/metadata.json`
- `outputs/corpus_querychat/research_summary.json`
- `outputs/corpus_querychat/corpus_analysis.duckdb` when DuckDB is installed

### Dashboard and QueryChat

`corpus_querychat_explorer/app.py`

This Shiny app:

- loads the prepared CSV tables;
- cleans data types;
- builds KPI cards and charts;
- displays evidence examples;
- starts QueryChat when `GEMINI_API_KEY` is available;
- passes the `verbs` table to QueryChat as the main natural-language query table.

## 2. Prompt Preparation

The prompt is assembled in `build_user_payload()` inside `Pipeline/run_annotation_pipeline.py`.

Conceptually, it does this:

```python
payload = (
    task_prompt
    + guidelines
    + json_template
    + transcript
    + "Return strictly valid JSON, and no extra prose."
)
```

In actual section order:

```text
TASK PROMPT

## ANNOTATION GUIDELINES
... full guidelines.md ...

## JSON TEMPLATE (Fill this schema, do not remove required keys)
... full json_template.json ...

## TRANSCRIPT TO ANNOTATE
... transcript text ...

Return strictly valid JSON, and no extra prose.
```

This matters because the LLM sees three kinds of instruction:

1. Task:
   "Annotate the transcript."

2. Rules:
   "These are the linguistic decisions you must make."

3. Schema:
   "This is the JSON shape you must fill."

The schema acts like a contract. If a field is not in `json_template.json`, the rest of the pipeline should not assume it exists.

## 3. Loading the Transcript

The pipeline accepts either:

- a local file path;
- a Google Doc URL;
- a Google Doc ID.

Example local run:

```powershell
python Pipeline/run_annotation_pipeline.py `
  --provider gemini `
  --model gemini-2.5-flash-lite `
  --source .\my_transcript.txt `
  --output .\TS_FA_session01_page01_annotation.json
```

Example Google Doc run:

```powershell
python Pipeline/run_annotation_pipeline.py `
  --provider openai `
  --model gpt-5-mini `
  --source "https://docs.google.com/document/d/<DOC_ID>/edit" `
  --output .\TS_FA_session01_page01_annotation.json
```

The function `load_transcript()` decides which source type you gave it:

```python
maybe_file = Path(source)
if maybe_file.exists() and maybe_file.is_file():
    return maybe_file.read_text(encoding="utf-8")
```

If it is not a local file, the script tries to extract a Google Doc ID:

```python
doc_id = extract_gdoc_id(source)
url = f"https://docs.google.com/document/d/{doc_id}/export?format=txt"
res = requests.get(url, headers=headers, timeout=60)
return res.text
```

If your Google Doc is private, the script can use:

```powershell
GOOGLE_DOCS_BEARER=<token>
```

## 4. Model Calls

The runner supports two providers:

- OpenAI through `call_openai()`;
- Gemini through `call_gemini()`.

### OpenAI call

OpenAI is called with the Responses API:

```python
data = {
    "model": model,
    "input": [
        {
            "role": "system",
            "content": [
                {
                    "type": "input_text",
                    "text": "You are a strict linguistic annotator. Output only valid JSON.",
                }
            ],
        },
        {"role": "user", "content": [{"type": "input_text", "text": user_payload}]},
    ],
    "text": {"format": {"type": "json_object"}},
}
```

The most important part is:

```python
"text": {"format": {"type": "json_object"}}
```

That tells the model to return JSON.

### Gemini call

Gemini is called through the REST endpoint:

```python
url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
```

The payload sets:

```python
"generationConfig": {
    "responseMimeType": "application/json",
    "temperature": 0.1,
}
```

Important:

- `responseMimeType` asks Gemini for JSON.
- `temperature = 0.1` reduces randomness.
- A lower temperature is useful for annotation consistency.

If Gemini returns a `503 UNAVAILABLE`, that usually means provider-side high demand. It is not caused by the JSON or Shiny conversion.

## 5. JSON Extraction

The function `extract_json()` tries to protect you against two common LLM behaviors:

1. Returning pure JSON:

```json
{"file_id": "...", "utterances": []}
```

2. Returning JSON wrapped in markdown fences:

````markdown
```json
{"file_id": "...", "utterances": []}
```
````

The function first strips markdown fences. Then it tries:

```python
json.loads(raw)
```

If that fails, it searches for the first large `{ ... }` block:

```python
match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
return json.loads(match.group(0))
```

Potential improvement:

- Add schema validation after parsing.
- Fail loudly when required keys are missing.
- Save the raw model response for debugging.

## 6. Final Annotation JSON Shape

Your current final JSON looks like this:

```json
{
  "file_id": "TS_FA_session01_page01",
  "utterance_unit": "line_break",
  "speaker_map": {
    "TS": {
      "participant_id": "P001",
      "role": "learner"
    }
  },
  "utterances": [
    {
      "utterance_id": "utt_001",
      "speaker": "TS",
      "participant_id": "P001",
      "raw_utterance": "Ouais [soupir], on va s'adresser a quel sujet ?",
      "normalized_utterance": "Ouais [soupir], on va s'adresser a quel sujet ?",
      "verbs": []
    }
  ]
}
```

When an utterance has verbs, each verb has this nested shape:

```json
{
  "verb_index": 1,
  "produced_form": "va s'adresser",
  "produced_agreement": "target",
  "target_form": "va s'adresser",
  "lemma": "s'adresser",
  "tense_mood": "futur_proche",
  "category": null,
  "category_coding_status": "pending_human_coding",
  "subject": {
    "subject_form": "on",
    "head": "on",
    "subject_type": "pronoun",
    "person": "3",
    "number": "singular"
  },
  "cue_strength": {
    "cue_present": false,
    "cue_expression": null,
    "cue_type": "none",
    "cue_number": null
  },
  "attraction": {
    "attraction_configuration": false,
    "attraction_error": false,
    "attractor_surface_form": null,
    "attractor_head": null,
    "attractor_number": null,
    "attractor_type": null,
    "structural_complexity_type": "infinitival_complement",
    "linear_distance_words": 0
  },
  "annotation_metadata": {
    "llm_confidence": 0.92,
    "low_confidence_reason": null,
    "comments": "Futur proche with subject on."
  }
}
```

## 7. Conversion From JSON to CSV

The final JSON is nested. CSV is flat. The data preparation script converts nested objects into rows and columns.

### One document

A source JSON file is treated as one document when it has:

```python
payload.get("utterances")
```

The document-level values are:

- `file_id`
- `utterance_unit`
- `speaker_map`

### One utterance

Each item in `utterances` becomes one row in `utterances.csv`.

Example JSON:

```json
{
  "utterance_id": "utt_001",
  "speaker": "TS",
  "participant_id": "P001",
  "raw_utterance": "Ouais ...",
  "normalized_utterance": "Ouais ...",
  "verbs": [{ "...": "..." }]
}
```

Example CSV row:

```text
utterance_uid,file_id,utterance_id,participant_id,speaker,verb_count,...
TS_FA_session01_page01:utt_001,TS_FA_session01_page01,utt_001,P001,TS,1,...
```

The `utterance_uid` is created so the row has a stable unique ID:

```python
utterance_uid = f"{file_id}:{utterance_id}"
```

### One verb

Each verb inside an utterance becomes one row in `verbs.csv`.

Example:

```json
"subject": {
  "subject_form": "on",
  "head": "on",
  "subject_type": "pronoun",
  "person": "3",
  "number": "singular"
}
```

becomes:

```text
subject_form,head,subject_type,person,number
on,on,pronoun,3,singular
```

Nested `cue_strength` becomes:

```text
cue_present,cue_expression,cue_type,cue_number
False,,none,
```

Nested `attraction` becomes:

```text
attraction_configuration,attraction_error,attractor_surface_form,attractor_head,attractor_number,attractor_type,structural_complexity_type,linear_distance_words
False,,,,,,infinitival_complement,0
```

Nested `annotation_metadata` becomes:

```text
llm_confidence,low_confidence_reason,comments
0.92,,Futur proche with subject on.
```

### Why flatten?

Shiny tables, Pandas groupby operations, DuckDB SQL, and QueryChat work much better with flat tables:

```sql
SELECT participant_id, AVG(agreement_accuracy)
FROM verbs
WHERE is_analyzable_agreement = TRUE
GROUP BY participant_id
```

That is much easier than querying deeply nested JSON.

## 8. Derived Fields

The preparation script does not only copy fields. It creates analysis-friendly derived fields.

### Agreement status

Raw model field:

```text
produced_agreement = target | non-target | unexpected
```

Derived field:

```text
agreement_status = target | non_target | unexpected | ambiguous | not_applicable | other | unknown
```

Why?

The older annotation drafts used values like `expected`, `correct`, or `error`. The function `_agreement_status()` normalizes those older variants.

Examples:

```python
_agreement_status("target")      # "target"
_agreement_status("non-target")  # "non_target"
_agreement_status("unexpected")  # "unexpected"
_agreement_status("expected")    # "target"
```

### Agreement accuracy

The dashboard needs numeric accuracy.

Rules:

- `target` -> `agreement_accuracy = 1`
- `non_target` -> `agreement_accuracy = 0`
- `unexpected` -> `agreement_accuracy = null`

Why is `unexpected` excluded?

Because unexpected means the target cannot be confidently inferred. Counting it as wrong would mix agreement errors with fragments, false starts, and unclear forms.

### Boolean flags

The script creates:

```text
is_agreement_target_like
is_agreement_non_target
is_unexpected_form
is_analyzable_agreement
```

These are useful for filtering and counting.

Example:

```python
analyzable = verbs["is_analyzable_agreement"].sum()
target_like = verbs["is_agreement_target_like"].sum()
accuracy = target_like / analyzable
```

### Cue group and RQ1 condition

Raw JSON has:

```text
cue_present
cue_expression
cue_type
cue_number
```

Derived fields:

```text
cue_presence_label
cue_group
rq1_condition
```

Current mapping:

```text
quantifier -> quantifier -> quantifier
numeral -> numeral -> numeral
none -> none -> no_cue
ambiguous -> ambiguous -> ambiguous
other unexpected explicit cue -> other_explicit_quantity -> other_explicit_quantity
```

Earlier drafts inferred `cue_group` from cue expressions and could produce legacy groups such as pronoun or morphological. The current pipeline no longer does that. It trusts the final-schema `cue_type` field, which reduces accidental category drift.

### Attraction configuration and RQ2 condition

Raw JSON has:

```text
attraction_configuration
attraction_error
attractor_surface_form
attractor_head
attractor_number
attractor_type
structural_complexity_type
linear_distance_words
```

Derived fields:

```text
has_attraction
has_attraction_error
rq2_configuration
distance_bin
```

Example mapping:

```text
attractor_type = pp -> rq2_configuration = prepositional_phrase
structural_complexity_type contains relative -> rq2_configuration = relative_clause
attraction_configuration = false -> rq2_configuration = no_attraction
```

### Distance bin

Raw value:

```text
linear_distance_words = 0, 1, 2, 3, ...
```

Derived group:

```text
0
1_2
3_5
6_plus
unknown
```

These bins make charts readable.

## 9. Output Tables

### verbs.csv

This is the most important table. It has one row per annotated verb.

Use it when asking:

- How many verbs are target-like?
- Which participant has more non-target forms?
- Which cue condition has lower accuracy?
- Which utterances include attraction configurations?
- Which examples have low confidence?

Example query in Python:

```python
import pandas as pd

verbs = pd.read_csv("outputs/corpus_querychat/verbs.csv")
print(verbs["agreement_status"].value_counts())
```

Example query in SQL:

```sql
SELECT
  participant_id,
  COUNT(*) AS verbs,
  SUM(is_agreement_non_target) AS non_target_verbs,
  AVG(agreement_accuracy) AS agreement_accuracy
FROM verbs
WHERE is_analyzable_agreement = TRUE
GROUP BY participant_id;
```

### utterances.csv

One row per utterance line. Includes counts of verbs inside the utterance.

Useful for:

- finding utterances with no verbs;
- counting transcript lines;
- joining back to raw utterance text;
- inspecting evidence.

### files.csv

One row per source file. Useful when you annotate multiple transcript files.

### learners.csv

Despite the legacy name, this is now a participant summary table. It groups by `participant_id` and `participant_role` when available.

Potential improvement:

Rename the file from `learners.csv` to `participants.csv`, then update the dashboard and docs. It currently remains `learners.csv` for backward compatibility.

### cue_summary.csv

Grouped by:

```text
rq1_condition
cue_group
cue_type
cue_number
```

### attraction_summary.csv

Grouped by:

```text
rq2_configuration
attractor_type
attractor_number
structural_complexity_type
```

## 10. Shiny App Load Phase

When `corpus_querychat_explorer/app.py` starts, it immediately loads prepared tables:

```python
verbs = _load_verbs()
utterances = _load_table(UTTERANCES_PATH, TEXT_COLUMNS | {"utterance_uid"})
files = _load_table(FILES_PATH, TEXT_COLUMNS)
learners = _load_table(LEARNERS_PATH, TEXT_COLUMNS)
metadata = _load_json_file(METADATA_PATH)
research_summary = _load_json_file(RESEARCH_SUMMARY_PATH)
```

This happens once at app startup.

If you regenerate CSV files while the Shiny process is running, restart the app to reload them.

## 11. Data Cleaning in the Shiny App

`_load_table()` reads a CSV with Pandas, then calls `_clean_frame()`.

`_clean_frame()` does three important things:

1. Text columns become Pandas string columns.
2. Numeric columns become numbers.
3. Boolean columns become booleans.

Example:

```python
for column in NUMERIC_COLUMNS.intersection(clean.columns):
    clean[column] = pd.to_numeric(clean[column], errors="coerce")
```

This is necessary because CSV files store everything as text unless you explicitly coerce types.

## 12. QueryChat Integration

The app only enables QueryChat when:

1. `chatlas` and `querychat` import successfully;
2. `GEMINI_API_KEY` exists in `.env`.

The client is created here:

```python
client = ChatGoogle(
    model=os.getenv("QUERYCHAT_GEMINI_MODEL", DEFAULT_QUERYCHAT_GEMINI_MODEL),
    api_key=gemini_api_key,
)
```

Default model:

```text
gemini-2.5-flash-lite
```

Then QueryChat receives the `verbs` table:

```python
qc = QueryChat(
    verbs,
    "verbs",
    client=client,
    greeting=...,
    data_description=...,
    tools=_querychat_tools(),
)
```

Important:

QueryChat queries only the main `verbs` table. The dashboard separately uses `utterances`, but QueryChat's natural-language SQL/dataframe operations are centered on `verbs`.

If you ask:

```text
Show non-target forms for P001
```

QueryChat can translate that into a filter over `verbs`.

If you ask:

```text
How many transcript lines have no verbs?
```

That may be harder because `utterances.csv` is not the main QueryChat table. Potential improvement: expose more tables to QueryChat if the library supports it, or merge utterance-level counts into `verbs` in a careful way.

## 13. Dashboard Reactivity

The server creates a reactive `filtered_verbs()` table.

If QueryChat is disabled:

```python
return verbs.copy()
```

If QueryChat is enabled:

```python
return _clean_frame(_as_pandas(qc_vals.df()))
```

That means every chart uses either:

- all verbs; or
- the QueryChat-filtered subset.

Example:

```python
@render.ui
def metric_tiles():
    return ui.HTML(_metric_tiles(filtered_verbs(), filtered_utterances()))
```

When QueryChat changes the selected dataframe, the KPI tiles automatically recalculate.

## 14. Dashboard KPIs

The KPI cards come from `_metrics()`.

Current KPIs:

```text
Files
Participants
Utterances
Verb Tokens
Analyzable
Accuracy
Non-target
Unexpected
Cued Verbs
Attraction
Low Confidence
Pending Coding
```

The main accuracy formula is:

```python
accuracy = target_like / analyzable
```

where:

```text
analyzable = target + non_target
```

and:

```text
unexpected is excluded
```

## 15. Example End-to-End Workflow

### Step 1: annotate transcript

```powershell
python Pipeline/run_annotation_pipeline.py `
  --provider gemini `
  --model gemini-2.5-flash-lite `
  --source .\transcript.txt `
  --output .\TS_FA_session01_page01_annotation.json
```

### Step 2: inspect the raw JSON

```powershell
Get-Content .\TS_FA_session01_page01_annotation.json -TotalCount 80
```

### Step 3: prepare dashboard data

```powershell
python corpus_querychat_explorer\prepare_corpus_data.py .\TS_FA_session01_page01_annotation.json
```

### Step 4: inspect generated tables

```powershell
Import-Csv outputs\corpus_querychat\verbs.csv |
  Select-Object -First 5 file_id,utterance_id,participant_id,produced_form,agreement_status
```

### Step 5: run dashboard

```powershell
shiny run --reload corpus_querychat_explorer\app.py
```

### Step 6: use QueryChat

Example prompts:

```text
Show all non-target agreement cases with participant_id, raw_utterance, produced_form, target_form, and comments.
```

```text
Compare agreement accuracy between P001 and P002.
```

```text
List unexpected forms with low confidence reasons.
```

```text
Show all verbs where attraction_configuration is true.
```

## 16. Common Failure Points

### The model returns invalid JSON

Symptoms:

- `json.JSONDecodeError`
- output file not written

Where to look:

- `Pipeline/templates/prompt.md`
- `Pipeline/templates/guidelines.md`
- `Pipeline/templates/json_template.json`
- `extract_json()` in `Pipeline/run_annotation_pipeline.py`

Improvements:

- Add a second retry prompt that includes the parse error.
- Save the raw response next to the parsed output.
- Add JSON Schema validation.

### The dashboard shows "Missing"

This usually means one of these:

- the JSON template does not contain a field the dashboard expects;
- the model left a field null;
- the field exists only for a subset, such as attraction rows.

Example already fixed:

- `intervener_type` was an old field.
- It is not in the final JSON template.
- It was removed from the dashboard.

### QueryChat gives a Gemini 503

That is a provider-side high-demand error.

Use:

```powershell
QUERYCHAT_GEMINI_MODEL=gemini-2.5-flash-lite
```

or retry later.

### Counts look different between JSON and dashboard

Remember:

- `utterances.csv` counts all utterance lines;
- `verbs.csv` counts only annotated verbs;
- many dashboard charts are verb-level, not utterance-level;
- `unexpected` forms are excluded from accuracy.

## 17. Ideas for Improvement

### Add schema validation

Create a JSON Schema for `json_template.json` and validate every output.

Example package:

```powershell
pip install jsonschema
```

Example validation:

```python
from jsonschema import validate

validate(instance=annotation_json, schema=annotation_schema)
```

### Add model retry and repair

If JSON parsing fails:

1. Save raw response.
2. Ask model to repair only the JSON.
3. Parse again.

### Add confidence review workflow

Use `low_confidence_reason` to produce a review queue:

```python
import pandas as pd

verbs = pd.read_csv("outputs/corpus_querychat/verbs.csv")
review = verbs[
    verbs["low_confidence_reason"].fillna("").str.strip().ne("")
    | (verbs["llm_confidence"] < 0.75)
]
review.to_csv("outputs/corpus_querychat/review_queue.csv", index=False)
```

### Rename `learners.csv`

It now represents participants, not only learners.

Migration plan:

1. Write both `learners.csv` and `participants.csv`.
2. Update app to read `participants.csv`.
3. Keep `learners.csv` for backward compatibility.
4. Remove it later if no longer needed.

### Separate utterance-level QueryChat

If QueryChat supports multiple tables, expose:

- `verbs`
- `utterances`
- `files`
- `participants`

If not, create a second QueryChat instance for `utterances`.

### Keep cue logic schema-driven

The final schema has clear cue types:

```text
quantifier
numeral
none
ambiguous
```

The pipeline now maps those values directly in `_cue_group()`. If you add new allowed cue types later, update `Pipeline/templates/json_template.json`, `Pipeline/templates/guidelines.md`, `_cue_group()`, and `_rq1_condition()` together.

### Add tests

Minimum useful tests:

```text
test final JSON loads
test participant_id is copied correctly
test unexpected forms are excluded from accuracy
test no obsolete columns appear
test research_summary.json contains no NaN
```

## 18. Mental Model

Think of the project as two separate pipelines:

### Annotation pipeline

```text
human transcript -> LLM prompt -> final annotation JSON
```

This part is about linguistic correctness and model reliability.

### Explorer pipeline

```text
final annotation JSON -> flat tables -> dashboard + QueryChat
```

This part is about data normalization, metrics, filtering, and visualization.

Keeping these separate is good. It means you can improve annotation quality without rewriting the dashboard, and improve the dashboard without changing the LLM prompt.
