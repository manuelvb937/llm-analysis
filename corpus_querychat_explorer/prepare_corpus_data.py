from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

try:
    import duckdb
except ModuleNotFoundError:
    duckdb = None


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "Annotation.py"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "corpus_querychat"

VERBS_FILENAME = "verbs.csv"
UTTERANCES_FILENAME = "utterances.csv"
FILES_FILENAME = "files.csv"
LEARNERS_FILENAME = "learners.csv"
CUE_SUMMARY_FILENAME = "cue_summary.csv"
ATTRACTION_SUMMARY_FILENAME = "attraction_summary.csv"
METADATA_FILENAME = "metadata.json"
RESEARCH_SUMMARY_FILENAME = "research_summary.json"
DUCKDB_FILENAME = "corpus_analysis.duckdb"

SOURCE_EXTENSIONS = {".json", ".py", ".txt"}


VERB_COLUMNS = [
    "verb_uid",
    "source_path",
    "file_id",
    "annotation_scope",
    "annotation_notes",
    "utterance_unit",
    "utterance_id",
    "learner_id",
    "speaker",
    "raw_utterance",
    "normalized_utterance",
    "verb_index",
    "produced_form",
    "produced_agreement",
    "target_form",
    "lemma",
    "tense_mood",
    "category",
    "subject_form",
    "head",
    "subject_type",
    "person",
    "number",
    "cue_present",
    "cue_expression",
    "cue_type",
    "cue_number",
    "attraction_configuration",
    "attraction_error",
    "attractor_surface_form",
    "attractor_head",
    "attractor_number",
    "attractor_type",
    "intervener_type",
    "structural_complexity_type",
    "linear_distance_words",
    "llm_confidence",
    "comments",
    "agreement_status",
    "agreement_accuracy",
    "is_agreement_target_like",
    "is_agreement_non_target",
    "is_unexpected_form",
    "is_analyzable_agreement",
    "cue_presence_label",
    "cue_group",
    "rq1_condition",
    "has_attraction",
    "has_attraction_error",
    "rq2_configuration",
    "distance_bin",
    "finite_status",
]

UTTERANCE_COLUMNS = [
    "utterance_uid",
    "source_path",
    "file_id",
    "annotation_scope",
    "utterance_id",
    "learner_id",
    "speaker",
    "raw_utterance",
    "normalized_utterance",
    "verb_count",
    "analyzable_verb_count",
    "target_like_verb_count",
    "non_target_verb_count",
    "unexpected_form_count",
    "cue_present_verb_count",
    "attraction_configuration_count",
    "attraction_error_count",
]


TEXT_ID_COLUMNS = {
    "verb_uid",
    "source_path",
    "file_id",
    "annotation_scope",
    "annotation_notes",
    "utterance_unit",
    "utterance_id",
    "learner_id",
    "speaker",
    "raw_utterance",
    "normalized_utterance",
    "produced_form",
    "produced_agreement",
    "target_form",
    "lemma",
    "tense_mood",
    "category",
    "subject_form",
    "head",
    "subject_type",
    "person",
    "number",
    "cue_expression",
    "cue_type",
    "cue_number",
    "attractor_surface_form",
    "attractor_head",
    "attractor_number",
    "attractor_type",
    "intervener_type",
    "structural_complexity_type",
    "comments",
    "agreement_status",
    "cue_presence_label",
    "cue_group",
    "rq1_condition",
    "rq2_configuration",
    "distance_bin",
    "finite_status",
}


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return value


def _label(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return " | ".join(_label(item) for item in value if _label(item))
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    return str(value).strip()


def _normal(value: Any) -> str:
    text = _label(value).lower().strip()
    return (
        text.replace("-", "_")
        .replace(" ", "_")
        .replace("/", "_")
        .replace("__", "_")
    )


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not pd.isna(value):
        return bool(value)
    text = _normal(value)
    return text in {"true", "1", "yes", "y", "present"}


def _as_number(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _get(mapping: Any, key: str, default: Any = None) -> Any:
    return mapping.get(key, default) if isinstance(mapping, dict) else default


def _load_json(path: Path) -> Any:
    last_error: Exception | None = None
    for encoding in ("utf-8", "utf-8-sig", "cp1252"):
        try:
            return json.loads(path.read_text(encoding=encoding))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            last_error = exc
    raise ValueError(f"Could not parse JSON from {path}: {last_error}")


def _source_files(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    if not input_path.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")
    files = [
        path
        for path in sorted(input_path.rglob("*"))
        if path.is_file() and path.suffix.lower() in SOURCE_EXTENSIONS
    ]
    if not files:
        raise FileNotFoundError(f"No JSON-like files found under {input_path}")
    return files


def _documents_from_payload(payload: Any) -> Iterable[dict[str, Any]]:
    if isinstance(payload, dict) and isinstance(payload.get("utterances"), list):
        yield payload
        return

    if isinstance(payload, list):
        for item in payload:
            yield from _documents_from_payload(item)
        return

    if isinstance(payload, dict):
        for key in ("documents", "files", "annotations", "records", "items", "data"):
            if key in payload:
                yield from _documents_from_payload(payload[key])


def _agreement_status(value: Any) -> str:
    text = _normal(value)
    if not text:
        return "unknown"
    if "not_applicable" in text or text in {"na", "n_a", "non_finite"}:
        return "not_applicable"
    if "ambiguous" in text or "uncertain" in text:
        return "ambiguous"
    if "unexpected" in text or "non_target" in text or "incorrect" in text or text == "error":
        return "non_target"
    if text in {"expected", "target", "correct", "accurate", "ok"} or "target_like" in text:
        return "target"
    return "other"


def _cue_group(cue_present: bool, cue_type: Any, cue_expression: Any) -> str:
    cue_type_text = _normal(cue_type)
    expression_text = _normal(cue_expression)
    combined = f"{cue_type_text} {expression_text}"

    if not cue_present:
        return "none"
    if "ambiguous" in combined or "variable" in combined:
        return "ambiguous"
    if any(token in combined for token in ("lexical", "semantic", "quant", "plusieurs", "beaucoup", "quelques", "nombreux", "tous")):
        return "lexical_semantic"
    if any(token in combined for token in ("pronoun", "pronom", "je", "tu", "il", "elle", "on", "nous", "vous", "ils", "elles", "ce", "c'")):
        return "pronoun"
    if any(token in combined for token in ("morph", "determiner", "determinant", "article", "les", "des", "ces", "mes", "tes", "ses", "nos", "vos", "leurs", "le", "la", "un", "une")):
        return "morphological"
    if cue_type_text in {"none", "null"}:
        return "none"
    return "other"


def _rq1_condition(cue_group: str) -> str:
    return {
        "lexical_semantic": "strong_lexical_semantic",
        "morphological": "weak_morphological",
        "pronoun": "pronoun",
        "none": "no_cue",
        "ambiguous": "ambiguous",
    }.get(cue_group, "other_cue")


def _rq2_configuration(has_attraction: bool, attractor_type: Any, intervener_type: Any, complexity_type: Any) -> str:
    if not has_attraction:
        return "no_attraction"

    combined = " ".join(
        _normal(value) for value in (attractor_type, intervener_type, complexity_type)
    )
    if "relative" in combined:
        return "relative_clause"
    if combined in {"", "none"}:
        return "other_attraction"
    if "pp" in combined or "prepositional" in combined:
        return "prepositional_phrase"
    if "object" in combined:
        return "object"
    if "clitic" in combined:
        return "clitic"
    if "false_start" in combined:
        return "false_start"
    return "other_attraction"


def _distance_bin(value: Any) -> str:
    number = _as_number(value)
    if number is None:
        return "unknown"
    if number <= 0:
        return "0"
    if number <= 2:
        return "1_2"
    if number <= 5:
        return "3_5"
    return "6_plus"


def _finite_status(tense_mood: Any, agreement_status: str) -> str:
    text = _normal(tense_mood)
    if agreement_status == "not_applicable":
        return "non_finite_or_not_applicable"
    if "infinitive" in text or "participle" in text:
        return "non_finite_or_not_applicable"
    if not text or text == "unknown":
        return "unknown"
    return "finite"


def _verb_row(
    source_path: Path,
    document: dict[str, Any],
    utterance: dict[str, Any],
    verb: dict[str, Any],
    verb_position: int,
) -> dict[str, Any]:
    subject = _get(verb, "subject", {})
    cue = _get(verb, "cue_strength", {})
    attraction = _get(verb, "attraction", {})
    metadata = _get(verb, "annotation_metadata", {})

    file_id = _label(_get(utterance, "file_id")) or _label(_get(document, "file_id")) or source_path.stem
    utterance_id = _label(_get(utterance, "utterance_id")) or f"utt_{verb_position:03d}"
    produced_agreement = _get(verb, "produced_agreement")
    agreement_status = _agreement_status(produced_agreement)
    cue_present = _as_bool(_get(cue, "cue_present"))
    cue_group = _cue_group(cue_present, _get(cue, "cue_type"), _get(cue, "cue_expression"))
    has_attraction = _as_bool(_get(attraction, "attraction_configuration"))
    has_attraction_error = _as_bool(_get(attraction, "attraction_error"))
    finite_status = _finite_status(_get(verb, "tense_mood"), agreement_status)

    return {
        "verb_uid": f"{file_id}:{utterance_id}:v{_get(verb, 'verb_index', verb_position)}",
        "source_path": str(source_path),
        "file_id": file_id,
        "annotation_scope": _get(document, "annotation_scope"),
        "annotation_notes": _get(document, "annotation_notes"),
        "utterance_unit": _get(document, "utterance_unit"),
        "utterance_id": utterance_id,
        "learner_id": _get(utterance, "learner_id"),
        "speaker": _get(utterance, "speaker"),
        "raw_utterance": _get(utterance, "raw_utterance"),
        "normalized_utterance": _get(utterance, "normalized_utterance"),
        "verb_index": _get(verb, "verb_index", verb_position),
        "produced_form": _get(verb, "produced_form"),
        "produced_agreement": produced_agreement,
        "target_form": _get(verb, "target_form"),
        "lemma": _get(verb, "lemma"),
        "tense_mood": _get(verb, "tense_mood"),
        "category": _get(verb, "category"),
        "subject_form": _get(subject, "subject_form"),
        "head": _get(subject, "head"),
        "subject_type": _get(subject, "subject_type"),
        "person": _get(subject, "person"),
        "number": _get(subject, "number"),
        "cue_present": cue_present,
        "cue_expression": _get(cue, "cue_expression"),
        "cue_type": _get(cue, "cue_type"),
        "cue_number": _get(cue, "cue_number"),
        "attraction_configuration": has_attraction,
        "attraction_error": has_attraction_error,
        "attractor_surface_form": _get(attraction, "attractor_surface_form"),
        "attractor_head": _get(attraction, "attractor_head"),
        "attractor_number": _get(attraction, "attractor_number"),
        "attractor_type": _get(attraction, "attractor_type"),
        "intervener_type": _get(attraction, "intervener_type"),
        "structural_complexity_type": _get(attraction, "structural_complexity_type"),
        "linear_distance_words": _get(attraction, "linear_distance_words"),
        "llm_confidence": _get(metadata, "llm_confidence"),
        "comments": _get(metadata, "comments"),
        "agreement_status": agreement_status,
        "agreement_accuracy": 1 if agreement_status == "target" else 0 if agreement_status == "non_target" else None,
        "is_agreement_target_like": agreement_status == "target",
        "is_agreement_non_target": agreement_status == "non_target",
        "is_unexpected_form": "unexpected" in _normal(produced_agreement),
        "is_analyzable_agreement": agreement_status in {"target", "non_target"},
        "cue_presence_label": "cue_present" if cue_present else "no_cue",
        "cue_group": cue_group,
        "rq1_condition": _rq1_condition(cue_group),
        "has_attraction": has_attraction,
        "has_attraction_error": has_attraction_error,
        "rq2_configuration": _rq2_configuration(
            has_attraction,
            _get(attraction, "attractor_type"),
            _get(attraction, "intervener_type"),
            _get(attraction, "structural_complexity_type"),
        ),
        "distance_bin": _distance_bin(_get(attraction, "linear_distance_words")),
        "finite_status": finite_status,
    }


def _rows_from_document(source_path: Path, document: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    verb_rows: list[dict[str, Any]] = []
    utterance_rows: list[dict[str, Any]] = []
    file_id = _label(_get(document, "file_id")) or source_path.stem

    for utterance_position, utterance in enumerate(_get(document, "utterances", []), start=1):
        if not isinstance(utterance, dict):
            continue
        utterance_id = _label(_get(utterance, "utterance_id")) or f"utt_{utterance_position:03d}"
        verbs = _get(utterance, "verbs", [])
        if not isinstance(verbs, list):
            verbs = []

        local_verb_rows = [
            _verb_row(source_path, document, utterance, verb, verb_position)
            for verb_position, verb in enumerate(verbs, start=1)
            if isinstance(verb, dict)
        ]
        verb_rows.extend(local_verb_rows)

        utterance_rows.append(
            {
                "utterance_uid": f"{file_id}:{utterance_id}",
                "source_path": str(source_path),
                "file_id": _label(_get(utterance, "file_id")) or file_id,
                "annotation_scope": _get(document, "annotation_scope"),
                "utterance_id": utterance_id,
                "learner_id": _get(utterance, "learner_id"),
                "speaker": _get(utterance, "speaker"),
                "raw_utterance": _get(utterance, "raw_utterance"),
                "normalized_utterance": _get(utterance, "normalized_utterance"),
                "verb_count": len(local_verb_rows),
                "analyzable_verb_count": sum(bool(row["is_analyzable_agreement"]) for row in local_verb_rows),
                "target_like_verb_count": sum(bool(row["is_agreement_target_like"]) for row in local_verb_rows),
                "non_target_verb_count": sum(bool(row["is_agreement_non_target"]) for row in local_verb_rows),
                "unexpected_form_count": sum(bool(row["is_unexpected_form"]) for row in local_verb_rows),
                "cue_present_verb_count": sum(bool(row["cue_present"]) for row in local_verb_rows),
                "attraction_configuration_count": sum(bool(row["has_attraction"]) for row in local_verb_rows),
                "attraction_error_count": sum(bool(row["has_attraction_error"]) for row in local_verb_rows),
            }
        )

    return verb_rows, utterance_rows


def _records_to_frame(records: list[dict[str, Any]], columns: list[str] | None = None) -> pd.DataFrame:
    frame = pd.DataFrame(records)
    if columns is not None:
        for column in columns:
            if column not in frame:
                frame[column] = pd.NA
        frame = frame.loc[:, columns]

    for column in frame.columns:
        frame[column] = frame[column].map(_csv_value)
    for column in TEXT_ID_COLUMNS.intersection(frame.columns):
        frame[column] = frame[column].astype("string")
    for column in ("verb_index", "linear_distance_words", "llm_confidence", "agreement_accuracy"):
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def _rate(numerator: float, denominator: float) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def _accuracy_summary(frame: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    for column in group_columns:
        if column not in frame:
            return pd.DataFrame()

    grouped = (
        frame.assign(
            analyzable=frame["is_analyzable_agreement"].astype(bool),
            target_like=frame["is_agreement_target_like"].astype(bool),
            non_target=frame["is_agreement_non_target"].astype(bool),
            unexpected=frame["is_unexpected_form"].astype(bool),
            attraction_config=frame["has_attraction"].astype(bool),
            attraction_err=frame["has_attraction_error"].astype(bool),
        )
        .groupby(group_columns, dropna=False)
        .agg(
            verbs=("verb_uid", "size"),
            analyzable_verbs=("analyzable", "sum"),
            target_like_verbs=("target_like", "sum"),
            non_target_verbs=("non_target", "sum"),
            unexpected_forms=("unexpected", "sum"),
            attraction_configurations=("attraction_config", "sum"),
            attraction_errors=("attraction_err", "sum"),
            avg_llm_confidence=("llm_confidence", "mean"),
        )
        .reset_index()
    )
    grouped["agreement_accuracy"] = [
        _rate(target, analyzable)
        for target, analyzable in zip(grouped["target_like_verbs"], grouped["analyzable_verbs"])
    ]
    grouped["attraction_error_rate"] = [
        _rate(errors, configs)
        for errors, configs in zip(grouped["attraction_errors"], grouped["attraction_configurations"])
    ]
    return grouped.sort_values(["verbs", "non_target_verbs"], ascending=False)


def _file_summary(verbs: pd.DataFrame, utterances: pd.DataFrame) -> pd.DataFrame:
    if utterances.empty:
        return pd.DataFrame()
    files = (
        utterances.groupby("file_id", dropna=False)
        .agg(
            source_path=("source_path", "first"),
            utterances=("utterance_uid", "size"),
            learners=("learner_id", lambda values: " | ".join(sorted({str(value) for value in values.dropna()}))),
            verb_count=("verb_count", "sum"),
            analyzable_verb_count=("analyzable_verb_count", "sum"),
        )
        .reset_index()
    )
    if verbs.empty:
        files["agreement_accuracy"] = pd.NA
        files["attraction_errors"] = 0
        return files

    verb_stats = _accuracy_summary(verbs, ["file_id"])
    keep = [
        "file_id",
        "target_like_verbs",
        "non_target_verbs",
        "unexpected_forms",
        "attraction_configurations",
        "attraction_errors",
        "agreement_accuracy",
        "avg_llm_confidence",
    ]
    return files.merge(verb_stats[keep], on="file_id", how="left")


def _learner_summary(verbs: pd.DataFrame) -> pd.DataFrame:
    if verbs.empty or "learner_id" not in verbs:
        return pd.DataFrame()
    return _accuracy_summary(verbs, ["learner_id"])


def _research_summary(verbs: pd.DataFrame, utterances: pd.DataFrame, source_files: list[Path]) -> dict[str, Any]:
    analyzable = int(verbs["is_analyzable_agreement"].sum()) if not verbs.empty else 0
    target_like = int(verbs["is_agreement_target_like"].sum()) if not verbs.empty else 0
    non_target = int(verbs["is_agreement_non_target"].sum()) if not verbs.empty else 0
    attraction_configs = int(verbs["has_attraction"].sum()) if not verbs.empty else 0
    attraction_errors = int(verbs["has_attraction_error"].sum()) if not verbs.empty else 0

    rq1 = _accuracy_summary(verbs, ["rq1_condition"]).to_dict(orient="records") if not verbs.empty else []
    rq2 = _accuracy_summary(verbs, ["rq2_configuration"]).to_dict(orient="records") if not verbs.empty else []

    return {
        "source_file_count": len(source_files),
        "file_count": int(verbs["file_id"].nunique(dropna=True)) if not verbs.empty else int(utterances["file_id"].nunique(dropna=True)) if not utterances.empty else 0,
        "utterance_count": int(len(utterances)),
        "verb_count": int(len(verbs)),
        "analyzable_verb_count": analyzable,
        "target_like_verb_count": target_like,
        "non_target_verb_count": non_target,
        "overall_agreement_accuracy": _rate(target_like, analyzable),
        "attraction_configuration_count": attraction_configs,
        "attraction_error_count": attraction_errors,
        "attraction_error_rate": _rate(attraction_errors, attraction_configs),
        "rq1_by_cue_condition": rq1,
        "rq2_by_configuration": rq2,
    }


def _write_duckdb(tables: dict[str, pd.DataFrame], duckdb_path: Path) -> bool:
    if duckdb is None:
        return False
    with duckdb.connect(str(duckdb_path)) as conn:
        for table_name, frame in tables.items():
            view_name = f"{table_name}_df"
            conn.register(view_name, frame)
            conn.execute(f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM {view_name}")
            conn.unregister(view_name)
    return True


def _resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Flatten L2 French agreement annotations into QueryChat-ready tables.",
    )
    parser.add_argument(
        "input_path",
        nargs="?",
        default=DEFAULT_INPUT,
        help="A structured annotation JSON file, a JSON-like .py file, or a folder of files.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for CSV, JSON, and optional DuckDB files.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    input_path = _resolve_path(args.input_path)
    output_dir = _resolve_path(args.output_dir)
    source_files = _source_files(input_path)

    verb_records: list[dict[str, Any]] = []
    utterance_records: list[dict[str, Any]] = []
    skipped_files: list[dict[str, str]] = []

    for source_file in source_files:
        try:
            payload = _load_json(source_file)
            documents = list(_documents_from_payload(payload))
        except Exception as exc:
            skipped_files.append({"source_path": str(source_file), "error": str(exc)})
            continue

        if not documents:
            skipped_files.append({"source_path": str(source_file), "error": "No document with an utterances list was found."})
            continue

        for document in documents:
            verbs, utterances = _rows_from_document(source_file, document)
            verb_records.extend(verbs)
            utterance_records.extend(utterances)

    output_dir.mkdir(parents=True, exist_ok=True)

    verbs = _records_to_frame(verb_records, VERB_COLUMNS)
    utterances = _records_to_frame(utterance_records, UTTERANCE_COLUMNS)
    cue_summary = _accuracy_summary(verbs, ["rq1_condition", "cue_group", "cue_type", "cue_number"])
    attraction_summary = _accuracy_summary(
        verbs,
        ["rq2_configuration", "intervener_type", "attractor_type", "structural_complexity_type"],
    )
    files = _file_summary(verbs, utterances)
    learners = _learner_summary(verbs)

    metadata = {
        "input_path": str(input_path),
        "output_dir": str(output_dir),
        "source_files": [str(path) for path in source_files],
        "skipped_files": skipped_files,
        "tables": {
            "verbs": VERBS_FILENAME,
            "utterances": UTTERANCES_FILENAME,
            "files": FILES_FILENAME,
            "learners": LEARNERS_FILENAME,
            "cue_summary": CUE_SUMMARY_FILENAME,
            "attraction_summary": ATTRACTION_SUMMARY_FILENAME,
        },
    }
    research_summary = _research_summary(verbs, utterances, source_files)

    tables = {
        "verbs": verbs,
        "utterances": utterances,
        "files": files,
        "learners": learners,
        "cue_summary": cue_summary,
        "attraction_summary": attraction_summary,
    }

    for table_name, frame in tables.items():
        frame.to_csv(output_dir / f"{table_name}.csv", index=False, encoding="utf-8")

    (output_dir / METADATA_FILENAME).write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / RESEARCH_SUMMARY_FILENAME).write_text(
        json.dumps(research_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    duckdb_written = _write_duckdb(tables, output_dir / DUCKDB_FILENAME)

    print(f"Read {len(source_files)} source file(s) from {input_path}")
    print(f"Wrote {len(verbs)} verb rows to {output_dir / VERBS_FILENAME}")
    print(f"Wrote {len(utterances)} utterance rows to {output_dir / UTTERANCES_FILENAME}")
    print(f"Wrote summaries to {output_dir}")
    if skipped_files:
        print(f"Skipped {len(skipped_files)} file(s); see {output_dir / METADATA_FILENAME}")
    if duckdb_written:
        print(f"Wrote DuckDB database to {output_dir / DUCKDB_FILENAME}")
    else:
        print("Skipped DuckDB database because duckdb is not installed.")


if __name__ == "__main__":
    main()
