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
# Prefer the new final-output naming pattern, but keep Annotation.py as an old fallback.
DEFAULT_INPUT = next(iter(sorted(ROOT.glob("*_annotation.json"))), ROOT / "Annotation.py")
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


# This is the canonical verb-level table schema consumed by Shiny and QueryChat.
VERB_COLUMNS = [
    "verb_uid",
    "source_path",
    "file_id",
    "annotation_scope",
    "annotation_notes",
    "utterance_unit",
    "utterance_id",
    "learner_id",
    "participant_id",
    "participant_role",
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
    "category_coding_status",
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
    "structural_complexity_type",
    "linear_distance_words",
    "llm_confidence",
    "low_confidence_reason",
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

# This is the utterance-level table schema; one row equals one transcript line.
UTTERANCE_COLUMNS = [
    "utterance_uid",
    "source_path",
    "file_id",
    "annotation_scope",
    "utterance_id",
    "learner_id",
    "participant_id",
    "participant_role",
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


# CSV readers can otherwise convert ids like 001 or booleans into the wrong type.
TEXT_ID_COLUMNS = {
    "verb_uid",
    "source_path",
    "file_id",
    "annotation_scope",
    "annotation_notes",
    "utterance_unit",
    "utterance_id",
    "learner_id",
    "participant_id",
    "participant_role",
    "speaker",
    "raw_utterance",
    "normalized_utterance",
    "produced_form",
    "produced_agreement",
    "target_form",
    "lemma",
    "tense_mood",
    "category",
    "category_coding_status",
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
    "structural_complexity_type",
    "low_confidence_reason",
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
    # CSV cells cannot hold nested Python objects, so preserve them as JSON strings.
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return value


def _json_safe(value: Any) -> Any:
    # json.dumps(..., allow_nan=False) rejects pandas/NumPy NaN values, so normalize them.
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(value, "item"):
        try:
            return _json_safe(value.item())
        except (TypeError, ValueError):
            pass
    return value


def _label(value: Any) -> str:
    # Convert any scalar-ish annotation value into a clean display/query string.
    if value is None:
        return ""
    if isinstance(value, list):
        return " | ".join(_label(item) for item in value if _label(item))
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    return str(value).strip()


def _normal(value: Any) -> str:
    # Normalized labels make string matching robust to spaces, hyphens, and case.
    text = _label(value).lower().strip()
    return (
        text.replace("-", "_")
        .replace(" ", "_")
        .replace("/", "_")
        .replace("__", "_")
    )


def _as_bool(value: Any) -> bool:
    # Model outputs and CSV reloads can represent booleans as bools, numbers, or strings.
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not pd.isna(value):
        return bool(value)
    text = _normal(value)
    return text in {"true", "1", "yes", "y", "present"}


def _as_number(value: Any) -> float | None:
    # Numeric annotation fields can be null, blank, strings, or actual numbers.
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
    # Safe dict access keeps malformed rows from crashing the whole preparation run.
    return mapping.get(key, default) if isinstance(mapping, dict) else default


def _participant_fields(document: dict[str, Any], utterance: dict[str, Any]) -> dict[str, str | None]:
    # Final JSON stores stable participant ids either on utterances or in top-level speaker_map.
    speaker = _label(_get(utterance, "speaker"))
    speaker_map = _get(document, "speaker_map", {})
    speaker_info = _get(speaker_map, speaker, {})

    # learner_id is retained as a backward-compatible alias for older dashboard queries.
    participant_id = (
        _label(_get(utterance, "participant_id"))
        or _label(_get(speaker_info, "participant_id"))
        or _label(_get(utterance, "learner_id"))
    )
    participant_role = (
        _label(_get(utterance, "participant_role"))
        or _label(_get(speaker_info, "role"))
    )
    learner_id = _label(_get(utterance, "learner_id")) or participant_id

    return {
        "learner_id": learner_id or None,
        "participant_id": participant_id or None,
        "participant_role": participant_role or None,
    }


def _load_json(path: Path) -> Any:
    # Try common encodings because transcript exports may include BOMs or Windows text.
    last_error: Exception | None = None
    for encoding in ("utf-8", "utf-8-sig", "cp1252"):
        try:
            return json.loads(path.read_text(encoding=encoding))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            last_error = exc
    raise ValueError(f"Could not parse JSON from {path}: {last_error}")


def _source_files(input_path: Path) -> list[Path]:
    # The prep script can process one file or recursively process a folder.
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
    # Accept the final single-document shape directly.
    if isinstance(payload, dict) and isinstance(payload.get("utterances"), list):
        yield payload
        return

    # Also accept wrapper lists or batch envelopes for future multi-file runs.
    if isinstance(payload, list):
        for item in payload:
            yield from _documents_from_payload(item)
        return

    if isinstance(payload, dict):
        for key in ("documents", "files", "annotations", "records", "items", "data"):
            if key in payload:
                yield from _documents_from_payload(payload[key])


def _agreement_status(value: Any) -> str:
    # Normalize final and legacy agreement labels into a small analysis vocabulary.
    text = _normal(value)
    if not text:
        return "unknown"
    if text == "unexpected" or text.startswith("unexpected_"):
        return "unexpected"
    if "not_applicable" in text or text in {"na", "n_a", "non_finite"}:
        return "not_applicable"
    if "ambiguous" in text or "uncertain" in text:
        return "ambiguous"
    if "unexpected" in text or "non_target" in text or "incorrect" in text or text == "error":
        return "non_target"
    if text in {"expected", "target", "correct", "accurate", "ok"} or "target_like" in text:
        return "target"
    return "other"


def _cue_group(cue_present: bool, cue_type: Any) -> str:
    # Final-schema cue_type is already tightly controlled, so do not infer cues from words.
    cue_type_text = _normal(cue_type)

    if cue_type_text in {"quantifier", "numeral", "ambiguous"}:
        return cue_type_text
    if cue_type_text in {"", "none", "null"}:
        return "none"
    if cue_present:
        return "other_explicit_quantity"
    return "none"


def _rq1_condition(cue_group: str) -> str:
    # RQ1 buckets now mirror the final cue_type schema instead of legacy heuristics.
    return {
        "quantifier": "quantifier",
        "numeral": "numeral",
        "none": "no_cue",
        "ambiguous": "ambiguous",
        "other_explicit_quantity": "other_explicit_quantity",
    }.get(cue_group, "other_cue")


def _rq2_configuration(has_attraction: bool, attractor_type: Any, complexity_type: Any) -> str:
    # RQ2 buckets are derived only from fields that exist in the final JSON template.
    if not has_attraction:
        return "no_attraction"

    combined = " ".join(
        _normal(value) for value in (attractor_type, complexity_type)
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
    # Binned distance keeps dashboard charts readable.
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
    # Accuracy denominators usually need to separate finite agreement from non-finite forms.
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
    # Flatten one nested verb annotation into one CSV-ready row.
    subject = _get(verb, "subject", {})
    cue = _get(verb, "cue_strength", {})
    attraction = _get(verb, "attraction", {})
    metadata = _get(verb, "annotation_metadata", {})

    file_id = _label(_get(utterance, "file_id")) or _label(_get(document, "file_id")) or source_path.stem
    utterance_id = _label(_get(utterance, "utterance_id")) or f"utt_{verb_position:03d}"
    participant = _participant_fields(document, utterance)
    produced_agreement = _get(verb, "produced_agreement")
    agreement_status = _agreement_status(produced_agreement)
    cue_present = _as_bool(_get(cue, "cue_present"))
    cue_group = _cue_group(cue_present, _get(cue, "cue_type"))
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
        "learner_id": participant["learner_id"],
        "participant_id": participant["participant_id"],
        "participant_role": participant["participant_role"],
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
        "category_coding_status": _get(verb, "category_coding_status"),
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
        "structural_complexity_type": _get(attraction, "structural_complexity_type"),
        "linear_distance_words": _get(attraction, "linear_distance_words"),
        "llm_confidence": _get(metadata, "llm_confidence"),
        "low_confidence_reason": _get(metadata, "low_confidence_reason"),
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
            _get(attraction, "structural_complexity_type"),
        ),
        "distance_bin": _distance_bin(_get(attraction, "linear_distance_words")),
        "finite_status": finite_status,
    }


def _rows_from_document(source_path: Path, document: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    # Build both verb-level and utterance-level rows in one pass over the document.
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
        participant = _participant_fields(document, utterance)

        # Each verb becomes a row in verbs.csv; utterances with zero verbs are still kept below.
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
                "learner_id": participant["learner_id"],
                "participant_id": participant["participant_id"],
                "participant_role": participant["participant_role"],
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
    # Convert collected dictionaries into a stable-column Pandas table.
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
    # Return None instead of crashing or emitting infinity when a group has no denominator.
    if denominator == 0:
        return None
    return numerator / denominator


def _accuracy_summary(frame: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    # Shared summary helper for cue, attraction, file, and participant tables.
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
    # File summaries start from utterances so transcript lines with zero verbs are counted.
    if utterances.empty:
        return pd.DataFrame()
    identity_column = "participant_id" if "participant_id" in utterances else "learner_id"
    files = (
        utterances.groupby("file_id", dropna=False)
        .agg(
            source_path=("source_path", "first"),
            utterances=("utterance_uid", "size"),
            participants=(identity_column, lambda values: " | ".join(sorted({str(value) for value in values.dropna()}))),
            verb_count=("verb_count", "sum"),
            analyzable_verb_count=("analyzable_verb_count", "sum"),
        )
        .reset_index()
    )
    if "participant_role" in utterances:
        roles = (
            utterances.groupby("file_id", dropna=False)["participant_role"]
            .apply(lambda values: " | ".join(sorted({str(value) for value in values.dropna()})))
            .reset_index(name="participant_roles")
        )
        files = files.merge(roles, on="file_id", how="left")
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
    # The output filename is legacy; the grouping now prefers final-schema participant ids.
    if verbs.empty or "learner_id" not in verbs:
        return pd.DataFrame()
    if "participant_id" in verbs and verbs["participant_id"].notna().any():
        group_columns = ["participant_id"]
        if "participant_role" in verbs and verbs["participant_role"].notna().any():
            group_columns.append("participant_role")
        return _accuracy_summary(verbs, group_columns)
    return _accuracy_summary(verbs, ["learner_id"])


def _research_summary(verbs: pd.DataFrame, utterances: pd.DataFrame, source_files: list[Path]) -> dict[str, Any]:
    # Compact JSON summary for quick checks and README-style reporting.
    analyzable = int(verbs["is_analyzable_agreement"].sum()) if not verbs.empty else 0
    target_like = int(verbs["is_agreement_target_like"].sum()) if not verbs.empty else 0
    non_target = int(verbs["is_agreement_non_target"].sum()) if not verbs.empty else 0
    attraction_configs = int(verbs["has_attraction"].sum()) if not verbs.empty else 0
    attraction_errors = int(verbs["has_attraction_error"].sum()) if not verbs.empty else 0
    participant_column = "participant_id" if "participant_id" in verbs else "learner_id"
    participant_count = int(verbs[participant_column].nunique(dropna=True)) if not verbs.empty and participant_column in verbs else 0
    learner_participant_count = participant_count
    if not verbs.empty and {"participant_id", "participant_role"}.issubset(verbs.columns):
        learner_participant_count = int(
            verbs.loc[
                verbs["participant_role"].astype("string").str.lower().eq("learner"),
                "participant_id",
            ].nunique(dropna=True)
        )
    low_confidence_count = 0
    pending_category_count = 0
    if not verbs.empty:
        low_reason = (
            verbs["low_confidence_reason"].astype("string").fillna("").str.strip().ne("")
            if "low_confidence_reason" in verbs
            else pd.Series(False, index=verbs.index)
        )
        low_score = (
            pd.to_numeric(verbs["llm_confidence"], errors="coerce").lt(0.75)
            if "llm_confidence" in verbs
            else pd.Series(False, index=verbs.index)
        )
        low_confidence_count = int((low_reason | low_score).sum())
        pending_category_count = int(
            verbs.get("category_coding_status", pd.Series(index=verbs.index))
            .astype("string")
            .eq("pending_human_coding")
            .sum()
        )

    rq1 = _accuracy_summary(verbs, ["rq1_condition"]).to_dict(orient="records") if not verbs.empty else []
    rq2 = _accuracy_summary(verbs, ["rq2_configuration"]).to_dict(orient="records") if not verbs.empty else []

    return {
        "source_file_count": len(source_files),
        "file_count": int(verbs["file_id"].nunique(dropna=True)) if not verbs.empty else int(utterances["file_id"].nunique(dropna=True)) if not utterances.empty else 0,
        "utterance_count": int(len(utterances)),
        "verb_count": int(len(verbs)),
        "participant_count": participant_count,
        "learner_participant_count": learner_participant_count,
        "analyzable_verb_count": analyzable,
        "target_like_verb_count": target_like,
        "non_target_verb_count": non_target,
        "unexpected_form_count": int(verbs["is_unexpected_form"].sum()) if not verbs.empty else 0,
        "overall_agreement_accuracy": _rate(target_like, analyzable),
        "attraction_configuration_count": attraction_configs,
        "attraction_error_count": attraction_errors,
        "attraction_error_rate": _rate(attraction_errors, attraction_configs),
        "low_confidence_count": low_confidence_count,
        "pending_category_coding_count": pending_category_count,
        "rq1_by_cue_condition": rq1,
        "rq2_by_configuration": rq2,
    }


def _write_duckdb(tables: dict[str, pd.DataFrame], duckdb_path: Path) -> bool:
    # DuckDB is optional; CSV remains the primary interchange format.
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
    # Resolve relative CLI paths from the repository root.
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def _parse_args() -> argparse.Namespace:
    # Keep the command-line interface small: input path plus optional output directory.
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
    # Entry point: discover sources, flatten records, write dashboard-ready tables.
    args = _parse_args()
    input_path = _resolve_path(args.input_path)
    output_dir = _resolve_path(args.output_dir)
    source_files = _source_files(input_path)

    verb_records: list[dict[str, Any]] = []
    utterance_records: list[dict[str, Any]] = []
    skipped_files: list[dict[str, str]] = []

    for source_file in source_files:
        try:
            # Parse one source file and discover one or more annotation documents inside it.
            payload = _load_json(source_file)
            documents = list(_documents_from_payload(payload))
        except Exception as exc:
            skipped_files.append({"source_path": str(source_file), "error": str(exc)})
            continue

        if not documents:
            skipped_files.append({"source_path": str(source_file), "error": "No document with an utterances list was found."})
            continue

        for document in documents:
            # Flatten nested utterances and verbs from this document.
            verbs, utterances = _rows_from_document(source_file, document)
            verb_records.extend(verbs)
            utterance_records.extend(utterances)

    output_dir.mkdir(parents=True, exist_ok=True)

    verbs = _records_to_frame(verb_records, VERB_COLUMNS)
    utterances = _records_to_frame(utterance_records, UTTERANCE_COLUMNS)
    # Summary tables are materialized so Shiny, QueryChat, and humans can inspect them directly.
    cue_summary = _accuracy_summary(verbs, ["rq1_condition", "cue_group", "cue_type", "cue_number"])
    attraction_summary = _accuracy_summary(
        verbs,
        ["rq2_configuration", "attractor_type", "attractor_number", "structural_complexity_type"],
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
        # CSV is the dashboard's main storage format.
        frame.to_csv(output_dir / f"{table_name}.csv", index=False, encoding="utf-8")

    (output_dir / METADATA_FILENAME).write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / RESEARCH_SUMMARY_FILENAME).write_text(
        json.dumps(_json_safe(research_summary), ensure_ascii=False, indent=2, allow_nan=False),
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
