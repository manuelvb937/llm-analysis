from __future__ import annotations

import html
import importlib.util
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
from shiny import App, Inputs, Outputs, Session, reactive, render, ui

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    def load_dotenv(*_: Any, **__: Any) -> bool:
        return False

try:
    from chatlas import ChatGoogle
    from querychat import QueryChat
except ModuleNotFoundError:
    ChatGoogle = None
    QueryChat = None


ROOT = Path(__file__).resolve().parents[1]
# All Shiny tables are read from the prepared output directory.
DATA_DIR = ROOT / "outputs" / "corpus_querychat"
VERBS_PATH = DATA_DIR / "verbs.csv"
UTTERANCES_PATH = DATA_DIR / "utterances.csv"
FILES_PATH = DATA_DIR / "files.csv"
LEARNERS_PATH = DATA_DIR / "learners.csv"
METADATA_PATH = DATA_DIR / "metadata.json"
RESEARCH_SUMMARY_PATH = DATA_DIR / "research_summary.json"
GREETING = Path(__file__).parent / "greeting.md"
DATA_DESCRIPTION = Path(__file__).parent / "data_description.md"
STYLES = Path(__file__).parent / "styles.css"
DEFAULT_QUERYCHAT_GEMINI_MODEL = "gemini-2.5-flash-lite"


# Columns that must stay textual after CSV reload.
TEXT_COLUMNS = {
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

# Columns that should be numeric for averages, charts, and KPI calculations.
NUMERIC_COLUMNS = {
    "verb_index",
    "linear_distance_words",
    "llm_confidence",
    "agreement_accuracy",
}

# Columns that should behave as booleans after CSV reload.
BOOL_COLUMNS = {
    "cue_present",
    "attraction_configuration",
    "attraction_error",
    "is_agreement_target_like",
    "is_agreement_non_target",
    "is_unexpected_form",
    "is_analyzable_agreement",
    "has_attraction",
    "has_attraction_error",
}

# Preferred visible order for the verb and utterance data tables.
DISPLAY_COLUMNS = [
    "file_id",
    "utterance_id",
    "speaker",
    "participant_id",
    "participant_role",
    "learner_id",
    "produced_form",
    "produced_agreement",
    "agreement_status",
    "agreement_accuracy",
    "target_form",
    "lemma",
    "tense_mood",
    "category",
    "category_coding_status",
    "subject_form",
    "head",
    "number",
    "cue_present",
    "cue_expression",
    "cue_type",
    "cue_group",
    "rq1_condition",
    "attraction_configuration",
    "attraction_error",
    "attractor_surface_form",
    "attractor_number",
    "attractor_type",
    "structural_complexity_type",
    "rq2_configuration",
    "linear_distance_words",
    "llm_confidence",
    "low_confidence_reason",
    "raw_utterance",
    "normalized_utterance",
    "comments",
]

HIDDEN_TABLE_COLUMNS = {
    "annotation_notes",
    "source_path",
}

STATUS_COLORS = {
    "target": "green",
    "non_target": "red",
    "unexpected": "amber",
    "ambiguous": "amber",
    "not_applicable": "slate",
    "other": "violet",
    "unknown": "slate",
}


def _read_text(path: Path, fallback: str) -> str:
    # Greeting and data-description files are optional conveniences for QueryChat.
    if path.exists():
        return path.read_text(encoding="utf-8")
    return fallback


def _load_json_file(path: Path) -> dict[str, Any]:
    # Metadata and research summaries are optional; the dashboard can run without them.
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as source:
        data = json.load(source)
    return data if isinstance(data, dict) else {}


def _coerce_bool(values: pd.Series) -> pd.Series:
    # CSV booleans may reload as True/False strings, numbers, or actual booleans.
    if values.dtype == bool:
        return values.fillna(False)
    text = values.astype("string").str.lower().str.strip()
    return text.isin({"true", "1", "yes", "y", "present"}) | values.eq(True)


def _load_table(path: Path, text_columns: set[str] | None = None) -> pd.DataFrame:
    # Load a prepared CSV and immediately normalize its types.
    if not path.exists():
        return pd.DataFrame()
    text_columns = text_columns or set()
    dtype = {column: "string" for column in text_columns}
    frame = pd.read_csv(path, dtype=dtype)
    return _clean_frame(frame)


def _load_verbs() -> pd.DataFrame:
    # The verb table is required because QueryChat and most dashboard charts depend on it.
    if not VERBS_PATH.exists():
        raise FileNotFoundError(
            "Missing prepared corpus table. Run "
            "`python corpus_querychat_explorer/prepare_corpus_data.py Annotation.py` first."
        )
    return _load_table(VERBS_PATH, TEXT_COLUMNS)


def _as_pandas(data: Any) -> pd.DataFrame:
    # QueryChat can return Pandas-like, Polars-like, or other dataframe objects.
    if isinstance(data, pd.DataFrame):
        return data.copy()
    if hasattr(data, "to_pandas"):
        return data.to_pandas()
    if hasattr(data, "collect"):
        collected = data.collect()
        if hasattr(collected, "to_pandas"):
            return collected.to_pandas()
        return pd.DataFrame(collected)
    return pd.DataFrame(data)


def _clean_frame(frame: pd.DataFrame) -> pd.DataFrame:
    # Apply the same type rules to startup tables and QueryChat-filtered tables.
    clean = frame.copy()
    for column in TEXT_COLUMNS.intersection(clean.columns):
        clean[column] = clean[column].astype("string")
    for column in NUMERIC_COLUMNS.intersection(clean.columns):
        clean[column] = pd.to_numeric(clean[column], errors="coerce")
    for column in BOOL_COLUMNS.intersection(clean.columns):
        clean[column] = _coerce_bool(clean[column])

    if "agreement_status" not in clean and "produced_agreement" in clean:
        clean["agreement_status"] = clean["produced_agreement"].astype("string")
    if "participant_id" not in clean and "learner_id" in clean:
        clean["participant_id"] = clean["learner_id"]
    if "learner_id" not in clean and "participant_id" in clean:
        clean["learner_id"] = clean["participant_id"]
    if "cue_presence_label" not in clean and "cue_present" in clean:
        clean["cue_presence_label"] = clean["cue_present"].map(
            lambda value: "cue_present" if bool(value) else "no_cue"
        )
    return clean


def _bool_series(frame: pd.DataFrame, column: str) -> pd.Series:
    # Missing boolean columns should behave like all-False filters.
    if column not in frame:
        return pd.Series(False, index=frame.index)
    return _coerce_bool(frame[column])


def _format_count(value: int | float) -> str:
    # KPI display helper.
    if pd.isna(value):
        return "0"
    return f"{int(value):,}"


def _format_percent(value: float | None) -> str:
    # KPI/chart display helper.
    if value is None or pd.isna(value):
        return "n/a"
    return f"{value * 100:.1f}%"


def _rate(numerator: int | float, denominator: int | float) -> float | None:
    # Avoid divide-by-zero in empty filtered selections.
    if denominator == 0:
        return None
    return float(numerator) / float(denominator)


def _empty_state(message: str) -> str:
    # Render a consistent empty panel for charts without data.
    return f"<div class='empty-state'>{html.escape(message)}</div>"


def _pretty_label(value: Any) -> str:
    # Convert stored snake_case values into dashboard labels.
    if value is None or pd.isna(value):
        return "Missing"
    text = str(value).strip()
    if not text:
        return "Missing"
    return text.replace("_", " ").title()


def _truncate(value: Any, limit: int = 220) -> str:
    # Evidence cards need compact text so long utterances do not dominate the layout.
    text = "" if value is None or pd.isna(value) else str(value)
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return f"{text[: limit - 1].rstrip()}..."


def _summary_by(frame: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    # Shared dashboard summary helper; QueryChat filters flow through this function.
    if frame.empty:
        return pd.DataFrame()
    for column in group_columns:
        if column not in frame:
            return pd.DataFrame()

    working = frame.copy()
    for column in group_columns:
        working[column] = working[column].astype("string").fillna("missing")

    grouped = (
        working.assign(
            analyzable=_bool_series(working, "is_analyzable_agreement"),
            target_like=_bool_series(working, "is_agreement_target_like"),
            non_target=_bool_series(working, "is_agreement_non_target"),
            unexpected=_bool_series(working, "is_unexpected_form"),
            attraction_config=_bool_series(working, "has_attraction"),
            attraction_err=_bool_series(working, "has_attraction_error"),
        )
        .groupby(group_columns, dropna=False)
        .agg(
            verbs=("verb_uid", "size") if "verb_uid" in working else (group_columns[0], "size"),
            analyzable_verbs=("analyzable", "sum"),
            target_like_verbs=("target_like", "sum"),
            non_target_verbs=("non_target", "sum"),
            unexpected_forms=("unexpected", "sum"),
            attraction_configurations=("attraction_config", "sum"),
            attraction_errors=("attraction_err", "sum"),
            avg_llm_confidence=("llm_confidence", "mean") if "llm_confidence" in working else (group_columns[0], "size"),
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


def _metrics(frame: pd.DataFrame, utterance_frame: pd.DataFrame | None = None) -> dict[str, int | float | None]:
    # Compute all KPI tile values from the current verb selection.
    verbs = len(frame)
    analyzable = int(_bool_series(frame, "is_analyzable_agreement").sum())
    target_like = int(_bool_series(frame, "is_agreement_target_like").sum())
    non_target = int(_bool_series(frame, "is_agreement_non_target").sum())
    unexpected = int(_bool_series(frame, "is_unexpected_form").sum())
    cue_present = int(_bool_series(frame, "cue_present").sum())
    attraction_configs = int(_bool_series(frame, "has_attraction").sum())
    attraction_errors = int(_bool_series(frame, "has_attraction_error").sum())
    identity_column = "participant_id" if "participant_id" in frame else "learner_id"
    participants = int(frame[identity_column].nunique(dropna=True)) if identity_column in frame else 0
    files = int(frame["file_id"].nunique(dropna=True)) if "file_id" in frame else 0
    utterance_count_frame = utterance_frame if utterance_frame is not None else frame
    utterances = 0
    if utterance_frame is not None and "utterance_uid" in utterance_count_frame:
        utterances = int(utterance_count_frame["utterance_uid"].nunique(dropna=True))
    elif {"file_id", "utterance_id"}.issubset(utterance_count_frame.columns):
        utterances = int(utterance_count_frame[["file_id", "utterance_id"]].drop_duplicates().shape[0])
    low_reason = (
        frame["low_confidence_reason"].astype("string").fillna("").str.strip().ne("")
        if "low_confidence_reason" in frame
        else pd.Series(False, index=frame.index)
    )
    low_score = (
        pd.to_numeric(frame["llm_confidence"], errors="coerce").lt(0.75)
        if "llm_confidence" in frame
        else pd.Series(False, index=frame.index)
    )
    pending_category = int(
        frame.get("category_coding_status", pd.Series(index=frame.index))
        .astype("string")
        .eq("pending_human_coding")
        .sum()
    )

    return {
        "verbs": verbs,
        "files": files,
        "participants": participants,
        "utterances": utterances,
        "analyzable": analyzable,
        "target_like": target_like,
        "non_target": non_target,
        "unexpected": unexpected,
        "accuracy": _rate(target_like, analyzable),
        "cue_present": cue_present,
        "cue_present_share": _rate(cue_present, verbs),
        "attraction_configs": attraction_configs,
        "attraction_errors": attraction_errors,
        "attraction_error_rate": _rate(attraction_errors, attraction_configs),
        "low_confidence": int((low_reason | low_score).sum()),
        "pending_category": pending_category,
    }


def _metric_tiles(frame: pd.DataFrame, utterance_frame: pd.DataFrame | None = None) -> str:
    # Convert KPI values into the dashboard's HTML card grid.
    values = _metrics(frame, utterance_frame)
    tiles = [
        ("Files", _format_count(values["files"]), "Source recordings", "blue"),
        ("Participants", _format_count(values["participants"]), "Speaker-map IDs", "teal"),
        ("Utterances", _format_count(values["utterances"]), "Transcript lines", "violet"),
        ("Verb Tokens", _format_count(values["verbs"]), "Current selection", "blue"),
        ("Analyzable", _format_count(values["analyzable"]), "Finite agreement cases", "slate"),
        ("Accuracy", _format_percent(values["accuracy"]), "Target-like / analyzable", "green"),
        ("Non-target", _format_count(values["non_target"]), "Agreement errors", "red"),
        ("Unexpected", _format_count(values["unexpected"]), "Excluded from accuracy", "amber"),
        ("Cued Verbs", _format_count(values["cue_present"]), _format_percent(values["cue_present_share"]), "teal"),
        ("Attraction", _format_count(values["attraction_configs"]), f"{_format_count(values['attraction_errors'])} errors", "red"),
        ("Low Confidence", _format_count(values["low_confidence"]), "< 0.75 or reason given", "amber"),
        ("Pending Coding", _format_count(values["pending_category"]), "Category needs human coding", "slate"),
    ]

    cards = []
    for label, value, detail, color in tiles:
        cards.append(
            f"<article class='metric-tile metric-{color}'>"
            f"<span>{html.escape(label)}</span>"
            f"<strong>{value}</strong>"
            f"<small>{html.escape(str(detail))}</small>"
            "</article>"
        )
    return f"<div class='metric-grid'>{''.join(cards)}</div>"


def _agreement_chart(frame: pd.DataFrame) -> str:
    # Stacked status bar for target, non-target, unexpected, and other labels.
    if frame.empty or "agreement_status" not in frame:
        return _empty_state("No agreement-status values in the current selection.")

    counts = frame["agreement_status"].astype("string").fillna("unknown").value_counts()
    total = int(counts.sum())
    segments = []
    legend = []
    for status, count in counts.items():
        color = STATUS_COLORS.get(str(status), "slate")
        width = max((int(count) / total) * 100, 3)
        label = _pretty_label(status)
        segments.append(
            f"<div class='stack-segment stack-{color}' style='width:{width:.2f}%' "
            f"title='{html.escape(label)}: {int(count)}'></div>"
        )
        legend.append(
            "<div class='legend-item'>"
            f"<span class='legend-dot dot-{color}'></span>"
            f"<span>{html.escape(label)}</span><strong>{int(count):,}</strong>"
            "</div>"
        )

    return (
        "<div class='stack-chart'>"
        f"<div class='status-stack'>{''.join(segments)}</div>"
        f"<div class='chart-legend'>{''.join(legend)}</div>"
        "</div>"
    )


def _accuracy_bars(frame: pd.DataFrame, group_column: str, limit: int = 10) -> str:
    # Horizontal bars where width represents agreement accuracy by category.
    summary = _summary_by(frame, [group_column])
    if summary.empty:
        return _empty_state(f"No {group_column} values in the current selection.")

    summary = summary.head(limit)
    rows = []
    for _, row in summary.iterrows():
        accuracy = row["agreement_accuracy"]
        accuracy_width = 0 if pd.isna(accuracy) else max(min(float(accuracy) * 100, 100), 1)
        error_count = int(row["non_target_verbs"])
        label = _pretty_label(row[group_column])
        detail = (
            f"{int(row['target_like_verbs']):,}/{int(row['analyzable_verbs']):,} "
            "target-like"
            if int(row["analyzable_verbs"])
            else "No analyzable verbs"
        )
        rows.append(
            "<div class='accuracy-row'>"
            f"<div class='bar-label' title='{html.escape(label)}'>{html.escape(label)}</div>"
            "<div class='accuracy-track'>"
            f"<div class='accuracy-fill' style='width:{accuracy_width:.2f}%'></div>"
            "</div>"
            "<div class='bar-value'>"
            f"<strong>{_format_percent(None if pd.isna(accuracy) else float(accuracy))}</strong>"
            f"<span>{html.escape(detail)}; {error_count:,} errors</span>"
            "</div>"
            "</div>"
        )
    return f"<div class='bar-chart'>{''.join(rows)}</div>"


def _count_bars(frame: pd.DataFrame, column: str, limit: int = 12) -> str:
    # Horizontal bars where width represents frequency by category.
    if frame.empty or column not in frame:
        return _empty_state(f"No {column} values in the current selection.")
    counts = frame[column].astype("string").fillna("missing").value_counts().head(limit)
    if counts.empty:
        return _empty_state(f"No {column} values in the current selection.")

    max_count = max(int(counts.max()), 1)
    rows = []
    for label_value, count in counts.items():
        label = _pretty_label(label_value)
        width = max((int(count) / max_count) * 100, 2)
        rows.append(
            "<div class='count-row'>"
            f"<div class='bar-label' title='{html.escape(label)}'>{html.escape(label)}</div>"
            "<div class='bar-track'>"
            f"<div class='bar-fill' style='width:{width:.2f}%'></div>"
            "</div>"
            f"<div class='count-value'>{int(count):,}</div>"
            "</div>"
        )
    return f"<div class='bar-chart'>{''.join(rows)}</div>"


def _attraction_only(frame: pd.DataFrame) -> pd.DataFrame:
    # Attractor fields are meaningful only when an attraction configuration exists.
    if frame.empty:
        return frame.copy()
    return frame[_bool_series(frame, "has_attraction")].copy()


def _rq_snapshot(frame: pd.DataFrame) -> str:
    # High-level research cards for cue presence, attraction, and coding quality.
    if frame.empty:
        return _empty_state("No rows in the current selection.")

    cue_summary = _summary_by(frame, ["cue_presence_label"])

    def lookup(summary: pd.DataFrame, column: str, key: str, metric: str) -> Any:
        if summary.empty or column not in summary:
            return None
        rows = summary[summary[column].astype("string").eq(key)]
        if rows.empty:
            return None
        return rows.iloc[0].get(metric)

    values = _metrics(frame)
    cued_acc = lookup(cue_summary, "cue_presence_label", "cue_present", "agreement_accuracy")
    no_cue_acc = lookup(cue_summary, "cue_presence_label", "no_cue", "agreement_accuracy")
    cued_n = lookup(cue_summary, "cue_presence_label", "cue_present", "analyzable_verbs") or 0
    no_cue_n = lookup(cue_summary, "cue_presence_label", "no_cue", "analyzable_verbs") or 0

    cards = [
        (
            "Cue Presence",
            [
                ("Cued verbs", _format_percent(cued_acc), f"{int(cued_n):,} analyzable"),
                ("No cue", _format_percent(no_cue_acc), f"{int(no_cue_n):,} analyzable"),
            ],
        ),
        (
            "Attraction",
            [
                ("Configurations", _format_count(values["attraction_configs"]), f"{_format_count(values['attraction_errors'])} errors"),
                ("Error rate", _format_percent(values["attraction_error_rate"]), "errors / configurations"),
            ],
        ),
        (
            "Coding Quality",
            [
                ("Low confidence", _format_count(values["low_confidence"]), "< 0.75 or reason given"),
                ("Pending category", _format_count(values["pending_category"]), "category_coding_status"),
            ],
        ),
    ]

    html_cards = []
    for title, rows in cards:
        facts = []
        for label, value, detail in rows:
            facts.append(
                "<div>"
                f"<span>{html.escape(label)}</span>"
                f"<strong>{html.escape(value)}</strong>"
                f"<small>{html.escape(detail)}</small>"
                "</div>"
            )
        html_cards.append(
            "<article class='research-card'>"
            f"<h4>{html.escape(title)}</h4>"
            f"<div class='research-facts'>{''.join(facts)}</div>"
            "</article>"
        )
    return f"<div class='research-grid'>{''.join(html_cards)}</div>"


def _attraction_cards(frame: pd.DataFrame) -> str:
    # Compact cards for actual attraction configurations, excluding no-attraction rows.
    summary = _summary_by(frame, ["rq2_configuration"])
    if summary.empty:
        return _empty_state("No attraction configuration values in the current selection.")

    summary = summary[summary["rq2_configuration"].astype("string").ne("no_attraction")]
    if summary.empty:
        return _empty_state("No attraction configurations in the current selection.")

    cards = []
    for _, row in summary.iterrows():
        label = _pretty_label(row["rq2_configuration"])
        configs = int(row["attraction_configurations"])
        errors = int(row["attraction_errors"])
        rate = row["attraction_error_rate"]
        cards.append(
            "<article class='attraction-card'>"
            f"<span>{html.escape(label)}</span>"
            f"<strong>{errors:,}</strong>"
            f"<small>{_format_percent(None if pd.isna(rate) else float(rate))} error rate; {configs:,} configurations</small>"
            "</article>"
        )
    return f"<div class='attraction-grid'>{''.join(cards)}</div>"


def _distance_chart(frame: pd.DataFrame) -> str:
    # Distance chart uses bins produced by prepare_corpus_data.py.
    ordered = ["0", "1_2", "3_5", "6_plus", "unknown"]
    summary = _summary_by(frame, ["distance_bin"])
    if summary.empty:
        return _empty_state("No distance values in the current selection.")
    summary["distance_bin"] = pd.Categorical(summary["distance_bin"], ordered, ordered=True)
    summary = summary.sort_values("distance_bin")
    rows = []
    max_verbs = max(int(summary["verbs"].max()), 1)
    for _, row in summary.iterrows():
        label = _pretty_label(row["distance_bin"])
        width = max((int(row["verbs"]) / max_verbs) * 100, 2)
        accuracy = row["agreement_accuracy"]
        rows.append(
            "<div class='count-row'>"
            f"<div class='bar-label'>{html.escape(label)}</div>"
            "<div class='bar-track'>"
            f"<div class='bar-fill bar-fill-alt' style='width:{width:.2f}%'></div>"
            "</div>"
            "<div class='bar-value compact'>"
            f"<strong>{int(row['verbs']):,}</strong>"
            f"<span>{_format_percent(None if pd.isna(accuracy) else float(accuracy))}</span>"
            "</div>"
            "</div>"
        )
    return f"<div class='bar-chart'>{''.join(rows)}</div>"


def _confidence_chart(frame: pd.DataFrame) -> str:
    # Bucket model confidence scores and reuse the accuracy bar renderer.
    if frame.empty or "llm_confidence" not in frame:
        return _empty_state("No confidence scores in the current selection.")
    working = frame.dropna(subset=["llm_confidence"]).copy()
    if working.empty:
        return _empty_state("No confidence scores in the current selection.")
    working["confidence_bucket"] = pd.cut(
        working["llm_confidence"],
        bins=[0, 0.6, 0.75, 0.9, 1.0],
        labels=["0-0.60", "0.61-0.75", "0.76-0.90", "0.91-1.00"],
        include_lowest=True,
    )
    return _accuracy_bars(working, "confidence_bucket", limit=8)


def _evidence_cards(frame: pd.DataFrame, limit: int = 8) -> str:
    # Evidence cards prioritize errors, unexpected forms, and attraction errors.
    if frame.empty:
        return _empty_state("No verb rows in the current selection.")
    flags = (
        _bool_series(frame, "is_agreement_non_target")
        | _bool_series(frame, "is_unexpected_form")
        | _bool_series(frame, "has_attraction_error")
    )
    evidence = frame[flags].copy()
    if evidence.empty:
        evidence = frame.copy()
    if "llm_confidence" in evidence:
        evidence = evidence.sort_values("llm_confidence", ascending=False, na_position="last")
    evidence = evidence.head(limit)

    cards = []
    for _, row in evidence.iterrows():
        produced = _truncate(row.get("produced_form"), 60)
        target = _truncate(row.get("target_form"), 60)
        status = _pretty_label(row.get("agreement_status"))
        subject = _truncate(row.get("subject_form"), 80)
        raw = _truncate(row.get("raw_utterance"), 260)
        comment = _truncate(row.get("comments"), 220)
        cue = _truncate(row.get("cue_expression"), 80) or _pretty_label(row.get("cue_group"))
        attraction = _pretty_label(row.get("rq2_configuration"))
        header = f"{row.get('file_id', '')} / {row.get('utterance_id', '')} / {row.get('learner_id', '')}"

        cards.append(
            "<article class='evidence-card'>"
            f"<div class='evidence-meta'>{html.escape(str(header))}</div>"
            f"<h4>{html.escape(produced)} <span>&rarr;</span> {html.escape(target or 'target unknown')}</h4>"
            "<div class='evidence-tags'>"
            f"<span>{html.escape(status)}</span>"
            f"<span>{html.escape('Cue: ' + cue)}</span>"
            f"<span>{html.escape('Attraction: ' + attraction)}</span>"
            "</div>"
            f"<p>{html.escape(raw)}</p>"
            "<dl>"
            f"<dt>Subject</dt><dd>{html.escape(subject or 'missing')}</dd>"
            f"<dt>Note</dt><dd>{html.escape(comment or 'No comment')}</dd>"
            "</dl>"
            "</article>"
        )
    return f"<div class='evidence-grid'>{''.join(cards)}</div>"


def _filtered_utterances_for(frame: pd.DataFrame) -> pd.DataFrame:
    # Keep utterance-level rows aligned with the current QueryChat verb filter.
    if utterances.empty or frame.empty or not {"file_id", "utterance_id"}.issubset(frame.columns):
        return utterances.iloc[0:0].copy() if frame.empty else utterances.copy()
    if len(frame) == len(verbs):
        if "verb_uid" not in frame or "verb_uid" not in verbs:
            return utterances.copy()
        selected_ids = set(frame["verb_uid"].astype("string").dropna())
        all_ids = set(verbs["verb_uid"].astype("string").dropna())
        if selected_ids == all_ids:
            return utterances.copy()
    keys = frame[["file_id", "utterance_id"]].drop_duplicates()
    return utterances.merge(keys, on=["file_id", "utterance_id"], how="inner")


def _display_table(frame: pd.DataFrame) -> pd.DataFrame:
    # Data tab table formatter: order columns and round numeric display fields.
    if frame.empty:
        return frame
    preferred = [column for column in DISPLAY_COLUMNS if column in frame.columns]
    remaining = [
        column
        for column in frame.columns
        if column not in preferred and column not in HIDDEN_TABLE_COLUMNS
    ]
    table = frame.loc[:, preferred + remaining].copy()
    if {"participant_id", "learner_id"}.issubset(table.columns):
        participant = table["participant_id"].astype("string").fillna("")
        learner = table["learner_id"].astype("string").fillna("")
        if participant.eq(learner).all():
            table = table.drop(columns=["learner_id"])
    for column in ("agreement_accuracy", "llm_confidence", "linear_distance_words"):
        if column in table:
            table[column] = pd.to_numeric(table[column], errors="coerce").round(3)
    return table


def _summary_table(frame: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    # Data-frame renderer helper for grouped summary tables.
    summary = _summary_by(frame, group_columns)
    if summary.empty:
        return summary
    for column in ("agreement_accuracy", "attraction_error_rate", "avg_llm_confidence"):
        if column in summary:
            summary[column] = pd.to_numeric(summary[column], errors="coerce").round(3)
    return summary


def _file_summary_table(frame: pd.DataFrame) -> pd.DataFrame:
    # File summary is recalculated from the current filtered verb set.
    summary = _summary_table(frame, ["file_id"])
    if summary.empty:
        return summary
    identity_column = "participant_id" if "participant_id" in frame else "learner_id"
    participants_by_file = (
        frame.groupby("file_id", dropna=False)[identity_column]
        .apply(lambda values: " | ".join(sorted({str(value) for value in values.dropna()})))
        .reset_index(name="participants")
        if {"file_id", identity_column}.issubset(frame.columns)
        else pd.DataFrame()
    )
    if not participants_by_file.empty:
        summary = summary.merge(participants_by_file, on="file_id", how="left")
    return summary


def _querychat_tools() -> tuple[str, ...]:
    # Visualization tools are enabled only when optional packages are installed.
    if os.getenv("QUERYCHAT_ENABLE_VIZ", "true").lower() in {"0", "false", "no"}:
        return ("update", "query")
    required = ("altair", "ggsql", "shinywidgets", "vl_convert")
    if all(importlib.util.find_spec(module) is not None for module in required):
        return ("update", "query", "visualize")
    return ("update", "query")


load_dotenv(ROOT / ".env")

# Startup data load: Shiny reads prepared CSV/JSON files once when the process starts.
verbs = _load_verbs()
utterances = _load_table(UTTERANCES_PATH, TEXT_COLUMNS | {"utterance_uid"})
files = _load_table(FILES_PATH, TEXT_COLUMNS)
learners = _load_table(LEARNERS_PATH, TEXT_COLUMNS)
metadata = _load_json_file(METADATA_PATH)
research_summary = _load_json_file(RESEARCH_SUMMARY_PATH)

querychat_status = ""
qc = None
if ChatGoogle is None or QueryChat is None:
    querychat_status = "Install requirements to enable QueryChat."
else:
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        querychat_status = "Set GEMINI_API_KEY in .env to enable QueryChat."
    else:
        # QueryChat receives the verb-level table as its main queryable dataset.
        client = ChatGoogle(
            model=os.getenv("QUERYCHAT_GEMINI_MODEL", DEFAULT_QUERYCHAT_GEMINI_MODEL),
            api_key=gemini_api_key,
        )
        qc = QueryChat(
            verbs,
            "verbs",
            client=client,
            greeting=_read_text(GREETING, "Explore the L2 French agreement corpus."),
            data_description=_read_text(DATA_DESCRIPTION, "A verb-level table of L2 French agreement annotations."),
            tools=_querychat_tools(),
        )


def _sidebar():
    # Use QueryChat's sidebar when configured; otherwise show setup guidance.
    if qc is not None:
        return qc.sidebar(width=420)
    return ui.sidebar(
        ui.h4("QueryChat"),
        ui.p(querychat_status),
        ui.tags.code("GEMINI_API_KEY"),
        width=420,
    )


def app_ui(request):
    # Static Shiny UI layout; dynamic content is provided by server render functions.
    return ui.page_sidebar(
        _sidebar(),
        ui.include_css(STYLES),
        ui.div(
            ui.div(
                ui.output_text("dashboard_title"),
                ui.output_text("dashboard_subtitle"),
                class_="dashboard-heading",
            ),
            ui.div(
                ui.input_action_button(
                    "reset_dashboard",
                    "Reset query",
                    class_="btn btn-outline-secondary btn-sm",
                ),
                class_="dashboard-actions",
            ),
            class_="dashboard-topbar",
        ),
        ui.div(
            ui.navset_card_tab(
                ui.nav_panel(
                    "Overview",
                    ui.accordion(
                        ui.accordion_panel(
                            "KPI Summary",
                            ui.output_ui("metric_tiles"),
                            value="metrics",
                        ),
                        ui.accordion_panel(
                            "Research Snapshot",
                            ui.output_ui("rq_snapshot"),
                            value="snapshot",
                        ),
                        id="overview_sections",
                        open=["metrics", "snapshot"],
                    ),
                    ui.layout_columns(
                        ui.card(
                            ui.card_header("Agreement Status"),
                            ui.output_ui("agreement_chart"),
                            class_="dashboard-card chart-card",
                        ),
                        ui.card(
                            ui.card_header("Cue Condition Accuracy"),
                            ui.output_ui("cue_condition_chart"),
                            class_="dashboard-card chart-card",
                        ),
                        col_widths=(4, 8),
                    ),
                    ui.layout_columns(
                        ui.card(
                            ui.card_header("Attraction Configurations"),
                            ui.output_ui("attraction_cards"),
                            class_="dashboard-card chart-card",
                        ),
                        ui.card(
                            ui.card_header("LLM Confidence"),
                            ui.output_ui("confidence_chart"),
                            class_="dashboard-card chart-card",
                        ),
                        col_widths=(5, 7),
                    ),
                    value="overview",
                ),
                ui.nav_panel(
                    "RQ1 Cues",
                    ui.layout_columns(
                        ui.card(
                            ui.card_header("Cue Type Accuracy"),
                            ui.output_ui("cue_type_chart"),
                            class_="dashboard-card chart-card",
                        ),
                        ui.card(
                            ui.card_header("Cue Number"),
                            ui.output_ui("cue_number_chart"),
                            class_="dashboard-card chart-card",
                        ),
                        col_widths=(7, 5),
                    ),
                    ui.card(
                        ui.card_header("Cue Summary"),
                        ui.output_data_frame("cue_summary_table"),
                        class_="dashboard-card data-card",
                    ),
                    value="rq1",
                ),
                ui.nav_panel(
                    "RQ2 Attraction",
                    ui.layout_columns(
                        ui.card(
                            ui.card_header("Structural Complexity"),
                            ui.output_ui("structural_chart"),
                            class_="dashboard-card chart-card",
                        ),
                        ui.card(
                            ui.card_header("Attractor Number"),
                            ui.output_ui("attractor_number_chart"),
                            class_="dashboard-card chart-card",
                        ),
                        col_widths=(7, 5),
                    ),
                    ui.layout_columns(
                        ui.card(
                            ui.card_header("Attractor Type"),
                            ui.output_ui("attractor_chart"),
                            class_="dashboard-card chart-card",
                        ),
                        ui.card(
                            ui.card_header("Subject-Verb Distance"),
                            ui.output_ui("distance_chart"),
                            class_="dashboard-card chart-card",
                        ),
                        col_widths=(5, 7),
                    ),
                    ui.card(
                        ui.card_header("Attraction Summary"),
                        ui.output_data_frame("attraction_summary_table"),
                        class_="dashboard-card data-card",
                    ),
                    value="rq2",
                ),
                ui.nav_panel(
                    "Participants",
                    ui.layout_columns(
                        ui.card(
                            ui.card_header("Participant Summary"),
                            ui.output_data_frame("learner_summary_table"),
                            class_="dashboard-card data-card",
                        ),
                        ui.card(
                            ui.card_header("File Summary"),
                            ui.output_data_frame("file_summary_table"),
                            class_="dashboard-card data-card",
                        ),
                        col_widths=(6, 6),
                    ),
                    value="participants",
                ),
                ui.nav_panel(
                    "Evidence",
                    ui.card(
                        ui.card_header("Selected Examples"),
                        ui.output_ui("evidence_cards"),
                        class_="dashboard-card evidence-panel",
                    ),
                    value="evidence",
                ),
                ui.nav_panel(
                    "Data",
                    ui.card(
                        ui.card_header("Verb-Level Table"),
                        ui.output_data_frame("verbs_table"),
                        class_="dashboard-card data-card",
                    ),
                    ui.card(
                        ui.card_header("Utterance-Level Table"),
                        ui.output_data_frame("utterances_table"),
                        class_="dashboard-card data-card",
                    ),
                    value="data",
                ),
                ui.nav_panel(
                    "SQL",
                    ui.card(
                        ui.card_header("Current Query"),
                        ui.output_ui("sql_output"),
                        class_="dashboard-card sql-card",
                    ),
                    value="sql",
                ),
                id="dashboard_tab",
                selected="overview",
            ),
            class_="dashboard-tabs",
        ),
        title=ui.span("L2 French Agreement Corpus"),
        fillable=False,
        class_="corpus-dashboard",
    )


def server(input: Inputs, output: Outputs, session: Session) -> None:
    # qc_vals exposes QueryChat's filtered dataframe, generated SQL, and title.
    qc_vals = qc.server(enable_bookmarking=False) if qc is not None else None

    @reactive.effect
    @reactive.event(input.reset_dashboard)
    def _reset_dashboard() -> None:
        if qc_vals is not None:
            qc_vals.sql.set(None)
            qc_vals.title.set(None)

    @reactive.calc
    def filtered_verbs() -> pd.DataFrame:
        # Every chart reads this reactive table, so QueryChat filters propagate everywhere.
        if qc_vals is None:
            return verbs.copy()
        return _clean_frame(_as_pandas(qc_vals.df()))

    @reactive.calc
    def filtered_utterances() -> pd.DataFrame:
        # Utterance rows follow the selected verb rows where possible.
        return _filtered_utterances_for(filtered_verbs())

    @render.text
    def dashboard_title() -> str:
        if qc_vals is not None and qc_vals.title():
            return qc_vals.title()
        return "L2 French Agreement Corpus"

    @render.text
    def dashboard_subtitle() -> str:
        frame = filtered_verbs()
        total = len(verbs)
        shown = len(frame)
        return f"Showing {shown:,} of {total:,} verb annotations"

    @render.ui
    def metric_tiles():
        return ui.HTML(_metric_tiles(filtered_verbs(), filtered_utterances()))

    @render.ui
    def rq_snapshot():
        return ui.HTML(_rq_snapshot(filtered_verbs()))

    @render.ui
    def agreement_chart():
        return ui.HTML(_agreement_chart(filtered_verbs()))

    @render.ui
    def cue_condition_chart():
        return ui.HTML(_accuracy_bars(filtered_verbs(), "rq1_condition"))

    @render.ui
    def confidence_chart():
        return ui.HTML(_confidence_chart(filtered_verbs()))

    @render.ui
    def cue_type_chart():
        return ui.HTML(_accuracy_bars(filtered_verbs(), "cue_type"))

    @render.ui
    def cue_number_chart():
        return ui.HTML(_count_bars(filtered_verbs(), "cue_number"))

    @render.ui
    def attraction_cards():
        return ui.HTML(_attraction_cards(filtered_verbs()))

    @render.ui
    def structural_chart():
        return ui.HTML(_accuracy_bars(filtered_verbs(), "structural_complexity_type", limit=12))

    @render.ui
    def attractor_number_chart():
        return ui.HTML(_count_bars(_attraction_only(filtered_verbs()), "attractor_number", limit=10))

    @render.ui
    def attractor_chart():
        return ui.HTML(_count_bars(_attraction_only(filtered_verbs()), "attractor_type"))

    @render.ui
    def distance_chart():
        return ui.HTML(_distance_chart(filtered_verbs()))

    @render.ui
    def evidence_cards():
        return ui.HTML(_evidence_cards(filtered_verbs()))

    @render.ui
    def sql_output():
        if qc_vals is None:
            return ui.HTML(
                "<pre class='sql-block'><code>"
                "SELECT * FROM verbs"
                "</code></pre>"
            )
        sql = qc_vals.sql() or "SELECT * FROM verbs"
        return ui.HTML(
            "<pre class='sql-block'><code>"
            f"{html.escape(sql)}"
            "</code></pre>"
        )

    @render.data_frame
    def cue_summary_table():
        return _summary_table(filtered_verbs(), ["rq1_condition", "cue_group", "cue_type", "cue_number"])

    @render.data_frame
    def attraction_summary_table():
        return _summary_table(
            _attraction_only(filtered_verbs()),
            ["rq2_configuration", "attractor_type", "attractor_number", "structural_complexity_type"],
        )

    @render.data_frame
    def learner_summary_table():
        frame = filtered_verbs()
        if "participant_id" in frame:
            group_columns = ["participant_id"]
            if "participant_role" in frame:
                group_columns.append("participant_role")
            return _summary_table(frame, group_columns)
        return _summary_table(frame, ["learner_id"])

    @render.data_frame
    def file_summary_table():
        return _file_summary_table(filtered_verbs())

    @render.data_frame
    def verbs_table():
        return _display_table(filtered_verbs())

    @render.data_frame
    def utterances_table():
        return _display_table(filtered_utterances())


app = App(app_ui, server, bookmark_store="url")
