#!/usr/bin/env python3
"""Annotation pipeline for learner transcription analysis.

Features:
- Provider switch: OpenAI or Gemini.
- Input source: local text/markdown file or Google Doc URL.
- Prompt assembly from separate template/guidelines files.
- JSON-only model output with retry-friendly parsing.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

import requests

ROOT = Path(__file__).resolve().parent
TEMPLATES = ROOT / "templates"


def extract_gdoc_id(source: str) -> str | None:
    # Accept either a normal Google Docs sharing URL or a URL with an id= query parameter.
    patterns = [
        r"/document/d/([a-zA-Z0-9-_]+)",
        r"id=([a-zA-Z0-9-_]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, source)
        if match:
            return match.group(1)
    return None


def load_transcript(source: str) -> str:
    # Prefer local files so the same script can run without network access.
    maybe_file = Path(source)
    if maybe_file.exists() and maybe_file.is_file():
        return maybe_file.read_text(encoding="utf-8")

    # If it is not a file, treat the source as a Google Doc URL or ID.
    doc_id = extract_gdoc_id(source)
    if not doc_id:
        raise ValueError("Source must be a local file path or a valid Google Doc URL/ID.")

    # Google Docs can export plain text directly when the document is accessible.
    url = f"https://docs.google.com/document/d/{doc_id}/export?format=txt"
    headers: dict[str, str] = {}
    docs_token = os.getenv("GOOGLE_DOCS_BEARER")
    if docs_token:
        # Optional bearer token lets you export private docs when your environment provides one.
        headers["Authorization"] = f"Bearer {docs_token}"

    res = requests.get(url, headers=headers, timeout=60)
    res.raise_for_status()
    return res.text


def load_text(path: Path) -> str:
    # Template files are required because they define the annotation contract.
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return path.read_text(encoding="utf-8")


def build_user_payload(guidelines: str, json_template: str, task_prompt: str, transcript: str) -> str:
    # The model sees the task, rules, schema, and transcript in one deterministic payload.
    return (
        f"{task_prompt.strip()}\n\n"
        "## ANNOTATION GUIDELINES\n"
        f"{guidelines.strip()}\n\n"
        "## JSON TEMPLATE (Fill this schema, do not remove required keys)\n"
        f"{json_template.strip()}\n\n"
        "## TRANSCRIPT TO ANNOTATE\n"
        f"{transcript.strip()}\n\n"
        "Return strictly valid JSON, and no extra prose."
    )


def extract_json(raw: str) -> dict[str, Any]:
    raw = raw.strip()
    # Some models wrap JSON in markdown fences even when asked not to.
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?", "", raw).strip()
        raw = raw.rstrip("`").strip()

    # First try the ideal case: the whole response is valid JSON.
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Fallback: recover the first JSON object from extra prose or markdown.
        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def call_openai(model: str, user_payload: str) -> str:
    # OpenAI calls require an API key in the environment.
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY is not set.")

    url = "https://api.openai.com/v1/responses"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
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
        # Ask the Responses API to enforce JSON-object output.
        "text": {"format": {"type": "json_object"}},
    }

    res = requests.post(url, headers=headers, json=data, timeout=180)
    res.raise_for_status()
    body = res.json()
    return body.get("output_text", "")


def call_gemini(model: str, user_payload: str) -> str:
    # Gemini calls require an API key in the environment.
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError("GEMINI_API_KEY is not set.")

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    data = {
        "contents": [{"parts": [{"text": user_payload}]}],
        "system_instruction": {
            "parts": [{"text": "You are a strict linguistic annotator. Output only valid JSON."}]
        },
        "generationConfig": {
            # Ask Gemini for JSON and keep temperature low for annotation consistency.
            "responseMimeType": "application/json",
            "temperature": 0.1,
        },
    }

    res = requests.post(url, json=data, timeout=180)
    res.raise_for_status()
    body = res.json()
    return body["candidates"][0]["content"]["parts"][0]["text"]


def main() -> None:
    # CLI arguments make the same runner work for local files, Google Docs, OpenAI, and Gemini.
    parser = argparse.ArgumentParser(description="Run transcript annotation with OpenAI or Gemini.")
    parser.add_argument("--source", required=True, help="Path to transcript file OR Google Doc URL/ID")
    parser.add_argument("--provider", choices=["openai", "gemini"], required=True)
    parser.add_argument("--model", help="Model name override")
    parser.add_argument("--guidelines", default=str(TEMPLATES / "guidelines.md"))
    parser.add_argument("--prompt", default=str(TEMPLATES / "prompt.md"))
    parser.add_argument("--json-template", default=str(TEMPLATES / "json_template.json"))
    parser.add_argument("--output", default=str(ROOT / "outputs" / "annotation_result.json"))
    args = parser.parse_args()

    # Load the four inputs that form the model request.
    transcript = load_transcript(args.source)
    guidelines = load_text(Path(args.guidelines))
    task_prompt = load_text(Path(args.prompt))
    json_template = load_text(Path(args.json_template))

    payload = build_user_payload(guidelines, json_template, task_prompt, transcript)

    # Call exactly one provider. Defaults can be overridden from the command line.
    if args.provider == "openai":
        model = args.model or "gpt-5-mini"
        raw = call_openai(model, payload)
    else:
        model = args.model or "gemini-2.0-flash"
        raw = call_gemini(model, payload)

    # Convert the model text into a Python dict before saving pretty JSON.
    parsed = extract_json(raw)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(parsed, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved annotation JSON to: {output_path}")


if __name__ == "__main__":
    main()
