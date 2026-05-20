# Manual Chat Test Pack (Copy/Paste)

Paste everything below into ChatGPT along with your transcript/doc attachment.

---

## ROLE
You are a strict linguistic annotator. Output only valid JSON.

## TASK
Annotate the attached transcription using the rules and schema below. Return JSON only.

## GUIDELINES
{{PASTE_CONTENTS_OF: Pipeline/templates/guidelines.md}}

## JSON TEMPLATE
{{PASTE_CONTENTS_OF: Pipeline/templates/json_template.json}}

## QUALITY CHECKLIST
- Keep utterance order exactly as produced.
- Annotate every relevant verb.
- Do not invent information not inferable from transcript.
- Use comments for uncertainty.
- Return valid JSON only (no markdown fences).

