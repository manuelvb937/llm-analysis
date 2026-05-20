The `verbs` table contains one row per annotated verb token from L2 French oral-transcription annotations.

Core identifiers:

- `file_id`, `utterance_id`, `learner_id`, `speaker`
- `raw_utterance`, `normalized_utterance`
- `verb_uid`, `verb_index`

Agreement fields:

- `produced_form`, `produced_agreement`, `target_form`, `lemma`, `tense_mood`, `category`
- `agreement_status`: derived category: `target`, `non_target`, `ambiguous`, `not_applicable`, `other`, or `unknown`
- `agreement_accuracy`: 1 for target-like analyzable agreement, 0 for non-target analyzable agreement, null otherwise
- `is_analyzable_agreement`, `is_agreement_target_like`, `is_agreement_non_target`, `is_unexpected_form`

Subject fields:

- `subject_form`, `head`, `subject_type`, `person`, `number`

Cue strength fields:

- Original annotation fields: `cue_present`, `cue_expression`, `cue_type`, `cue_number`
- Derived analysis fields: `cue_group`, `rq1_condition`, `cue_presence_label`
- `rq1_condition` uses `strong_lexical_semantic`, `weak_morphological`, `pronoun`, `no_cue`, `ambiguous`, or `other_cue`

Attraction and structural-complexity fields:

- Original annotation fields: `attraction_configuration`, `attraction_error`, `attractor_surface_form`, `attractor_head`, `attractor_number`, `attractor_type`, `intervener_type`, `structural_complexity_type`, `linear_distance_words`
- Derived analysis fields: `has_attraction`, `has_attraction_error`, `rq2_configuration`, `distance_bin`
- `rq2_configuration` highlights `relative_clause` and `prepositional_phrase` when detectable

Annotation-quality fields:

- `llm_confidence`, `comments`, `annotation_scope`, `annotation_notes`, `source_path`

For accuracy calculations, exclude `not_applicable`, `ambiguous`, `other`, and `unknown` from the denominator unless the user explicitly asks to include them.
