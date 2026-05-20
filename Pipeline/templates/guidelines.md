# Annotation Guidelines for French Learner Corpus

## Task

Annotate spoken French learner transcriptions at the utterance level. Each transcript line is one utterance. For each utterance, identify all verbs and annotate subject–verb agreement, explicit subject quantity cues, and possible attraction configurations.

Return valid JSON only.

## Speaker and participant IDs

Use the provided speaker_map to assign stable participant IDs.

Example:

TS -> P001
FA -> P002

The same speaker code may appear in several files. If it refers to the same person, keep the same participant_id across files.

Do not create learner IDs from utterance IDs. For example, do not create TS_001, TS_002, etc., because those would incorrectly look like different learners.

If a speaker is not found in the speaker_map, set participant_id to null and mention this in comments.

## Raw and normalized utterance

raw_utterance must preserve the original transcript exactly.

normalized_utterance may lightly normalize the utterance to support interpretation, but do not fully correct the learner’s grammar.

Example:

Raw:
"j'ai plusieurs voyages de prévu"

Normalized:
"j'ai plusieurs voyages de prévu"

Do not change it to:
"j'ai plusieurs voyages prévus"

unless the normalization is necessary for interpretation. Learner errors should remain visible.

## Verb annotation

Annotate every verb form produced in the utterance.

For each verb, include:
- produced_form
- produced_agreement
- target_form
- lemma
- tense_mood
- subject information
- cue_strength
- attraction
- annotation metadata

For now, do not code verb category. Always set:

"category": null
"category_coding_status": "pending_human_coding"

## Produced agreement

Use only these values:

- target
- non-target
- unexpected

target:
The produced verb form agrees with the expected subject in the intended sentence.

non-target:
The produced verb form does not agree with the expected subject.

unexpected:
Use this when the form is difficult to classify using ordinary agreement rules. This includes fragments, false starts, unclear repairs, incomplete forms, or forms whose intended target cannot be confidently inferred.

## Subject annotation

Identify the grammatical subject of the verb.

subject_form:
The full subject expression.

head:
The head of the subject.

Examples:
- "plusieurs personnes" -> head: "personnes"
- "les étudiants" -> head: "étudiants"
- "je" -> head: "je"
- "qui" -> head: "qui"

subject_type values:
- pronoun
- np
- relative pronoun
- demonstrative pronoun
- other

person values:
- 1
- 2
- 3
- null

number values:
- singular
- plural
- ambiguous
- null

## Cue strength: explicit subject quantity cue

This variable does not mean general agreement cue. It only captures whether the subject contains an explicit quantifier or numeral that makes quantity/number salient.

Set cue_present = true only when the subject contains an explicit quantifier or numeral.

Examples where cue_present = true:
- plusieurs personnes
- beaucoup de personnes
- deux personnes
- trois étudiants
- quelques étudiants

Set cue_present = false for normal articles, determiners, pronouns, or plural morphology alone.

Examples where cue_present = false:
- les personnes
- des personnes
- ces personnes
- la personne
- le garçon
- je
- tu
- il
- elle
- nous
- ils
- elles

Important:
The plural -s on the noun is not enough by itself. Articles such as les, des, and ces are not counted as explicit quantity cues for this variable.

cue_type values:
- quantifier
- numeral
- none
- ambiguous

cue_expression:
Only write the explicit quantifier or numeral, such as "plusieurs", "beaucoup de", "deux", "trois".
If there is no explicit quantity cue, use null.

cue_number:
Usually plural for plusieurs, beaucoup de, deux, trois, etc. Use ambiguous if unclear.

## Attraction

Attraction means that an intervening noun phrase or phrase between the subject and the verb could interfere with agreement.

Set attraction_configuration = true when there is an intervening element that could create number interference.

Examples:
- "le frère des filles arrive"
Subject head: frère, singular
Attractor: filles, plural
Attraction configuration: true

- "les étudiants de la classe arrivent"
Subject head: étudiants, plural
Attractor: classe, singular
Attraction configuration: true

Set attraction_configuration = false when there is no relevant intervening attractor.

attraction_error:
Set attraction_error = true only when:
1. produced_agreement is non-target, and
2. the error is compatible with the attractor’s number.

If there is an attractor but the verb agreement is target-like, then:
attraction_configuration = true
attraction_error = false

## Linear distance

linear_distance_words is the number of words between the subject head and the verb.

Example:
"plusieurs personnes dans la classe reçoivent"

Subject head: personnes
Verb: reçoivent
Words between: dans, la, classe
linear_distance_words = 3

If unclear, use null.

## Confidence

llm_confidence should be a number between 0 and 1.

Use lower confidence for:
- false starts
- repairs
- unclear subject
- unclear verb form
- ambiguous target form
- unclear attraction configuration

If confidence is low, explain briefly in low_confidence_reason.
