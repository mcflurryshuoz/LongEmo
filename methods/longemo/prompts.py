"""Question-independent memory and evidence-grounded question answering."""

PERCEPTION = """Build an emotional memory of the supplied audiovisual window. You do not receive any benchmark question or reference answer.
Use only these media and the preceding memory. Identify visible people from appearance, voice and grounded dialogue; never identify actors from outside knowledge. Reuse supplied P IDs only when supported. Give new people local IDs such as new_1. Unknown names must be null. The cast descriptions are hypotheses, not proof.
Separate observable cues (face, gesture, words, vocal tone, listener reactions) from inferred emotion. Capture mixed feelings and different emotional targets separately. Do not equate emotion target with its cause. Record observable intensity rather than arbitrary numerical ratings. Do not infer continuity across unseen gaps. For every event, preserve the concrete action, objects/items, participants and any explicit comparison signals (score, count, ordinal, superlative, quoted label) stated in the media; never invent numeric values.
All spans use ABSOLUTE VIDEO SECONDS, within the supplied media range. The core interval owns new records; padding supplies context only. Emit only events overlapping the core. Capture each distinct occurrence; use continues_event ONLY if this is the same ongoing occurrence as a supplied earlier event, not a repeated similar action. Existing events may be linked by temporal, causal, changes_to, or coexists_with edges ONLY with evidence; temporal order alone does not establish causation.
Return one JSON object with entities, observations, events, relations, corrections. Required schema:
entities: [{id: local or existing P ID, name: string|null, description: visual/voice identity description}]
observations: [{id: local O ID, subject: entity ID, span:[start,end], cue: concrete observed words/behavior/tone, modality: visual|audio|subtitle|multimodal}]
events: [{id: local ID such as new_event_1 (distinct from existing E IDs), span:[start,end], event_type: observation|action|interaction|speech|reaction|transition|outcome|evaluation, summary: concrete occurrence, action: concrete observed action|null, objects:[concrete items or topics], participants:[entity IDs], signals:[{kind:explicit_score|count|ordinal|superlative|quote|label, value:string, text:verbatim or faithful evidence}], confidence:0..1|null, observation_ids:[local O IDs], continues_event: existing E ID|null, states:[{subject:entity ID, target: what this feeling concerns, emotion: open emotion description, intensity: observed manifestations, evidence_ids:[local O IDs], uncertainty: limitations, appraisal: evidence-grounded interpretation or null}]}]
relations: [{source: local or existing event ID, target: local or existing event ID, type: temporal|causal|changes_to|coexists_with, evidence_ids:[local O IDs]}]
Reference rules: declare new people with local IDs such as new_1 and use that EXACT ID in every subject and participants field in this response. An existing P ID may be used only if supplied in the cast; do not guess the P ID that the program will later assign. For example, entities:[{id:"new_1",name:null,description:"woman in blue"}], observations:[{id:"o1",subject:"new_1",span:[1,2],cue:"smiles",modality:"visual"}] permits observation_ids:["o1"], not the cue text or an invented O ID. Use timestamps from the actual supplied window, not this illustrative example.
Every emitted event must have at least one supporting observation_id from THIS response's observations. The same rule applies to every state's and relation's evidence_ids. If no supporting observation can be given, omit that event/state/relation; do not emit an empty evidence list, invent evidence, or weaken this requirement. Empty top-level lists are valid when nothing person-grounded is observed. Do not invent a person to represent music, ambience, a title card, or scenery: these may contextualize a person-grounded observation but cannot themselves be a person's subject ID.
corrections: [] unless corrections are explicitly enabled. If later evidence corrects the interpretation of an EARLIER same-time state, use {state_id:existing S ID, expected_version:integer, emotion:string, uncertainty:string, evidence_ids:[local O IDs], reason:string}. Do not use correction for a real later emotional change. Never silently erase an earlier state or invent psychological motives. Empty arrays are allowed when there is no evidence. Keep observations detailed enough to later answer questions about emotional developments, causes, intensity comparisons and distinct repeated occurrences."""

PERCEPTION_NOEVENT = """Describe the supplied audiovisual window without constructing events or a graph.
You do not receive any benchmark question or reference answer. Keep every claim tied to this fixed window;
do not link it to earlier windows, infer continuity, or create states, relations, causes, or event IDs. The supplied
cast is an identity index only: reuse its IDs when supported; descriptions must contain stable appearance/voice
identity cues, not earlier actions, emotions or narrative history. Identify people only from the supplied frames,
voice and grounded dialogue. Record concrete visual/audio/subtitle cues,
actions, objects, explicit counts/quotes/labels, and cautious emotion cues with their evidence observations.
Every observations[].subject, emotion_cues[].subject and participants[] value must exactly match either a
local person ID declared in this response's entities or an existing P ID supplied in cast. Existing cast IDs
may be referenced directly without redeclaring them in entities. Give newly observed people distinct local IDs
such as new_1 and new_2; do not guess the P IDs the program will assign. For example, declaring entities[0].id
as "new_1" means its observation uses subject:"new_1", not the person's name or a guessed "P1". If that
observation has id:"o1", a supporting emotion cue uses subject:"new_1" and evidence_ids:["o1"]. Declare each
person ID at most once in entities; repeated observations can reuse it. Names remain null when unknown.
All observations and emotion cues are person-bound. Put background music, ambience and other content that
cannot be attributed to a person in the window summary or actions; never invent a person or use null as an
observation subject. When no person is observable, entities, observations, participants and emotion_cues may
all be empty while summary/actions describe the window. Do not infer a person's emotion from music alone.
The window summary is a compact description of what is observable, not a narrative reconstruction. All spans are
ABSOLUTE VIDEO SECONDS and must overlap the core interval. Return exactly one JSON object:
{entities:[{id:local person ID or supplied cast P ID,name:string|null,description:string}],
 observations:[{id:local observation ID,subject:declared local person ID or supplied cast P ID,span:[number,number],cue:string,modality:visual|audio|subtitle|multimodal}],
 summary:string, actions:[string], objects:[string], signals:[string], participants:[entity IDs],
 emotion_cues:[{subject:entity ID,target:string,emotion:string,intensity:string,evidence_ids:[observation IDs]}]}.
Emotion cues describe evidence in this window only, not persistent emotional states. A directly observed causal
statement may be recorded as a quote or observation. Do not output events, states, relations, continues_event,
causes, or unsupported psychological motives. Empty lists are valid."""

PLANNER = """Plan evidence retrieval for an emotion question about a video. You receive only the question and a cast index, never reference annotations. Return JSON:
{mode: trajectory|comparison|count|causal|local, entity_terms:[names/descriptions], target_terms:[events/topics/emotions], query_terms:[short lexical retrieval terms], time_range:[start,end]|null}.
Trajectory, comparison, count and multi-event causal questions need coverage across the entire question scope, not just the most similar moment. Resolve pronouns cautiously. time_range is a restriction explicitly in the question, not an invented guess. Keep all terms in the question language, adding useful synonyms only."""

ANSWER = """Answer this video-emotion question using the supplied memory and any inspected media. Do not rely on knowledge of the show or story outside the evidence. Distinguish observed cues from uncertain interpretations. Ground each statement in the correct person, emotional target and time. For trajectories, cover the necessary stages in order, including reversals and concurrent feelings; avoid compressing away a stage. For causes, include all supported necessary factors, without inventing motives. For comparisons, check the relevant candidate moments; for counts, distinguish unique occurrences from repeated observations of the same event. Navigation edges are not causal evidence.
Return JSON {answer: nonempty final answer in the question's language, evidence_ids:[event/observation IDs], uncertainty:short factual evidence limitations, inspect:[{start:number,end:number,question:neutral evidence question}]}.
If a missing audiovisual observation could materially change the answer, you may request a bounded inspect interval from the allowed budget. Do not request to confirm a preferred hypothesis. Always include your best current answer; if no more inspection budget remains, inspect must be empty. The final answer should directly satisfy the question, with enough concrete detail for completeness. Do not include internal node IDs, grading instructions or rubric guesses in answer."""

ANSWER_NOEVENT = """Answer using only the supplied time-window records and any inspected media. This is a window-only
ablation: the stored representation has no event nodes, state nodes, relation edges, or continuity links. Do not
claim that such graph annotations were supplied, and do not rely on knowledge of the show outside the evidence.
You may reason across the disclosed windows, including emotional changes, causes, comparisons, and whether
observations refer to the same occurrence, when the observations support those conclusions. For trajectory,
comparison, count and causal questions, compare relevant windows in chronological order, avoid counting padding
observations twice, and state uncertainty when the evidence does not establish a claim. Answer in the question's
language with enough concrete detail. Return JSON
{answer:nonempty string,evidence_ids:[window or observation IDs],uncertainty:string,inspect:[]}.
Do not include internal IDs in the answer text or grading instructions."""

ANSWER_PROGRESSIVE = """Answer this video-emotion question using the currently disclosed evidence from a frozen event graph. The evidence is intentionally partial. Do not assume that an undisclosed event exists or does not exist, and do not rely on knowledge of the show outside the evidence. Distinguish observations from interpretations and ground claims in the correct person, emotional target and time.
The evidence may include a compact coverage_map. It is only a navigation index: event_hints are not sufficient evidence for a final claim, but their IDs may be used as anchors for a retrieve_more request. You may request another evidence page when the current evidence cannot establish a required time stage, comparison candidate, count, relation, or cause. Return JSON:
{answer: nonempty best answer in the question's language, evidence_ids:[event/observation IDs], uncertainty:short factual limitation, retrieve_more:[{anchor_ids:[event IDs], scope:same_target|same_subject|relations|time_range, direction:before|after|both, time_range:[start,end]|null, relation_types:[temporal|causal|changes_to|coexists_with], page_size:1..16, reason:string}], inspect:[]}.
Use retrieve_more only for a concrete missing evidence need, keep anchor_ids from the disclosed evidence, and do not repeat an already disclosed page. For trajectory, comparison, count, and multi-event causal questions, request at least one expansion unless the initial packet explicitly covers every required stage or candidate. If the evidence is sufficient, or no supported expansion is needed, set retrieve_more to []. Always include the best current answer even when requesting more evidence. Navigation edges are not causal evidence. Do not include internal node IDs in the answer text itself."""
