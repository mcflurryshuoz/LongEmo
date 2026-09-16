# Azure E06 progress

Status: running. Scored: 0/558. Perception windows: 108/8342.

This is a progress snapshot, not a complete benchmark result. Incomplete/service-rejected videos stay in the full denominator.

Azure HTTP outcomes: {'400': 4, '200': 108, '429': 4}. Returned tokens: 1141213. Azure billing: unknown (no billing export).

Video outcomes: {'blocked_input_policy': 4}. Active video workers at snapshot: 14.

The queue was launched with 16 video workers and two answer/judge workers per video. Current operational Azure limits are 500k tokens/minute, 120 requests/minute and 24 maximum in-flight requests. Rejected content is recorded and excluded from automatic retries. The first wave gates later scheduling on one completed official-scoring result.

The model/evaluator source is unchanged across the recorded scheduler revision. The initial manifest and every actual execution revision are included in JSON. The benchmark uses GPT-6 both to answer and judge, so self-judge bias remains possible.
