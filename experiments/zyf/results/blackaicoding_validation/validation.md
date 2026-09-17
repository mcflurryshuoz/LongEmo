# BlackAI Gemini resource validation

Validated from g450 on 2026-09-17 using the newly supplied key, which exactly matches the existing private Gemini credential. The credential is absent from this report.

| Capability | Result |
|---|---|
| Native model inventory | HTTP 200; includes gemini-3.8-flash |
| Gemini 3.8 text | HTTP 200, modelVersion gemini-3.8-flash, STOP, returned OK |
| Gemini 3.8 image + audio | Real four-second V16 excerpt, two frames and audio, no subtitles; JSON valid; IMAGE and AUDIO token usage returned |
| Embedding endpoint | HTTP 404: Embeddings API is not supported for this platform |

Use the explicit gemini-3.8-flash model rather than the supplied example's older gemini-2.0-flash. Native generation endpoint: https://www.blackaicoding.com/v1beta/models/gemini-3.8-flash:generateContent.

This validates connectivity, model identity as reported by the provider, and a small multimodal request. It is not a benchmark score or a concurrency/capacity test. The gateway did not return billed costs or an account balance. The two model-list formats expose different catalogs; successful generation is the stronger availability evidence. No benchmark job was started or modified by these diagnostics. Gemini Embedding 2 still needs a separate endpoint; the existing OpenRouter account had about USD 0.2410 remaining at this check.
