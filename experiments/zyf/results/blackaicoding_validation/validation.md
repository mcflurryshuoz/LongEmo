# BlackAI Gemini resource validation

Validated from g450 on 2026-09-17 using the newly supplied key, which exactly matches the existing private Gemini credential. The credential is absent from this report.

| Capability | Result |
|---|---|
| Native model inventory | HTTP 200; includes gemini-3.8-flash |
| Gemini 3.8 text | HTTP 200, modelVersion gemini-3.8-flash, STOP, returned OK |
| Gemini 3.8 image + audio | Real four-second V16 excerpt, two frames and audio, no subtitles; JSON valid; IMAGE and AUDIO token usage returned |
| OpenAI-compatible /v1/embeddings | HTTP 404: Embeddings API is not supported for this platform |
| Native Embedding 2 embedContent and batchEmbedContents | HTTP 404 NOT_FOUND: model is not configured for this account group |
| Native Embedding 001 and 2-preview embedContent | HTTP 404 NOT_FOUND: model is not configured for this account group |

Use the explicit gemini-3.8-flash model rather than the supplied example's older gemini-2.0-flash. Native generation endpoint: https://www.blackaicoding.com/v1beta/models/gemini-3.8-flash:generateContent.

This validates connectivity, model identity as reported by the provider, and a small multimodal request. It is not a benchmark score or a concurrency/capacity test. The gateway did not return billed costs or an account balance. The two model-list formats expose different catalogs; successful generation is the stronger availability evidence. No benchmark job was started or modified by these diagnostics.

Correction: the original compatible-endpoint check alone did not establish native embedding availability. Follow-up checks use the documented native URI, `/v1beta/models/gemini-embedding-2:embedContent`, and its batch counterpart, with the same authorized key. Both return model-not-configured errors, as do the two alternate embedding models. The evidence establishes that this account group cannot currently serve those models; it does not establish that every BlackAI account or native Gemini service lacks embeddings. See [native probes](native_embedding_probes.json) and [alternate model probes](native_embedding_alternatives.json). Enabling Embedding 2 on this account or supplying another authorized Gemini endpoint is required to complete native graph retrieval. OpenRouter credit was approximately USD 0.2410, but the user selected the native Gemini API, so E09 does not fall back to OpenRouter.

Implementation reference: [Google embedding guide](https://ai.google.dev/gemini-api/docs/embeddings) and [native REST reference](https://ai.google.dev/api/embeddings). Embedding 2 uses task prefixes in text, 3072 dimensions here, and independent batch entries for separate documents.
