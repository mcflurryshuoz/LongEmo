# Matrix API validation — 2026-09-17

The user supplied an authorized Matrix API key. All probes below used that key; it is omitted from artifacts. Model-call payloads contained short synthetic text, no benchmark questions or answers.

| Check | Result |
|---|---|
| Local machine through its existing proxy: GET /v1/models | HTTP 200, valid model catalog |
| GPT-6 Astra: POST /v1/chat/completions | HTTP 200, returned model gpt-6-astra, exactly OK, 15 total tokens |
| Gemini Embedding 2 Preview: POST /v1/embeddings | HTTP 404, unsupported_operation |
| Gemini Embedding 2 Preview: native /v1beta/models/...:embedContent | HTTP 404 |
| Gemini Embedding 2 Preview: native /v1/models/...:embedContent | HTTP 404 |
| g450 direct connection | DNS resolves; TCP connection to port 443 times out |

The catalog lists gemini-embedding-2-preview, but a listed model is not proof that a particular endpoint/key can invoke embeddings. No vector was returned in these probes. The platform-specific embedding documentation or working request example is still needed. GPT-6 chat is verified from the local network/proxy only; direct use from g450 remains unverified.

No benchmark provider, predictions, scores or credential files were modified. Existing Azure GPT-6 answering/judging and native BlackAI perception remain configured as before. This diagnostic is not a benchmark result.
