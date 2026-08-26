# Retry LLM calls at the provider boundary

Every model created from a Model Profile uses the provider SDK's native retry at the boundary of one HTTP inference. The provider classifies transient connection failures, timeouts, rate limits and retryable HTTP statuses, applies exponential backoff with jitter, and honors `Retry-After`; the environment controls provider attempts, safe stream resumes, and request, stream-chunk, and auxiliary-call timeouts.

Deep Agents' summarization middleware normally adds a second broad retry loop. The application replaces that inner runnable with the same provider-configured model, so `LLM_MAX_RETRIES` remains authoritative for each inference and provider retry loops cannot nest.

Deep Agents exposes no public option for disabling only this inner loop. The adapter therefore verifies the supported middleware shape and fails at graph construction if it changes; dependency upgrades must keep the integration test green instead of silently restoring nested retries.

## Consequences

Provider retry stops once a stream has produced a chunk, because repeating a partially visible response would duplicate output. `LLM_STREAM_RETRIES` gives zero-chunk stream timeouts a separate finite budget around only the innermost model handler. It sits inside Deep Agents' summarization wrapper, so context offload, the Agent graph, Turn and tools are not replayed. Each new inference still has the per-inference provider budget. Permanent request, authentication, and validation errors fail immediately.

Deep Agents' existing context-overflow recovery is a separate semantic operation: it may compact the Agent Memory context and issue a new inference. It is not a transport retry of the rejected HTTP inference and does not replay a completed tool call.
