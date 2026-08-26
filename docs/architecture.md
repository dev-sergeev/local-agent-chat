# MVP architecture

## Module boundaries

| Module | Owns | Deliberately does not own |
| --- | --- | --- |
| `app.py` | Dependency composition and Chainlit callbacks | Model construction, durable Chat configuration, rollback algorithms |
| `chat_configuration.py` + `chat_bindings.py` | Profile/mode recovery and the SQLite-authoritative Chat binding | Deep Agents graphs and UI session state |
| `deep_agent_execution.py` | Graph construction, Agent Memory checkpoints, event translation and execution cleanup | Profile/mode selection, Turn rollback and cosmetic labels |
| `runtime.py` | One serialized Turn and compensating runtime rollback held through the UI commit | Chainlit storage algorithms and Deep Agents internals |
| `chainlit_data.py` | Chainlit History, title state and transactional Revision archive | Agent Memory and Sandbox contents |
| `sandbox_files.py` + `sandbox_provider.py` | Revisioned files/artifacts and mode-aware read scopes | Chat History, Agent Memory and Agent-visible mutation |
| `llm_retry.py` + `auxiliary_labels.py` | One provider retry policy and failure-isolated Chat labels | Retrying a whole Turn or persisting UI state |
| `long_term_memory.py` | Bounded Long-term Memory Markdown, atomic semantic upsert and Deep Agents/LangChain adapters | Chat transcript search, Agent Memory checkpoints or Sandbox permissions |

Dependencies point inward through these narrow interfaces. `ChatBindings` is the only authoritative source for a Chat's Model Profile, Agent Mode and active LangGraph memory thread; Chainlit metadata and `user_session` are mirrors used by the UI, never competing registries.

## Runtime

- Chainlit serves the UI under `APP_ROOT_PATH`, supplied to `chainlit run --root-path` behind Jupyter Server Proxy.
- Browser-side DELETE requests with JSON bodies are tunneled as POST across Jupyter Server Proxy and restored to DELETE by a narrow ASGI middleware before Chainlit routing.
- A silent, fixed Local User identity enables Chainlit's built-in history without a login form.
- Each new Chat selects one Model Profile loaded from YAML. Provider secrets come only from environment variables.
- A persisted available Model Profile is immutable. If configuration removes it, resume migrates the Chat to the first valid persisted/client hint or the configured default, so an older Chat remains usable without authorizing an arbitrary profile change.
- Each new Chat starts in `CHAT_FILES` Agent Mode. The user may select `HOST_FILES` in Chat settings until the first message. A signature-guarded Socket.IO pre-dispatch adapter serializes mode-setting and valid-message frames before Chainlit schedules concurrent tasks, then atomically locks the mode in the Chat registry. Stop, settings update, and resume retain a persisted-message recovery path. Revision, metadata, and model checkpoints cannot change the mode. Legacy `read_only` and `extended` values migrate to `HOST_FILES`, preserving their former host-read scope while deliberately removing mutation and execution; their stored lock state is preserved, and schemas that predate mode selection migrate as locked.
- A Model Profile may set `streaming: false`; LangChain then routes event streaming through the model's complete `ainvoke` result, and the UI renders that final answer without text deltas.
- One Turn per Chat may run at a time; concurrent submissions are rejected.
- Deep Agents emits typed text/tool events through `ChatRuntime`; a Chainlit-only adapter persists text segments and tool Steps as one ordered sibling timeline inside the Turn. A new text segment starts after every tool so live rendering and resumed Chat History keep the same chronology.
- Structured tool input is rendered as JSON. Tool output is sanitized and persisted as a bounded preformatted text log so untrusted text cannot be interpreted as chat Markdown.
- On the first request, the same Model Profile generates a three-to-five-word semantic Chat title in a background task that never delays or cancels the Turn. The UI immediately receives a deterministic compact title derived from the request and bounded attachment names for a file-only Turn; a validated model title replaces it when available, while provider failure retains the deterministic title and retries on a later Turn. Atomic title state prevents late Chainlit writes or the LLM task from overwriting a manual rename.
- Tool Steps are collapsed on their first render and are never auto-collapsed later, preventing layout jumps while preserving manual expansion; failed Steps remain marked as errors.
- Cancelling a Turn restores its pre-turn Agent Memory and Sandbox snapshot before another Turn can start.

## Persistence

- Chainlit history uses a local SQLite data layer.
- Writes for one Chainlit Step are serialized so a background `create_step` cannot overwrite a newer streamed output or tool result during reload.
- Legacy raw tool output is normalized when Chat History is read, so saved chats remain readable after upgrading.
- LangGraph checkpoints use a separate SQLite database. Each Chat starts with its Chainlit ID as the memory thread and tracks every later materialized restore thread for rollback and deletion.
- Each Chat owns a durable file directory with a 1 GiB limit; each Uploaded File is limited to 100 MiB.
- Spontaneous uploads accept every MIME type and valid zero-byte files. Basename collisions receive ` (2)`, ` (3)`, … suffixes unless an explicit internal replacement is requested; user uploads never overwrite a same-named Chat file silently.
- Binary files are stored in the Chat directory, not in SQLite.
- Upload, snapshot, restore and deletion are serialized per Chat and run outside the event loop. Uploads are published by an atomic rename, and restore stages both revisioned trees before replacing the current state, so copy failures leave the previous files intact.
- Canonical active Turn text is mirrored transactionally into a local FTS5 index. `superseded_turns`, checkpoints, tool logs, snapshots, and file contents are never indexed.
- Global Memory is progressive and on demand: `search_past_chats` returns bounded candidates from other Chats, then `read_past_chat` returns a bounded source window. Revision and deletion remove sources through the same active-history transaction.
- Long-term Memory is the current compact snapshot at `APP_DATA_DIR/memory/MEMORY.md`. Stable keys are serialized under both an application lock and an OS file lock, then published by atomic replacement with mode `0600`; identical writes are idempotent and updates replace the previous value instead of growing an event log.
- Long-term Memory is limited to 128 facts, 500 characters per fact and 32 KiB for both the source file and the escaped prompt snapshot. Reads are bounded and do not follow the final path component when it is a symlink. Non-canonical, oversized or non-UTF-8 Markdown is not sent to the model and does not break a Turn; it also cannot be silently overwritten by an Agent tool. Recognizable credential keys and values are rejected on write and omitted on read.

## Agent execution

- Agent and Chat-title system prompts are centralized in `local_agent_chat/prompts.py`.
- Deep Agents runs in the application process and has no command or code-execution interface.
- `DeepAgentExecution` exposes only checkpoint, restore, run, delete and close operations. It reads the locked authoritative `ChatBinding`; configuration callers never pass a second profile or mode through an execution facade.
- Every main, summarization and auxiliary model is created by the same `RetryBlock`. Provider retry applies to one HTTP inference. A zero-chunk stream timeout retries only the innermost model handler; the surrounding summarization/offload middleware, Agent graph, Turn and tools are never replayed. LangGraph uses synchronous checkpoint durability for persisted Agent steps.
- `AuxiliaryLabels` owns the bounded prompt, normalization and model cache for Chat titles. Provider errors, invalid output and timeout retain the deterministic request-derived fallback, while task cancellation still propagates.
- Project-local Skills are loaded through Deep Agents' native `skills=` interface from the absolute `skills/` path. Metadata is checkpointed on the first Turn, while full `SKILL.md` instructions use progressive disclosure. The source is routed read-only in `CHAT_FILES`, is directly readable in `HOST_FILES`, and is inherited by the default general-purpose subagent. Skills are therefore available in both modes but cannot expand a mode's scope or toolset. Existing Chats retain their loaded metadata until a new Chat is created.
- Deep Agents' native `MemoryMiddleware` loads Long-term Memory into the main Agent prompt as untrusted reference data. A narrow adapter deliberately refreshes it before every Turn instead of reusing checkpointed `memory_contents`, so both new and already-open Chats see the latest snapshot. The general-purpose subagent receives the same fresh reference snapshot without the narrow mutation tools and reports durable corrections or results back to the main Agent.
- `remember_context` and `forget_context` are main-Agent application tools in both Agent Modes. They change only Long-term Memory and do not add generic filesystem writes; delegated Agents cannot call them. The main Agent decides during its existing inference when an explicit user fact, stable preference, recurring constraint or verified reusable result is worth preserving; no second extraction LLM call is made.
- One mode-aware adapter is the filesystem seam. Both `CHAT_FILES` and `HOST_FILES` expose exactly `ls`, `read_file`, `glob`, and `grep` through explicit replacement `FilesystemMiddleware` instances on the main Agent and general-purpose subagent. Their backends also reject write, edit, delete and upload calls, and no shell or code-execution backend is constructed.
- In `CHAT_FILES`, the default backend maps the Chat's uploaded-file directory to virtual `/`; traversal, absolute host paths and symbolic-link escapes are rejected. An explicit read-only route exposes trusted Project Skill packages, while a separate private route serves internal artifacts. In `HOST_FILES`, the default backend maps real `/`, so absolute process-readable host paths retain their usual meaning.
- Deep Agents context offloads are routed by their actual absolute path into a private Chat `artifacts/` directory rather than the user-visible manifest. Middleware may write there internally, but no Agent-visible mutation tool exposes that route. Both `files/` and `artifacts/` are revisioned together.
- Host Files is confidentiality-sensitive, not an OS sandbox: process-readable host content can be sent to the model provider. A dedicated container or VM remains the deployment boundary.

## Revision

- Chainlit's native user-message editor is used without a custom frontend.
- Before every Turn, the application records the corresponding agent checkpoint and Sandbox snapshot.
- Editing a user message must reproduce Chainlit's revision behavior across both the UI and the agent backend; updating only the displayed or persisted message text is invalid.
- When an edited message is submitted, the application performs the following operation as one coordinated workflow:
  1. Identify the checkpoint and Sandbox snapshot immediately before the original Turn.
  2. Remove every later Turn from the active Chat History, including the original agent result and all downstream tool results.
  3. Restore Agent Memory to the identified pre-turn checkpoint so the model cannot see superseded messages or results.
  4. Restore Uploaded Files and internal artifacts to the matching pre-turn snapshot so offloaded context is not reused against stale state.
  5. Persist the edited user message as the new active Turn.
  6. Invoke the Agent again and stream a newly generated result to Chainlit.
- Chainlit Steps, elements and feedback from the superseded continuation are staged before replay, removed during replay, and restored if replay fails. `ChatRuntime` keeps the previous Agent checkpoint, Sandbox snapshot and exact active runtime-history continuation until the Chainlit context commits; a late rendering, persistence or cancellation failure compensates the tentative runtime history (including FTS and audit rows) before releasing the Chat lock. Database changes for each individual revision decision are transactional; unreferenced blob cleanup happens only afterward.
- The edited request and regenerated result become the canonical continuation of the Chat. A page reload or application restart must restore this revised state, not the superseded continuation.
- If either Agent Memory or Sandbox restoration fails, the Agent must not be rerun and the active history must not be partially replaced.
- Restoring Agent Memory materializes the selected checkpoint as the durable head of a new tracked LangGraph thread. An empty pre-first-Turn checkpoint starts a genuinely empty thread; neither case depends on an in-process pinned checkpoint that would be lost on restart.
- Superseded records remain only in a local technical audit log.
- Long-term Memory mutations are intentionally independent of the Turn transaction. Once accepted, a mutation reaches an atomic publish even if the Turn is cancelled; a published change survives a later Turn failure or cancellation, Revision, and deletion of the originating Chat. A corrected fact reuses its stable key; explicit removal uses `forget_context`.

## Lifecycle

- Resuming a Chat restores its Model Profile, locked Agent Mode, Agent Memory, and Sandbox association.
- Deleting a Chat first blocks new Turns and waits for the active Turn transaction, then deletes its checkpoints, stored files, snapshots, artifacts, and in-memory backend.
- Deleting a Chat does not delete shared Long-term Memory; the user or Agent removes an obsolete fact explicitly with `forget_context`.
- Uploaded Files survive application restarts; the in-memory backend is recreated from the Chat directory.
