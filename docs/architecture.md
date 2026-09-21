# Architecture

LocalChat consists of Chainlit and one LangChain `create_agent` ReAct loop. See [LangChain agents](https://docs.langchain.com/oss/python/langchain/agents) and [middleware](https://docs.langchain.com/oss/python/langchain/middleware/built-in). ADR 0011 supersedes the earlier Deep Agents composition.

| Module | Responsibility |
|---|---|
| `local_agent_chat/app.py` | Compose dependencies; handle Chainlit start/resume/message/edit/Stop; publish titles |
| `agent_execution.py` | Load messages, run model/tools, translate events, save successful context |
| `agent_context.py` | Rolling summaries, input budget, isolated model retries |
| `agent_memory.py` | Current messages and immutable pre-Turn context snapshots in SQLite |
| `sandbox_tools.py` | Exactly four bounded read tools, rooted in the current Chat's uploads |
| `sandbox_files.py` | Application-managed upload, snapshot, restore and removal |
| `chat_bindings.py` | Immutable available Model Profile; fallback if a profile was removed |
| `runtime.py` | Serialize each Turn and coordinate compensating rollback across context/files/history |
| `sqlite_history.py` | Current completed Turns and legacy visible-context import |
| `sqlite_storage.py` | SQLite connection policy and per-file transaction ownership in opt-in NFS mode |
| `chainlit_data.py`, `chainlit_revision.py`, `chainlit_persistence.py` | Ordered UI history, native editing, temporary recovery and awaited background writes |
| `llm_retry.py`, `auxiliary_labels.py` | Provider retry policy and bounded asynchronous titles |

## Agent and context

The graph has no application filesystem backend, subagents, planning tools, project skill loader or shared memory. Tools are ordinary functions built for one Sandbox. Agent state crossing a Turn consists only of LangChain messages, including summaries. Model-call counters are per invocation, so reaching a limit does not permanently exhaust a Chat. Model Profiles accept `summary_options` for provider-specific summary configuration; the OpenRouter example disables reasoning for this small-output inference.

Middleware order is explicit: `ContextSummary` (standard `SummarizationMiddleware` with bounded, complete summary input), `ModelGuardrails` (request budget and retry immediately around the model handler), `ModelCallLimitMiddleware` (finite per-Turn inference budget). The summary model is a separately configured instance of the same Model Profile with its own output limit and streaming disabled. Internal summary events never become UI text.

Context compaction preserves recent tool-call/result groups using the upstream retention algorithm. Old data is serialized into bounded fragments and summarized sequentially with the preceding summary. This supports importing older, already oversized histories without silently discarding their beginning. Empty or over-budget summaries fail the Turn and leave the prior context intact. The final guard rejects an oversized uncompressible request before primary inference. Token counts are conservative estimates; the configured context budget must fit the real provider window.

OpenAI-compatible and native GigaChat models share an asynchronous inference retry adapter, with SDK retries disabled. Up to ten retries wait 1, 2, 4, 8, 16, 32, 64, 128, 256 and 300 seconds; Retry-After can lengthen a wait up to 300 seconds. Request deadlines apply to each attempt. A separate finite budget retries zero-chunk stream timeouts with the same backoff; a started stream is never replayed. GigaChat uses a ready access token and configurable base URL. Agent, summary and title models share the provider factory and one inference lock, held through attempts and backoff. Auxiliary deadlines start after lock acquisition. Retry never restarts tools, a graph or the whole Turn. Summarization bypasses the upstream nested retry runnable. The main model's call limit ends a runaway tool loop with an error and runtime rollback.

New Model Profiles default to streaming disabled and max_tokens=128000. The Agent's default total context budget is 160000 and its output cap is 128000; each main inference uses the smaller of the profile and Agent output limits, including the same limit in context guardrails. Summary and title output remain bounded separately. These are configurable application budgets, not claims about an endpoint's capacity.

`chainlit_requests.py` rejects overlapping messages and edits across every chat and socket before Chainlit persists them or replaces the task targeted by Stop. Rejection restores the visible history, removes unpersisted optimistic messages and reports that the model is busy. There is no user-request queue. The runtime independently rejects simultaneous Turns across chats, including Revision. Model inference remains asynchronous so cancellation and the UI stay responsive, but the shared lock prevents concurrent provider calls. See [ADR 0014](adr/0014-serial-inference-and-shared-backoff.md).

## Files

Virtual `/` maps only to the current Chat's `files/` directory. `ls`, `read_file`, `glob`, `grep` cannot resolve a host path. Each path component is opened relative to a directory descriptor with `O_NOFOLLOW`; traversal, symlink escapes and special files are rejected. A model cannot mutate files or run commands. File output is bounded; large regular files are read by pages or searched for a literal phrase. Very long lines are explicitly truncated. Uploaded data is untrusted evidence in the system prompt.

The application retains ownership of uploads and snapshots. Earlier `artifacts/` directories are still copied/restored as legacy Sandbox data; the new agent creates no context offload files and cannot read that tree.

## Persistence and Revision

`LOCALCHAT_SQLITE_NOLOCK=1` explicitly enables SQLite URI `nolock` for NFS without functional SQLite locks. Store construction reads the setting after CLI dotenv loading and freezes it until restart. `SQLiteDatabase` shares a process-local gate by resolved file path across schema setup, synchronous stores and complete Chainlit sessions. This includes the two adapters using `checkpoints.sqlite3`. Chainlit uses a one-connection pool without overflow; the shared session gate also covers multiple layer instances and all inherited SQL methods. Async acquisition never blocks the event loop. Cancellation drains active SQL and transaction cleanup before releasing the gate, including repeated cancellation. Normal mode retains SQLite locking and its default journal/sync policy.

NFS connections enforce DELETE journals and EXTRA synchronization. Persistent WAL headers are rejected before opening SQLite rather than migrated on a filesystem with broken locks. The CLI's existing `flock` plus atomic-directory guard remains the cross-process boundary. It is never automatically reclaimed; a missing parent PID does not prove its child or another pod stopped. Direct server launches and external database readers/writers must not bypass that boundary. This mode relies on correct NFS synchronization and does not add crash-atomic commits spanning multiple database files or Sandbox/blob state.

The infrastructure constraint and durability trade-off are recorded in [ADR 0015](adr/0015-serialize-sqlite-access-on-lockless-nfs.md).

`agent_context` stores current serialized messages by Chat. `agent_snapshots` stores messages before a Turn; a version-4 checkpoint token identifies an owned snapshot. The ReAct graph itself is ephemeral between calls. This avoids dependencies on LangGraph DeltaChannel ancestry, branch IDs and node schemas.

`ChatRuntime` captures context and Sandbox snapshots before execution. A Revision restores the selected pre-Turn state, runs the edited request and replaces the canonical continuation. It holds rollback state through the Chainlit UI commit. Background Message/Step writes are drained before commit or compensation. Failure or cancellation restores context (including its prior summary), files, runtime Turns and UI steps. UI recovery archives are temporary.

UI steps have a durable per-Chat `stepOrder`, preserved through edits and recovery. Revision selection and rendering use this order instead of timestamps or UUID order, since distinct messages can share the same timestamp. New values are allocated inside the step upsert SQL statement. Existing steps migrate in chronological order with original row order as the tie-breaker.

Stop cancels the owning Turn task once. That task alone renders cancellation inside the tracked write scope before rollback; there is no second UI writer in an `on_stop` callback. Repeated Stop events cannot interrupt cleanup. Failed revisions report a toast after restoring history, and model failures are recorded in the service log.

The current UI history contains full requests and answers even when model context has been summarized. It is independent of the summary representation. Confirming unchanged text is a no-op; a message without a completed Turn is not a valid revision target. After success or failure, the live UI is republished from authoritative SQLite.

## Existing data

Old binding tables migrate only their Chat/model mapping; capability modes and branch registries are retired. The first execution of a legacy Chat imports its current completed user requests and final answers, with stable message IDs. Old tool invocations, filesystem routes and graph nodes are not deserialized or executed. Existing tool traces remain in the UI; the new agent can reread uploaded files if it needs raw evidence.

For a legacy checkpoint token, the runtime history locates the corresponding active Turn and imports only earlier pairs. New checkpoints always capture the actual compacted context. A missing or foreign checkpoint is an error. Old internal LangGraph tables and Markdown memory may remain on disk, but are neither loaded nor used by the new execution path. The retired derived cross-Chat search index is removed; canonical Turn history is preserved.

## Installed application

`cli.py` implements `localchat init/run` without importing Chainlit before paths and environment are ready. Configuration defaults to the current directory; persistent SQLite and uploads default to `.local-agent-chat/` beside `.env`. Explicit config directories retain that same relative layout. Explicit command arguments override environment variables, which override literal `.env` values. Relative configured paths are anchored to the config directory. Init creates private files exclusively, generates a session secret and never overwrites existing settings.

`installation.py` owns the installed resource layout and process lock. The wheel includes UI assets, translations, templates and the application module. Each start copies resources to a fresh writable workspace under the data directory and sets `CHAINLIT_APP_ROOT` before importing Chainlit. The CLI supervises a child Chainlit process and forwards SIGINT/SIGTERM once. This keeps cleanup outside Chainlit's `os._exit()` shutdown path; application resources close before the Chainlit lifespan exits, and the parent removes the workspace while durable data stays in place. No runtime writes target site-packages. Durable data and temporary UI workspaces stay inside the selected data directory; technical logs remain on the console. See [ADR 0012](adr/0012-package-resources-and-separate-user-data.md).

## Hosting

`localchat run` reserves the first free port starting at the configured port and passes the listening socket to `server.py`, which initializes Chainlit and starts Uvicorn on that socket. Holding the socket eliminates the check/bind race. The `run.sh` wrapper delegates literal dotenv loading to the CLI. The default `APP_ROOT_PATH=auto` uses the JupyterHub/VS Code proxy when `JUPYTERHUB_SERVICE_PREFIX` is present, or no prefix locally. The exact `${JUPYTERHUB_SERVICE_PREFIX%/}/vscode/proxy/$APP_PORT` template is also supported without executing shell code. Automatic paths use the final port; explicit static or empty paths take precedence. Under JupyterHub the prefix is the full public route, including `/user/.../vscode/proxy/<port>`. The narrow ASGI adapter restores a prefix stripped by the proxy. DELETE bodies use the existing proxy-compatible POST override. Validate UI assets and WebSocket history through that public route, not only a direct local port.

Project-local defaults supersede the system-directory defaults in ADR 0012; see [ADR 0013](adr/0013-project-local-provider-configuration.md). Existing system directories require an explicit config path or an offline copy; SQLite schemas are unchanged.
