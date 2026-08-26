# Bind Agent Mode to the Chat at its first Turn

Every Chat has an Agent Mode: `chat_files` by default or `host_files`. The UI may change the selection before the first user message. A synchronous Socket.IO pre-dispatch adapter orders Host Files-setting frames and the first valid message frame before Chainlit creates concurrent handler and persistence tasks. The message frame atomically locks the last accepted mode in the authoritative Chat registry; a failed or cancelled Turn does not unlock it. Stop, settings update, and Chat resume also recover the lock from a persisted request created by older or interrupted code. Chainlit metadata mirrors the value but never authorizes capabilities.

Both modes expose exactly `ls`, `read_file`, `glob`, and `grep`. Chat Files maps Uploaded Files of the current Chat to virtual `/`; Host Files maps real `/` and therefore accepts process-readable absolute host paths. Explicit replacement Deep Agents filesystem middleware and read-only backends enforce the boundary on the main Agent and general-purpose subagent; the prompt only explains it. Project Skills are routed read-only to both modes and do not expand capabilities.

The capability decision and removal of mutation/execution are recorded in [ADR 0009](0009-scope-agent-file-reading-by-chat-mode.md).

Legacy `read_only` and `extended` values migrate to `host_files`, preserving the host-read scope those Chats previously had while deliberately removing mutation and execution. Their existing lock flag is preserved; schemas that predate mode selection migrate as locked `host_files`. Unrecognized values fail closed to `chat_files` without escalating their stored lock state.

## Consequences

Resume, restart, Revision, and model checkpoints cannot change a Chat's read scope. New modes require an explicit domain and migration change rather than a presentation-only switch. Host Files can disclose process-readable content to the model provider, while Revision restores only Chat-owned files and agent artifacts; the deployment container or VM remains the security boundary.

Chainlit exposes no synchronous pre-message callback, so the adapter isolates one private `python-socketio` seam and verifies its expected `_handle_event` signature at startup. A dependency update that changes this seam fails explicitly rather than silently weakening the invariant.
