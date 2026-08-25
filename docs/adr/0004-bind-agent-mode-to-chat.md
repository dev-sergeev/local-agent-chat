# Bind Agent Mode to the Chat at its first Turn

Every Chat has an Agent Mode: `read_only` by default or `extended`. The UI may change the selection before the first user message. A synchronous Socket.IO pre-dispatch adapter orders Extended-setting frames and the first valid message frame before Chainlit creates concurrent handler and persistence tasks. The message frame atomically locks the last accepted mode in the authoritative Chat registry; a failed or cancelled Turn does not unlock it. Stop, settings update, and Chat resume also recover the lock from a persisted request created by older or interrupted code. Chainlit metadata mirrors the value but never authorizes capabilities.

Read-only exposes only explicit file read/list/search tools over real absolute host paths. Extended is their strict superset and additionally exposes create, edit, delete, and command execution. A non-executable read-only backend and an explicit Deep Agents filesystem-tool allowlist enforce the boundary; the prompt only explains it. Legacy Chats are migrated to locked Extended mode because that preserves their former capabilities.

## Consequences

Resume, restart, Revision, and model checkpoints cannot change a Chat's capabilities. New modes require an explicit domain and migration change rather than a presentation-only switch. Extended operations may change arbitrary host paths, while Revision can restore only Chat-owned files and agent artifacts; the deployment container or VM remains the security boundary.

Chainlit exposes no synchronous pre-message callback, so the adapter isolates one private `python-socketio` seam and verifies its expected `_handle_event` signature at startup. A dependency update that changes this seam fails explicitly rather than silently weakening the invariant.
