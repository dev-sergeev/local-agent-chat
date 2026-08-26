# Persist model, memory, and sandbox at the chat boundary

Each chat selects one model profile and owns separate durable agent memory and a durable sandbox. This keeps resumed chats reproducible, prevents files from leaking between chats, and allows a revision of a user request to roll both memory and files back to the same pre-turn state.

## Consequences

Uploaded files live in the chat sandbox rather than SQLite. Deleting a chat deletes its sandbox, and supporting Chainlit's native revision of any user message requires both an agent checkpoint and a sandbox snapshot for every turn.

A revision is a backend state transition, not merely a UI update: it truncates the active history after the edited request, restores agent memory and sandbox files to their matching pre-turn state, persists the replacement request, and reruns the agent to produce a new result. These changes must succeed as one coordinated operation; the agent must never continue from a mixture of revised messages, superseded memory, or stale filesystem side effects. Superseded turns remain in a technical audit log but disappear from the active chat timeline.

Files attached to a revised request are imported only after its pre-turn Sandbox snapshot has been restored and before the Agent reruns. The import participates in the Revision transaction: provider failure, cancellation, or persistence failure restores the previously active Sandbox and removes tentative revised uploads.

The runtime transaction remains open until Chainlit has persisted and committed the replacement UI continuation. It retains the current Agent checkpoint, Sandbox snapshot, and an exact snapshot of every active runtime Turn from the edited Turn onward. If rendering or the Chainlit commit fails after runtime history was replaced, that snapshot compensates the runtime write, restores its FTS sources and audit boundary, and restores Agent Memory and Sandbox before the per-Chat lock is released.

LangGraph root graphs do not support using `checkpoint_ns` as an application-level memory branch. Every restore therefore materializes the selected checkpoint as the durable head of a new tracked LangGraph thread; restoring the pre-first-Turn state creates an empty thread. The active thread pointer is persisted before the next Turn, survives restart, and all threads owned by a Chat are removed on deletion.
