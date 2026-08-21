# Persist model, memory, and sandbox at the chat boundary

Each chat selects one model profile and owns separate durable agent memory and a durable sandbox. This keeps resumed chats reproducible, prevents files from leaking between chats, and allows a revision of a user request to roll both memory and files back to the same pre-turn state.

## Consequences

Uploaded files live in the chat sandbox rather than SQLite. Deleting a chat deletes its sandbox, and supporting Chainlit's native revision of any user message requires both an agent checkpoint and a sandbox snapshot for every turn.

A revision is a backend state transition, not merely a UI update: it truncates the active history after the edited request, restores agent memory and sandbox files to their matching pre-turn state, persists the replacement request, and reruns the agent to produce a new result. These changes must succeed as one coordinated operation; the agent must never continue from a mixture of revised messages, superseded memory, or stale filesystem side effects. Superseded turns remain in a technical audit log but disappear from the active chat timeline.
