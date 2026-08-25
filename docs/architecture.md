# MVP architecture

## Runtime

- Chainlit serves the UI under `APP_ROOT_PATH`, supplied to `chainlit run --root-path` behind Jupyter Server Proxy.
- Browser-side DELETE requests with JSON bodies are tunneled as POST across Jupyter Server Proxy and restored to DELETE by a narrow ASGI middleware before Chainlit routing.
- A silent, fixed Local User identity enables Chainlit's built-in history without a login form.
- Each new Chat selects one Model Profile loaded from YAML. Provider secrets come only from environment variables.
- One Turn per Chat may run at a time; concurrent submissions are rejected.
- Deep Agents emits typed text/tool events through `ChatRuntime`; a Chainlit-only adapter persists text segments and tool Steps as one ordered sibling timeline inside the Turn. A new text segment starts after every tool so live rendering and resumed Chat History keep the same chronology.
- Tool input is rendered according to its type (a shell command is Bash, structured file arguments are JSON). Tool output is sanitized and persisted as a bounded preformatted text log so terminal text cannot be interpreted as chat Markdown.
- Shell Steps never expose the raw command in their title. A separate bounded call to the Chat's Model Profile asynchronously replaces the neutral initial title with a three-to-five-word Russian description of the command's purpose; invalid, failed, or timed-out descriptions retain the neutral fallback and never fail the Turn.
- Tool Steps are collapsed on their first render and are never auto-collapsed later, preventing layout jumps while preserving manual expansion; failed Steps remain marked as errors.
- Cancelling a Turn restores its pre-turn Agent Memory and Sandbox snapshot before another Turn can start.

## Persistence

- Chainlit history uses a local SQLite data layer.
- Writes for one Chainlit Step are serialized so a background `create_step` cannot overwrite a newer streamed output or tool result during reload.
- Legacy raw tool output is normalized when Chat History is read, so saved chats remain readable after upgrading.
- LangGraph checkpoints use a separate SQLite database keyed by the Chainlit thread ID.
- Each Chat owns a durable file directory with a 1 GiB limit; each Uploaded File is limited to 100 MiB.
- Binary files are stored in the Chat directory, not in SQLite.

## Agent execution

- Agent and tool-title system prompts are centralized in `local_agent_chat/prompts.py`.
- Deep Agents runs in the application process; model credentials are not injected into the command environment.
- A local adapter supplies filesystem tools and Python/shell execution.
- The local adapter keeps one command backend in memory per Chat and roots file operations in that Chat's Sandbox directory.
- Commands receive a minimal environment but execute on the application host with the application's permissions. The deployment must provide a dedicated container or VM as the actual isolation boundary.

## Revision

- Chainlit's native user-message editor is used without a custom frontend.
- Before every Turn, the application records the corresponding agent checkpoint and Sandbox snapshot.
- Editing a user message must reproduce Chainlit's revision behavior across both the UI and the agent backend; updating only the displayed or persisted message text is invalid.
- When an edited message is submitted, the application performs the following operation as one coordinated workflow:
  1. Identify the checkpoint and Sandbox snapshot immediately before the original Turn.
  2. Remove every later Turn from the active Chat History, including the original agent result and all downstream tool results.
  3. Restore Agent Memory to the identified pre-turn checkpoint so the model cannot see superseded messages or results.
  4. Restore Sandbox files to the matching pre-turn snapshot so commands are not rerun against stale side effects.
  5. Persist the edited user message as the new active Turn.
  6. Invoke the Agent again and stream a newly generated result to Chainlit.
- Chainlit Steps and elements from the superseded branch are staged before replay, removed during replay, and restored if replay fails.
- The edited request and regenerated result become the canonical continuation of the Chat. A page reload or application restart must restore this revised state, not the superseded branch.
- If either Agent Memory or Sandbox restoration fails, the Agent must not be rerun and the active history must not be partially replaced.
- Superseded records remain only in a local technical audit log.

## Lifecycle

- Resuming a Chat restores its model profile, Agent Memory, and Sandbox association.
- Deleting a Chat deletes its checkpoints, stored files, snapshots, and in-memory backend.
- Files survive application restarts; the in-memory backend is recreated from the Chat directory.
