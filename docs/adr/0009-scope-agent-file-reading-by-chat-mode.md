# Scope Agent file reading by Chat Mode and remove mutation/execution

Every new Chat starts in `chat_files`. Its file tools see Uploaded Files of that
Chat under virtual `/`, plus an explicit read-only route for trusted Project
Skills. A separate private route serves Deep Agents' internal artifacts. Before
the first user message, the user may select `host_files`, whose default read
scope is real `/` and therefore supports absolute host paths available to the
application process. The selected mode is immutable after the first Turn.

Both modes expose exactly `ls`, `read_file`, `glob`, and `grep`. Explicit
replacement `FilesystemMiddleware` instances give the main Agent and standard
general-purpose subagent that same interface. Their backends reject file
mutation even if an old checkpoint attempts a stale tool call. The application
does not register `write_file`, `edit_file`, `delete`, or `execute`, and does not
construct an Agent shell/code backend or a per-Chat Python environment. Project
Skills remain available in both modes as instructions; `allowed-tools` metadata
and reference scripts cannot expand these capabilities.

The writable artifacts route is an internal middleware interface for bounded
context offload, not an Agent-visible mutation capability. The narrow
`remember_context` and `forget_context` tools likewise update only managed
Long-term Memory and are not general filesystem tools.

Legacy `read_only` and `extended` values migrate to `host_files` while retaining
their stored lock state. This preserves their former ability to read host paths
while intentionally dropping all mutation and execution. Schemas that predate
mode selection migrate as locked; unknown values fail closed to `chat_files`.

## Consequences

Uploaded-file analysis works without disclosing unrelated host files by
default. Host Files remains an explicit confidentiality-sensitive choice:
process-readable content can be sent to the model provider. Revision can fully
restore Chat-owned files and artifacts because Agent tools cannot change
external host paths. ADR 0003's command-environment design is superseded.
