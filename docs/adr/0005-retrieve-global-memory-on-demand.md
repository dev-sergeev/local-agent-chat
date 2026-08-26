# Retrieve Global Memory from canonical active Turns on demand

Global Memory is implemented as two read-only Agent tools: bounded search across other Chats and bounded reading of a selected result with nearby Turns. The tools query an SQLite FTS5 index derived transactionally from canonical active `turns`. They never search the current Chat or `superseded_turns`, and retrieved text is marked as untrusted historical data rather than instructions.

The complete Chat History is not added to every model request. Embeddings and automatic extraction by a second model call remain deferred; FTS5 is local, deterministic, and effective for paths, identifiers, commands, errors, and quoted prior decisions. ADR 0008 separately introduces a small curated Markdown snapshot that is always loaded, without adding the full history.

## Consequences

Append, answer update, Revision, and Chat deletion update the index in the same SQLite transaction as canonical history, so stale branches disappear immediately. The Agent decides when prior context is relevant and first sees short candidates before reading a source. Semantic paraphrases may have lower recall than a future hybrid vector index, but no external embedding provider receives the history and the index can be rebuilt from active Turns.
