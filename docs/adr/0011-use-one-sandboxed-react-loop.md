# Use one sandboxed ReAct loop with explicit persisted context

Replace Deep Agents with LangChain `create_agent`, four read-only Sandbox tools and an explicit middleware list. Persist only messages and rolling summaries between Turns, keeping the graph ephemeral: Revision then restores a small application-owned snapshot instead of reconstructing internal checkpoint ancestry. This trades resuming a partially completed graph for simpler, predictable Turn rollback and context migration.

This decision supersedes ADRs 0003–0005 and 0007–0009's agent capabilities, modes, subagents, skills and shared memory. ADR 0006's provider retry boundary and ADR 0010's coordinated UI commit remain. Existing Chats retain their visible history; legacy context is imported from completed requests and answers, never by executing obsolete tools or graph state. Every Chat now reads only its uploaded files, including Chats previously configured for host access.
