# Load Project Skills from versioned packages

Project Skills live in `skills/<name>/SKILL.md` packages and are loaded through Deep Agents' native Skills source rather than being stored in Chat History, Sandbox files or model configuration. This keeps procedural instructions reviewable with the application, supports progressive disclosure and gives the main Agent and its standard subagent one canonical source.

## Consequences

Skills are trusted project code, shared by all Chats and cannot expand Agent Mode. Their metadata enters Agent Memory on the first Turn, so a new or changed Skill becomes visible in a new Chat rather than mutating the behavior of an existing Chat midstream.
