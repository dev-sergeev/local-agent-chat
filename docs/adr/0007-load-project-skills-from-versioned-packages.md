# Load Project Skills from versioned packages

Project Skills live in `skills/<name>/SKILL.md` packages and are loaded through Deep Agents' native Skills source rather than being stored in Chat History, Uploaded Files or model configuration. This keeps procedural instructions reviewable with the application, supports progressive disclosure and gives the main Agent and its standard subagent one canonical source. Chat Files receives an explicit read-only route to this trusted source; Host Files can read the same absolute source directly.

## Consequences

Skills are trusted project instructions, shared by all Chats and available in both Agent Modes. They cannot expand a mode's filesystem scope, add tools, or make reference scripts executable. Their metadata enters Agent Memory on the first Turn, so a new or changed Skill becomes visible in a new Chat rather than mutating the behavior of an existing Chat midstream.
