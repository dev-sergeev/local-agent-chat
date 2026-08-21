# Execute commands in the local Chat Sandbox

Python and shell commands execute through a local Deep Agents backend rooted in the Chat's Sandbox directory. The backend receives a minimal environment without model credentials. The surrounding Docker container is the command-isolation boundary.

## Consequences

Uploaded files remain in the JupyterHub container. Chat files are durable, while backend objects are recreated after application restart. Shell commands are not process-isolated from the container, so this mode is suitable only when the Docker container itself is the trusted isolation boundary.
