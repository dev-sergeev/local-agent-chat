# Execute commands in the local Chat Sandbox

Python and shell commands execute through a local Deep Agents backend rooted in the Chat's Sandbox directory. The backend receives a minimal environment without model credentials, but it does not restrict host filesystem or network access. The deployment, not this application, must provide a dedicated container or VM as the command-isolation boundary.

## Consequences

Uploaded files remain in the deployment environment. Chat files are durable, while backend objects are recreated after application restart. This mode is suitable only for a trusted single user when the surrounding container or VM is the trusted isolation boundary.
