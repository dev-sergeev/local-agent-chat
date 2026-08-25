# Execute Extended-mode commands in a Chat-specific Python environment

Only Extended Agent Mode exposes command execution. Python and shell commands execute through a local Deep Agents backend whose working directory is the Chat's `files/` directory, while absolute paths retain their real host meaning. Every Extended Chat owns a persistent `environment/venv` plus private `HOME`, temp and cache directories. The service environment is removed from `PATH`, user site-packages are disabled and normal `pip` operations target only the Chat venv. Read-only Chat creates no environment.

## Consequences

Chat dependencies survive application restarts and remain separate from revisioned files and internal artifacts; editing an earlier Turn rolls those revisioned directories back but does not uninstall packages. Deleting the Chat removes files, artifacts, snapshots, and its environment.

This isolates Python dependencies, not arbitrary code. Extended commands and file tools can still access and modify the host filesystem and network with the service user's permissions, and those external changes are not rolled back by Revision. The deployment must provide a dedicated container or VM as the security boundary.
