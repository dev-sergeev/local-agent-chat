# Start with local persistence behind adapters

The application starts with separate local SQLite persistence for Chainlit chat history and LangGraph agent checkpoints, linked by the Chainlit thread identifier. Storage boundaries remain replaceable because SQLite support for the Chainlit data layer is not an officially guaranteed deployment path, while SQLite checkpointing is supported by LangGraph.

## Consequences

Resuming a chat must restore both persistence planes without replaying the same history twice. The application will identify one local user silently so Chainlit can expose its built-in chat history without presenting a login screen.
