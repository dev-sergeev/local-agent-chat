# Serialize SQLite access when NFS cannot provide file locks

Some installations have only persistent NFS without working SQLite locks and ephemeral local storage, with no persistent block volume or database service available. Keep Chat History, Agent Memory and Model Profiles on NFS through an explicit `LOCALCHAT_SQLITE_NOLOCK=1` mode: all adapters share transaction ownership by resolved database path, Chainlit limits its connection pool, and cancellation drains SQL before releasing ownership. A single application instance alone is insufficient because its connections can overlap; ordinary SQLite locking remains the default.

## Consequences

NFS mode uses DELETE journals and EXTRA synchronization, rejects existing WAL databases without migrating them, and retains the CLI's cross-process guard without automatic stale-lock removal. Operators must confirm that all previous owners have stopped before reclaiming a stale guard; direct server launches and other database clients must not bypass it. Durability still depends on NFS synchronization, and this decision does not make a Revision crash-atomic across the three databases and uploaded files; ephemeral SQLite with periodic snapshots was rejected because it deliberately loses changes since the last snapshot.
