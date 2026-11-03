# Deployment companion

FluidLayer remains a research library, not a production-ready model-serving
system. The helpers here add application-side lifecycle controls without
changing the core package. They do not make the core suitable for sensitive
data, establish model quality, or replace a security review.

## Process-local session isolation

`SessionRegistry` creates one independent branch for each authenticated
`(tenant_id, session_id)` pair. A lease serializes requests for that pair,
while unrelated sessions can proceed independently. Idle sessions expire
after the configured TTL and the registry has a strict session-count limit.
Capacity exhaustion raises `SessionCapacityError` instead of silently
evicting a live user's state.

Example, using an already-loaded model and its embedding table:

```python
from deployment.session_registry import SessionRegistry
from fluidlayer import SAABranch

registry = SessionRegistry(
    lambda: SAABranch(
        M=64,
        r=8,
        d=model.emb.shape[1],
        emb=model.emb,
        ctx_len=1,
        capacity=8192,
    ),
    max_sessions=500,
    ttl_seconds=900,
)

# Obtain both identifiers from the authenticated application request.
with registry.lease(authenticated_tenant_id, conversation_id) as branch:
    hidden, info = branch.forward(residual, content, token_ids)
    # Keep all reads/writes for this request inside the lease.
```

Call `reset_session(tenant_id, session_id)` when a conversation starts over,
or `close_session(...)` when it ends. Both operations fail explicitly if the
session has an active lease. Capacity sizing must account for the per-branch
memory budget and actual Python object overhead, not only the documented
packed-slot estimate.

## Required controls outside this helper

- Authenticate and authorize callers before deriving tenant/session IDs.
- Never accept tenant IDs supplied by an untrusted client as authority.
- Do not share branch objects between users or retain them outside a lease.
- Add rate limits, request/token limits, observability, and graceful shutdown
  handling at the service boundary.
- Use process affinity or an application-owned session service when multiple
  workers must share a conversation. This registry is neither distributed nor
  persistent; restarting a worker loses all state.
- Treat retrieved text as untrusted input. Apply the serving model's normal
  prompt-injection, output validation, and data-retention controls.
- Do not store sensitive data until encryption, access control, retention, and
  incident-response requirements are independently designed and reviewed.

## Release gate before production use

The library has no supported production service contract. Before deploying a
specific integration, validate it on the target model and hardware: correctness
against the unmodified model, memory/latency under realistic concurrency,
session cleanup, failure behavior, tenant isolation, dependency and artifact
provenance, and rollback. The research results in this repository are not a
substitute for those checks.

Run the companion's tests with:

```bash
python -m pytest -q tests/test_session_registry.py
```
