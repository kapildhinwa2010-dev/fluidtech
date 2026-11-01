Security Policy
Supported versions
| version|supported|
| --- | ---|
| 1.2.x|✅|
| 1.1.x|✅ (security fixes only)|
| < 1.1|❌|
Reporting a vulnerability
Do not open a public issue. Use GitHub's private vulnerability
reporting (`Security` → `Report a vulnerability`).
Please include: the affected version, a minimal reproduction, and the impact
you believe it has.
Threat model — read this before using it
This is a research prototype, and it stores untrusted text.  Two things
follow from that:
The layer is a prompt-injection surface.  The exact tier stores token
ids retrieved from the stream and can inject them into the residual
stream.  A malicious document can therefore attempt to steer the model
through the memory.  The register is append-only and first-write-wins, so
a fact can be corroborated before it is trusted — but nothing here
authenticates content.
The store is a cache of raw text.  It is not encrypted and not
tenant-isolated by default: one `SAABranch` instance = one memory.  If you
serve multiple users, give each its own instance (or its own namespace
inside the store keys).
The framework refuses to overflow the register rather than silently growing
it, and a signature mismatch makes a recall fail rather than return
another fact's value.  Those are the only safety guarantees claimed here.
Non-goals
No sandboxing of the injected content.
No cryptographic integrity for the stored facts.
No defence against a model that has been fine-tuned to obey the memory.