# Security policy

## Supported versions

| Version | Security support |
|---|---|
| 1.2.x | Supported |
| 1.1.x | Security fixes only |
| Earlier versions | Unsupported |

## Report a vulnerability

Do not disclose vulnerabilities in a public issue. Use GitHub's private
vulnerability reporting feature from the repository's **Security** tab.
Include the affected version, a minimal reproduction, and the potential
impact. Avoid including real user data or secrets in a report.

## Threat model

Superfluid is a research prototype that can store and retrieve content from
an input stream. Retrieved content can influence model outputs, so
untrusted documents may act as prompt-injection inputs.

The memory is not encrypted, tenant-isolated, or protected by cryptographic
integrity checks by default. A `SAABranch` instance holds session memory;
applications that serve multiple users must isolate instances and manage
their lifecycle and access controls themselves. Stored values and majority
support counts do not establish that retrieved content is true.

The low-level `FrozenRegister` checks a supplied signature during recall and
refuses writes after its capacity is reached. These checks do not provide
authentication, tamper evidence, or protection against poisoning.

## Non-goals

This project does not currently provide:

- sandboxing for retrieved or injected content;
- encryption at rest or in transit;
- tenant isolation or access control;
- cryptographic integrity or authenticated provenance;
- a defense against prompt injection, memory poisoning, or malicious models.

Do not use the project as a security boundary or with sensitive production
data without implementing and independently validating the controls your
application requires.
