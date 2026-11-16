# ADR 0003 — Authentication and isolation

**Status:** Accepted
**Date:** 2026-11-16

## Context

The platform is multi-tenant: knowledge bases, documents, traces, chats, and
graph artifacts belong to a workspace. Cross-workspace leakage is a critical
failure.

## Decision

- Bearer API keys, generated with `secrets.token_urlsafe(32)`.
- Only a SHA-256 digest is stored; a display prefix is stored for identification.
- Keys carry scopes `read`, `write`, `admin`.
- The principal's workspace is resolved server-side; clients never supply a
  trusted workspace identifier.
- All queries filter by `workspace_id`; vector queries additionally filter by
  `knowledge_base_id` and `active`.
- Cross-tenant access returns 404 for object lookups to avoid existence leaks.

## Consequences

- Simple, testable auth suitable for API and dashboard bootstrap.
- Key rotation is manual via CLI (revocation supported by schema).
- No OAuth/SSO; documented as a limitation.
