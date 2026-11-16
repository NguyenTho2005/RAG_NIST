# Reference Analysis — GRC-RAG

**Date of inspection:** 2026-11-16
**Reference URL:** https://github.com/NguynHung-VN/GRC-RAG
**Inspection mode:** read-only public HTTP requests (GitHub web + REST API)

## 1. Access result: `REFERENCE_UNAVAILABLE`

| Probe | URL | Result | Evidence |
|---|---|---|---|
| Repository page | `https://github.com/NguynHung-VN/GRC-RAG` | HTTP 404 | Web fetch returned `404 GET` |
| Repository API | `https://api.github.com/repos/NguynHung-VN/GRC-RAG` | HTTP 404 `Not Found` | GitHub REST API |
| Owner account | `https://api.github.com/users/NguynHung-VN` | HTTP 200, `public_repos: 1` | GitHub REST API |
| Owner public repositories | `https://api.github.com/users/NguynHung-VN/repos?per_page=100` | One repository: `maison` | GitHub REST API |
| Owner public events | `https://api.github.com/users/NguynHung-VN/events/public?per_page=100` | Empty list (`[]`) | GitHub REST API |
| GitHub repository search | `https://api.github.com/search/repositories?q=GRC-RAG` | 44 results, none owned by `NguynHung-VN` | GitHub REST API |

The account exists. The requested repository is not publicly accessible under that
name. It may be private, renamed, deleted, or never public. None of these
possibilities can be verified from outside.

The search results contain unrelated projects with similar names (for example
`Viglyn/viglyn-grc-rag`, `lovish-dhillon/grc-rag`, `EamMengHor/GRC_RAG`,
`amdwave/grc-rag`, `SdSarthak/AegisAI`). **They are not the requested reference
and were not used as a substitute.**

## 2. Evidence classification

Every architectural property of the requested repository is **UNKNOWN**.
No README, file listing, issue, release, diagram, or source file from
`NguynHung-VN/GRC-RAG` was available to inspect.

| Property | Classification | Evidence |
|---|---|---|
| Application purpose (beyond the name "GRC-RAG") | UNKNOWN | Repository 404 |
| Technology stack | UNKNOWN | Repository 404 |
| Repository structure | UNKNOWN | Repository 404 |
| Ingestion architecture | UNKNOWN | Repository 404 |
| Document parsers | UNKNOWN | Repository 404 |
| Chunking strategy | UNKNOWN | Repository 404 |
| Embedding pipeline | UNKNOWN | Repository 404 |
| Vector database | UNKNOWN | Repository 404 |
| Retrieval algorithms | UNKNOWN | Repository 404 |
| Graph components | UNKNOWN | Repository 404 |
| LLM orchestration | UNKNOWN | Repository 404 |
| Query routing | UNKNOWN | Repository 404 |
| Reranking | UNKNOWN | Repository 404 |
| API interface | UNKNOWN | Repository 404 |
| Frontend | UNKNOWN | Repository 404 |
| Persistence | UNKNOWN | Repository 404 |
| Evaluation methodology | UNKNOWN | Repository 404 |
| Deployment | UNKNOWN | Repository 404 |
| Known limitations | UNKNOWN | Repository 404 |

**No parity claim about the reference architecture can be verified.** Any claim
of "same architecture as GRC-RAG" would be fabrication. Reference-specific
parity is therefore recorded as **UNVERIFIED** in
`docs/02-feature-parity-matrix.md`.

## 3. What was not done

- No `git clone`, fork, archive download, or mirror.
- No reference script executed.
- No unrelated GRC-RAG repository substituted.
- No source code, prompt, test, or documentation copied.

## 4. Consequences for this project

The implementation follows the independent architecture in this repository's
specification. Features are justified by retrieval-engineering evidence and by
the project requirements — not by asserted reference parity. Where a feature
would normally be described as "matching the reference", this project instead
says: **independently implemented; reference parity UNVERIFIED**.

## 5. Reproducing this inspection

```bash
curl -s -o /dev/null -w '%{http_code}\n' https://github.com/NguynHung-VN/GRC-RAG
# 404
curl -s https://api.github.com/repos/NguynHung-VN/GRC-RAG
# {"message":"Not Found", ...}
curl -s https://api.github.com/users/NguynHung-VN
# public_repos: 1
curl -s 'https://api.github.com/users/NguynHung-VN/repos?per_page=100'
# [{"name":"maison", ...}]
```

If the repository becomes public later, rerun Phase 0 and update this document;
do not retroactively describe this inspection as having succeeded.
