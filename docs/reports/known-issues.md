# Known Issues and Limitations

## External / environmental

1. **Reference repository unavailable.** `NguynHung-VN/GRC-RAG` returns 404; owner exposes only `maison`. No feature parity verified; architecture is independent.
2. **No LLM model pulled.** Ollama 0.11.3 runs but `qwen2.5:1.5b-instruct` is not downloaded. `/models` and generation paths report unavailable/unconfigured honestly; no real generation E2E run.
3. **No real embedding model downloaded.** Default `intfloat/multilingual-e5-small` (MIT, 384d) not fetched. Tests use deterministic finite fake embeddings (dim 8) and do not prove semantic retrieval quality.
4. **No git repo initialized** in `/home/tho/Documents/RAG`; no commits.

## Functional gaps

5. **Frontend missing:** `apps/web` absent; 10 dashboard pages not implemented; compose `full` profile references missing web Dockerfile.
6. **GraphRAG not implemented:** `/graph` is gated and reports unavailable honestly (no fake data).
7. **Evaluation gaps:** dataset `grc-vi-v1` (74 questions, synthetic corpus) is **AI-authored draft, human review NOT_VERIFIED**. Retrieval metrics harness is implemented/tested; answer faithfulness, citation support, abstention/completeness, and LLM-judge runs are not performed (blocked without real LLM and human review). Ablation/report export automation is limited.
8. **Real-model validation blocked:** reranker weights, semantic retrieval quality, latency/cost under real models unmeasured.

## Correctness / hardening notes

9. Ingestion retries clear partial chunks/sections/vectors for reused revisions (exercised for early failures). Add regression for failure injected after Qdrant upsert but before DB publish.
10. URL ingestion disabled by default; SSRF controls exist, but redirect-chain and DNS-rebinding need dedicated adversarial tests.
11. Qdrant requires raised `nofile` when multiple collections are active (configured in `docker-compose.yml`).
12. `docker-compose.yml` still references `apps/api/Dockerfile`-style expectations in some environments? (actual build uses `infra/api.Dockerfile` for api/worker; verify compose paths if running `full` profile.)

## Reporting rule

The final report must state **PARTIALLY COMPLETE** unless every critical gate (real-model E2E, human-reviewed evaluation, security review/backup-restore, frontend, and reference parity or explicit waiver) is verified.
