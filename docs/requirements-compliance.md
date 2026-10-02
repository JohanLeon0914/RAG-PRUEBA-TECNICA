# Requirements Compliance Audit

| Requirement | Implementation | Files/components | Validation | Status |
|---|---|---|---|---|
| 1. Python | Backend implemented with FastAPI/Python 3.12. | `app/`, `pyproject.toml`, `Dockerfile` | `pytest`, Docker build | Complete |
| 2. Web scraping | Bancolombia scraper with domain boundary, rate limit, robots-aware filters and configurable seeds. | `app/scraping/`, `scripts/scrape_bancolombia.py` | Phase 2 diagnostics and scraper tests | Complete |
| 3. Raw storage | Raw documents stored as JSONL. | `data/raw/`, `app/scraping/storage.py` | Phase 2 corpus validation | Complete |
| 4. Processed storage | Clean documents stored as JSONL. | `data/processed/`, `data/corpus/processed_documents.jsonl` | Cleaner tests and corpus report | Complete |
| 5. Vectorization | Local embeddings via BGE-M3/GTE strategies. | `app/embeddings/` | Phase 3 ingestion and embedding tests | Complete |
| 6. Vector DB | Qdrant stores chunks, vectors and payload metadata. | `app/repositories/qdrant_repository.py`, `docker-compose.yml` | Qdrant count/dimension validation | Complete |
| 7. Conversational interface | `/chat` endpoint and React frontend. | `app/api/routes/chat.py`, `frontend/` | API/frontend tests and E2E Docker validation | Complete |
| 8. History by ID | `session_id` accepted/returned by `/chat`. | `app/rag/service.py`, `frontend/src/App.tsx` | Phase 9 tests and E2E multi-turn check | Complete |
| 9. N previous messages configurable | `CONVERSATION_HISTORY_N_MESSAGES` controls prior messages sent to prompt. | `app/config/settings.py`, `app/rag/service.py` | Repository/RAGService tests | Complete |
| 10. Persisted history | SQLite sessions/messages persist in Docker volume. | `app/repositories/sqlite_conversation_repository.py`, `docker-compose.yml` | Restart persistence test | Complete |
| 11. Dockerfile | API Dockerfile uses Python 3.12 and uv. | `Dockerfile` | `docker compose build` | Complete |
| 12. docker-compose | Qdrant, API and frontend services with persistent volumes. | `docker-compose.yml` | `docker compose config/up` | Complete |
| 13. Public repository readiness | Project is structured for GitHub with `.env.example` and docs. | repository root | Needs final push/visibility check by owner | Ready / requires user action |
| 14. Logical commit history | Incremental commits exist for productization and frontend fixes. | Git history | `git log --oneline` review | Ready / no rewrite planned |
| 15. 3 design patterns | Repository, Strategy and Factory used for real seams. | `app/repositories/`, `app/embeddings/`, `app/reranking/`, `app/llm/` | README/design docs | Complete |
| 16. Analysis of conversation history | Analytics endpoint aggregates persisted conversations. | `app/api/routes/conversations.py`, SQLite repository | Analytics tests and E2E summary | Complete |
| 17. README prerequisites | Docker and Compose documented. | `README.md` | Manual review | Complete |
| 18. README setup | `.env`, Qdrant, ingestion, stack startup documented. | `README.md` | Manual review | Complete |
| 19. README usage | `/chat`, sessions, history and analytics examples documented. | `README.md` | Manual review | Complete |
| 20. README patterns | Design Patterns section explains practical reasons. | `README.md` | Manual review | Complete |
| 21. README stack justification | Qdrant, SQLite, BGE, reranker and LLM Factory decisions documented. | `README.md`, `docs/architecture-decisions.md` | Manual review | Complete |
| 22. README limitations | Local SQLite, no auth, no cloud, no business KPI claims documented. | `README.md` | Manual review | Complete |
| 23. README future improvements | Query rewriting, production DB, cloud deployment implied in limitations/decisions. | `README.md`, `docs/architecture-decisions.md` | Manual review | Complete |
| 24. Reranker bonus | BGE reranker implemented, evaluated and optional. | `app/reranking/`, `scripts/evaluate_reranker.py` | Phase 6 benchmark | Complete |
| 25. Error handling bonus | API maps validation/runtime/unexpected errors without stack traces. | `app/api/routes/chat.py` | API tests | Complete |
| 26. Externalized config bonus | Settings centralized with `.env.example`; no hardcoded secrets. | `app/config/settings.py`, `.env.example` | Settings tests | Complete |

## Notes

- BBVA was replaced by Bancolombia after HTTP 403 diagnostics with standard Python clients. No antibot evasion was implemented.
- Runtime analytics are not business KPIs. They are technical indicators derived from persisted conversation data.
- Cloud deployment is intentionally out of scope for this frozen local deliverable.
