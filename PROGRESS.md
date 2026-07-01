# AskMyDocs — Suivi d'avancement

> Coche les tickets au fur et à mesure. 1-2 par jour. Privé, zéro trace publique.

---

## Phase 0 : Setup

- [ ] **0.1** — Créer `.gitignore`, `README.md`, `LICENSE`, `pyproject.toml`, `src/askmydocs/__init__.py`
- [ ] **0.2** — `requirements.txt` + `requirements-dev.txt` (langgraph, fastapi, lancedb, mcp, pytest…)
- [ ] **0.3** — `Makefile` (install, test, lint, run-api, run-ui) + `.github/workflows/test.yml`

## Phase 1 : RAG — Ingestion & recherche

- [x] **1.1** — `src/askmydocs/rag/loader.py` — `load_pdf(path)` avec pymupdf (+ tests)
- [x] **1.2** — `src/askmydocs/rag/chunker.py` — `chunk_text(text, size, overlap)` (+ tests)
- [ ] **1.3** — `src/askmydocs/rag/store.py` — `VectorStore` LanceDB avec `add()` et `search()` (+ tests)
- [ ] **1.4** — `src/askmydocs/rag/ingest.py` — `ingest_document(pdf_path)` pipeline complet (+ tests)
- [ ] **1.5** — `src/askmydocs/rag/query.py` — `rag_query(question, store)` retrieve + LLM (+ tests)
- [ ] **1.6** — `src/askmydocs/rag/session.py` — `DocSession` (lister, supprimer, basculer) (+ tests)
- [ ] **1.7** — `data/ia_france_2026.md` — document démo + `make ingest-demo`
- [ ] **1.8** — `README.md` — section RAG Pipeline

## Phase 2 : Agent LangGraph

- [ ] **2.1** — `src/askmydocs/agent/state.py` — `AgentState` TypedDict
- [ ] **2.2** — `src/askmydocs/agent/nodes.py` — `retrieve_node(state)` (+ tests)
- [ ] **2.3** — `src/askmydocs/agent/nodes.py` — `reason_node(state)` décision RETRIEVE/TOOL/ANSWER (+ tests)
- [ ] **2.4** — `src/askmydocs/agent/nodes.py` — `answer_node(state)` réponse finale (+ tests)
- [ ] **2.5** — `src/askmydocs/agent/graph.py` — StateGraph retrieve→reason→answer (+ tests)
- [ ] **2.6** — `src/askmydocs/agent/tools.py` — `web_search(query)` duckduckgo, intégré au graphe (+ tests)
- [ ] **2.7** — Boucle agent : max 5 itérations, fallback (+ tests)
- [ ] **2.8** — `README.md` — section Architecture Agent + diagramme ASCII

## Phase 3 : MCP — Model Context Protocol

- [ ] **3.1** — `src/askmydocs/mcp/server.py` — serveur MCP stdio, outil `ping` (+ tests)
- [ ] **3.2** — `src/askmydocs/mcp/tools/calculator.py` — outil calculatrice (+ tests)
- [ ] **3.3** — `src/askmydocs/mcp/tools/files.py` — outil métadonnées fichier (+ tests)
- [ ] **3.4** — `src/askmydocs/mcp/client.py` — client MCP intégré à l'agent (+ tests)
- [ ] **3.5** — Intégration MCP dans le graphe LangGraph (+ tests)
- [ ] **3.6** — `docs/mcp.md` + `README.md` section MCP

## Phase 4 : API FastAPI

- [ ] **4.1** — `src/askmydocs/api/main.py` + `routes.py` — `POST /documents` (+ tests)
- [ ] **4.2** — `POST /ask` — question → agent LangGraph → réponse (+ tests)
- [ ] **4.3** — `GET /health`, `GET /documents`, `DELETE /documents/{id}` (+ tests)
- [ ] **4.4** — Documentation OpenAPI + gestion erreurs 404/422/500 (+ tests)
- [ ] **4.5** — `docs/api.md` + `README.md` section API

## Phase 5 : UI Streamlit

- [ ] **5.1** — `src/askmydocs/ui/app.py` — page upload PDF
- [ ] **5.2** — Page chat : question → réponse + étapes agent
- [ ] **5.3** — Visualisation du raisonnement (expanders Streamlit)
- [ ] **5.4** — `README.md` — section Interface + screenshot

## Phase 6 : Docker & déploiement

- [ ] **6.1** — `Dockerfile` — Python 3.11-slim, expose 8000
- [ ] **6.2** — `docker-compose.yml` — api (8000) + ui (8501)
- [ ] **6.3** — `scripts/demo.sh` — one-liner : docker compose up + ingest + navigateur
- [ ] **6.4** — `tests/e2e/test_docker.py` — test end-to-end Docker

## Phase 7 : Polish

- [ ] **7.1** — `README.md` final (badges, quickstart, architecture, stack)
- [ ] **7.2** — `docs/architecture.md` — diagramme Mermaid
- [ ] **7.3** — `ruff check --fix` + `mypy src/` + docstrings → `make lint` vert

---

**Total : 41 tickets** | 1-2/jour = 4-5 semaines

Dernière mise à jour : _(date)_
