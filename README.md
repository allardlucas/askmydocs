# 🤖 AskMyDocs

[![Tests](https://github.com/allardlucas/askmydocs/actions/workflows/test.yml/badge.svg)](https://github.com/allardlucas/askmydocs/actions/workflows/test.yml)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue?logo=python)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> Pose des questions à tes documents. AskMyDocs ingère tes PDF, cherche sémantiquement et répond avec un agent LangGraph.

AskMyDocs est un **agent IA documentaire** construit en Python :

- 📄 **RAG** : ingestion PDF, chunking, recherche vectorielle avec LanceDB
- 🧠 **Agent** : graphe LangGraph `retrieve → reason → answer`
- 🔧 **Outils** : outils MCP custom (calculatrice, métadonnées fichier)
- 🚀 **API** : FastAPI avec OpenAPI
- 📱 **UI** : interface Streamlit pour démonstrations

Le repo est public et pensé comme vitrine pour du consulting Python / IA.

---

## 🚀 Quickstart

```bash
# Cloner le repo
git clone https://github.com/allardlucas/askmydocs.git
cd askmydocs

# Créer l'environnement
python -m venv .venv
source .venv/bin/activate

# Installer les dépendances de dev
make install-dev

# Lancer les tests
make test

# Linter
make lint
```

---

## 🛠️ Commandes utiles

| Commande | Description |
|----------|-------------|
| `make install` | Installe les dépendances runtime |
| `make install-dev` | Installe les dépendances de développement |
| `make test` | Lance la suite de tests |
| `make lint` | Vérifie le style avec ruff |
| `make format` | Formate et corrige le code |
| `make run-api` | Lance l'API FastAPI sur le port 8000 |
| `make run-ui` | Lance l'interface Streamlit |

---

## 📁 Structure du projet

```
askmydocs/
├── src/askmydocs/     # Code source
│   ├── api/           # Routes FastAPI
│   ├── agent/         # Graphe LangGraph
│   ├── mcp/           # Serveur et outils MCP
│   ├── rag/           # Pipeline RAG
│   └── ui/            # Application Streamlit
├── tests/             # Tests avec pytest
├── docs/              # Architecture et guides
├── scripts/           # Scripts utilitaires
├── data/              # Documents d'exemple
├── Makefile
└── pyproject.toml
```

---

## 📖 Stack technique

- [LangGraph](https://langchain-ai.github.io/langgraph/) — orchestration d'agents
- [LangChain](https://python.langchain.com/) — abstractions LLM
- [LanceDB](https://lancedb.github.io/lancedb/) — base vectorielle locale
- [FastAPI](https://fastapi.tiangolo.com/) — API REST
- [Streamlit](https://streamlit.io/) — UI de démo
- [MCP](https://modelcontextprotocol.io/) — Model Context Protocol
- [pytest](https://docs.pytest.org/) + [ruff](https://docs.astral.sh/ruff/)

---

## 📋 Feuille de route

Le projet est découpé en tickets dans Linear. On avance à raison de 1–2 tickets par jour :

1. **Phase 0** — Setup projet, dépendances, CI
2. **Phase 1** — RAG : chargement PDF, chunking, stockage vectoriel
3. **Phase 2** — Agent LangGraph : retrieve, reason, answer
4. **Phase 3** — MCP : serveur d'outils et intégration
5. **Phase 4** — API FastAPI
6. **Phase 5** — UI Streamlit
7. **Phase 6** — Docker & déploiement
8. **Phase 7** — Polish final

---

## 👨‍💻 Auteur

Fait avec soin par [Lucas Allard](https://github.com/allardlucas).

Licence MIT — libre à toi de t'en servir.
