# 📚 AskMyDocs

> Un agent IA documentaire qui répond à vos questions à partir de vos documents — propulsé par LangGraph, MCP et le RAG.

[![CI](https://github.com/lucasallard/askmydocs/actions/workflows/test.yml/badge.svg)](https://github.com/lucasallard/askmydocs/actions/workflows/test.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🎯 Le projet en deux mots

AskMyDocs est un assistant intelligent qui lit vos documents (PDF, Markdown,
texte brut) et répond à vos questions en s'appuyant sur leur contenu. Il ne se
contente pas d'une recherche par mots-clés : il comprend le contexte, peut
enchaîner plusieurs étapes de raisonnement et faire appel à des outils
externes quand sa base interne ne suffit pas.

La stack technique a été choisie pour être à la fois moderne et pédagogique :

| Brique | Technologie | Rôle |
|--------|-------------|------|
| 🧠 Agent | **LangGraph** | Orchestre le raisonnement (retrieve → reason → answer) |
| 🔎 RAG | **LanceDB** | Vectorisation et recherche sémantique dans vos docs |
| 🔧 MCP | **Model Context Protocol** | Expose des outils à l'agent (calculatrice, fichiers…) |
| 🚀 API | **FastAPI** | Endpoints REST pour programmer l'agent |
| 🖥️ UI | **Streamlit** | Interface web prête à l'emploi |

## 🚀 Quickstart

```bash
# 1. Cloner le dépôt
git clone https://github.com/lucasallard/askmydocs.git
cd askmydocs

# 2. Créer un environnement virtuel (Python 3.11+)
python -m venv .venv
source .venv/bin/activate

# 3. Installer les dépendances et le package en mode éditable
make install

# 4. Lancer les tests
make test

# 5. Démarrer l'API (FastAPI + Uvicorn)
make run-api

# 6. Démarrer l'interface web (Streamlit)
make run-ui
```

## 📁 Structure du dépôt

```
askmydocs/
├── src/askmydocs/        # Code source du package
├── tests/                # Suite de tests (pytest)
├── .github/workflows/    # CI GitHub Actions
├── Makefile              # Cibles : install, test, lint, run-api, run-ui
├── pyproject.toml        # Configuration du package et des outils
├── requirements.txt      # Dépendances d'exécution
└── requirements-dev.txt  # Dépendances de développement
```

## 🧪 Tests & qualité

Le projet suit une approche **TDD** (Test-Driven Development) : chaque
fonctionnalité est d'abord couverte par un test, puis implémentée.

```bash
make test   # lance la suite pytest
make lint   # vérifie le style avec ruff
```

## 👤 Auteur

**Lucas Allard** — Ingénieur IA & données  
[GitHub](https://github.com/lucasallard) · [LinkedIn](https://www.linkedin.com/in/lucasallard)

## 📄 Licence

Distribué sous licence **MIT**. Voir le fichier [`LICENSE`](LICENSE) pour
plus de détails.
