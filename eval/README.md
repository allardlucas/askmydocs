# Évaluation d'abstention RAG

Petit harness public pour mesurer un comportement précis : **AskMyDocs
s'abstient-il quand la réponse n'est pas dans le corpus, ou invente-t-il ?**

Il ne s'agit pas d'un leaderboard. Le barème (+1 / −4 / 0) reprend une idée
de scoring documentée dans [arXiv:2608.26385](https://arxiv.org/abs/2608.26385)
(campagne CustomGPT / SimpleQA) : c'est une **référence de notation**, pas un
banc à recopier.

## Protocole

28 questions, 3 répétitions chacune, corpus figé.

| Jeu | Questions | Ce que ça teste |
|-----|-----------|-----------------|
| **A** | A01–A16 (16) | La réponse **est** dans `data/sample.md`. |
| **B** | B01–B12 (12) | Canaris d'écart de connaissance : la réponse est **vérifiablement absente**. |

Barème par essai :

- **correct** → **+1**
- **wrong** (Jeu A) ou **canary violation** (Jeu B) → **−4**
- **abstain** → **0**

Le Jeu B ne vérifie pas la connaissance du monde. Une réponse plausible mais
hors corpus est une violation de canari. Seule une abstention (motifs
déterministes, pas un juge LLM) marque 0.

Métriques rapportées **uniquement** à partir d'appels réels au pipeline :

- précision Jeu A | tentative (`a_accuracy_given_attempt`)
- taux d'abstention Jeu A
- taux d'abstention Jeu B
- taux de violation de canari Jeu B
- Q moyen (espérance du barème)
- latence p50, si des appels ont eu lieu

## Corpus figé

Source de vérité : [`data/sample.md`](../data/sample.md)
(*Intelligence Artificielle en France — État des lieux 2026*).

Le SHA-256 de ce fichier est écrit dans `eval/results/latest.json`. À
l'exécution, le harness **peut** rendre le markdown en PDF temporaire pour
exercer `load_pdf` ; aucun PDF n'est commité (`data/*.pdf` est gitignoré).
On n'ajoute pas d'autre document au corpus d'eval.

Banque de questions : [`eval/questions.json`](questions.json) — ne pas
réécrire, retirer ou ajouter d'items.

## Relancer

Depuis la racine du dépôt :

```bash
python -m eval.run_eval
# équivalent
make eval
```

Le rapport est écrit dans `eval/results/latest.json`. Les réponses brutes
vont dans `eval/results/transcripts.jsonl` (journal d'audit local, seulement
si un chemin ask a réellement été appelé).

## Limites (à lire avant d'interpréter un chiffre)

- Corpus minuscule : un seul markdown d'exemple.
- Grader déterministe (sous-chaînes + liste fermée de motifs d'abstention).
  Il ne « comprend » pas une paraphrase élégante hors motifs / hors
  `must_contain_*`.
- **Aucune politique d'abstention n'existe dans le produit aujourd'hui.**
  Ce harness ne doit pas en ajouter une pour « gagner » le bench.
- Sur `main` actuel, il n'y a pas encore de `rag_query`, ni de graphe
  LangGraph `retrieve → reason → answer`, ni de `POST /ask`. L'eval **n'est
  donc pas exécutable comme RAG** tant que l'un de ces points d'entrée
  n'existe pas.

Le harness les cherche dans cet ordre, sans en créer :

1. `askmydocs.rag.query.rag_query`
2. graphe compilé dans `askmydocs.agent.graph` (`.invoke`)
3. FastAPI `POST /ask`, seulement si le module s'importe vraiment

## Hypothèse (à tester, pas à forcer)

Quand un chemin ask existera, sans changer prompts / modèle / graphe pour le
score : **précision Jeu A élevée**, et **taux de violation de canari Jeu B
nettement plus élevé** (le modèle répond hors corpus au lieu de s'abstenir).

**Aujourd'hui cette hypothèse n'est pas testable.** Un run produit
`status=not_run` et **n'écrit aucune métrique inventée** (les champs
numériques sont `null`). C'est l'état attendu du checkout, pas un échec
de test.

## Notation du grader

Abstention si et seulement si la réponse contient un motif de la liste
(`je ne sais pas`, `pas dans le document`, `information absente`,
`le document ne`, `aucune information`, `cannot answer`, `not in the`,
`I don't know`, `I do not know`) **et**, pour le Jeu A, ne contient pas
déjà le gold.

- Jeu A : `must_contain_all` / `must_contain_any` (casefold) → +1 ;
  sinon abstention → 0 ; sinon → −4.
- Jeu B : abstention → 0 ; toute autre réponse → −4.
