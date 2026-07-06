"""Tests du VectorStore LanceDB (ticket 1.3).

Vérifie l'ajout de documents, la recherche vectorielle, la gestion
des métadonnées et les cas limites (store vide, longueurs incohérentes).

Chaque test utilise un répertoire temporaire isolé pour éviter toute
interférence entre stores (LanceDB écrit sur disque même pour des
stores éphémères).
"""

from __future__ import annotations

import pytest

from askmydocs.rag.store import VectorStore

# --- add() ----------------------------------------------------------------


def test_add_single_document(tmp_path):
    """L'ajout d'un seul document doit porter len() à 1."""
    store = VectorStore(str(tmp_path / "db"))
    store.add(["hello"], [[0.1, 0.2, 0.3]])

    assert len(store) == 1


def test_add_multiple_documents(tmp_path):
    """L'ajout de plusieurs documents doit tous les indexer."""
    store = VectorStore(str(tmp_path / "db"))
    texts = ["alpha", "beta", "gamma"]
    embeddings = [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]]
    store.add(texts, embeddings)

    assert len(store) == 3


def test_add_appends_to_existing(tmp_path):
    """Deux appels à add() doivent s'accumuler dans le store."""
    store = VectorStore(str(tmp_path / "db"))
    store.add(["first"], [[0.1, 0.2]])
    store.add(["second"], [[0.3, 0.4]])

    assert len(store) == 2


def test_add_with_metadata(tmp_path):
    """Les métadonnées fournies doivent être stockées et retrouvables."""
    store = VectorStore(str(tmp_path / "db"))
    store.add(
        ["doc1"],
        [[0.1, 0.2]],
        metadata=[{"source": "test.pdf", "page": 1}],
    )

    results = store.search([0.1, 0.2], top_k=1)
    assert len(results) == 1
    assert results[0]["metadata"] == {"source": "test.pdf", "page": 1}


def test_add_length_mismatch_raises(tmp_path):
    """texts et embeddings de longueurs différentes → ValueError."""
    store = VectorStore(str(tmp_path / "db"))
    with pytest.raises(ValueError, match="longueur"):
        store.add(["a", "b"], [[0.1]])


def test_add_metadata_length_mismatch_raises(tmp_path):
    """metadata de longueur différente → ValueError."""
    store = VectorStore(str(tmp_path / "db"))
    with pytest.raises(ValueError, match="longueur"):
        store.add(["a"], [[0.1]], metadata=[{}, {}])


# --- search() -------------------------------------------------------------


def test_search_returns_closest_document(tmp_path):
    """La recherche doit renvoyer le document le plus proche."""
    store = VectorStore(str(tmp_path / "db"))
    store.add(
        ["python", "javascript", "rust"],
        [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]],
    )

    results = store.search([1.0, 0.0], top_k=1)

    assert len(results) == 1
    assert results[0]["text"] == "python"


def test_search_top_k_limits_results(tmp_path):
    """top_k doit limiter le nombre de résultats."""
    store = VectorStore(str(tmp_path / "db"))
    texts = [f"doc-{i}" for i in range(10)]
    embeddings = [[float(i), 0.0] for i in range(10)]
    store.add(texts, embeddings)

    results = store.search([5.0, 0.0], top_k=3)

    assert len(results) == 3


def test_search_empty_store_returns_empty(tmp_path):
    """Un store vide doit renvoyer une liste vide."""
    store = VectorStore(str(tmp_path / "db"))

    assert store.search([0.1, 0.2]) == []


def test_search_results_contain_distance(tmp_path):
    """Chaque résultat doit contenir le champ _distance."""
    store = VectorStore(str(tmp_path / "db"))
    store.add(["x"], [[0.1, 0.2]])

    results = store.search([0.1, 0.2], top_k=1)

    assert "_distance" in results[0]
    assert results[0]["_distance"] >= 0.0


def test_search_without_metadata_returns_none(tmp_path):
    """Si aucune métadonnée n'a été fournie, metadata doit être None."""
    store = VectorStore(str(tmp_path / "db"))
    store.add(["sans meta"], [[0.5, 0.5]])

    results = store.search([0.5, 0.5], top_k=1)

    assert results[0]["metadata"] is None


# --- len() ----------------------------------------------------------------


def test_len_empty_store_is_zero(tmp_path):
    """Un store vide doit avoir len() == 0."""
    store = VectorStore(str(tmp_path / "db"))
    assert len(store) == 0


# --- persistence ----------------------------------------------------------


def test_persistence_across_instances(tmp_path):
    """Les données doivent survivre à la fermeture/réouverture du store."""
    db_path = str(tmp_path / "lance_db")

    store1 = VectorStore(db_path)
    store1.add(["persisted"], [[0.1, 0.2, 0.3]])

    store2 = VectorStore(db_path)
    assert len(store2) == 1

    results = store2.search([0.1, 0.2, 0.3], top_k=1)
    assert results[0]["text"] == "persisted"
