"""Tests du VectorStore LanceDB (ticket VEL-54).

Vérifie l'ajout, la recherche, la suppression et le vidage du store
vectoriel, ainsi que les cas limites (store vide, métadonnées, etc.).
"""

from __future__ import annotations

import uuid
from pathlib import Path

import pytest

from askmydocs.rag.store import VectorStore


@pytest.fixture
def embeddings() -> list[list[float]]:
    """Quatre vecteurs de dimension 4."""
    return [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0, 0.0],
    ]


@pytest.fixture
def texts() -> list[str]:
    """Quatre textes associés aux embeddings."""
    return [
        "Python est un langage de programmation",
        "Les chiens sont des mammifères",
        "La cuisine française est réputée",
        "Apprendre à coder en Python",
    ]


@pytest.fixture
def filled_store(tmp_path: Path, texts: list[str], embeddings: list[list[float]]) -> VectorStore:
    """VectorStore pré-rempli avec 4 documents."""
    store = VectorStore(tmp_path / "test_db", "documents")
    store.add(texts=texts, embeddings=embeddings)
    return store


# --- Ajout et recherche ---------------------------------------------------


class TestAddAndSearch:
    """Vérifie que l'ajout et la recherche fonctionnent correctement."""

    def test_search_returns_results(
        self, filled_store: VectorStore, embeddings: list[list[float]]
    ) -> None:
        """Une recherche doit retourner des résultats."""
        results = filled_store.search(embeddings[0], top_k=2)
        assert len(results) == 2

    def test_search_result_structure(
        self, filled_store: VectorStore, embeddings: list[list[float]]
    ) -> None:
        """Chaque résultat doit contenir id, text, metadata et score."""
        results = filled_store.search(embeddings[0], top_k=1)
        assert len(results) == 1
        r = results[0]
        assert set(r.keys()) == {"id", "text", "metadata", "score"}
        assert isinstance(r["id"], str)
        assert isinstance(r["text"], str)
        assert isinstance(r["metadata"], dict)
        assert isinstance(r["score"], float)

    def test_exact_match_has_lowest_score(
        self, filled_store: VectorStore, embeddings: list[list[float]]
    ) -> None:
        """L'embedding de la requête exacte doit avoir le score le plus bas."""
        results = filled_store.search(embeddings[1], top_k=4)
        assert results[0]["text"] == "Les chiens sont des mammifères"
        assert results[0]["score"] <= results[1]["score"]

    def test_top_k_limits_results(
        self, filled_store: VectorStore, embeddings: list[list[float]]
    ) -> None:
        """top_k doit limiter le nombre de résultats."""
        results = filled_store.search(embeddings[0], top_k=1)
        assert len(results) == 1

    def test_results_sorted_by_score_ascending(
        self, filled_store: VectorStore, embeddings: list[list[float]]
    ) -> None:
        """Les résultats doivent être triés par score ascendant."""
        results = filled_store.search(embeddings[0], top_k=4)
        scores = [r["score"] for r in results]
        assert scores == sorted(scores)


# --- Métadonnées -----------------------------------------------------------


class TestMetadata:
    """Vérifie la gestion des métadonnées."""

    def test_add_with_metadata(self, tmp_path: Path, embeddings: list[list[float]]) -> None:
        """Les métadonnées passées à `add` doivent être restituées dans `search`."""
        store = VectorStore(tmp_path / "meta_db", "docs")
        meta = {"source": "livre.pdf", "page": 42}
        store.add(
            texts=["Contenu du livre"],
            embeddings=[embeddings[0]],
            metadatas=[meta],
        )
        results = store.search(embeddings[0], top_k=1)
        assert results[0]["metadata"] == meta

    def test_custom_id_in_metadata(self, tmp_path: Path, embeddings: list[list[float]]) -> None:
        """Un id fourni dans les métadonnées doit être utilisé comme identifiant."""
        store = VectorStore(tmp_path / "id_db", "docs")
        custom_id = "doc-abc-123"
        store.add(
            texts=["Texte avec id personnalisé"],
            embeddings=[embeddings[0]],
            metadatas=[{"id": custom_id, "tag": "important"}],
        )
        results = store.search(embeddings[0], top_k=1)
        assert results[0]["id"] == custom_id
        # L'id ne doit pas apparaître dans les métadonnées restituées
        assert "id" not in results[0]["metadata"]

    def test_auto_generated_id_is_valid_uuid(
        self, tmp_path: Path, embeddings: list[list[float]]
    ) -> None:
        """Sans id fourni, un UUID4 valide doit être généré."""
        store = VectorStore(tmp_path / "uuid_db", "docs")
        store.add(texts=["Sans id"], embeddings=[embeddings[0]])
        results = store.search(embeddings[0], top_k=1)
        doc_id = results[0]["id"]
        # Doit pouvoir être parsé en UUID4
        parsed = uuid.UUID(doc_id)
        assert parsed.version == 4

    def test_metadatas_defaults_to_empty_dicts(
        self, tmp_path: Path, embeddings: list[list[float]]
    ) -> None:
        """Sans métadonnées explicites, un dict vide est stocké."""
        store = VectorStore(tmp_path / "no_meta_db", "docs")
        store.add(texts=["Hello"], embeddings=[embeddings[0]])
        results = store.search(embeddings[0], top_k=1)
        assert results[0]["metadata"] == {}


# --- Suppression -----------------------------------------------------------


class TestDelete:
    """Vérifie la suppression de documents."""

    def test_delete_removes_document(
        self, filled_store: VectorStore, embeddings: list[list[float]]
    ) -> None:
        """Un document supprimé ne doit plus apparaître dans les recherches."""
        results_before = filled_store.search(embeddings[1], top_k=1)
        doc_id = results_before[0]["id"]

        filled_store.delete([doc_id])

        results_after = filled_store.search(embeddings[1], top_k=1)
        assert all(r["id"] != doc_id for r in results_after)

    def test_delete_nonexistent_id_does_nothing(
        self, filled_store: VectorStore, embeddings: list[list[float]]
    ) -> None:
        """Supprimer un id inexistant ne doit pas lever d'erreur."""
        filled_store.delete(["does-not-exist"])
        results = filled_store.search(embeddings[0], top_k=4)
        assert len(results) == 4  # aucun document perdu

    def test_delete_multiple_ids(
        self, filled_store: VectorStore, embeddings: list[list[float]]
    ) -> None:
        """Supprimer plusieurs ids d'un coup doit fonctionner."""
        results = filled_store.search(embeddings[3], top_k=4)
        ids_to_delete = [r["id"] for r in results[:2]]

        filled_store.delete(ids_to_delete)
        remaining = filled_store.search(embeddings[3], top_k=4)
        assert len(remaining) == 2

    def test_delete_empty_list(
        self, filled_store: VectorStore, embeddings: list[list[float]]
    ) -> None:
        """Supprimer une liste vide ne fait rien."""
        filled_store.delete([])
        results = filled_store.search(embeddings[0], top_k=4)
        assert len(results) == 4


# --- Vidage -----------------------------------------------------------------


class TestClear:
    """Vérifie le vidage intégral du store."""

    def test_clear_empties_store(
        self, filled_store: VectorStore, embeddings: list[list[float]]
    ) -> None:
        """Après `clear`, une recherche ne retourne rien."""
        filled_store.clear()
        results = filled_store.search(embeddings[0], top_k=5)
        assert results == []

    def test_clear_is_idempotent(self, filled_store: VectorStore) -> None:
        """Appeler `clear` plusieurs fois ne doit pas lever d'erreur."""
        filled_store.clear()
        filled_store.clear()  # ne doit pas planter


# --- Store vide ------------------------------------------------------------


class TestEmptyStore:
    """Vérifie le comportement avec un store non initialisé."""

    def test_search_on_empty_store(self, tmp_path: Path, embeddings: list[list[float]]) -> None:
        """Rechercher dans un store vide retourne une liste vide."""
        store = VectorStore(tmp_path / "empty_db", "docs")
        results = store.search(embeddings[0], top_k=5)
        assert results == []

    def test_delete_on_empty_store(self, tmp_path: Path) -> None:
        """Supprimer dans un store vide ne lève pas d'erreur."""
        store = VectorStore(tmp_path / "empty_delete_db", "docs")
        store.delete(["anything"])

    def test_add_then_search(self, tmp_path: Path, embeddings: list[list[float]]) -> None:
        """Ajouter puis rechercher dans un store fraîchement créé."""
        store = VectorStore(tmp_path / "fresh_db", "docs")
        store.add(texts=["A"], embeddings=[embeddings[0]])
        results = store.search(embeddings[0], top_k=1)
        assert len(results) == 1
        assert results[0]["text"] == "A"


# --- Cas d'erreur -----------------------------------------------------------


class TestErrors:
    """Vérifie la levée d'exceptions pour des entrées invalides."""

    def test_length_mismatch_raises(self, tmp_path: Path, embeddings: list[list[float]]) -> None:
        """Des listes de longueurs différentes doivent lever ValueError."""
        store = VectorStore(tmp_path / "err_db", "docs")
        with pytest.raises(ValueError):
            store.add(texts=["a", "b"], embeddings=embeddings[:1])

    def test_empty_add_is_noop(self, tmp_path: Path) -> None:
        """Ajouter des listes vides ne fait rien."""
        store = VectorStore(tmp_path / "empty_add_db", "docs")
        store.add(texts=[], embeddings=[])
        results = store.search([0.1, 0.2, 0.3, 0.4], top_k=1)
        assert results == []
