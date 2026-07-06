"""Vector store LanceDB (ticket 1.3).

Stocke et interroge les embeddings des chunks de documents via LanceDB.
Le store est le dernier maillon de la chaîne d'indexation RAG : après
avoir chargé un document et l'avoir découpé en chunks, on génère des
embeddings et on les persiste ici pour la recherche.

Exemple d'utilisation :
    >>> from askmydocs.rag.store import VectorStore
    >>> store = VectorStore("/tmp/db")
    >>> store.add(["Hello", "World"], [[0.1, 0.2], [0.3, 0.4]])
    >>> results = store.search([0.1, 0.2], top_k=1)
    >>> len(results)
    1
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

import lancedb
import pyarrow as pa

__all__ = ["VectorStore"]


def _build_schema(dim: int) -> pa.Schema:
    """Construit le schéma pyarrow pour une dimension de vecteur donnée.

    LanceDB requiert un ``FixedSizeList`` pour les colonnes vectorielles.
    La dimension est figée au premier ``add()`` et tous les appels
    suivants doivent respecter cette même dimension.
    """
    return pa.schema(
        [
            pa.field("id", pa.string()),
            pa.field("text", pa.string()),
            pa.field("vector", pa.list_(pa.float32(), dim)),
            pa.field("metadata", pa.string(), nullable=True),
        ]
    )


class VectorStore:
    """Store vectoriel persistant basé sur LanceDB.

    Chaque document est stocké avec son embedding, son texte et des
    métadonnées optionnelles (sérialisées en JSON). La recherche
    sémantique exploite l'index vectoriel de LanceDB (distance cosinus
    par défaut).

    Parameters
    ----------
    path:
        Chemin vers le répertoire de la base LanceDB. Par défaut, un
        répertoire ``.lancedb`` dans le CWD est utilisé.
    table_name:
        Nom de la table LanceDB, par défaut ``"chunks"``.

    Notes
    -----
    Les métadonnées sont sérialisées en JSON dans une colonne texte
    pour éviter les contraintes de schéma struct (chaque chunk peut
    avoir des clés différentes).
    """

    def __init__(
        self,
        path: str | Path = ".lancedb",
        table_name: str = "chunks",
    ) -> None:
        self._db = lancedb.connect(str(path))
        self._table_name = table_name
        self._table = self._open_if_exists()

    # ------------------------------------------------------------------
    # API publique
    # ------------------------------------------------------------------

    def add(
        self,
        texts: list[str],
        embeddings: list[list[float]],
        metadata: list[dict[str, Any]] | None = None,
    ) -> None:
        """Ajoute des documents et leurs embeddings au store.

        Parameters
        ----------
        texts:
            Liste des textes (chunks) à indexer. Doit avoir la même
            longueur que ``embeddings``.
        embeddings:
            Liste des vecteurs d'embedding, un par texte. Doit avoir
            la même longueur que ``texts``. Tous les vecteurs doivent
            avoir la même dimension.
        metadata:
            Métadonnées associées à chaque chunk (source, page, etc.).
            Si ``None``, ``None`` est stocké. Si fournie, doit avoir
            la même longueur que ``texts``. Les dicts sont sérialisés
            en JSON.

        Raises
        ------
        ValueError
            Si ``texts`` et ``embeddings`` n'ont pas la même longueur,
            ou si ``metadata`` est fourni mais de longueur différente.
        """
        n_texts = len(texts)
        n_emb = len(embeddings)

        if n_texts != n_emb:
            raise ValueError(
                f"texts et embeddings doivent avoir la même longueur ({n_texts} != {n_emb})"
            )

        if metadata is not None and len(metadata) != n_texts:
            raise ValueError(
                f"metadata doit avoir la même longueur que texts ({len(metadata)} != {n_texts})"
            )

        # Détection de la dimension depuis le premier vecteur.
        dim = len(embeddings[0])
        schema = _build_schema(dim)

        rows: list[dict[str, Any]] = []
        for i in range(n_texts):
            meta_raw = metadata[i] if metadata is not None else None
            rows.append(
                {
                    "id": str(uuid.uuid4()),
                    "text": texts[i],
                    "vector": [float(x) for x in embeddings[i]],
                    "metadata": json.dumps(meta_raw, ensure_ascii=False)
                    if meta_raw is not None
                    else None,
                }
            )

        # Création ou append via RecordBatch typé (FixedSizeList).
        batch = pa.RecordBatch.from_pylist(rows, schema=schema)

        if self._table is None:
            self._table = self._db.create_table(self._table_name, data=batch, schema=schema)
        else:
            self._table.add(batch)

    def search(self, query_embedding: list[float], top_k: int = 5) -> list[dict[str, Any]]:
        """Recherche les ``top_k`` chunks les plus proches du vecteur requête.

        Parameters
        ----------
        query_embedding:
            Vecteur d'embedding de la requête.
        top_k:
            Nombre de résultats à retourner, par défaut 5.

        Returns
        -------
        list[dict[str, Any]]
            Liste de dictionnaires contenant les champs ``text``,
            ``metadata`` et ``_distance`` pour chaque résultat. Liste
            vide si le store ne contient aucun document.
        """
        if self._table is None or len(self) == 0:
            return []

        raw = (
            self._table.search([float(x) for x in query_embedding], vector_column_name="vector")
            .limit(top_k)
            .to_list()
        )

        results: list[dict[str, Any]] = []
        for r in raw:
            meta_raw = r.get("metadata")
            meta = None
            if meta_raw:
                try:
                    meta = json.loads(meta_raw)
                except (TypeError, json.JSONDecodeError):
                    meta = meta_raw
            results.append(
                {
                    "text": r["text"],
                    "metadata": meta,
                    "_distance": r["_distance"],
                }
            )
        return results

    def __len__(self) -> int:
        """Nombre de documents présents dans le store."""
        if self._table is None:
            return 0
        return self._table.count_rows()

    # ------------------------------------------------------------------
    # Interne
    # ------------------------------------------------------------------

    def _open_if_exists(self):
        """Tente d'ouvrir la table si elle existe déjà, sinon None."""
        try:
            return self._db.open_table(self._table_name)
        except Exception:
            return None
