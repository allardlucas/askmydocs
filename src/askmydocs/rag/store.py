"""VectorStore LanceDB (ticket VEL-54).

Stocke et interroge des embeddings vectoriels à l'aide de LanceDB, une
base de données vectorielle locale sans serveur. Ce module fait le lien
entre les chunks de texte produits par le chunker et le moteur de
recherche sémantique du pipeline RAG.

Exemple d'utilisation :
    >>> from askmydocs.rag.store import VectorStore
    >>> store = VectorStore("/tmp/mydb")
    >>> store.add(
    ...     texts=["Hello world"],
    ...     embeddings=[[0.1, 0.2, 0.3]],
    ...     metadatas=[{"source": "doc1"}],
    ... )
    >>> results = store.search([0.1, 0.2, 0.3], top_k=3)
    >>> results[0]["text"]
    'Hello world'
"""

from __future__ import annotations

import contextlib
import json
import uuid
from pathlib import Path
from typing import Any

import lancedb
import pyarrow as pa

__all__ = ["VectorStore"]

_SCHEMA_ID = pa.field("id", pa.string())
_SCHEMA_TEXT = pa.field("text", pa.string())
_SCHEMA_METADATA = pa.field("metadata", pa.string())


class VectorStore:
    """Stockage et recherche vectorielle avec LanceDB.

    Chaque document est représenté par un identifiant unique, un texte
    brut, un vecteur d'embedding et des métadonnées optionnelles.
    LanceDB stocke les données localement sur disque via des tables
    Apache Arrow.

    Parameters
    ----------
    db_path:
        Chemin vers le répertoire de la base LanceDB. Créé s'il
        n'existe pas.
    table_name:
        Nom de la table Arrow utilisée pour stocker les documents.
    """

    def __init__(self, db_path: str | Path, table_name: str = "documents") -> None:
        self.db_path = Path(db_path)
        self.table_name = table_name
        self.db = lancedb.connect(str(self.db_path))
        self._dimension: int | None = None

    # ------------------------------------------------------------------
    # API publique
    # ------------------------------------------------------------------

    def add(
        self,
        texts: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, Any]] | None = None,
    ) -> None:
        """Ajoute des documents au vector store.

        Parameters
        ----------
        texts:
            Liste des textes à indexer.
        embeddings:
            Liste des embeddings associés, un vecteur par texte.
        metadatas:
            Métadonnées optionnelles. Si la clé ``"id"`` est présente
            dans un dict, elle est extraite comme identifiant du
            document (sinon un ``uuid4`` est généré).

        Raises
        ------
        ValueError
            Si les listes n'ont pas la même longueur.
        """
        if metadatas is None:
            metadatas = [{} for _ in texts]

        n = len(texts)
        if len(embeddings) != n or len(metadatas) != n:
            raise ValueError(
                f"Longueurs incompatibles : texts={len(texts)}, "
                f"embeddings={len(embeddings)}, metadatas={len(metadatas)}"
            )

        if n == 0:
            return

        dim = len(embeddings[0])
        records = self._build_records(texts, embeddings, metadatas, dim)

        table = self._open_existing()
        if table is None:
            schema = pa.schema(
                [
                    _SCHEMA_ID,
                    _SCHEMA_TEXT,
                    pa.field("vector", pa.list_(pa.float32(), dim)),
                    _SCHEMA_METADATA,
                ]
            )
            self._dimension = dim
            self.db.create_table(self.table_name, records, schema=schema, mode="create")
        else:
            table.add(records)

    def search(self, query_embedding: list[float], top_k: int = 5) -> list[dict[str, Any]]:
        """Recherche les `top_k` documents les plus proches d'un embedding.

        La distance utilisée est la distance euclidienne (L2) par
        défaut. Les résultats sont triés par score ascendant (plus la
        distance est petite, plus le document est pertinent).

        Parameters
        ----------
        query_embedding:
            Vecteur d'embedding de la requête.
        top_k:
            Nombre maximum de résultats à retourner.

        Returns
        -------
        list[dict]
            Liste de dictionnaires ordonnés par ``score`` ascendant.
            Chaque entrée contient les clés ``id``, ``text``,
            ``metadata`` et ``score``.
        """
        table = self._open_existing()
        if table is None:
            return []

        results = table.search(query_embedding).limit(top_k).to_list()

        return [
            {
                "id": row["id"],
                "text": row["text"],
                "metadata": json.loads(row["metadata"]),
                "score": row["_distance"],
            }
            for row in results
        ]

    def delete(self, ids: list[str]) -> None:
        """Supprime les documents correspondant aux identifiants donnés.

        Parameters
        ----------
        ids:
            Liste des identifiants de documents à supprimer.
        """
        if not ids:
            return
        table = self._open_existing()
        if table is None:
            return
        quoted = ", ".join(f"'{doc_id}'" for doc_id in ids)
        table.delete(f"id IN ({quoted})")

    def clear(self) -> None:
        """Supprime la totalité de la table (base vide après l'appel)."""
        with contextlib.suppress(ValueError):
            self.db.drop_table(self.table_name)

    # ------------------------------------------------------------------
    # Méthodes privées
    # ------------------------------------------------------------------

    def _open_existing(self):
        """Ouvre la table si elle existe, sinon renvoie ``None``."""
        try:
            return self.db.open_table(self.table_name)
        except ValueError:
            return None

    @staticmethod
    def _build_records(
        texts: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, Any]],
        dim: int,
    ) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for i in range(len(texts)):
            meta = dict(metadatas[i])
            doc_id = meta.pop("id", str(uuid.uuid4()))
            emb = embeddings[i]
            if len(emb) != dim:
                raise ValueError(f"Embedding {i} a une dimension {len(emb)} au lieu de {dim}")
            records.append(
                {
                    "id": doc_id,
                    "text": texts[i],
                    "vector": emb,
                    "metadata": json.dumps(meta),
                }
            )
        return records
