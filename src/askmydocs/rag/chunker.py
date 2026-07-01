"""Text chunker (ticket VEL-53).

Découpe un texte en chunks de taille fixe (en caractères) avec un
chevauchement configurable entre chunks consécutifs. C'est l'étape
qui précède l'embedding dans le pipeline RAG.

Exemple d'utilisation :
    >>> from askmydocs.rag.chunker import chunk_text
    >>> chunks = chunk_text("abcdefghij", chunk_size=4, overlap=2)
    >>> chunks
    ['abcd', 'cdef', 'efgh', 'ghij']
"""

from __future__ import annotations

__all__ = ["chunk_text"]


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Découpe `text` en chunks de `chunk_size` caractères avec overlap.

    Le découpage se fait **caractère par caractère** (pas de respect des
    mots). Chaque chunk (sauf le premier) commence `overlap` caractères
    avant la fin du précédent, ce qui garantit un contexte partagé entre
    chunks adjacents.

    Parameters
    ----------
    text:
        Texte à découper.
    chunk_size:
        Taille maximale de chaque chunk, en caractères. Doit être > 0.
    overlap:
        Nombre de caractères de chevauchement entre chunks consécutifs.
        Doit respecter ``0 <= overlap < chunk_size``.

    Returns
    -------
    list[str]
        Liste des chunks. Liste vide si ``text`` est vide.

    Raises
    ------
    ValueError
        Si ``chunk_size <= 0`` ou si ``overlap`` est négatif ou
        supérieur ou égal à ``chunk_size``.

    Examples
    --------
    >>> chunk_text("abcdefghij", chunk_size=4, overlap=2)
    ['abcd', 'cdef', 'efgh', 'ghij']

    >>> chunk_text("court", chunk_size=500, overlap=50)
    ['court']

    >>> chunk_text("")
    []
    """
    if chunk_size <= 0:
        raise ValueError(
            f"chunk_size doit être strictement positif, reçu {chunk_size}"
        )
    if overlap < 0:
        raise ValueError(f"overlap doit être >= 0, reçu {overlap}")
    if overlap >= chunk_size:
        raise ValueError(
            f"overlap ({overlap}) doit être strictement inférieur à "
            f"chunk_size ({chunk_size})"
        )

    if not text:
        return []

    step = chunk_size - overlap
    chunks: list[str] = []

    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start += step

    return chunks
