"""Document loader PDF (ticket VEL-52).

Extrait le texte brut d'un fichier PDF à l'aide de `pymupdf`. Ce module
constitue la première brique du pipeline RAG d'AskMyDocs : avant de
découper et d'embedder, il faut obtenir du texte exploitable.

Exemple d'utilisation :
    >>> from askmydocs.rag.loader import load_pdf
    >>> text = load_pdf("docs/manuel.pdf")
    >>> print(text[:80])
"""

from __future__ import annotations

import pymupdf

__all__ = ["load_pdf"]


def load_pdf(path: str) -> str:
    """Extrait et concatène le texte de toutes les pages d'un PDF.

    Parameters
    ----------
    path:
        Chemin vers le fichier PDF à lire.

    Returns
    -------
    str
        Texte extrait, une page par ligne de séparation (pages jointes
        par un saut de ligne).

    Raises
    ------
    FileNotFoundError
        Si le fichier n'existe pas.
    RuntimeError
        Si `pymupdf` ne parvient pas à ouvrir le document.

    Examples
    --------
    >>> text = load_pdf("sample.pdf")  # doctest: +SKIP
    >>> isinstance(text, str)
    True
    """
    import os

    if not os.path.exists(path):
        raise FileNotFoundError(f"Fichier PDF introuvable : {path}")

    pages_text: list[str] = []
    try:
        with pymupdf.open(path) as doc:
            for page in doc:
                page_text = page.get_text()
                # `get_text()` peut renvoyer un type union selon les options ;
                # on force str pour le typage statique.
                pages_text.append(str(page_text))
    except Exception as exc:  # pragma: no cover - erreur pymupdf interne
        raise RuntimeError(f"Impossible de lire le PDF {path}: {exc}") from exc

    return "\n".join(pages_text)
