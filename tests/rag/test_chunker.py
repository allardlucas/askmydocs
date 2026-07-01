"""Tests du text chunker (ticket VEL-53).

Vérifie la taille des chunks, l'overlap entre chunks consécutifs,
et les cas limites (texte vide, chunk_size trop petit, fin du texte).
"""

from __future__ import annotations

import pytest

from askmydocs.rag.chunker import chunk_text

# --- Comportement nominal -------------------------------------------------

def test_chunk_size_respected():
    """Aucun chunk ne doit dépasser `chunk_size` caractères."""
    text = "a" * 1200
    chunks = chunk_text(text, chunk_size=500, overlap=50)

    assert len(chunks) >= 1
    for chunk in chunks:
        assert len(chunk) <= 500


def test_overlap_between_chunks():
    """L'overlap entre deux chunks consécutifs doit être exact."""
    # texte prévisible : abcdefghij (10 caractères)
    text = "abcdefghij"
    chunks = chunk_text(text, chunk_size=4, overlap=2)

    assert chunks == ["abcd", "cdef", "efgh", "ghij"]
    # vérifie explicitement l'overlap
    for i in range(len(chunks) - 1):
        assert chunks[i][-2:] == chunks[i + 1][:2]


def test_last_chunk_reaches_end_of_text():
    """Le dernier chunk doit contenir la fin du texte."""
    text = "0123456789" * 100  # 1000 caractères
    chunks = chunk_text(text, chunk_size=300, overlap=50)

    assert chunks[-1].endswith("9")
    assert len(chunks[-1]) <= 300


def test_text_shorter_than_chunk_returns_single_chunk():
    """Un texte plus court que `chunk_size` renvoie un seul chunk."""
    text = "court"
    chunks = chunk_text(text, chunk_size=500, overlap=50)

    assert chunks == ["court"]


def test_default_parameters():
    """Sans arguments optionnels, les valeurs par défaut s'appliquent."""
    text = "x" * 600
    chunks = chunk_text(text)

    assert len(chunks) >= 2
    for chunk in chunks:
        assert len(chunk) <= 500


# --- Cas limites ----------------------------------------------------------

def test_empty_text_returns_empty_list():
    """Un texte vide doit renvoyer une liste vide."""
    assert chunk_text("") == []


def test_chunk_size_too_small_raises():
    """Un `chunk_size` <= 0 doit lever une ValueError."""
    with pytest.raises(ValueError):
        chunk_text("du texte", chunk_size=0, overlap=0)


def test_negative_chunk_size_raises():
    """Un `chunk_size` négatif doit lever une ValueError."""
    with pytest.raises(ValueError):
        chunk_text("du texte", chunk_size=-10, overlap=0)


def test_overlap_greater_or_equal_to_chunk_size_raises():
    """Un `overlap` >= `chunk_size` doit lever une ValueError."""
    with pytest.raises(ValueError):
        chunk_text("du texte", chunk_size=100, overlap=100)

    with pytest.raises(ValueError):
        chunk_text("du texte", chunk_size=100, overlap=150)


def test_negative_overlap_raises():
    """Un `overlap` négatif doit lever une ValueError."""
    with pytest.raises(ValueError):
        chunk_text("du texte", chunk_size=100, overlap=-10)


def test_zero_overlap_works():
    """Un overlap de 0 est valide : les chunks se suivent sans chevauchement."""
    text = "abcdefghij"
    chunks = chunk_text(text, chunk_size=5, overlap=0)

    assert chunks == ["abcde", "fghij"]


def test_no_chunk_is_empty():
    """Aucun chunk renvoyé ne doit être une chaîne vide."""
    text = "a" * 550
    chunks = chunk_text(text, chunk_size=500, overlap=50)

    assert all(len(c) > 0 for c in chunks)
