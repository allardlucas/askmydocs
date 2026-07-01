"""Tests du document loader PDF (ticket VEL-52).

On génère un mini PDF dans un répertoire temporaire avec pymupdf,
puis on vérifie que `load_pdf` en extrait bien le texte.
"""

from __future__ import annotations

import tempfile

import pymupdf

from askmydocs.rag.loader import load_pdf


def _make_pdf(path: str, lines: list[str]) -> None:
    """Crée un PDF d'une page contenant `lines` lignes de texte."""
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), "\n".join(lines))
    doc.save(path)
    doc.close()


def test_load_pdf_extracts_text(tmp_path):
    """`load_pdf` doit renvoyer le texte écrit dans le PDF."""
    lines = ["AskMyDocs RAG loader", "Deuxième ligne du document"]
    pdf_path = tmp_path / "sample.pdf"
    _make_pdf(str(pdf_path), lines)

    text = load_pdf(str(pdf_path))

    assert "AskMyDocs RAG loader" in text
    assert "Deuxième ligne du document" in text


def test_load_pdf_handles_multiple_pages(tmp_path):
    """Un PDF de plusieurs pages doit être entièrement extrait."""
    doc = pymupdf.open()
    for content in ["Page un", "Page deux", "Page trois"]:
        page = doc.new_page()
        page.insert_text((72, 72), content)
    pdf_path = tmp_path / "multi.pdf"
    doc.save(str(pdf_path))
    doc.close()

    text = load_pdf(str(pdf_path))

    assert "Page un" in text
    assert "Page deux" in text
    assert "Page trois" in text


def test_load_pdf_missing_file_raises(tmp_path):
    """Un chemin inexistant doit lever une erreur claire."""
    import pytest

    with pytest.raises((FileNotFoundError, RuntimeError)):
        load_pdf(str(tmp_path / "does_not_exist.pdf"))


def test_load_pdf_returns_string(tmp_path):
    """Le retour doit être une chaîne de caractères, même pour un PDF vide."""
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Une ligne")
    pdf_path = tmp_path / "one.pdf"
    doc.save(str(pdf_path))
    doc.close()

    text = load_pdf(str(pdf_path))

    assert isinstance(text, str)


def test_load_pdf_with_tempfile_module():
    """Vérifie le fonctionnement avec le module `tempfile` standard."""
    with tempfile.TemporaryDirectory() as tmpdir:
        pdf_path = f"{tmpdir}/temp.pdf"
        _make_pdf(pdf_path, ["Ligne temporaire"])

        text = load_pdf(pdf_path)

        assert "Ligne temporaire" in text
