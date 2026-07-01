"""Tests de smoking — le package est-il importable et versionné ?

Ces tests sont volontairement minimaux : ils garantissent que le
squelette du projet (ticket 0.1) tient debout avant d'attaquer le RAG.
"""

import askmydocs


def test_package_importable():
    """Le package `askmydocs` doit s'importer sans erreur."""
    assert askmydocs is not None


def test_version_exposee():
    """`askmydocs.__version__` doit être une chaîne non vide."""
    assert isinstance(askmydocs.__version__, str)
    assert askmydocs.__version__ != ""


def test_version_format_semver():
    """La version doit respecter le format MAJOR.MINOR.PATCH."""
    parts = askmydocs.__version__.split(".")
    assert len(parts) == 3
    assert all(part.isdigit() for part in parts)
