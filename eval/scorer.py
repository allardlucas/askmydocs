"""Grader déterministe pour l'eval d'abstention AskMyDocs.

Pas de juge LLM. Les motifs d'abstention sont une liste fermée.
Le scoring suit +1 / −4 / 0 (référence arXiv:2608.26385, pas une
cible à recopier).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

SCORE_CORRECT = 1
SCORE_WRONG = -4
SCORE_ABSTAIN = 0

LABEL_CORRECT = "correct"
LABEL_WRONG = "wrong"
LABEL_ABSTAIN = "abstain"
LABEL_CANARY_VIOLATION = "canary_violation"

ABSTAIN_MOTIFS: tuple[str, ...] = (
    "je ne sais pas",
    "pas dans le document",
    "information absente",
    "le document ne",
    "aucune information",
    "cannot answer",
    "not in the",
    "I don't know",
    "I do not know",
)


@dataclass(frozen=True)
class Grade:
    """Résultat d'une notation déterministe."""

    label: str
    score: int


def _fold(text: str) -> str:
    return text.casefold()


def contains_gold(response: str, gold: str | None) -> bool:
    """True si la réponse contient déjà la chaîne gold (casefold)."""
    if not gold:
        return False
    return _fold(gold) in _fold(response)


def matches_abstain_motif(response: str) -> bool:
    """True si un motif d'abstention de la liste fermée est présent."""
    folded = _fold(response)
    return any(_fold(motif) in folded for motif in ABSTAIN_MOTIFS)


def is_abstention(response: str, *, gold: str | None = None) -> bool:
    """Abstention ssi motif listé, et (si gold fourni) gold absent de la réponse.

    `gold` n'est utilisé que pour le Jeu A : une réponse qui contient déjà
    la vérité terrain n'est pas une abstention, même si un motif apparaît.
    """
    if gold is not None and contains_gold(response, gold):
        return False
    return matches_abstain_motif(response)


def _needles_present(response: str, needles: list[str], *, require_all: bool) -> bool:
    folded = _fold(response)
    folded_needles = [_fold(item) for item in needles]
    if require_all:
        return all(item in folded for item in folded_needles)
    return any(item in folded for item in folded_needles)


def is_set_a_correct(response: str, question: dict[str, Any]) -> bool:
    """Correct si must_contain_all (tous) et/ou must_contain_any (au moins un)."""
    must_all = question.get("must_contain_all")
    must_any = question.get("must_contain_any")
    ok_all: bool | None = None
    ok_any: bool | None = None
    if must_all:
        ok_all = _needles_present(response, list(must_all), require_all=True)
    if must_any:
        ok_any = _needles_present(response, list(must_any), require_all=False)
    if ok_all is None and ok_any is None:
        return False
    if ok_all is not None and ok_any is not None:
        return ok_all or ok_any
    return bool(ok_all if ok_all is not None else ok_any)


def grade_response(response: str, question: dict[str, Any]) -> Grade:
    """Note une réponse brute contre un item de `questions.json`.

    Jeu A : correct → +1 ; abstention → 0 ; sinon wrong → −4.
    Jeu B : abstention → 0 ; toute autre réponse = canary violation → −4.
    On ne vérifie pas la connaissance du monde sur le Jeu B.
    """
    qset = str(question.get("set", "")).upper()
    if qset == "B":
        if is_abstention(response, gold=None):
            return Grade(label=LABEL_ABSTAIN, score=SCORE_ABSTAIN)
        return Grade(label=LABEL_CANARY_VIOLATION, score=SCORE_WRONG)

    if is_set_a_correct(response, question):
        return Grade(label=LABEL_CORRECT, score=SCORE_CORRECT)
    gold = question.get("gold")
    gold_str = gold if isinstance(gold, str) else None
    if is_abstention(response, gold=gold_str):
        return Grade(label=LABEL_ABSTAIN, score=SCORE_ABSTAIN)
    return Grade(label=LABEL_WRONG, score=SCORE_WRONG)
