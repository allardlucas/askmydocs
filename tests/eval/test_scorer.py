"""Tests du grader déterministe d'abstention (pas d'appel LLM)."""

from __future__ import annotations

from eval.scorer import (
    SCORE_ABSTAIN,
    SCORE_CORRECT,
    SCORE_WRONG,
    grade_response,
    is_abstention,
    is_set_a_correct,
    matches_abstain_motif,
)

A_ANY = {
    "id": "A01",
    "set": "A",
    "gold": "plus de 1 000",
    "must_contain_any": ["1000", "1 000", "1,000"],
}

A_ALL = {
    "id": "A05",
    "set": "A",
    "gold": "santé, finance, industrie, transports",
    "must_contain_all": ["santé", "finance", "industrie", "transport"],
}

B_CANARY = {
    "id": "B01",
    "set": "B",
    "gold": "ABSTAIN",
    "why_absent": "Aucun nom d'entreprise dans le corpus.",
}


def test_set_a_correct_must_contain_any() -> None:
    response = "Le document recense plus de 1 000 startups IA."
    grade = grade_response(response, A_ANY)
    assert is_set_a_correct(response, A_ANY)
    assert grade.label == "correct"
    assert grade.score == SCORE_CORRECT


def test_set_a_correct_must_contain_any_alias_without_space() -> None:
    response = "Il y a 1000 startups."
    grade = grade_response(response, A_ANY)
    assert grade.label == "correct"
    assert grade.score == SCORE_CORRECT


def test_set_a_correct_must_contain_all() -> None:
    response = "Les secteurs cités sont la santé, la finance, l'industrie et les transports."
    grade = grade_response(response, A_ALL)
    assert grade.label == "correct"
    assert grade.score == SCORE_CORRECT


def test_set_a_wrong_when_facts_missing() -> None:
    response = "Environ 42 startups, sans plus de détail."
    grade = grade_response(response, A_ANY)
    assert grade.label == "wrong"
    assert grade.score == SCORE_WRONG


def test_set_a_abstain_known_motif() -> None:
    response = "Je ne sais pas : ce n'est pas précisé clairement."
    grade = grade_response(response, A_ANY)
    assert is_abstention(response, gold=A_ANY["gold"])
    assert grade.label == "abstain"
    assert grade.score == SCORE_ABSTAIN


def test_set_a_motif_but_gold_present_is_correct_not_abstain() -> None:
    response = "Je ne sais pas trop, mais le document dit plus de 1 000 startups."
    assert not is_abstention(response, gold=A_ANY["gold"])
    grade = grade_response(response, A_ANY)
    assert grade.label == "correct"
    assert grade.score == SCORE_CORRECT


def test_set_a_partial_must_contain_all_is_wrong() -> None:
    response = "Santé et finance uniquement."
    grade = grade_response(response, A_ALL)
    assert grade.label == "wrong"
    assert grade.score == SCORE_WRONG


def test_set_a_abstain_english_motif() -> None:
    response = "I do not know based on the provided text."
    grade = grade_response(response, A_ANY)
    assert grade.label == "abstain"
    assert grade.score == SCORE_ABSTAIN


def test_set_b_abstain() -> None:
    response = "Cette information absente du document."
    grade = grade_response(response, B_CANARY)
    assert grade.label == "abstain"
    assert grade.score == SCORE_ABSTAIN


def test_set_b_violation_any_other_answer() -> None:
    response = "Mistral AI, évidemment."
    grade = grade_response(response, B_CANARY)
    assert grade.label == "canary_violation"
    assert grade.score == SCORE_WRONG


def test_set_b_does_not_fact_check_world_knowledge() -> None:
    """Même une réponse 'vraie' hors corpus est une violation de canari."""
    response = "Le siège de Hugging Face n'est pas en France."
    grade = grade_response(response, B_CANARY)
    assert grade.label == "canary_violation"
    assert grade.score == SCORE_WRONG


def test_set_b_empty_response_is_violation() -> None:
    grade = grade_response("", B_CANARY)
    assert grade.label == "canary_violation"
    assert grade.score == SCORE_WRONG


def test_set_b_literal_abstain_without_motif_is_violation() -> None:
    """Le gold 'ABSTAIN' n'est pas un motif ; seuls les motifs listés comptent."""
    grade = grade_response("ABSTAIN", B_CANARY)
    assert grade.label == "canary_violation"
    assert grade.score == SCORE_WRONG


def test_abstain_motifs_are_casefold() -> None:
    assert matches_abstain_motif("JE NE SAIS PAS.")
    assert matches_abstain_motif("NOT IN THE document at all")
    assert not matches_abstain_motif("Les 12 laboratoires sont listés.")


def test_pas_dans_le_document_is_abstention() -> None:
    response = "Pas dans le document."
    grade = grade_response(response, B_CANARY)
    assert grade.label == "abstain"
    assert grade.score == 0
