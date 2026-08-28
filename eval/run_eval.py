"""Harness d'évaluation d'abstention RAG.

Mesure si AskMyDocs s'abstient ou hallucine quand la réponse est
vérifiablement absente du corpus figé (`data/sample.md`).

Ce script ne crée pas de chemin ask/query/agent. S'il n'existe pas
sur le checkout courant, il écrit `eval/results/latest.json` avec
`status=not_run` (sans métriques inventées) et sort 0.

Usage :
    python -m eval.run_eval
    make eval
"""

from __future__ import annotations

import hashlib
import importlib
import inspect
import json
import statistics
import sys
import tempfile
import time
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from eval.scorer import Grade, grade_response  # noqa: E402

QUESTIONS_PATH = ROOT / "eval" / "questions.json"
RESULTS_DIR = ROOT / "eval" / "results"
LATEST_PATH = RESULTS_DIR / "latest.json"
TRANSCRIPTS_PATH = RESULTS_DIR / "transcripts.jsonl"
CORPUS_PATH = ROOT / "data" / "sample.md"

NOT_RUN_REASON = "no retrieve→reason→answer entrypoint on this checkout"

AskFn = Callable[[str], str]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_bank(path: Path = QUESTIONS_PATH) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _question_counts(bank: dict[str, Any]) -> dict[str, int]:
    questions = list(bank.get("questions") or [])
    n_a = sum(1 for item in questions if str(item.get("set", "")).upper() == "A")
    n_b = sum(1 for item in questions if str(item.get("set", "")).upper() == "B")
    return {
        "n_questions": len(questions),
        "n_set_a": n_a,
        "n_set_b": n_b,
        "repeats": int(bank.get("repeats") or 3),
    }


def _scoring_block(bank: dict[str, Any]) -> dict[str, Any]:
    scoring = dict(bank.get("scoring") or {})
    return {
        "correct": scoring.get("correct", 1),
        "wrong": scoring.get("wrong", -4),
        "abstain": scoring.get("abstain", 0),
        "reference": scoring.get(
            "reference",
            "arxiv.org/abs/2608.26385 (référence, pas une cible à copier)",
        ),
    }


def _corpus_block(bank: dict[str, Any], digest: str) -> dict[str, Any]:
    corpus = dict(bank.get("corpus") or {})
    return {
        "id": corpus.get("id", "sample-md-2026"),
        "path": corpus.get("path", "data/sample.md"),
        "sha256": digest,
    }


def _null_metrics() -> dict[str, None]:
    return {
        "a_accuracy_given_attempt": None,
        "a_abstention_rate": None,
        "b_abstention_rate": None,
        "b_canary_violation_rate": None,
        "mean_q": None,
        "p50_latency_ms": None,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def not_run_payload(bank: dict[str, Any], digest: str) -> dict[str, Any]:
    """Rapport honnête : pas d'entrypoint, donc pas de chiffres inventés."""
    return {
        "status": "not_run",
        "reason": NOT_RUN_REASON,
        "entrypoint": None,
        "corpus": _corpus_block(bank, digest),
        "bank": _question_counts(bank),
        "scoring": _scoring_block(bank),
        "metrics": _null_metrics(),
    }


# --- Découverte d'entrypoint (lecture seule, n'en crée aucun) --------------


def _try_import(module_name: str) -> Any | None:
    try:
        return importlib.import_module(module_name)
    except Exception:
        return None


def _coerce_answer(result: Any) -> str:
    if result is None:
        return ""
    if isinstance(result, str):
        return result
    if isinstance(result, dict):
        for key in ("answer", "output", "response", "text", "content"):
            value = result.get(key)
            if isinstance(value, str):
                return value
        messages = result.get("messages")
        if isinstance(messages, list) and messages:
            last = messages[-1]
            content = getattr(last, "content", None)
            if isinstance(content, str):
                return content
            if isinstance(last, dict) and isinstance(last.get("content"), str):
                return str(last["content"])
    content = getattr(result, "content", None)
    if isinstance(content, str):
        return content
    answer = getattr(result, "answer", None)
    if isinstance(answer, str):
        return answer
    return str(result)


def _call_with_question(fn: Callable[..., Any], question: str, extra: dict[str, Any]) -> str:
    sig = inspect.signature(fn)
    kwargs: dict[str, Any] = {}
    names = list(sig.parameters)
    if not names:
        result = fn()
        return _coerce_answer(result)
    first = names[0]
    if first in {"self", "cls"} and len(names) > 1:
        first = names[1]
    if first in {"question", "query", "q", "prompt", "input", "text"}:
        kwargs[first] = question
    else:
        kwargs[first] = question
    for name, value in extra.items():
        if name in sig.parameters and name not in kwargs:
            kwargs[name] = value
    result = fn(**kwargs)
    return _coerce_answer(result)


def _discover_rag_query() -> tuple[str, AskFn] | None:
    mod = _try_import("askmydocs.rag.query")
    if mod is None:
        return None
    fn = getattr(mod, "rag_query", None)
    if not callable(fn):
        return None

    def ask(question: str) -> str:
        extra: dict[str, Any] = {}
        store = _try_existing_store()
        if store is not None:
            extra["store"] = store
        return _call_with_question(fn, question, extra)

    return "askmydocs.rag.query.rag_query", ask


def _try_existing_store() -> Any | None:
    """Réutilise un store déjà exposé, sans en construire un."""
    for module_name, attr in (
        ("askmydocs.rag.store", "default_store"),
        ("askmydocs.rag.store", "STORE"),
        ("askmydocs.rag.session", "current_store"),
    ):
        mod = _try_import(module_name)
        if mod is None:
            continue
        obj = getattr(mod, attr, None)
        if obj is not None:
            return obj() if callable(obj) else obj
    return None


def _find_invokeable(mod: Any) -> Any | None:
    for name in ("graph", "app", "compiled_graph", "agent"):
        obj = getattr(mod, name, None)
        if obj is not None and callable(getattr(obj, "invoke", None)):
            return obj
    for name in ("get_graph", "compile_graph", "build_graph", "create_graph"):
        factory = getattr(mod, name, None)
        if not callable(factory):
            continue
        try:
            obj = factory()
        except TypeError:
            continue
        except Exception:
            continue
        if obj is not None and callable(getattr(obj, "invoke", None)):
            return obj
    return None


def _discover_agent_graph() -> tuple[str, AskFn] | None:
    mod = _try_import("askmydocs.agent.graph")
    if mod is None:
        return None
    compiled = _find_invokeable(mod)
    if compiled is None:
        return None

    def ask(question: str) -> str:
        payload_keys = ("question", "query", "input")
        last_error: Exception | None = None
        for key in payload_keys:
            try:
                result = compiled.invoke({key: question})
                return _coerce_answer(result)
            except Exception as exc:  # noqa: BLE001 — on tente la clé suivante
                last_error = exc
        if last_error is not None:
            raise last_error
        return ""

    return "askmydocs.agent.graph.invoke", ask


def _route_is_post_ask(route: Any) -> bool:
    path = getattr(route, "path", None) or getattr(route, "path_format", None)
    if path not in {"/ask", "/ask/", "/api/ask", "/api/ask/"}:
        return False
    methods = getattr(route, "methods", None) or set()
    methods_up = {str(item).upper() for item in methods}
    return not methods_up or "POST" in methods_up


def _discover_fastapi_ask() -> tuple[str, AskFn] | None:
    app = None
    source = ""
    for module_name in ("askmydocs.api.main", "askmydocs.api.routes", "askmydocs.api.app"):
        mod = _try_import(module_name)
        if mod is None:
            continue
        candidate = getattr(mod, "app", None)
        if candidate is None:
            continue
        app = candidate
        source = module_name
        break
    if app is None:
        return None
    routes = getattr(app, "routes", [])
    if not any(_route_is_post_ask(route) for route in routes):
        return None
    try:
        from fastapi.testclient import TestClient
    except Exception:
        return None

    def ask(question: str) -> str:
        with TestClient(app) as client:
            last: Any = None
            for payload in (
                {"question": question},
                {"query": question},
                {"q": question},
            ):
                response = client.post("/ask", json=payload)
                last = response
                if response.status_code < 400:
                    try:
                        body = response.json()
                    except ValueError:
                        return response.text
                    return _coerce_answer(body)
            if last is None:
                return ""
            return last.text

    return f"{source}:POST /ask", ask


def discover_ask_entrypoint() -> tuple[str, AskFn] | None:
    """Découvre un chemin ask existant, sans en créer un.

    Ordre : ``rag_query`` → graphe LangGraph compilé → ``POST /ask``.
    """
    for finder in (_discover_rag_query, _discover_agent_graph, _discover_fastapi_ask):
        found = finder()
        if found is not None:
            return found
    return None


# --- Exécution éventuelle contre le pipeline courant -----------------------


def _maybe_render_pdf_and_load(md_text: str) -> None:
    """Rend le markdown en PDF temporaire pour exercer `load_pdf` s'il existe.

    Le markdown commité reste la source de vérité ; on n'écrit pas de PDF
    dans `data/` (gitignore).
    """
    loader_mod = _try_import("askmydocs.rag.loader")
    if loader_mod is None or not callable(getattr(loader_mod, "load_pdf", None)):
        return
    try:
        import pymupdf
    except Exception:
        return
    load_pdf = loader_mod.load_pdf
    with tempfile.TemporaryDirectory(prefix="askmydocs-eval-") as tmp:
        pdf_path = Path(tmp) / "sample_frozen.pdf"
        doc = pymupdf.open()
        page = doc.new_page()
        page.insert_text((72, 72), md_text[:4000])
        doc.save(str(pdf_path))
        doc.close()
        load_pdf(pdf_path)
        ingest_mod = _try_import("askmydocs.rag.ingest")
        ingest = getattr(ingest_mod, "ingest_document", None) if ingest_mod else None
        if callable(ingest):
            try:
                ingest(pdf_path)
            except TypeError:
                ingest(str(pdf_path))


def _safe_ask(ask: AskFn, question: str) -> tuple[str, float, str | None]:
    started = time.perf_counter()
    try:
        answer = ask(question)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return answer, elapsed_ms, None
    except Exception as exc:  # noqa: BLE001 — une erreur d'appel n'est pas une métrique
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        detail = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
        return f"[eval_error] {detail}", elapsed_ms, detail


def _ratio(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return statistics.fmean(values)


def _p50(values: list[float]) -> float | None:
    if not values:
        return None
    return float(statistics.median(values))


def run_against_pipeline(
    bank: dict[str, Any],
    digest: str,
    entrypoint_name: str,
    ask: AskFn,
) -> dict[str, Any]:
    md_text = CORPUS_PATH.read_text(encoding="utf-8")
    _maybe_render_pdf_and_load(md_text)

    questions = list(bank.get("questions") or [])
    repeats = int(bank.get("repeats") or 3)
    transcripts: list[dict[str, Any]] = []
    a_correct = a_wrong = a_abstain = 0
    b_abstain = b_violation = 0
    q_scores: list[float] = []
    latencies: list[float] = []

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with TRANSCRIPTS_PATH.open("w", encoding="utf-8") as transcript_file:
        for question in questions:
            qid = question.get("id")
            qset = str(question.get("set", "")).upper()
            prompt = str(question.get("question", ""))
            for repeat in range(1, repeats + 1):
                answer, latency_ms, error = _safe_ask(ask, prompt)
                grade: Grade = grade_response(answer, question)
                q_scores.append(float(grade.score))
                latencies.append(latency_ms)
                if qset == "A":
                    if grade.label == "correct":
                        a_correct += 1
                    elif grade.label == "abstain":
                        a_abstain += 1
                    else:
                        a_wrong += 1
                else:
                    if grade.label == "abstain":
                        b_abstain += 1
                    else:
                        b_violation += 1
                row = {
                    "id": qid,
                    "set": qset,
                    "repeat": repeat,
                    "question": prompt,
                    "response": answer,
                    "label": grade.label,
                    "score": grade.score,
                    "latency_ms": round(latency_ms, 3),
                    "error": error,
                }
                transcripts.append(row)
                transcript_file.write(json.dumps(row, ensure_ascii=False) + "\n")

    a_attempts = a_correct + a_wrong
    a_total = a_correct + a_wrong + a_abstain
    b_total = b_abstain + b_violation
    metrics = {
        "a_accuracy_given_attempt": _ratio(a_correct, a_attempts),
        "a_abstention_rate": _ratio(a_abstain, a_total),
        "b_abstention_rate": _ratio(b_abstain, b_total),
        "b_canary_violation_rate": _ratio(b_violation, b_total),
        "mean_q": _mean(q_scores),
        "p50_latency_ms": _p50(latencies),
    }
    counts = {
        "a_correct": a_correct,
        "a_wrong": a_wrong,
        "a_abstain": a_abstain,
        "b_abstain": b_abstain,
        "b_canary_violation": b_violation,
        "n_graded": len(q_scores),
    }
    return {
        "status": "completed",
        "reason": None,
        "entrypoint": entrypoint_name,
        "corpus": _corpus_block(bank, digest),
        "bank": _question_counts(bank),
        "scoring": _scoring_block(bank),
        "counts": counts,
        "metrics": metrics,
        "n_transcripts": len(transcripts),
        "transcripts_path": str(TRANSCRIPTS_PATH.relative_to(ROOT)),
    }


def main() -> int:
    if not CORPUS_PATH.is_file():
        print(f"AskMyDocs eval: corpus introuvable ({CORPUS_PATH})", file=sys.stderr)
        return 1
    bank = load_bank()
    digest = sha256_file(CORPUS_PATH)
    discovered = discover_ask_entrypoint()
    if discovered is None:
        payload = not_run_payload(bank, digest)
        write_json(LATEST_PATH, payload)
        print(
            "AskMyDocs eval: status=not_run — no retrieve→reason→answer entrypoint on this checkout"
        )
        return 0

    entrypoint_name, ask = discovered
    payload = run_against_pipeline(bank, digest, entrypoint_name, ask)
    write_json(LATEST_PATH, payload)
    metrics = payload["metrics"]
    print(
        "AskMyDocs eval: status=completed "
        f"entrypoint={entrypoint_name} "
        f"mean_q={metrics['mean_q']} "
        f"B_canary_violation_rate={metrics['b_canary_violation_rate']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
