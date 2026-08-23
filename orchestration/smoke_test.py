"""Smoke test — valide la chaîne complète en quelques minutes.

    python -m orchestration.smoke_test

À lancer avant toute campagne. Il vérifie, sur un **vrai document du corpus**, tout ce
qui peut faire échouer un run de 12 à 20 h après coup : disponibilité des modèles,
fenêtre de contexte effective, lisibilité de la mémoire GPU, un appel par agent, et
surtout la capacité du juge à rendre une sortie structurée — c'est le point le plus
fragile de la chaîne, un modèle de 8 B n'y parvient pas toujours.

Le plus petit modèle de chaque tâche est utilisé, pour que le test reste court.
"""

from __future__ import annotations

import argparse
import logging
import time
from typing import Any, Callable

import requests

from common import logging_setup
from config import (
    JUDGE_BASE_URL,
    JUDGE_MODEL,
    MODELS_BY_TASK,
    NUM_CTX,
    Task,
    assert_judge_is_isolated,
    safe_document_tokens,
)
from orchestration import corpus, mlflow_setup, store

log = logging.getLogger("smoke")

#: Modèle le plus léger de chaque liste : le smoke test valide la plomberie,
#: pas la qualité.
SMOKE_MODELS: dict[Task, str] = {
    Task.EXTRACTION: "phi4-mini:latest",
    Task.SUMMARY: "phi4-mini:latest",
    Task.TRANSLATION: "translategemma:latest",
    Task.QA: "phi4-mini:latest",
}


class Check:
    """Un contrôle nommé, dont l'échec n'interrompt pas les suivants."""

    def __init__(self) -> None:
        self.failures: list[str] = []

    def run(self, label: str, fn: Callable[[], Any], *, fatal: bool = False) -> Any:
        started = time.perf_counter()
        try:
            result = fn()
        except Exception as exc:
            elapsed = time.perf_counter() - started
            log.error("[ECHEC ] %-42s (%.1fs) %s: %s", label, elapsed, type(exc).__name__, exc)
            self.failures.append(label)
            if fatal:
                raise SystemExit(self._summary())
            return None

        log.info("[  OK  ] %-42s (%.1fs)", label, time.perf_counter() - started)
        return result

    def _summary(self) -> int:
        if self.failures:
            log.error("Smoke test en échec sur : %s", ", ".join(self.failures))
            return 1
        log.info("Smoke test complet : la chaîne est prête pour une campagne.")
        return 0


# --------------------------------------------------------------------------- #
# Contrôles
# --------------------------------------------------------------------------- #


def check_ollama_models() -> list[str]:
    """Tous les modèles configurés sont-ils réellement installés ?"""
    resp = requests.get(f"{JUDGE_BASE_URL}/api/tags", timeout=15)
    resp.raise_for_status()
    installed = {m["name"] for m in resp.json().get("models", [])}

    needed = {JUDGE_MODEL} | {m for models in MODELS_BY_TASK.values() for m in models}
    # `ollama list` suffixe implicitement `:latest`.
    resolved = {m if ":" in m else f"{m}:latest" for m in needed}

    missing = sorted(m for m in resolved if m not in installed)
    if missing:
        raise RuntimeError(f"modèles absents d'Ollama : {missing} (ollama pull ...)")
    return sorted(installed)


def check_context_length() -> str:
    """La fenêtre de contexte effective du serveur correspond-elle à la config ?

    On ne peut pas la lire directement : on charge un modèle et on relit la valeur
    que le serveur lui a réellement attribuée dans /api/ps.
    """
    from agents.ollama_client import warm_model

    model = SMOKE_MODELS[Task.EXTRACTION]
    warm_model(model)

    resp = requests.get(f"{JUDGE_BASE_URL}/api/ps", timeout=15)
    resp.raise_for_status()

    for entry in resp.json().get("models", []):
        if entry.get("name") == model:
            actual = entry.get("context_length")
            if actual is None:
                return "context_length non rapporté par /api/ps"
            if int(actual) != NUM_CTX:
                raise RuntimeError(
                    f"le serveur alloue {actual} tokens de contexte, config.NUM_CTX="
                    f"{NUM_CTX}. Redémarrez Ollama avec OLLAMA_CONTEXT_LENGTH={NUM_CTX}, "
                    "sinon les documents seront tronqués en silence."
                )
            return f"{actual} tokens"

    raise RuntimeError(f"{model} introuvable dans /api/ps après préchargement")


def check_memory_probe() -> str:
    from agents.ollama_client import read_memory

    sample = read_memory(SMOKE_MODELS[Task.EXTRACTION])
    if sample is None:
        raise RuntimeError("/api/ps ne rapporte aucun modèle chargé")
    return (
        f"{sample.total_bytes / 1e9:.1f} Go dont {sample.vram_bytes / 1e9:.1f} Go en "
        f"VRAM ({sample.gpu_fraction:.0%} sur GPU)"
    )


def check_corpus() -> dict[str, Any]:
    documents = corpus.load_documents()
    langs = sorted({d["lang"] for d in documents})

    over_budget = [
        d["document_id"]
        for d in documents
        if d["n_tokens_estimated"] > safe_document_tokens(Task.TRANSLATION)
    ]
    if over_budget:
        log.warning(
            "  %d document(s) dépassent le budget de traduction (%d tokens) : %s",
            len(over_budget),
            safe_document_tokens(Task.TRANSLATION),
            over_budget[:5],
        )

    return {
        "documents": documents,
        "summary": f"{len(documents)} document(s), langues {langs}",
    }


def check_agent(task: Task, document: dict[str, Any], question: str | None) -> str:
    from agents import run_agent

    model = SMOKE_MODELS[task]
    kwargs = {"question": question} if task is Task.QA else {}
    call = run_agent(task, document, model, **kwargs)

    if call.error:
        raise RuntimeError(call.error)
    if not call.output.strip():
        raise RuntimeError("sortie vide")

    store.save(
        call,
        task=task,
        model=model,
        document_id=f"smoke__{document['document_id']}",
        extra={"smoke": True, "source_lang": document["lang"]},
    )

    return (
        f"{model} -> {len(call.output)} car. en {call.latency_s:.1f}s "
        f"({call.tokens_per_second:.1f} tok/s)"
        if call.tokens_per_second
        else f"{model} -> {len(call.output)} car. en {call.latency_s:.1f}s"
    )


def check_judge(document: dict[str, Any], summary_output: str) -> str:
    """Le juge rend-il une note structurée exploitable ?

    Point le plus fragile de la chaîne : MLflow attend un verdict typé, et les
    modèles de 8 B échouent régulièrement à le produire. Mieux vaut le découvrir ici
    qu'après huit heures de génération.
    """
    from evaluation.judges import judges_for

    judge = judges_for(Task.SUMMARY)[0]
    feedback = judge(
        inputs={
            "task": Task.SUMMARY.value,
            "document": document["text"],
            "lang": document["lang"],
            "expected_output_lang": document["lang"],
        },
        outputs=summary_output,
    )

    if feedback.error:
        raise RuntimeError(f"le juge a échoué : {feedback.error}")
    if feedback.value is None:
        raise RuntimeError(
            f"{JUDGE_MODEL} n'a pas produit de note exploitable — il n'est pas "
            "capable de respecter le format de sortie structurée attendu par MLflow."
        )

    rationale = (feedback.rationale or "").strip().replace("\n", " ")
    return f"{JUDGE_MODEL} note {feedback.value} — {rationale[:100]}"


def check_code_scorers(document: dict[str, Any], summary_output: str) -> str:
    from evaluation.code_scorers import language_conformity, summary_length_in_range

    inputs = {
        "task": Task.SUMMARY.value,
        "document": document["text"],
        "lang": document["lang"],
        "expected_output_lang": document["lang"],
    }
    lang_fb = language_conformity(inputs=inputs, outputs=summary_output)
    len_fb = summary_length_in_range(outputs=summary_output)

    return f"langue={lang_fb.value} ({lang_fb.rationale}) | longueur={len_fb.value}"


# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--document-id",
        help="Document du corpus à utiliser. Par défaut, le plus court.",
    )
    parser.add_argument(
        "--skip-judge",
        action="store_true",
        help="Sauter l'appel au juge (le plus lent des contrôles).",
    )
    args = parser.parse_args(argv)

    logging_setup.setup()

    checks = Check()

    checks.run("juge isolé des listes de candidats", assert_judge_is_isolated, fatal=True)
    checks.run("MLflow initialisé", mlflow_setup.init, fatal=True)
    checks.run("modèles Ollama installés", check_ollama_models, fatal=True)

    corpus_info = checks.run("corpus ingéré", check_corpus, fatal=True)
    log.info("         %s", corpus_info["summary"])

    documents = corpus_info["documents"]
    if args.document_id:
        matches = [d for d in documents if d["document_id"] == args.document_id]
        if not matches:
            log.error("Document %r introuvable dans le corpus.", args.document_id)
            return 1
        document = matches[0]
    else:
        # Le plus court : le smoke test doit rester rapide.
        document = min(documents, key=lambda d: d["n_tokens_estimated"])

    log.info(
        "         document retenu : %s (%s, ~%d tokens)",
        document["document_id"],
        document["lang"],
        document["n_tokens_estimated"],
    )

    checks.run("fenêtre de contexte du serveur", check_context_length)
    checks.run("sonde mémoire /api/ps", check_memory_probe)

    question = _first_question(document["document_id"])

    outputs: dict[Task, str] = {}
    for task in Task:
        if task is Task.QA and question is None:
            log.warning("[ SKIP ] agent qa — aucune question définie pour ce document")
            continue
        result = checks.run(
            f"agent {task.value}",
            lambda t=task: check_agent(t, document, question),
        )
        if result:
            log.info("         %s", result)
            outputs[task] = _last_output(task, document)

    summary_output = outputs.get(Task.SUMMARY)
    if summary_output:
        result = checks.run(
            "scorers déterministes",
            lambda: check_code_scorers(document, summary_output),
        )
        if result:
            log.info("         %s", result)

        if not args.skip_judge:
            result = checks.run(
                f"juge {JUDGE_MODEL} (sortie structurée)",
                lambda: check_judge(document, summary_output),
            )
            if result:
                log.info("         %s", result)

    return checks._summary()


def _first_question(document_id: str) -> str | None:
    try:
        questions = corpus.questions_for(document_id, corpus.load_qa_questions())
    except FileNotFoundError as exc:
        log.warning("%s", exc)
        return None
    return questions[0]["question"] if questions else None


def _last_output(task: Task, document: dict[str, Any]) -> str:
    path = store.result_path(
        task, SMOKE_MODELS[task], f"smoke__{document['document_id']}"
    )
    import json

    return json.loads(path.read_text("utf-8"))["output"]


if __name__ == "__main__":
    raise SystemExit(main())
