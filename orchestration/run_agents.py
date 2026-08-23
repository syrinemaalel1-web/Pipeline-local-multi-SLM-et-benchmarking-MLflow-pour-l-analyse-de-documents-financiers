"""Phase 1 — génération : exécute les agents sur tout le corpus.

    python -m orchestration.run_agents                 # tout
    python -m orchestration.run_agents --task summary  # une tâche
    python -m orchestration.run_agents --limit-docs 1  # échantillon
    python -m orchestration.run_agents --doc proposal_north  # un document précis

La boucle est **modèle-majeure** : pour chaque modèle, on précharge une fois puis on
enchaîne tous les documents. Une boucle document-majeure rechargerait 5 Go depuis le
disque à chaque appel, sur une machine dont la VRAM ne tient qu'un modèle partiel.

La génération est séparée de l'évaluation pour que les 12 à 20 h d'inférence ne soient
jamais rejouées quand on ajuste un scorer.
"""

from __future__ import annotations

import argparse
import logging
import time
from typing import Any

import mlflow

from agents import run_agent
from common import logging_setup
from config import MODELS_BY_TASK, Task
from orchestration import corpus, mlflow_setup, store

log = logging.getLogger("run_agents")


def _tags(
    task: Task, model: str, document: dict[str, Any], item_id: str | None
) -> dict[str, str]:
    tags = {
        "task": task.value,
        "model_name": model,
        "document_id": document["document_id"],
        "source_lang": document["lang"],
    }
    if item_id:
        tags["item_id"] = item_id
    return tags


def _work_items(
    task: Task, document: dict[str, Any], questions: list[dict[str, Any]]
) -> list[tuple[str | None, dict[str, Any]]]:
    """Unités de travail d'un document : une seule, sauf en Q&A où il y en a une par question."""
    if task is not Task.QA:
        return [(None, {})]
    return [
        (q["id"], {"question": q["question"]})
        for q in corpus.questions_for(document["document_id"], questions)
    ]


def _select_documents(
    documents: list[dict[str, Any]], wanted: list[str]
) -> list[dict[str, Any]]:
    """Sous-ensemble du corpus désigné par identifiant, ou par préfixe d'identifiant.

    Les identifiants se terminent par un hash : accepter un préfixe évite de le
    recopier. Un préfixe ambigu lève une erreur plutôt que de trancher tout seul, un
    run pouvant durer des heures sur le mauvais document.
    """
    selected: list[dict[str, Any]] = []

    for prefix in wanted:
        matches = [d for d in documents if d["document_id"].startswith(prefix)]
        if not matches:
            available = "\n  ".join(d["document_id"] for d in documents)
            raise SystemExit(
                f"Aucun document ne correspond à '{prefix}'. Disponibles :\n  {available}"
            )
        if len(matches) > 1:
            ambiguous = ", ".join(d["document_id"] for d in matches)
            raise SystemExit(f"'{prefix}' est ambigu : {ambiguous}")
        selected.append(matches[0])

    return selected


def _has_work(
    task: Task,
    model: str,
    documents: list[dict[str, Any]],
    questions: list[dict[str, Any]],
    force: bool,
) -> bool:
    """Reste-t-il quelque chose à générer ? Évite de charger 5 Go pour rien."""
    if force:
        return True
    return any(
        not store.exists(task, model, document["document_id"], item_id)
        for document in documents
        for item_id, _ in _work_items(task, document, questions)
    )


def _warm(model: str) -> bool:
    from agents.ollama_client import warm_model

    log.info("Préchargement de %s...", model)
    started = time.perf_counter()
    if not warm_model(model):
        log.error("%s indisponible, ignoré pour cette tâche.", model)
        return False
    log.info("  chargé en %.1fs", time.perf_counter() - started)
    return True


def run_task(
    task: Task,
    *,
    models: list[str] | None = None,
    doc_ids: list[str] | None = None,
    limit_docs: int | None = None,
    force: bool = False,
) -> None:
    documents = corpus.load_documents()
    if doc_ids:
        documents = _select_documents(documents, doc_ids)
    if limit_docs:
        documents = documents[:limit_docs]

    questions = corpus.load_qa_questions() if task is Task.QA else []
    models = models or MODELS_BY_TASK[task]

    if task is Task.QA:
        # data/qa_questions.json est un fichier curé à la main (voir
        # corpus.load_qa_questions), pas régénéré par le pipeline — un document
        # tout juste ingéré (ex. via le site) n'y a par construction aucune
        # entrée tant que personne ne l'a complété. Sans cet avertissement,
        # "déjà complet, rien à générer" en aval (_has_work, vacuously true sur
        # une liste de travail vide) est indiscernable d'un vrai skip de reprise.
        missing = [
            d["document_id"]
            for d in documents
            if not corpus.questions_for(d["document_id"], questions)
        ]
        if missing:
            log.warning(
                "qa : aucune question définie dans data/qa_questions.json pour %s — "
                "cette tâche restera vide pour ce(s) document(s) tant qu'elle n'aura "
                "pas été complétée à la main.",
                ", ".join(missing),
            )

    for model in models:
        log.info("=== %s / %s ===", task.value, model)
        # Marqueur consommé par api/jobs.py::progress (barre de progression
        # réelle côté site) — un par (tâche, modèle), que le travail soit
        # réellement effectué ou déjà couvert. Le total correspondant est
        # annoncé une fois, dans main(), avant la boucle sur les tâches.
        log.info("PROGRESS_STEP")

        if not _has_work(task, model, documents, questions, force):
            log.info("  déjà complet, rien à générer.")
            continue

        # Le chargement du modèle doit rester hors des latences mesurées, sinon le
        # premier document de chaque modèle porte seul le coût des 5 Go lus sur disque.
        if not _warm(model):
            continue

        for document in documents:
            for item_id, kwargs in _work_items(task, document, questions):
                if not force and store.exists(
                    task, model, document["document_id"], item_id
                ):
                    continue

                with mlflow.start_run(
                    run_name=f"{task.value}:{model}:{document['document_id']}"
                ):
                    mlflow.set_tags(_tags(task, model, document, item_id))
                    call = run_agent(task, document, model, **kwargs)
                    mlflow.log_metrics(
                        {
                            key: value
                            for key, value in {
                                "latency_s": call.latency_s,
                                "prompt_tokens": call.prompt_tokens,
                                "completion_tokens": call.completion_tokens,
                                "tokens_per_second": call.tokens_per_second,
                                "vram_bytes": (call.memory or {}).get("vram_bytes"),
                                "gpu_fraction": (call.memory or {}).get("gpu_fraction"),
                            }.items()
                            if value is not None
                        }
                    )

                store.save(
                    call,
                    task=task,
                    model=model,
                    document_id=document["document_id"],
                    item_id=item_id,
                    extra={
                        "source_lang": document["lang"],
                        "target_lang": document.get("target_lang"),
                        "question": kwargs.get("question"),
                    },
                )

                log.info(
                    "  %s%s : %s en %.1fs (%s)",
                    document["document_id"],
                    f"/{item_id}" if item_id else "",
                    "ERREUR" if call.error else "ok",
                    call.latency_s,
                    f"{call.tokens_per_second:.1f} tok/s"
                    if call.tokens_per_second
                    else "débit inconnu",
                )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=[t.value for t in Task])
    parser.add_argument("--model", action="append", dest="models")
    parser.add_argument(
        "--doc",
        action="append",
        dest="doc_ids",
        help="Identifiant de document, ou préfixe. Répétable.",
    )
    parser.add_argument("--limit-docs", type=int)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    logging_setup.setup()
    mlflow_setup.init()

    tasks = [Task(args.task)] if args.task else list(Task)
    total = sum(len(args.models or MODELS_BY_TASK[t]) for t in tasks)
    log.info("PROGRESS_TOTAL=%d", total)

    for task in tasks:
        run_task(
            task,
            models=args.models,
            doc_ids=args.doc_ids,
            limit_docs=args.limit_docs,
            force=args.force,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
