"""Chargement du corpus ingéré et du jeu de questions Q&A."""

from __future__ import annotations

import json
import logging
from typing import Any

from config import DATA_PROCESSED_DIR, QA_QUESTIONS_PATH

log = logging.getLogger(__name__)


def load_documents(*, require_lang: bool = True) -> list[dict[str, Any]]:
    """Documents ingérés, triés par identifiant.

    Les documents dont la langue n'a pas pu être déterminée sont écartés : toute la
    chaîne (direction de traduction, langue de sortie attendue, scorer de langue) en
    dépend.
    """
    if not DATA_PROCESSED_DIR.exists():
        raise FileNotFoundError(
            f"{DATA_PROCESSED_DIR} est absent. Lancez d'abord "
            "`python -m extraction.ingest`."
        )

    documents = []
    for path in sorted(DATA_PROCESSED_DIR.glob("*.json")):
        record = json.loads(path.read_text("utf-8"))
        if require_lang and not record.get("lang"):
            log.warning(
                "%s écarté : langue indéterminée à l'ingestion.", record["document_id"]
            )
            continue
        documents.append(record)

    if not documents:
        raise RuntimeError(
            f"Aucun document exploitable dans {DATA_PROCESSED_DIR}. Déposez vos PDF/DOCX "
            "dans data/ puis lancez `python -m extraction.ingest`."
        )

    return documents


def load_qa_questions() -> list[dict[str, Any]]:
    """Questions Q&A, groupées par document.

    Format attendu, une entrée par question :

    ``{"id", "document_id", "question", "expect_abstention": true|false|null}``

    ``expect_abstention`` vaut ``true`` pour une question dont la réponse est
    volontairement absente du document. Sans au moins quelques cas de ce type, la
    règle « dis que tu ne sais pas » du spec n'est jamais mise à l'épreuve.
    """
    if not QA_QUESTIONS_PATH.exists():
        raise FileNotFoundError(
            f"{QA_QUESTIONS_PATH} est absent. Copiez "
            f"{QA_QUESTIONS_PATH.with_suffix('.example.json').name} et adaptez-le à "
            "vos documents : l'agent Q&A ne peut pas être benchmarké sans questions."
        )

    questions = json.loads(QA_QUESTIONS_PATH.read_text("utf-8"))

    known = {d["document_id"] for d in load_documents()}
    orphans = {q["document_id"] for q in questions} - known
    if orphans:
        log.warning(
            "Questions rattachées à des documents inconnus, ignorées : %s",
            sorted(orphans),
        )

    kept = [q for q in questions if q["document_id"] in known]

    if not any(q.get("expect_abstention") for q in kept):
        log.warning(
            "Aucune question marquée `expect_abstention: true`. Le scorer "
            "d'abstention restera sans effet, et la capacité des modèles à refuser "
            "d'halluciner ne sera pas mesurée."
        )

    return kept


def questions_for(document_id: str, questions: list[dict[str, Any]]):
    return [q for q in questions if q["document_id"] == document_id]
