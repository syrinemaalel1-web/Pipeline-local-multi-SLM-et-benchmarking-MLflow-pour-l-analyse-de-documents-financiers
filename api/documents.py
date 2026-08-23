"""Statut de chaque document du corpus (GET /documents) — lecture seule.

Recompute à chaque appel depuis les fichiers réellement présents sur disque
(comme `reporting.pdf` régénère son Markdown à chaque conversion) : pas d'état
à synchroniser, pas de risque d'afficher un statut périmé.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from config import DATA_PROCESSED_DIR, DATA_RAW_DIR, RESULTS_DIR, REPORTS_DIR
from extraction.docling_loader import SUPPORTED_SUFFIXES
from extraction.ingest import _document_id
from reporting.report import analysed_documents


def _raw_files() -> list[Path]:
    return sorted(
        p
        for p in DATA_RAW_DIR.rglob("*")
        if p.is_file()
        and p.suffix.lower() in SUPPORTED_SUFFIXES
        and DATA_PROCESSED_DIR not in p.parents
    )


def _evaluated_document_ids(eval_dir: Path) -> set[str]:
    ids: set[str] = set()
    if not eval_dir.exists():
        return ids
    for path in eval_dir.rglob("*.json"):
        try:
            data = json.loads(path.read_text("utf-8"))
        except json.JSONDecodeError:
            continue
        for row in data.get("rows", []):
            doc_id = row.get("document_id")
            if doc_id:
                ids.add(doc_id)
    return ids


def list_with_status() -> list[dict[str, Any]]:
    processed_ids = {p.stem for p in DATA_PROCESSED_DIR.glob("*.json")}
    generated_ids = set(analysed_documents())
    evaluated_judge = _evaluated_document_ids(RESULTS_DIR / "evaluation")
    evaluated_metrics = _evaluated_document_ids(RESULTS_DIR / "evaluation_metrics")

    out: list[dict[str, Any]] = []
    for path in _raw_files():
        doc_id = _document_id(path)
        out.append(
            {
                "document_id": doc_id,
                "source_file": path.name,
                "ingested": doc_id in processed_ids,
                "generated": doc_id in generated_ids,
                "evaluated_judge": doc_id in evaluated_judge,
                "evaluated_metrics": doc_id in evaluated_metrics,
                # `GET /reports/{id}` construit le rapport à la volée depuis
                # results/ (voir api/main.py) — il n'écrit jamais de .md sur
                # disque, contrairement à `reporting.report` en CLI. Se fier
                # uniquement à l'existence du fichier masquait donc le bouton
                # "voir" pour tout document évalué seulement via le site.
                # Un rapport est disponible dès que le document a été évalué,
                # que ce soit ici (evaluated_*) ou via la CLI (fichier présent).
                "report_judge": doc_id in evaluated_judge
                or (REPORTS_DIR / f"{doc_id}_rapport.md").exists(),
                "report_metrics": doc_id in evaluated_metrics
                or (REPORTS_DIR / f"{doc_id}_rapport_metrics.md").exists(),
            }
        )
    return out
