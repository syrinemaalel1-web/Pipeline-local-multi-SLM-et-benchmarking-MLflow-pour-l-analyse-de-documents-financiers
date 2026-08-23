"""Persistance des sorties d'agents, pour un benchmark reprenable.

Une campagne complète dure 12 à 20 h sur cette machine (docs/architecture-review.md
§2.1). Tout appel réussi est donc écrit sur disque immédiatement, et un run relancé
saute ce qui existe déjà. Corollaire : le rapport final se régénère depuis ces
artefacts sans relancer une seule inférence.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterator

from agents.ollama_client import AgentCall
from config import RESULTS_DIR, Task

GENERATION_DIR = RESULTS_DIR / "generation"

_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")


def slug(value: str) -> str:
    """Nom de fichier sûr à partir d'un tag de modèle (`mannix/llamax3` -> `mannix_llamax3`)."""
    return _UNSAFE.sub("_", value).strip("_")


def result_path(
    task: Task, model: str, document_id: str, item_id: str | None = None
) -> Path:
    stem = document_id if item_id is None else f"{document_id}__{item_id}"
    return GENERATION_DIR / task.value / slug(model) / f"{stem}.json"


def exists(
    task: Task, model: str, document_id: str, item_id: str | None = None
) -> bool:
    return result_path(task, model, document_id, item_id).exists()


def save(
    call: AgentCall,
    *,
    task: Task,
    model: str,
    document_id: str,
    item_id: str | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    path = result_path(task, model, document_id, item_id)
    path.parent.mkdir(parents=True, exist_ok=True)

    record = {
        "task": task.value,
        "model": model,
        "document_id": document_id,
        "item_id": item_id,
        **asdict(call),
        "tokens_per_second": call.tokens_per_second,
        **(extra or {}),
    }
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2), "utf-8")
    return path


def load_all(task: Task | None = None) -> Iterator[dict[str, Any]]:
    """Parcourt les résultats enregistrés, éventuellement filtrés par tâche."""
    root = GENERATION_DIR if task is None else GENERATION_DIR / task.value
    if not root.exists():
        return

    for path in sorted(root.rglob("*.json")):
        try:
            yield json.loads(path.read_text("utf-8"))
        except json.JSONDecodeError:
            continue
