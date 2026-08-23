"""Gestion des jobs d'arrière-plan (ingestion, génération, évaluation).

Chaque job lance la commande CLI existante (`python -m ...`) comme un
**sous-processus séparé** — pas d'import direct des fonctions `main()`. Deux
raisons : (1) isolation réelle, une erreur dans un run ne peut pas faire tomber
l'API ; (2) la sortie standard du sous-processus, déjà bien formée par le
pipeline existant (`common/logging_setup.py`), s'écrit directement dans un
fichier — consultable ligne par ligne par polling, sans rien changer au
pipeline ni dupliquer sa logique de logging.

Registre de jobs en mémoire : suffisant pour une v1 (voir CLAUDE.md/le prompt de
conception) — un redémarrage de l'API perd le suivi des jobs en cours, mais pas
les logs déjà écrits sur disque, ni les résultats déjà persistés par le pipeline
lui-même (qui reste la source de vérité).
"""

from __future__ import annotations

import re
import subprocess
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from config import ROOT

JOBS_LOG_DIR = ROOT / "api" / "job_logs"
JOBS_LOG_DIR.mkdir(parents=True, exist_ok=True)

Status = Literal["running", "done", "error"]


@dataclass
class Job:
    id: str
    command: list[str]
    process: subprocess.Popen
    log_path: Path
    started_at: str

    @property
    def status(self) -> Status:
        code = self.process.poll()
        if code is None:
            return "running"
        return "done" if code == 0 else "error"

    @property
    def return_code(self) -> int | None:
        return self.process.poll()


_JOBS: dict[str, Job] = {}


def start(command: list[str]) -> Job:
    """Lance `python -m <command[0]> <command[1:]>` en sous-processus.

    `command[0]` est un nom de module (`"extraction.ingest"`,
    `"orchestration.run_agents"`, ...), le reste des arguments CLI tels quels.
    """
    job_id = uuid.uuid4().hex[:12]
    log_path = JOBS_LOG_DIR / f"{job_id}.log"

    log_file = log_path.open("w", encoding="utf-8")
    try:
        process = subprocess.Popen(
            [sys.executable, "-u", "-m", *command],
            cwd=ROOT,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
    finally:
        # Le sous-processus hérite du descripteur de fichier à la création ;
        # le refermer ici ne l'affecte pas et évite une fuite de handle côté API.
        log_file.close()

    job = Job(
        id=job_id,
        command=command,
        process=process,
        log_path=log_path,
        started_at=datetime.now(timezone.utc).isoformat(),
    )
    _JOBS[job_id] = job
    return job


def get(job_id: str) -> Job | None:
    return _JOBS.get(job_id)


_TOTAL_RE = re.compile(r"PROGRESS_TOTAL=(\d+)")


def read_logs(job_id: str, since: int = 0) -> list[str] | None:
    """Lignes de log depuis la ligne `since` (exclue) — `None` si le job n'existe pas."""
    job = get(job_id)
    if job is None:
        return None
    if not job.log_path.exists():
        return []
    # Les marqueurs PROGRESS_TOTAL/PROGRESS_STEP (voir progress() ci-dessous) sont
    # à usage interne — retirés du journal affiché à l'utilisateur, qui ne les
    # comprendrait pas.
    lines = [
        line
        for line in job.log_path.read_text("utf-8", errors="replace").splitlines()
        if "PROGRESS_STEP" not in line and not _TOTAL_RE.search(line)
    ]
    return lines[since:]


def progress(job_id: str) -> dict[str, int] | None:
    """Avancement réel (pas une barre indéterminée) : les scripts CLI
    (`extraction.ingest`, `orchestration.run_agents`, `orchestration.run_eval`)
    annoncent leur nombre total d'unités de travail une seule fois au démarrage
    (``PROGRESS_TOTAL=N``) puis une ligne ``PROGRESS_STEP`` par unité traitée
    (générée ou déjà présente — les deux comptent, l'important est l'avancement
    dans la liste de travail, pas seulement l'inférence réelle). On relit le
    fichier de log en entier à chaque appel plutôt que de maintenir un compteur
    en mémoire : cohérent avec le reste de l'API, qui ne met jamais rien en
    cache (voir api/documents.py).

    `None` si le job n'existe pas ou si aucun marqueur n'a encore été écrit
    (tout début du job, ou script qui n'en émet pas).
    """
    job = get(job_id)
    if job is None or not job.log_path.exists():
        return None
    text = job.log_path.read_text("utf-8", errors="replace")
    match = _TOTAL_RE.search(text)
    if not match:
        return None
    return {"completed": text.count("PROGRESS_STEP"), "total": int(match.group(1))}
