"""API FastAPI de pilotage du pipeline benchmark — n'appelle que du code existant,
ne le modifie pas (extraction/, agents/, orchestration/, evaluation/, reporting/).

    uvicorn api.main:app --reload --port 8000

3 étapes séquentielles, chacune lancée en sous-processus (voir api/jobs.py) :
ingestion (Docling) -> génération (agents) -> évaluation (judge|metrics).
Pas de WebSocket : le suivi "ligne par ligne" se fait par polling sur
GET /jobs/{id}/logs?since=N (voir api/jobs.py pour le choix).
"""

from __future__ import annotations

from typing import Literal

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from config import DATA_RAW_DIR, MODELS_BY_TASK, Task
from extraction.docling_loader import SUPPORTED_SUFFIXES
from extraction.ingest import _document_id

from . import documents as documents_module
from . import jobs

app = FastAPI(title="Finance SLM Benchmark API")

# Ports par défaut de Vite (5173) et Create React App (3000) — à ajuster une fois
# le frontend choisi.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------- #
# Documents
# --------------------------------------------------------------------------- #


@app.get("/documents")
def list_documents():
    return documents_module.list_with_status()


@app.post("/documents/upload")
async def upload_document(file: UploadFile):
    """Dépose un PDF/DOCX dans data/ (le dossier lu par extraction.ingest) et lance
    l'ingestion immédiatement — l'utilisateur envoie un fichier, il n'a pas besoin
    d'un second geste pour déclencher son traitement."""
    suffix = "." + file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if suffix not in SUPPORTED_SUFFIXES:
        raise HTTPException(
            400,
            f"Format {suffix or 'inconnu'} non supporté "
            f"(formats acceptés : {', '.join(sorted(SUPPORTED_SUFFIXES))}).",
        )

    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    dest = DATA_RAW_DIR / file.filename
    content = await file.read()
    dest.write_bytes(content)

    document_id = _document_id(dest)
    job = jobs.start(["extraction.ingest"])
    return {"document_id": document_id, "filename": file.filename, "job_id": job.id}


# --------------------------------------------------------------------------- #
# Jobs — ingestion / génération / évaluation
# --------------------------------------------------------------------------- #


class GenerateRequest(BaseModel):
    document_id: str


class EvaluateRequest(BaseModel):
    document_id: str
    eval_mode: Literal["judge", "metrics"] = "judge"


@app.post("/ingest")
def ingest():
    """Lance `python -m extraction.ingest` (tout le corpus brut, pas par document —
    c'est déjà comme ça côté CLI, voir extraction/ingest.py)."""
    job = jobs.start(["extraction.ingest"])
    return {"job_id": job.id}


@app.post("/generate")
def generate(req: GenerateRequest):
    job = jobs.start(["orchestration.run_agents", "--doc", req.document_id])
    return {"job_id": job.id}


@app.post("/evaluate")
def evaluate(req: EvaluateRequest):
    job = jobs.start(
        [
            "orchestration.run_eval",
            "--doc",
            req.document_id,
            "--eval-mode",
            req.eval_mode,
        ]
    )
    return {"job_id": job.id}


@app.get("/jobs/{job_id}")
def job_status(job_id: str):
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "Job inconnu.")
    return {
        "job_id": job.id,
        "command": job.command,
        "status": job.status,
        "started_at": job.started_at,
        "progress": jobs.progress(job_id),
    }


@app.get("/jobs/{job_id}/logs")
def job_logs(job_id: str, since: int = 0):
    """`since` = nombre de lignes déjà reçues côté client. Le frontend rappelle
    avec `next_since` à chaque poll pour ne recevoir que les nouvelles lignes."""
    lines = jobs.read_logs(job_id, since)
    if lines is None:
        raise HTTPException(404, "Job inconnu.")
    return {"lines": lines, "next_since": since + len(lines)}


# --------------------------------------------------------------------------- #
# Rapports / dashboard — toujours recalculés depuis results/, jamais mis en
# cache : même principe que le fix de reporting/pdf.py (ne jamais afficher un
# résultat périmé).
# --------------------------------------------------------------------------- #


@app.get("/reports/{document_id}")
def report(document_id: str, eval_mode: Literal["judge", "metrics"] = "judge"):
    from reporting.report import build_metrics_report, build_report

    build = build_metrics_report if eval_mode == "metrics" else build_report
    try:
        markdown = build(document_id=document_id)
    except Exception as exc:  # noqa: BLE001 - remonté tel quel au frontend
        raise HTTPException(500, str(exc)) from exc
    return {"document_id": document_id, "eval_mode": eval_mode, "markdown": markdown}


@app.get("/dashboard")
def dashboard(eval_mode: Literal["judge", "metrics"] = "judge"):
    if eval_mode == "metrics":
        return _metrics_dashboard()
    return _judge_dashboard()


def _judge_dashboard() -> dict[str, list[dict]]:
    from reporting.report import task_table

    return {task.value: task_table(task, document_id=None) for task in Task}


def _metrics_dashboard() -> dict[str, list[dict]]:
    """Équivalent du dashboard judge pour le pipeline metrics : pas de score
    composite (pipeline non calibré, voir evaluation/legacy_metrics.py), donc
    juste les métriques brutes par modèle, réutilisées telles quelles."""
    from evaluation.legacy_metrics import METRIC_NAMES
    from reporting.report import _eval_rows, _load_metrics_evaluation, _metric_from_rows

    result: dict[str, list[dict]] = {}
    for task in Task:
        names = METRIC_NAMES.get(task, ())
        if not names:
            continue
        rows = []
        for model in MODELS_BY_TASK[task]:
            evaluation = _load_metrics_evaluation(task, model)
            eval_rows = _eval_rows(evaluation, None)
            if not eval_rows:
                continue
            rows.append(
                {
                    "model": model,
                    **{name: _metric_from_rows(eval_rows, name) for name in names},
                }
            )
        result[task.value] = rows
    return result
