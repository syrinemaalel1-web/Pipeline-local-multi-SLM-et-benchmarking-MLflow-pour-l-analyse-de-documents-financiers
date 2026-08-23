"""Initialisation MLflow, commune à tous les scripts du benchmark."""

from __future__ import annotations

import logging
import os

from config import (
    EVAL_MAX_WORKERS,
    JUDGE_BASE_URL,
    MLFLOW_EXPERIMENT,
    MLFLOW_TRACKING_URI,
    NUM_CTX,
    REQUEST_TIMEOUT_S,
    assert_judge_is_isolated,
)

log = logging.getLogger(__name__)

_initialised = False


def _pin_eval_workers() -> None:
    """Sérialise l'évaluation MLflow.

    `mlflow.genai.evaluate` parallélise ses scorers par défaut. Avec un unique serveur
    Ollama sur 4 Go de VRAM, plusieurs requêtes concurrentes déclenchent des
    chargements/déchargements de modèles en boucle et, à l'occasion, des OOM.
    Ces variables doivent être posées avant tout appel à evaluate().
    """
    os.environ.setdefault("MLFLOW_GENAI_EVAL_MAX_WORKERS", str(EVAL_MAX_WORKERS))
    os.environ.setdefault("MLFLOW_GENAI_EVAL_MAX_SCORER_WORKERS", str(EVAL_MAX_WORKERS))


def _check_ollama_context() -> None:
    """Avertit si le serveur Ollama n'a pas la fenêtre de contexte attendue.

    L'endpoint compatible OpenAI n'expose pas `num_ctx` : la seule façon de fixer la
    fenêtre est `OLLAMA_CONTEXT_LENGTH` côté serveur. Si elle n'est pas posée, Ollama
    tronque l'entrée en silence et les scores deviennent ininterprétables.
    """
    declared = os.environ.get("OLLAMA_CONTEXT_LENGTH")

    if declared is None:
        log.warning(
            "OLLAMA_CONTEXT_LENGTH n'est pas défini dans cet environnement. Vérifiez "
            "que le serveur Ollama a bien été démarré avec OLLAMA_CONTEXT_LENGTH=%d, "
            "sinon les documents seront tronqués sans avertissement.",
            NUM_CTX,
        )
    elif declared != str(NUM_CTX):
        log.warning(
            "OLLAMA_CONTEXT_LENGTH=%s alors que config.NUM_CTX=%d. Les budgets de "
            "tokens calculés à l'ingestion ne correspondent plus au serveur.",
            declared,
            NUM_CTX,
        )


def init(*, autolog: bool = True) -> None:
    """Configure MLflow une fois par processus."""
    global _initialised
    if _initialised:
        return

    import mlflow

    assert_judge_is_isolated()
    _pin_eval_workers()
    _check_ollama_context()

    # LiteLLM, qui porte les appels du juge, lit l'adresse d'Ollama ici. C'est le
    # seul moyen propre de la lui donner : passer `base_url` à make_judge ferait
    # basculer MLflow sur son adaptateur AI Gateway.
    os.environ.setdefault("OLLAMA_API_BASE", JUDGE_BASE_URL)

    # 2026-08-12 : hypothèse ci-dessus invalidée pour cette version de MLflow — Ollama
    # est un "native gateway provider" (`mlflow.gateway.providers.ollama` existe), donc
    # `mlflow.genai.judges.invoke_judge_model` route un modèle `ollama:/...` vers
    # `GatewayAdapter` de toute façon (voir sa docstring : "GatewayAdapter: For native
    # AI Gateway providers"), que `base_url` soit passé ou non. Conséquence trouvée en
    # pratique (`judge_probe` sur northbridge/thalassa, 2026-08-12) : le timeout de
    # `JUDGE_INFERENCE_PARAMS` (litellm) ne s'applique pas à ce chemin — GatewayAdapter
    # lit son propre timeout HTTP via la variable d'environnement
    # `MLFLOW_GATEWAY_ROUTE_TIMEOUT_SECONDS` (défaut MLflow : 300s), qui a fait échouer
    # des appels `judge(inputs=..., outputs=...)` directs (retry de run_eval.py,
    # judge_probe.py) sur des documents où `granite3.3:8b` dépasse 300s. Alignée sur
    # REQUEST_TIMEOUT_S comme les deux autres timeouts déjà posés dans ce projet
    # (JUDGE_INFERENCE_PARAMS, MLFLOW_GENAI_EVAL_LLM_TIMEOUT).
    os.environ.setdefault("MLFLOW_GATEWAY_ROUTE_TIMEOUT_SECONDS", str(REQUEST_TIMEOUT_S))

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    if autolog:
        # Trace chaque appel du SDK openai, y compris vers un base_url custom :
        # c'est ce qui rend le tracing des agents gratuit en lignes de code.
        mlflow.openai.autolog()

    log.info(
        "MLflow prêt — tracking=%s experiment=%s", MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT
    )
    _initialised = True
