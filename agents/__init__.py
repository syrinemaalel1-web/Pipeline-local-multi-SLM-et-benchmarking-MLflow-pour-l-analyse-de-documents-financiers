"""Les quatre agents, indépendants et stateless.

Chacun reçoit le document ingéré et un nom de modèle Ollama, et renvoie un
:class:`~agents.ollama_client.AgentCall`. Aucune mémoire partagée, aucun multi-tour.
"""

from __future__ import annotations

from typing import Any

from agents import extraction, qa, summarize, translate
from agents.ollama_client import AgentCall
from config import Task

AGENTS = {
    Task.EXTRACTION: extraction.run,
    Task.SUMMARY: summarize.run,
    Task.TRANSLATION: translate.run,
    Task.QA: qa.run,
}


def run_agent(
    task: Task,
    document: dict[str, Any],
    model_name: str,
    **kwargs: Any,
) -> AgentCall:
    """Point d'entrée unique du routeur : dispatch vers l'agent de la tâche."""
    return AGENTS[task](document, model_name, **kwargs)


__all__ = ["AGENTS", "AgentCall", "run_agent"]
