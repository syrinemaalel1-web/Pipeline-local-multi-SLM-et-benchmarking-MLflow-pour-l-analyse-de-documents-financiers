"""Scorers du benchmark : déterministes (code) et juges LLM.

`scorers_for(task)` renvoie l'ensemble applicable à une tâche. `Safety` n'y figure
pas : sur des propositions financières il passe à 100 % pour tous les candidats,
n'apporte aucun pouvoir discriminant et coûte un appel LLM par ligne
(docs/architecture-review.md §3.3). Il reste disponible via `safety_guardrail()`
pour un contrôle ponctuel hors tableau comparatif.
"""

from __future__ import annotations

from config import Task
from evaluation import code_scorers as _code
from evaluation.judges import judges_for

CODE_SCORERS: dict[Task, list] = {
    Task.EXTRACTION: [
        _code.json_parseable,
        _code.schema_valid,
        _code.required_fields_present,
        _code.language_conformity,
    ],
    Task.SUMMARY: [
        _code.summary_length_in_range,
        _code.language_conformity,
    ],
    Task.TRANSLATION: [
        _code.numbers_preserved,
        _code.translation_not_truncated,
        _code.language_conformity,
    ],
    Task.QA: [
        _code.abstention_correct,
        _code.language_conformity,
    ],
}


def scorers_for(task: Task, *, include_judges: bool = True) -> list:
    """Scorers d'une tâche.

    `include_judges=False` permet de rejouer les seuls scorers déterministes, qui
    sont gratuits — utile pour itérer sur le rapport sans relancer le juge.
    """
    scorers = list(CODE_SCORERS[task])
    if include_judges:
        scorers += judges_for(task)
    return scorers


def safety_guardrail():
    """Scorer `Safety` de MLflow, hors tableau comparatif."""
    from mlflow.genai.scorers import Safety

    from config import JUDGE_MODEL_URI

    return Safety(model=JUDGE_MODEL_URI)


__all__ = ["CODE_SCORERS", "judges_for", "safety_guardrail", "scorers_for"]
