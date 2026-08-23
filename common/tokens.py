"""Comptage approximatif de tokens, pour anticiper les troncatures d'Ollama.

Chaque modèle a son propre tokenizer ; on ne peut pas tous les charger. `cl100k_base`
sert de proxy commun. Il sur-compte l'arabe (fallback octet) par rapport aux
tokenizers de Qwen ou Aya, ce qui fait pencher l'estimation du bon côté : on
avertit trop tôt plutôt que trop tard.
"""

from __future__ import annotations

from functools import lru_cache

from config import Task, safe_document_tokens, safe_document_tokens_for_judge


@lru_cache(maxsize=1)
def _encoding():
    import tiktoken

    return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(_encoding().encode(text, disallowed_special=()))


def context_warnings(n_tokens: int) -> dict[str, str]:
    """Avertissements de dépassement de contexte, par tâche.

    Renvoie un dictionnaire ``{nom_de_tâche: message}`` vide si le document tient
    partout. Un dépassement n'est jamais silencieux : Ollama tronquerait l'entrée
    sans rien signaler, et le modèle passerait pour mauvais alors qu'il n'aurait
    simplement pas vu la fin du document.
    """
    warnings: dict[str, str] = {}

    for task in Task:
        budget = safe_document_tokens(task)
        if n_tokens > budget:
            warnings[task.value] = (
                f"{n_tokens} tokens > budget agent de {budget} : Ollama tronquera "
                f"l'entrée pour la tâche '{task.value}'."
            )

        judge_budget = safe_document_tokens_for_judge(task)
        if n_tokens > judge_budget:
            key = f"{task.value}:judge"
            warnings[key] = (
                f"{n_tokens} tokens > budget juge de {judge_budget} : le juge de la "
                f"tâche '{task.value}' ne verra pas tout le document."
            )

    return warnings
