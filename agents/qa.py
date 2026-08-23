"""Agent 4 — question-réponse sur le document entier.

La réponse est rendue dans la langue du document source, même si la question est
posée dans une autre langue (PROJECT.md §1).
"""

from __future__ import annotations

from typing import Any

import mlflow

from agents import prompts
from agents.ollama_client import AgentCall, chat
from config import LANG_NAMES, NOT_FOUND_MARKER, Lang, Task


@mlflow.trace(name="agent_qa", span_type="AGENT")
def run(document: dict[str, Any], model_name: str, question: str) -> AgentCall:
    lang = Lang(document["lang"])

    call = chat(
        model=model_name,
        messages=prompts.render(
            Task.QA,
            lang_name=LANG_NAMES[lang],
            not_found=NOT_FOUND_MARKER,
            document=document["text"],
            question=question,
        ),
        task=Task.QA,
    )

    call.metadata = {
        "source_lang": lang.value,
        "question": question,
        "abstained": NOT_FOUND_MARKER in call.output.upper(),
    }
    return call
