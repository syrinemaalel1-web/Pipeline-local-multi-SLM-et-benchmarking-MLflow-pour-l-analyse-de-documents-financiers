"""Agent 3 — traduction, direction imposée par la langue source.

FR vers EN, EN vers FR, AR vers FR. La langue cible n'est jamais un paramètre libre :
elle se déduit de la langue détectée à l'ingestion (PROJECT.md §1).
"""

from __future__ import annotations

from typing import Any

import mlflow

from agents import prompts
from agents.ollama_client import AgentCall, chat
from config import LANG_NAMES, TRANSLATION_DIRECTION, Lang, Task


@mlflow.trace(name="agent_translation", span_type="AGENT")
def run(document: dict[str, Any], model_name: str) -> AgentCall:
    source_lang = Lang(document["lang"])
    target_lang = TRANSLATION_DIRECTION[source_lang]

    call = chat(
        model=model_name,
        messages=prompts.render(
            Task.TRANSLATION,
            source_lang_name=LANG_NAMES[source_lang],
            target_lang_name=LANG_NAMES[target_lang],
            document=document["text"],
        ),
        task=Task.TRANSLATION,
    )

    call.metadata = {
        "source_lang": source_lang.value,
        "target_lang": target_lang.value,
    }
    return call
