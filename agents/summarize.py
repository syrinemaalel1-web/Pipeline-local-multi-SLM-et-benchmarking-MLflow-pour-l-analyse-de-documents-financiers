"""Agent 2 — résumé exécutif, dans la langue du document source."""

from __future__ import annotations

from typing import Any

import mlflow

from agents import prompts
from agents.ollama_client import AgentCall, chat
from config import LANG_NAMES, SUMMARY_WORD_RANGE, Lang, Task


@mlflow.trace(name="agent_summary", span_type="AGENT")
def run(document: dict[str, Any], model_name: str) -> AgentCall:
    lang = Lang(document["lang"])
    min_words, max_words = SUMMARY_WORD_RANGE

    call = chat(
        model=model_name,
        messages=prompts.render(
            Task.SUMMARY,
            lang_name=LANG_NAMES[lang],
            min_words=min_words,
            max_words=max_words,
            document=document["text"],
        ),
        task=Task.SUMMARY,
    )

    call.metadata = {
        "source_lang": lang.value,
        "word_count": len(call.output.split()),
    }
    return call
