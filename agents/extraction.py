"""Agent 1 — extraction des champs structurés en JSON."""

from __future__ import annotations

from typing import Any

import mlflow

from agents import prompts
from agents.json_utils import parse_json_object
from agents.ollama_client import AgentCall, chat
from config import LANG_NAMES, NOT_FOUND_MARKER, Lang, Task
from schemas.proposal import validate_extraction


@mlflow.trace(name="agent_extraction", span_type="AGENT")
def run(document: dict[str, Any], model_name: str) -> AgentCall:
    """Extrait la proposition en JSON, dans la langue du document source."""
    lang = Lang(document["lang"])

    call = chat(
        model=model_name,
        messages=prompts.render(
            Task.EXTRACTION,
            lang_name=LANG_NAMES[lang],
            not_found=NOT_FOUND_MARKER,
            document=document["text"],
        ),
        task=Task.EXTRACTION,
    )

    parsed, parse_error = parse_json_object(call.output)
    schema_ok, schema_errors = (
        validate_extraction(parsed) if parsed is not None else (False, [])
    )

    call.metadata = {
        "parsed": parsed,
        "parse_error": parse_error,
        "schema_valid": schema_ok,
        "schema_errors": schema_errors,
        "source_lang": lang.value,
    }
    return call
