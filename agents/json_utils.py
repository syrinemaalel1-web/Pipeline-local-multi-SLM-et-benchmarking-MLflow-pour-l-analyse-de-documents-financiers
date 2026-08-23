"""Récupération d'un objet JSON dans une sortie de modèle.

Le benchmark n'active pas le décodage contraint (cf. docs/architecture-review.md
§4.6) : la capacité à produire du JSON propre fait justement partie de ce qu'on
mesure. On tolère donc les enrobages les plus courants (bloc de code, phrase
d'introduction), mais rien de plus : un modèle qui rend du JSON cassé doit être
compté comme tel.
"""

from __future__ import annotations

import json
import re
from typing import Any

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def _balanced_object(text: str) -> str | None:
    """Extrait le premier objet JSON équilibré, en ignorant les accolades en chaîne."""
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escaped = False

    for i, char in enumerate(text[start:], start=start):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]

    return None


def parse_json_object(text: str) -> tuple[dict[str, Any] | None, str | None]:
    """Renvoie ``(objet, erreur)``. L'un des deux est toujours ``None``."""
    if not text or not text.strip():
        return None, "sortie vide"

    candidates: list[str] = []

    fenced = _FENCE.search(text)
    if fenced:
        candidates.append(fenced.group(1).strip())

    candidates.append(text.strip())

    balanced = _balanced_object(text)
    if balanced:
        candidates.append(balanced)

    last_error = "aucun objet JSON trouvé"
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError as exc:
            last_error = f"JSON invalide : {exc.msg} (ligne {exc.lineno}, col {exc.colno})"
            continue
        if isinstance(parsed, dict):
            return parsed, None
        last_error = f"racine JSON de type {type(parsed).__name__}, objet attendu"

    return None, last_error
