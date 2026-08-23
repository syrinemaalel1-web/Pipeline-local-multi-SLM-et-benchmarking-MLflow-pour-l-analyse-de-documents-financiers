"""Chargement des gabarits de prompt depuis prompts/.

Les prompts vivent hors du code parce qu'ils sont la principale variable
expérimentale du benchmark : les versionner séparément permet de voir dans un diff
git ce qui a changé entre deux campagnes de mesure.

Le gabarit est strictement identique pour tous les modèles d'une même tâche
(cf. docs/architecture-review.md §4.7) : seules les valeurs injectées varient.
"""

from __future__ import annotations

import re
from functools import lru_cache

from config import PROMPTS_DIR, Task

_SECTION = re.compile(r"^\[(SYSTEM|USER)\]\s*$", re.MULTILINE)


@lru_cache(maxsize=None)
def _load(task: Task) -> tuple[str, str]:
    path = PROMPTS_DIR / f"{task.value}.txt"
    raw = path.read_text(encoding="utf-8")

    parts = _SECTION.split(raw)
    # split() renvoie [préambule, nom, contenu, nom, contenu, ...]
    sections = dict(zip(parts[1::2], parts[2::2]))

    missing = {"SYSTEM", "USER"} - set(sections)
    if missing:
        raise ValueError(f"{path} : sections manquantes {sorted(missing)}")

    return sections["SYSTEM"].strip(), sections["USER"].strip()


def render(task: Task, **values: object) -> list[dict[str, str]]:
    """Construit les messages chat pour une tâche donnée."""
    system, user = _load(task)
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user.format(**values)},
    ]
