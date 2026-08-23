"""Détection de langue restreinte aux trois langues du corpus.

`langdetect` (proposé dans PROJECT.md §3.3) est peu fiable sur du texte court et sur
des documents saturés de chiffres, ce qui décrit exactement une proposition
financière. `py3langid` est utilisé à la place, avec deux garde-fous : restriction du
modèle à fr/en/ar, et seuil de confiance en dessous duquel on renvoie « indéterminé »
plutôt qu'une réponse fausse.
"""

from __future__ import annotations

import re
from functools import lru_cache

from config import LANGID_MIN_CHARS, LANGID_MIN_CONFIDENCE, Lang

#: Ce qui n'aide pas un détecteur de langue : montants, dates, codes, ponctuation.
_NOISE = re.compile(r"[\d.,;:%€$/\\|\-_()\[\]{}<>#*+=~^\s]+")

_ARABIC = re.compile(r"[\u0600-\u06FF\u0750-\u077F]")


@lru_cache(maxsize=1)
def _identifier():
    from py3langid.langid import MODEL_FILE, LanguageIdentifier

    ident = LanguageIdentifier.from_pickled_model(MODEL_FILE, norm_probs=True)
    ident.set_languages([lang.value for lang in Lang])
    return ident


def _denoise(text: str) -> str:
    return _NOISE.sub(" ", text).strip()


def detect_language(text: str) -> tuple[Lang | None, float]:
    """Renvoie ``(langue, confiance)``, ou ``(None, confiance)`` si indéterminé.

    Un texte trop court ou une confiance sous le seuil renvoie ``None`` : mieux vaut
    une abstention explicite qu'un scorer de langue qui pénalise un modèle sur une
    erreur de détection.
    """
    cleaned = _denoise(text)
    if len(cleaned) < LANGID_MIN_CHARS:
        return None, 0.0

    # L'écriture arabe est sans ambiguïté ; inutile de risquer un faux négatif
    # statistique sur un texte truffé de chiffres.
    arabic_ratio = len(_ARABIC.findall(cleaned)) / len(cleaned)
    if arabic_ratio > 0.20:
        return Lang.AR, 1.0

    code, confidence = _identifier().classify(cleaned)
    if confidence < LANGID_MIN_CONFIDENCE:
        return None, float(confidence)
    try:
        return Lang(code), float(confidence)
    except ValueError:
        return None, float(confidence)


def detect_language_strict(text: str, *, what: str = "texte") -> Lang:
    """Comme :func:`detect_language`, mais lève si la langue reste indéterminée."""
    lang, confidence = detect_language(text)
    if lang is None:
        raise ValueError(
            f"Langue indéterminée pour {what} (confiance {confidence:.2f} < "
            f"{LANGID_MIN_CONFIDENCE}). Un document doit être dans une seule langue "
            "parmi fr/en/ar."
        )
    return lang
