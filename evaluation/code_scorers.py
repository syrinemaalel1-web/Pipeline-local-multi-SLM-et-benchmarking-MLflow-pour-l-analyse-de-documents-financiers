"""Scorers déterministes.

Règle de partage avec les juges (docs/architecture-review.md §4.3) : **le code vérifie
la forme, le juge évalue le sens**, sans recouvrement. Compter des mots, valider un
schéma ou vérifier qu'un chiffre a survécu à une traduction sont des tâches où un LLM
est à la fois plus cher et moins fiable qu'une expression régulière.
"""

from __future__ import annotations

import re
from typing import Any

from mlflow.entities import AssessmentSource, AssessmentSourceType, Feedback
from mlflow.genai.scorers import scorer

from agents.json_utils import parse_json_object
from common.language import detect_language
from config import NOT_FOUND_MARKER, SUMMARY_WORD_RANGE, Lang
from schemas.proposal import REQUIRED_FIELDS, validate_extraction

_CODE_SOURCE = AssessmentSource(
    source_type=AssessmentSourceType.CODE, source_id="code_scorer"
)

#: Chiffres arabo-indiens, présents dans les documents en arabe.
_ARABIC_INDIC = str.maketrans("٠١٢٣٤٥٦٧٨٩٫٬", "0123456789.,")

#: Un nombre, éventuellement avec séparateurs de milliers et décimales.
_NUMBER = re.compile(r"\d[\d\s.,]*\d|\d")


def _feedback(value: Any, rationale: str) -> Feedback:
    return Feedback(value=value, rationale=rationale, source=_CODE_SOURCE)


# --------------------------------------------------------------------------- #
# Conformité linguistique — partagée par les quatre agents
# --------------------------------------------------------------------------- #


def _free_text_of_extraction(output: str) -> str:
    """Texte libre d'un JSON d'extraction, clés et codes exclus.

    Les clés sont en français par convention et `devise`/`date` sont des codes
    normalisés : les inclure ferait détecter « français » sur un document arabe
    correctement extrait.
    """
    parsed, _ = parse_json_object(output)
    if not parsed:
        return ""

    chunks: list[str] = []
    for key in ("client", "duree"):
        value = parsed.get(key)
        if isinstance(value, str) and value != NOT_FOUND_MARKER:
            chunks.append(value)

    for clause in parsed.get("clauses_cles") or []:
        if isinstance(clause, dict):
            chunks.extend(
                str(v) for v in (clause.get("titre"), clause.get("contenu")) if v
            )

    return "\n".join(chunks)


@scorer(name="language_conformity")
def language_conformity(inputs: dict[str, Any], outputs: str) -> Feedback:
    """La sortie est-elle dans la langue exigée par la règle de langue du projet ?

    Cette règle vaut pour les quatre agents (PROJECT.md §1), mais le spec ne prévoyait
    de vérification que pour la traduction. Sans ce scorer, la contrainte centrale du
    projet resterait non mesurée sur trois agents sur quatre.
    """
    expected = Lang(inputs["expected_output_lang"])

    text = (
        _free_text_of_extraction(outputs)
        if inputs["task"] == "extraction"
        else outputs
    )

    if not text.strip():
        return _feedback(False, "Aucun texte exploitable pour détecter la langue.")

    detected, confidence = detect_language(text)

    if detected is None:
        # Ne pas pénaliser un modèle pour une incertitude du détecteur.
        return _feedback(
            None,
            f"Langue indéterminée (confiance {confidence:.2f}) : scorer non concluant.",
        )

    return _feedback(
        detected is expected,
        f"Attendu {expected.value}, détecté {detected.value} "
        f"(confiance {confidence:.2f}).",
    )


# --------------------------------------------------------------------------- #
# Extraction
# --------------------------------------------------------------------------- #


@scorer(name="json_parseable")
def json_parseable(outputs: str) -> Feedback:
    parsed, error = parse_json_object(outputs)
    return _feedback(parsed is not None, error or "Objet JSON valide.")


@scorer(name="schema_valid")
def schema_valid(outputs: str) -> Feedback:
    """Conformité au schéma : champs requis, types, ISO 4217, date ISO 8601."""
    parsed, error = parse_json_object(outputs)
    if parsed is None:
        return _feedback(False, f"Schéma non vérifiable : {error}")

    ok, errors = validate_extraction(parsed)
    return _feedback(ok, "; ".join(errors) if errors else "Conforme au schéma.")


@scorer(name="required_fields_present")
def required_fields_present(outputs: str) -> Feedback:
    """Part des champs obligatoires renseignés ou explicitement marqués non trouvés.

    Distinct de `schema_valid` : mesure la couverture plutôt que la validité, et
    donne une note continue là où la validité est binaire.
    """
    parsed, error = parse_json_object(outputs)
    if parsed is None:
        return _feedback(0.0, f"JSON illisible : {error}")

    present = [f for f in REQUIRED_FIELDS if parsed.get(f) not in (None, "")]
    missing = sorted(set(REQUIRED_FIELDS) - set(present))

    return _feedback(
        len(present) / len(REQUIRED_FIELDS),
        f"{len(present)}/{len(REQUIRED_FIELDS)} champs renseignés."
        + (f" Manquants : {missing}." if missing else ""),
    )


# --------------------------------------------------------------------------- #
# Résumé
# --------------------------------------------------------------------------- #


@scorer(name="summary_length_in_range")
def summary_length_in_range(outputs: str) -> Feedback:
    min_words, max_words = SUMMARY_WORD_RANGE
    count = len(outputs.split())
    return _feedback(
        min_words <= count <= max_words,
        f"{count} mots (fourchette cible {min_words}-{max_words}).",
    )


# --------------------------------------------------------------------------- #
# Traduction
# --------------------------------------------------------------------------- #


def _numbers(text: str) -> set[str]:
    """Nombres normalisés d'un texte, séparateurs de milliers retirés."""
    normalized = text.translate(_ARABIC_INDIC)
    found: set[str] = set()

    for raw in _NUMBER.findall(normalized):
        cleaned = raw.replace(" ", "").replace("\u00a0", "")
        # Une virgule suivie de trois chiffres est un séparateur de milliers ;
        # sinon c'est une virgule décimale (convention francophone).
        cleaned = re.sub(r",(?=\d{3}\b)", "", cleaned)
        cleaned = cleaned.replace(",", ".").rstrip(".")
        cleaned = re.sub(r"\.(?=\d{3}\b)", "", cleaned)
        if cleaned:
            found.add(cleaned.lstrip("0") or "0")

    return found


@scorer(name="numbers_preserved")
def numbers_preserved(inputs: dict[str, Any], outputs: str) -> Feedback:
    """Part des nombres du document source retrouvés dans la traduction.

    Un montant altéré ou une clause omise se voit ici sans faire appel à un juge,
    et c'est le mode de défaillance le plus coûteux d'une traduction financière.
    """
    source_numbers = _numbers(inputs["document"])
    if not source_numbers:
        return _feedback(None, "Aucun nombre dans le document source.")

    output_numbers = _numbers(outputs)
    kept = source_numbers & output_numbers
    lost = sorted(source_numbers - output_numbers)[:10]

    return _feedback(
        len(kept) / len(source_numbers),
        f"{len(kept)}/{len(source_numbers)} nombres préservés."
        + (f" Perdus (10 premiers) : {lost}." if lost else ""),
    )


@scorer(name="translation_not_truncated")
def translation_not_truncated(inputs: dict[str, Any], outputs: str) -> Feedback:
    """Détecte une traduction manifestement écourtée.

    Une sortie très courte face au document source signale soit une omission
    massive, soit une troncature par la fenêtre de contexte — deux causes qu'il faut
    distinguer d'un simple problème de qualité.
    """
    source_len = len(inputs["document"])
    if source_len == 0:
        return _feedback(None, "Document source vide.")

    ratio = len(outputs) / source_len
    return _feedback(
        ratio >= 0.5,
        f"Traduction à {ratio:.0%} de la longueur du source "
        f"({len(outputs)} vs {source_len} caractères).",
    )


# --------------------------------------------------------------------------- #
# Question-Réponse
# --------------------------------------------------------------------------- #


@scorer(name="abstention_correct")
def abstention_correct(inputs: dict[str, Any], outputs: str) -> Feedback:
    """Le modèle s'abstient-il quand, et seulement quand, il le doit ?

    Ne s'applique qu'aux questions dont le dataset indique si la réponse figure ou
    non dans le document. Sans questions sans réponse dans le jeu de test, la
    guideline « dis que tu ne sais pas » du spec (§3.4) n'est jamais mise à
    l'épreuve.
    """
    expected = inputs.get("expect_abstention")
    if expected is None:
        return _feedback(None, "Question non annotée en abstention : scorer ignoré.")

    abstained = NOT_FOUND_MARKER in outputs.upper()

    if expected and not abstained:
        return _feedback(
            False, "Information absente du document, mais le modèle a répondu quand même."
        )
    if not expected and abstained:
        return _feedback(
            False, "Information présente dans le document, mais le modèle s'est abstenu."
        )
    return _feedback(
        True,
        "Abstention correcte." if expected else "Réponse fournie, comme attendu.",
    )
