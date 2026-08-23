"""Schéma du JSON d'extraction.

Trois usages à partir d'une seule définition : documentation du contrat attendu,
validation par le scorer code-based, et génération du JSON schema si l'on active
un jour le décodage contraint.

Convention de langue (cf. docs/architecture-review.md §4.7) : les **clés** sont
toujours en français, quelle que soit la langue du document. Seules les valeurs de
texte libre suivent la langue source. `devise` (ISO 4217) et `date` (ISO 8601) sont
indépendantes de la langue.
"""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, Field, ValidationError, field_validator

from config import ALLOWED_CURRENCIES, NOT_FOUND_MARKER

#: Champs que tout modèle doit produire, même vides. Un champ absent est une faute
#: de forme distincte d'un champ renseigné à NON_TROUVE (qui est une abstention
#: légitime).
REQUIRED_FIELDS = ("client", "montant", "devise", "taux", "duree", "date")

ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


class Clause(BaseModel):
    titre: str
    contenu: str


class FinancialProposal(BaseModel):
    """Proposition financière tunisienne extraite d'un document."""

    client: str | None = Field(
        default=None, description="Nom du client, dans la langue du document source."
    )
    montant: float | None = Field(
        default=None, description="Montant principal, en valeur numérique nue."
    )
    devise: str | None = Field(
        default=None, description="Code ISO 4217. TND est la valeur la plus courante."
    )
    taux: float | None = Field(
        default=None, description="Taux d'intérêt en pourcentage (5.25 et non 0.0525)."
    )
    duree: str | None = Field(
        default=None, description="Durée de l'engagement, telle qu'exprimée."
    )
    date: str | None = Field(
        default=None,
        description="Date de la proposition, normalisée en ISO 8601 (AAAA-MM-JJ).",
    )
    clauses_cles: list[Clause] = Field(default_factory=list)

    @field_validator("devise")
    @classmethod
    def _currency_is_iso(cls, v: str | None) -> str | None:
        if v is None or v == NOT_FOUND_MARKER:
            return v
        code = v.strip().upper()
        if code not in ALLOWED_CURRENCIES:
            raise ValueError(
                f"devise {code!r} hors ISO 4217 attendu : {sorted(ALLOWED_CURRENCIES)}"
            )
        return code

    @field_validator("date")
    @classmethod
    def _date_is_iso(cls, v: str | None) -> str | None:
        if v is None or v == NOT_FOUND_MARKER:
            return v
        if not ISO_DATE.match(v.strip()):
            raise ValueError(
                f"date {v!r} non normalisée : attendu AAAA-MM-JJ. "
                "Le format tunisien courant JJ/MM/AAAA doit être converti."
            )
        return v.strip()


def validate_extraction(payload: dict[str, Any]) -> tuple[bool, list[str]]:
    """Valide un JSON d'extraction sans lever d'exception.

    Renvoie ``(conforme, erreurs)``. Les champs manquants sont signalés séparément
    des erreurs de type, pour que le rapport puisse distinguer un modèle qui oublie
    des champs d'un modèle qui les remplit mal.
    """
    errors: list[str] = []

    missing = [f for f in REQUIRED_FIELDS if f not in payload]
    if missing:
        errors.append(f"champs obligatoires absents : {missing}")

    # Les valeurs sentinelles textuelles doivent traverser la validation typée.
    coerced = {
        k: (None if v == NOT_FOUND_MARKER else v)
        for k, v in payload.items()
        if k in FinancialProposal.model_fields
    }

    try:
        FinancialProposal.model_validate(coerced)
    except ValidationError as exc:
        for err in exc.errors():
            loc = ".".join(str(p) for p in err["loc"]) or "<racine>"
            errors.append(f"{loc}: {err['msg']}")

    unknown = set(payload) - set(FinancialProposal.model_fields)
    if unknown:
        errors.append(f"champs hors schéma : {sorted(unknown)}")

    return not errors, errors
