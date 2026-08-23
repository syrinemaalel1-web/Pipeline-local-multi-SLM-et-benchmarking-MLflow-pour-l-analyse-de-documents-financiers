"""Tests des briques déterministes.

Volontairement limités à ce qui ne dépend d'aucun modèle : schéma, parsing JSON,
détection de langue, préservation des nombres, abstention. Ce sont les composants
dont une régression fausserait silencieusement tout un rapport.

    python -m pytest tests -q
"""

from __future__ import annotations

import json

import pytest

from agents.json_utils import parse_json_object
from agents.ollama_client import clean_output
from common.language import detect_language
from config import Lang, Task
from evaluation import code_scorers as sc
from schemas.proposal import validate_extraction

# --------------------------------------------------------------------------- #
# Schéma d'extraction
# --------------------------------------------------------------------------- #

VALID = {
    "client": "STÉ ATLAS AGRO-INDUSTRIE SARL",
    "montant": 1450000,
    "devise": "TND",
    "taux": 7.4,
    "duree": "84 mois",
    "date": "2026-07-05",
    "clauses_cles": [],
}


def test_schema_accepts_tnd_and_iso_date():
    ok, errors = validate_extraction(VALID)
    assert ok, errors


def test_schema_rejects_non_iso_date():
    ok, errors = validate_extraction({**VALID, "date": "05/07/2026"})
    assert not ok
    assert any("AAAA-MM-JJ" in e for e in errors)


def test_schema_rejects_unknown_currency():
    ok, errors = validate_extraction({**VALID, "devise": "DINAR"})
    assert not ok
    assert any("ISO 4217" in e for e in errors)


def test_schema_reports_missing_required_fields():
    payload = {k: v for k, v in VALID.items() if k != "montant"}
    ok, errors = validate_extraction(payload)
    assert not ok
    assert any("montant" in e for e in errors)


def test_not_found_marker_is_not_a_type_error():
    ok, errors = validate_extraction({**VALID, "taux": "NON_TROUVE"})
    assert ok, errors


# --------------------------------------------------------------------------- #
# Parsing JSON
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "raw",
    [
        json.dumps(VALID),
        f"```json\n{json.dumps(VALID)}\n```",
        f"Voici le résultat :\n{json.dumps(VALID)}\nJ'espère que cela convient.",
    ],
)
def test_parse_json_tolerates_common_wrappers(raw):
    parsed, error = parse_json_object(raw)
    assert error is None
    assert parsed["devise"] == "TND"


def test_parse_json_reports_broken_output():
    parsed, error = parse_json_object('{"client": "X", "montant": }')
    assert parsed is None
    assert error


def test_parse_json_ignores_braces_inside_strings():
    parsed, _ = parse_json_object('{"client": "A { B } C", "montant": 1}')
    assert parsed["client"] == "A { B } C"


# --------------------------------------------------------------------------- #
# Langue
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "text,expected",
    [
        (
            "La société sollicite un financement de 1 450 000 dinars tunisiens "
            "sur une durée de quatre-vingt-quatre mois auprès de la banque.",
            Lang.FR,
        ),
        (
            "The company requests a facility of two million three hundred thousand "
            "dinars over a term of seventy-two months from the bank.",
            Lang.EN,
        ),
        (
            "تتقدم الشركة بطلب تمويل بمبلغ خمسة ملايين ومائتي ألف دينار تونسي "
            "على مدة اثنين وسبعين شهرا لدى البنك",
            Lang.AR,
        ),
    ],
)
def test_language_detection(text, expected):
    detected, confidence = detect_language(text)
    assert detected is expected
    assert confidence > 0.5


def test_language_abstains_on_digit_soup():
    detected, _ = detect_language("2 300 000 / 6.85 % / 72 / 2026-07-06")
    assert detected is None


def test_language_conformity_uses_json_values_not_keys():
    """Les clés sont en français par convention : elles ne doivent pas fausser la détection."""
    arabic_json = json.dumps(
        {
            "client": "شركة ألفا للمقاولات والإنشاءات المحدودة",
            "duree": "اثنان وسبعون شهرا من تاريخ التوقيع على العقد",
            "clauses_cles": [
                {
                    "titre": "ضمانات",
                    "contenu": "رهن عقاري على المقر الاجتماعي للشركة بمدينة تونس",
                }
            ],
        },
        ensure_ascii=False,
    )
    feedback = sc.language_conformity(
        inputs={"task": Task.EXTRACTION.value, "expected_output_lang": Lang.AR.value},
        outputs=arabic_json,
    )
    assert feedback.value is True


# --------------------------------------------------------------------------- #
# Traduction
# --------------------------------------------------------------------------- #


def test_numbers_preserved_across_thousand_separators():
    feedback = sc.numbers_preserved(
        inputs={"document": "Montant de 2 300 000 TND au taux de 6,85 % sur 72 mois."},
        outputs="Amount of 2,300,000 TND at a rate of 6.85% over 72 months.",
    )
    assert feedback.value == 1.0


def test_numbers_preserved_detects_a_lost_amount():
    feedback = sc.numbers_preserved(
        inputs={"document": "Montant de 2 300 000 TND sur 72 mois."},
        outputs="Amount over 72 months.",
    )
    assert feedback.value < 1.0


def test_numbers_preserved_handles_arabic_indic_digits():
    feedback = sc.numbers_preserved(
        inputs={"document": "المبلغ ٥٢٠٠٠٠٠ دينار على ٧٢ شهرا"},
        outputs="Montant de 5200000 dinars sur 72 mois.",
    )
    assert feedback.value == 1.0


def test_truncated_translation_is_flagged():
    feedback = sc.translation_not_truncated(
        inputs={"document": "x" * 10000}, outputs="y" * 1000
    )
    assert feedback.value is False


# --------------------------------------------------------------------------- #
# Q&A
# --------------------------------------------------------------------------- #


def test_abstention_expected_and_given():
    feedback = sc.abstention_correct(
        inputs={"expect_abstention": True},
        outputs="NON_TROUVE — le document ne mentionne aucun IBAN.",
    )
    assert feedback.value is True


def test_abstention_expected_but_hallucinated():
    feedback = sc.abstention_correct(
        inputs={"expect_abstention": True},
        outputs="L'IBAN du bénéficiaire est TN59 1000 6035 0000 1234 5678.",
    )
    assert feedback.value is False


def test_abstention_skipped_when_unannotated():
    feedback = sc.abstention_correct(inputs={}, outputs="Une réponse quelconque.")
    assert feedback.value is None


# --------------------------------------------------------------------------- #
# Nettoyage des modèles de raisonnement
# --------------------------------------------------------------------------- #


def test_think_block_is_stripped_for_reasoning_models():
    raw = "<think>The user asks in French about an English doc...</think>\nTND 2.3 million"
    assert clean_output(raw, "deepseek-r1:8b") == "TND 2.3 million"


def test_unterminated_think_block_is_stripped():
    raw = "Some prefix\n<think>reasoning cut off by max_tokens"
    assert clean_output(raw, "deepseek-r1:8b") == "Some prefix"


def test_think_block_is_left_alone_for_other_models():
    raw = "<think>literal text</think> answer"
    assert clean_output(raw, "qwen2.5:7b") == raw
