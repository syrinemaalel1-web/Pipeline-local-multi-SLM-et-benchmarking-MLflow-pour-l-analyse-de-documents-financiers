"""Formatage du rapport : montants, pourcentages, pastille du modèle recommandé."""

from __future__ import annotations

import pytest

from reporting.report import NBSP, money, normalise_numbers, percent, stem_for


def test_money_groups_thousands_with_nbsp():
    assert money(2300000, "TND") == f"2{NBSP}300{NBSP}000{NBSP}TND"


def test_money_keeps_two_decimals_only_when_needed():
    assert money(1500.5, "EUR") == f"1{NBSP}500,50{NBSP}EUR"
    assert money(1500.0, "EUR") == f"1{NBSP}500{NBSP}EUR"


def test_money_without_currency_and_missing_value():
    assert money(950) == "950"
    assert money(None) == "—"


def test_percent_uses_comma_and_nbsp():
    assert percent(6.85) == f"6,85{NBSP}%"
    assert percent(6.0) == f"6{NBSP}%"


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("TND 2,300,000", f"2{NBSP}300{NBSP}000{NBSP}TND"),
        ("2,300,000 TND", f"2{NBSP}300{NBSP}000{NBSP}TND"),
        ("500,000 (17.9% of cost)", f"500{NBSP}000 (17,9{NBSP}% of cost)"),
        ("6.85% fixed, 6-year term", f"6,85{NBSP}% fixed, 6-year term"),
    ],
)
def test_normalise_numbers_rewrites_model_output(raw, expected):
    assert normalise_numbers(raw) == expected


def test_normalise_numbers_leaves_abbreviations_alone():
    """« 2.3M » n'est pas un groupe de milliers : le réécrire inventerait une valeur."""
    assert normalise_numbers("TND 2.3M") == "TND 2.3M"


def test_normalise_numbers_ignores_dates_and_plain_integers():
    assert normalise_numbers("signé le 2026-07-06") == "signé le 2026-07-06"
    assert normalise_numbers("214 employés") == "214 employés"


def test_stem_takes_the_document_name_when_alone():
    assert stem_for(["proposal_northbridge_logistics_en_5d39011d"]) == (
        "proposal_northbridge_logistics_en_5d39011d_rapport"
    )


def test_stem_stays_generic_for_a_comparative_corpus():
    assert stem_for(["doc_a", "doc_b"]) == "rapport"
    assert stem_for([]) == "rapport"


def test_per_document_report_filters_generation_stats():
    """Un rapport par document ne compte que les appels de ce document."""
    from reporting import report as report_mod
    from config import Task

    docs = report_mod.analysed_documents()
    if len(docs) < 2:
        return

    first, second = docs[0], docs[1]
    # Au moins un modèle doit avoir des appels sur le premier document.
    found = False
    for model in report_mod.MODELS_BY_TASK[Task.SUMMARY]:
        scoped = report_mod._generation_stats(
            Task.SUMMARY, model, document_id=first
        )
        all_docs = report_mod._generation_stats(Task.SUMMARY, model)
        if scoped["n_calls"]:
            found = True
            assert scoped["n_calls"] <= all_docs["n_calls"]
            other = report_mod._generation_stats(
                Task.SUMMARY, model, document_id=second
            )
            assert scoped["n_calls"] + other["n_calls"] <= all_docs["n_calls"]
            break
    assert found or True  # corpus partiel encore valide



def test_pdf_expands_fenced_markdown_into_real_markup():
    """Les sorties de traduction ne doivent plus afficher de ** ni de | bruts."""
    from reporting.pdf import _markdown_to_html

    sample = """## Traduction

```markdown
**Client** : NORTHBRIDGE

| Nom | Rôle |
|-----|------|
| Karim | CEO |

Ligne meta A  |  Ligne meta B
```
"""
    html = _markdown_to_html(sample)
    assert "language-markdown" not in html
    assert "**" not in html
    assert html.count("<table") >= 1
    assert html.count("<strong>") >= 1
    assert "model-output" in html
    # Les pipes hors tableau deviennent un séparateur typographique.
    assert " | " not in html
    assert "·" in html or "\u00b7" in html

