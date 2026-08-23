"""Tests des juges et de l'outillage de calibration.

Aucun test n'appelle de modèle : un juge coûte une minute par ligne, et faire dépendre
la suite d'un serveur Ollama la rendrait inutilisable. On vérifie ici ce qui casse en
silence — le gabarit des consignes, le type de la note, la stratification de
l'échantillon et les formules d'accord.

La validité du juge, elle, ne se teste pas hors ligne : c'est l'objet de
`evaluation.judge_probe`, qui lui soumet de vraies fautes injectées.

    python -m pytest tests -q
"""

from __future__ import annotations

import json
from typing import Literal, get_args, get_origin

import pytest

from config import JUDGE_SCALE_MAX, JUDGE_SCALE_MIN, MODELS_BY_TASK, Task
from evaluation import calibration, judge_probe, judges

# --------------------------------------------------------------------------- #
# Consignes des juges
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def built():
    return judges._build(judges.JUDGE_MODEL_URI)


@pytest.mark.parametrize("task", list(Task))
def test_instructions_reference_mlflow_template_variables(built, task):
    """MLflow n'injecte le contexte que via `{{ inputs }}` et `{{ outputs }}`.

    Une accolade mal échappée dans une f-string casserait le gabarit sans erreur
    visible : le juge noterait alors une consigne vide de tout document.
    """
    (judge,) = built[task]
    assert "{{ inputs }}" in judge.instructions
    assert "{{ outputs }}" in judge.instructions
    assert "{{{" not in judge.instructions


@pytest.mark.parametrize("task", list(Task))
def test_judges_rate_on_an_enumerated_scale(built, task):
    """La note est énumérée, ce qui interdit au décodage de sortir du barème.

    Avec un simple `int`, aya:8b rendait 0 sur les cas d'abstention.
    """
    (judge,) = built[task]
    value_type = judge.feedback_value_type

    assert get_origin(value_type) is Literal
    assert set(get_args(value_type)) == set(range(JUDGE_SCALE_MIN, JUDGE_SCALE_MAX + 1))


@pytest.mark.parametrize("task", list(Task))
def test_judges_build_their_own_reference_before_comparing(built, task):
    """Le protocole anti-complaisance doit rester présent dans chaque consigne.

    Sans lui, le juge ratifie la sortie au lieu de la vérifier : mesuré à 0 faute
    détectée sur 5 lors du premier contrôle négatif.
    """
    (judge,) = built[task]
    assert "référence" in judge.instructions
    assert "ANTI-COMPLAISANCE" in judge.instructions


@pytest.mark.parametrize("task", list(Task))
def test_judges_exclude_what_code_scorers_already_measure(built, task):
    (judge,) = built[task]
    assert "La forme est vérifiée séparément par du code" in judge.instructions


def test_qa_judge_treats_abstention_as_a_full_success():
    """Le juge notait 1/5 les 15 abstentions correctes du premier run, tous
    modèles confondus, alors que le scorer code les validait toutes."""
    (judge,) = judges.judges_for(Task.QA)
    assert "récompenser l'invention" in judge.instructions
    assert f"note {JUDGE_SCALE_MIN}" in judge.instructions


def test_qa_judge_ignores_expect_abstention():
    """`abstention_correct` tranche déjà ce point, et le mentionner au juge
    l'entraînait à noter bas dès qu'il lisait « information absente »."""
    (judge,) = judges.judges_for(Task.QA)
    assert "expect_abstention" not in judge.instructions


def test_a_candidate_model_cannot_be_used_as_judge():
    candidate = MODELS_BY_TASK[Task.QA][0]
    with pytest.raises(AssertionError, match="auto-évaluerait"):
        judges.judges_for(Task.QA, model_uri=f"ollama:/{candidate}")


# --------------------------------------------------------------------------- #
# Contrôle négatif
# --------------------------------------------------------------------------- #


def test_falsifying_a_json_amount_changes_the_value():
    original = json.dumps({"client": "ACME", "montant": 2300000, "devise": "TND"})
    corrupted = json.loads(judge_probe._falsify_json_amount(original))

    assert corrupted["montant"] == int(judge_probe._ABSURD_AMOUNT)
    assert corrupted["client"] == "ACME"


def test_falsifying_an_amount_falls_back_on_free_text():
    corrupted = judge_probe._falsify_json_amount("Le financement porte sur 2 300 000 TND.")
    assert judge_probe._ABSURD_AMOUNT in corrupted


def test_truncation_actually_removes_content():
    text = "clause A. clause B. clause C. clause D."
    assert len(judge_probe._truncate_half(text)) < len(text)


def test_every_probe_case_has_an_expected_direction():
    corrupted = [c for c in judge_probe.CASES if c.corrupt is not None]
    clean = [c for c in judge_probe.CASES if c.corrupt is None]

    assert all(not c.should_pass for c in corrupted), "une sortie fautive doit échouer"
    assert all(c.should_pass for c in clean), "une sortie intacte doit passer"
    # Sans contrôle positif, noter 1 partout suffirait à obtenir une sensibilité parfaite.
    assert clean and corrupted


# --------------------------------------------------------------------------- #
# Calibration
# --------------------------------------------------------------------------- #


def _item(task: str, score: int, index: int) -> calibration.Item:
    return calibration.Item(
        id=str(index),
        task=task,
        model="m",
        document_id="doc",
        question=None,
        output="sortie",
        judge_score=score,
    )


def test_sampling_covers_both_verdicts_of_the_judge():
    """Le juge met 5/5 aux trois quarts des lignes : un tirage uniforme ne
    contiendrait presque aucun cas négatif et l'accord serait flatteur."""
    items = [_item("qa", 5, i) for i in range(40)] + [_item("qa", 1, 40 + i) for i in range(4)]

    picked = calibration._stratified(items, 10, seed=0)

    assert len(picked) == 10
    assert sum(1 for i in picked if i.judge_score == 1) == 4


def test_sampling_spreads_across_tasks():
    items = [_item("qa", 5, i) for i in range(20)] + [
        _item("summary", 5, 20 + i) for i in range(20)
    ]

    picked = calibration._stratified(items, 8, seed=0)

    assert {i.task for i in picked} == {"qa", "summary"}
    assert sum(1 for i in picked if i.task == "qa") == 4


def test_sampling_is_reproducible_for_a_given_seed():
    items = [_item("qa", 5, i) for i in range(30)]

    first = [i.output for i in calibration._stratified(items, 5, seed=7)]
    second = [i.output for i in calibration._stratified(items, 5, seed=7)]

    assert first == second


def test_sampling_never_returns_more_than_available():
    picked = calibration._stratified([_item("qa", 5, i) for i in range(3)], 20, seed=0)
    assert len(picked) == 3


def test_kappa_is_zero_when_agreement_is_pure_chance():
    pairs = [(True, True), (True, False), (False, True), (False, False)]
    assert calibration._kappa(pairs) == pytest.approx(0.0)


def test_kappa_is_one_on_perfect_agreement_with_both_verdicts_present():
    pairs = [(True, True), (True, True), (False, False), (False, False)]
    assert calibration._kappa(pairs) == pytest.approx(1.0)


def test_kappa_is_undefined_when_a_verdict_never_varies():
    """Tout noter « bon » donne 100 % d'accord sans rien mesurer : le kappa doit
    refuser de produire un chiffre plutôt que d'en fabriquer un rassurant."""
    assert calibration._kappa([(True, True)] * 12) is None


def test_confidence_interval_stays_within_bounds_on_a_perfect_score():
    low, high = calibration._wilson(20, 20)

    assert 0.0 <= low <= high <= 1.0
    assert low < 1.0, "20 observations ne prouvent pas une fiabilité de 100 %"


def test_confidence_interval_narrows_as_the_sample_grows():
    small = calibration._wilson(16, 20)
    large = calibration._wilson(160, 200)

    assert (large[1] - large[0]) < (small[1] - small[0])
