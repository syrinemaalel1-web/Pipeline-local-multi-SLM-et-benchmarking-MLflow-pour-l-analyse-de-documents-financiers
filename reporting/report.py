"""Rapport comparatif final : un tableau par tâche, plus la meilleure sortie réelle.

    python -m reporting.report            # un Markdown par document analysé
    python -m reporting.report --all      # rapport global combiné (rapport.md)

Toute la logique vit ici plutôt que dans le notebook, pour qu'elle soit testable et
rejouable en une commande. Le notebook ne fait qu'afficher ce que ce module calcule.

Le rapport se construit exclusivement depuis `results/` : il se régénère sans relancer
une seule inférence, ce qui est indispensable quand une campagne coûte 12 à 20 h.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import statistics
from typing import Any

from common import logging_setup
from config import (
    BENCHMARK_CAVEATS,
    CODE_SCORE_FLOOR,
    COMPOSITE_WEIGHTS,
    JUDGE_PASS_THRESHOLD,
    JUDGE_SCALE_MAX,
    JUDGE_SCALE_MIN,
    MODELS_BY_TASK,
    NOT_FOUND_MARKER,
    QUALITY_MIX,
    REPORTS_DIR,
    RESULTS_DIR,
    EvalMode,
    Task,
)
from evaluation.judges import judges_for
from evaluation.legacy_metrics import METRIC_NAMES
from orchestration import store

log = logging.getLogger("report")

EVALUATION_DIR = RESULTS_DIR / "evaluation"
METRICS_EVALUATION_DIR = RESULTS_DIR / "evaluation_metrics"

#: Libellés lisibles des 4 métriques du pipeline `metrics` (mlflow.evaluate(), voir
#: evaluation/legacy_metrics.py) — dans l'ordre où elles doivent apparaître au tableau.
_METRIC_LABELS: dict[str, str] = {
    "faithfulness": "Faithfulness /5",
    "answer_relevance": "Answer relevance /5",
    "ari_grade_level": "ARI",
    "flesch_kincaid_grade_level": "Flesch-Kincaid",
}

#: Sous-ensemble de METRIC_NAMES utilisé pour classer les sorties dans "meilleure
#: sortie" (`metrics_best_output`). Pour le résumé, les 3 métriques comptent
#: ensemble depuis le 2026-08-14, sur demande explicite de l'utilisateur —
#: `ari_grade_level`/`flesch_kincaid_grade_level` sont des scores de lisibilité,
#: pas de qualité au sens strict (un document financier dense a naturellement un
#: niveau élevé, ce n'est pas un défaut), donc leur direction "meilleur" n'est
#: pas universelle : convention assumée ici, "plus bas = plus accessible = mieux"
#: (voir `_HIGHER_IS_BETTER`).
_RANKING_METRICS: dict[Task, tuple[str, ...]] = {
    Task.TRANSLATION: ("faithfulness",),
    Task.SUMMARY: ("faithfulness", "ari_grade_level", "flesch_kincaid_grade_level"),
    Task.QA: ("answer_relevance", "faithfulness"),
}

#: Direction de chaque métrique de classement — True si "plus haut = meilleur".
#: ari_grade_level/flesch_kincaid_grade_level : plus BAS = plus accessible, donc
#: compté comme meilleur (convention assumée, pas universelle — voir ci-dessus).
_HIGHER_IS_BETTER: dict[str, bool] = {
    "faithfulness": True,
    "answer_relevance": True,
    "ari_grade_level": False,
    "flesch_kincaid_grade_level": False,
}


# --------------------------------------------------------------------------- #
# Lecture des artefacts
# --------------------------------------------------------------------------- #


def _load_evaluation(task: Task, model: str) -> dict[str, Any] | None:
    path = EVALUATION_DIR / task.value / f"{store.slug(model)}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text("utf-8"))


def _generation_stats(
    task: Task, model: str, document_id: str | None = None
) -> dict[str, Any]:
    """Latence, débit et mémoire, agrégés depuis les sorties d'agents."""
    latencies: list[float] = []
    throughputs: list[float] = []
    vram: list[int] = []
    total: list[int] = []
    gpu_fraction: list[float] = []
    errors = 0

    for record in store.load_all(task):
        if record["model"] != model or record.get("smoke"):
            continue
        if document_id is not None and record.get("document_id") != document_id:
            continue
        if record.get("error"):
            errors += 1
            continue

        latencies.append(record["latency_s"])
        if record.get("tokens_per_second"):
            throughputs.append(record["tokens_per_second"])
        memory = record.get("memory") or {}
        if memory:
            vram.append(memory["vram_bytes"])
            total.append(memory["total_bytes"])
            gpu_fraction.append(memory["gpu_fraction"])

    return {
        "n_calls": len(latencies) + errors,
        "n_errors": errors,
        "latency_mean_s": statistics.mean(latencies) if latencies else None,
        "tokens_per_second": statistics.mean(throughputs) if throughputs else None,
        "vram_gb": max(vram) / 1e9 if vram else None,
        "total_gb": max(total) / 1e9 if total else None,
        "gpu_fraction": statistics.mean(gpu_fraction) if gpu_fraction else None,
    }


def _metric(metrics: dict[str, Any], scorer_name: str) -> float | None:
    """Valeur agrégée d'un scorer, quel que soit le suffixe employé par MLflow.

    Les clés varient (`<scorer>/mean`, `<scorer>/true_ratio`...) selon le type de
    retour du scorer ; on prend la première correspondance numérique.
    """
    exact = metrics.get(scorer_name)
    if isinstance(exact, (int, float)):
        return float(exact)

    for key, value in metrics.items():
        if key.startswith(f"{scorer_name}/") and isinstance(value, (int, float)):
            return float(value)
    return None


# --------------------------------------------------------------------------- #
# Agrégation par tâche
# --------------------------------------------------------------------------- #


def _normalise_judge(value: float | None) -> float | None:
    """Ramène une note de 1 à 5 sur [0, 1]."""
    if value is None:
        return None
    span = JUDGE_SCALE_MAX - JUDGE_SCALE_MIN
    return max(0.0, min(1.0, (value - JUDGE_SCALE_MIN) / span))


def _code_scores(task: Task, metrics: dict[str, Any]) -> dict[str, float]:
    from evaluation import CODE_SCORERS

    scores: dict[str, float] = {}
    for scorer in CODE_SCORERS[task]:
        value = _metric(metrics, scorer.name)
        if value is not None:
            scores[scorer.name] = value
    return scores


def _min_max(values: dict[str, float | None], *, higher_is_better: bool) -> dict[str, float]:
    """Normalise sur [0, 1] parmi les candidats de la tâche.

    Un modèle sans valeur reçoit 0 : ne pas avoir produit de mesure ne doit pas être
    plus avantageux que d'en avoir produit une mauvaise.
    """
    present = {k: v for k, v in values.items() if v is not None}
    if not present:
        return {k: 0.0 for k in values}

    low, high = min(present.values()), max(present.values())
    if high == low:
        return {k: (1.0 if k in present else 0.0) for k in values}

    normalised = {}
    for key, value in values.items():
        if value is None:
            normalised[key] = 0.0
        elif higher_is_better:
            normalised[key] = (value - low) / (high - low)
        else:
            normalised[key] = (high - value) / (high - low)
    return normalised


def _eval_rows(
    evaluation: dict[str, Any] | None, document_id: str | None
) -> list[dict[str, Any]]:
    rows = (evaluation or {}).get("rows") or []
    if document_id is None:
        return rows
    return [r for r in rows if r.get("document_id") == document_id]


def _judge_error_count(
    evaluation: dict[str, Any] | None, judge_name: str, document_id: str | None
) -> int:
    """Lignes où l'appel au juge a échoué (timeout, erreur réseau...) plutôt que noté.

    `mlflow.genai.evaluate` avale ces échecs dans un `<scorer>/error_message` sans
    valeur associée : sans ce compte, la ligne ressort comme une note absente (`—`),
    indiscernable d'un cas où le juge n'a simplement pas été sollicité.
    """
    error_key = f"{judge_name}/error_message"
    return sum(1 for row in _eval_rows(evaluation, document_id) if row.get(error_key))


def _judge_excluded(evaluation: dict[str, Any] | None, document_id: str | None) -> bool:
    """Le juge a-t-il échoué deux fois de suite (premier essai + retry) sur une ligne ?

    Posé par `orchestration.run_eval._retry_judge_failures` sur la ligne elle-même
    (`row["judge_excluded"]`) plutôt que déduit d'un `error_message` isolé : un premier
    échec suivi d'un retry réussi ne doit pas compter, seul un échec confirmé compte.
    """
    return any(row.get("judge_excluded") for row in _eval_rows(evaluation, document_id))


def _metric_from_rows(rows: list[dict[str, Any]], scorer_name: str) -> float | None:
    """Moyenne d'un scorer sur les lignes d'évaluation, pour un document ciblé."""
    values: list[float] = []
    for row in rows:
        value: float | None = None
        for key, raw in row.items():
            if not isinstance(raw, (int, float)):
                continue
            if key == f"{scorer_name}/value" or key == scorer_name:
                value = float(raw)
                break
        if value is None:
            for key, raw in row.items():
                if (
                    isinstance(raw, (int, float))
                    and scorer_name in str(key)
                    and str(key).endswith("/value")
                ):
                    value = float(raw)
                    break
        if value is not None:
            values.append(value)
    return statistics.mean(values) if values else None


def task_table(task: Task, document_id: str | None = None) -> list[dict[str, Any]]:
    """Une ligne par modèle candidat, avec métriques brutes et score composite."""
    judge_name = judges_for(task)[0].name
    rows: list[dict[str, Any]] = []

    for model in MODELS_BY_TASK[task]:
        evaluation = _load_evaluation(task, model)
        generation = _generation_stats(task, model, document_id=document_id)

        if document_id is not None:
            eval_rows = _eval_rows(evaluation, document_id)
            judge_raw = _metric_from_rows(eval_rows, judge_name)
            code = _code_scores_from_rows(task, eval_rows)
            judge_pass = _judge_pass_rate(evaluation, judge_name, document_id)
        else:
            metrics = (evaluation or {}).get("metrics", {})
            judge_raw = _metric(metrics, judge_name)
            code = _code_scores(task, metrics)
            judge_pass = _judge_pass_rate(evaluation, judge_name)

        code_mean = statistics.mean(code.values()) if code else None
        judge_norm = _normalise_judge(judge_raw)
        quality = _blend(judge_norm, code_mean)
        judge_errors = _judge_error_count(evaluation, judge_name, document_id)

        # Un modèle n'est jamais éligible à "recommandé" (mais reste affiché avec ses
        # vraies métriques) dans deux cas : le juge a échoué deux fois de suite dessus
        # (premier essai + retry, voir run_eval._retry_judge_failures), ou les scorers
        # déterministes le placent sous CODE_SCORE_FLOOR malgré une bonne note de juge —
        # cas observé sur doc_8dcc7fc6/traduction, où une sortie quasi vide avait note 5
        # du juge (point mort d'omission) mais 35 % de scorers code. Voir config.py.
        ineligible = _judge_excluded(evaluation, document_id) or (
            code_mean is not None and code_mean < CODE_SCORE_FLOOR
        )

        rows.append(
            {
                "model": model,
                "judge_score": judge_raw,
                "judge_pass_rate": judge_pass,
                "code_scores": code,
                "code_mean": code_mean,
                "quality": quality,
                "n_calls": generation["n_calls"],
                "n_errors": generation["n_errors"] + judge_errors,
                "judge_errors": judge_errors,
                "ineligible": ineligible,
                "latency_mean_s": generation["latency_mean_s"],
                "tokens_per_second": generation["tokens_per_second"],
                "vram_gb": generation["vram_gb"],
                "total_gb": generation["total_gb"],
                "gpu_fraction": generation["gpu_fraction"],
                "caveat": BENCHMARK_CAVEATS.get(model),
            }
        )

    _add_composite(rows)
    # Les candidats éligibles d'abord (triés par composite), les exclus ensuite : un
    # modèle exclu ne peut jamais devenir rows[0], donc jamais "recommandé" — sauf si
    # tous les candidats de la tâche le sont, auquel cas le rapport doit encore afficher
    # un ordre plutôt que planter.
    rows.sort(key=lambda r: (r["ineligible"], -r["composite"]))
    return rows


def _code_scores_from_rows(task: Task, rows: list[dict[str, Any]]) -> dict[str, float]:
    from evaluation import CODE_SCORERS

    scores: dict[str, float] = {}
    for scorer in CODE_SCORERS[task]:
        value = _metric_from_rows(rows, scorer.name)
        if value is not None:
            scores[scorer.name] = value
    return scores


def _blend(judge_norm: float | None, code_mean: float | None) -> float | None:
    if judge_norm is None and code_mean is None:
        return None
    if judge_norm is None:
        return code_mean
    if code_mean is None:
        return judge_norm
    return QUALITY_MIX["judge"] * judge_norm + QUALITY_MIX["code"] * code_mean


def _judge_pass_rate(
    evaluation: dict[str, Any] | None,
    judge_name: str,
    document_id: str | None = None,
) -> float | None:
    """Part des sorties notées au-dessus du seuil de réussite.

    Conservé en plus de la note moyenne parce que le spec le demande, mais la note
    moyenne reste le signal principal : sur une dizaine de documents, un pass-rate
    n'a pas assez de granularité pour départager deux modèles.
    """
    values = [
        v
        for row in _eval_rows(evaluation, document_id)
        for key, v in row.items()
        if judge_name in str(key) and isinstance(v, (int, float))
    ]
    if not values:
        return None
    return sum(v >= JUDGE_PASS_THRESHOLD for v in values) / len(values)


def _add_composite(rows: list[dict[str, Any]]) -> None:
    quality = _min_max({r["model"]: r["quality"] for r in rows}, higher_is_better=True)
    latency = _min_max(
        {r["model"]: r["latency_mean_s"] for r in rows}, higher_is_better=False
    )
    memory = _min_max({r["model"]: r["total_gb"] for r in rows}, higher_is_better=False)

    for row in rows:
        model = row["model"]
        row["composite"] = (
            COMPOSITE_WEIGHTS["quality"] * quality[model]
            + COMPOSITE_WEIGHTS["latency"] * latency[model]
            + COMPOSITE_WEIGHTS["memory"] * memory[model]
        )


# --------------------------------------------------------------------------- #
# Meilleure sortie concrète
# --------------------------------------------------------------------------- #


def best_output(
    task: Task, model: str, document_id: str | None = None
) -> dict[str, Any] | None:
    """Meilleure sortie réellement produite par un modèle sur une tâche.

    Le spec insiste sur ce point : des métriques agrégées ne disent pas si la sortie
    est utilisable en production. On sélectionne par note de juge, en départageant
    par la latence.
    """
    evaluation = _load_evaluation(task, model)
    judge_name = judges_for(task)[0].name

    scored: list[tuple[float, str, str | None]] = []
    for row in _eval_rows(evaluation, document_id):
        values = [
            v
            for key, v in row.items()
            if judge_name in str(key) and isinstance(v, (int, float))
        ]
        if values and row.get("document_id"):
            scored.append((max(values), row["document_id"], row.get("question")))

    candidates = [
        r
        for r in store.load_all(task)
        if r["model"] == model
        and not r.get("error")
        and not r.get("smoke")
        and (document_id is None or r.get("document_id") == document_id)
    ]
    if not candidates:
        return None

    if scored:
        scored.sort(key=lambda s: -s[0])
        best_score, best_doc, best_question = scored[0]
        # Pour Q&A, plusieurs lignes partagent le même document_id (une par
        # question) : sans filtrer aussi sur la question, `next()` retombe sur
        # le premier candidat dans l'ordre de `store.load_all` (alphabétique
        # par item_id), qui peut être une question différente de celle qui a
        # réellement obtenu `best_score` — score et sortie affichés désaccordés.
        match = next(
            (
                c
                for c in candidates
                if c["document_id"] == best_doc
                and (best_question is None or c.get("question") == best_question)
            ),
            candidates[0],
        )
        return {**match, "judge_score": best_score}

    # Sans note de juge exploitable, on montre au moins une sortie non vide.
    return {**min(candidates, key=lambda c: c["latency_s"]), "judge_score": None}


# --------------------------------------------------------------------------- #
# Rendu Markdown
# --------------------------------------------------------------------------- #

_TASK_LABELS = {
    Task.EXTRACTION: "Extraction (JSON structuré)",
    Task.SUMMARY: "Résumé exécutif",
    Task.TRANSLATION: "Traduction",
    Task.QA: "Question-Réponse",
}


def _fmt(value: Any, spec: str = "", fallback: str = "—") -> str:
    if value is None:
        return fallback
    return format(value, spec) if spec else str(value)


# --------------------------------------------------------------------------- #
# Formatage des nombres et montants
#
# Les modèles écrivent les montants à l'anglaise (`TND 2,300,000`) ou à la
# française selon leur humeur et la langue du document. Le rapport les réécrit
# tous dans une seule convention : espace insécable pour les milliers, virgule
# décimale, code devise après le nombre. La transformation est purement
# typographique — aucune valeur n'est modifiée.
# --------------------------------------------------------------------------- #

NBSP = "\u00a0"

#: Marqueur du modèle recommandé. Écrit en clair pour rester lisible dans le
#: Markdown ; `reporting.pdf` le transforme en pastille colorée.
RECOMMENDED_MARK = "**recommandé**"

#: Nombre à séparateurs de milliers anglais, précédé ou suivi d'un code ISO 4217.
#: Les abréviations du type « 2.3M » ne correspondent pas et restent intactes.
_MONEY_IN_TEXT = re.compile(
    r"\b(?:(?P<pre>[A-Z]{3})\s*)?"
    r"(?P<number>\d{1,3}(?:,\d{3})+(?:\.\d+)?)"
    r"(?:\s*(?P<post>[A-Z]{3})\b)?"
)

_PERCENT_IN_TEXT = re.compile(r"(?<![\d.,])(?P<number>\d+(?:\.\d+)?)\s*%")


def _number(value: float, decimals: int) -> str:
    """Nombre à la française : milliers en espace insécable, virgule décimale."""
    integer, _, fraction = f"{value:,.{decimals}f}".partition(".")
    integer = integer.replace(",", NBSP)
    return f"{integer},{fraction}" if fraction else integer


def _decimals_for(value: float) -> int:
    """Décimales réellement portées par la valeur, jusqu'à quatre.

    Un taux de 17.9 doit ressortir « 17,9 » : le compléter en « 17,90 » afficherait
    une précision que le document ne donne pas.
    """
    value = float(value)
    for decimals in range(5):
        if abs(round(value, decimals) - value) < 1e-9:
            return decimals
    return 4


def money(value: float | None, currency: str | None = None) -> str:
    """Montant formaté, code devise en suffixe.

    Contrairement aux taux, un montant non entier est cadré sur deux décimales :
    c'est la convention comptable, et « 1 500,5 TND » se lit mal.
    """
    if value is None:
        return "—"
    decimals = 0 if float(value).is_integer() else max(2, _decimals_for(value))
    text = _number(float(value), decimals)
    return f"{text}{NBSP}{currency}" if currency else text


def percent(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{_number(float(value), _decimals_for(value))}{NBSP}%"


def normalise_numbers(text: str) -> str:
    """Uniformise montants et pourcentages dans un texte libre de modèle."""

    def money_repl(match: re.Match[str]) -> str:
        raw = match.group("number").replace(",", "")
        currency = match.group("pre") or match.group("post")
        return money(float(raw), currency)

    def percent_repl(match: re.Match[str]) -> str:
        return percent(float(match.group("number")))

    return _PERCENT_IN_TEXT.sub(percent_repl, _MONEY_IN_TEXT.sub(money_repl, text))


def _render_task(task: Task, document_id: str | None = None) -> str:
    rows = task_table(task, document_id=document_id)
    judge_name = judges_for(task)[0].name

    lines = [f"## {_TASK_LABELS[task]}", ""]

    if not any(r["n_calls"] for r in rows):
        lines += ["_Aucune sortie générée pour cette tâche._", ""]
        return "\n".join(lines)

    lines += [
        f"| Modèle | Note juge ({judge_name}, /{JUDGE_SCALE_MAX}) | Pass-rate juge "
        "| Scorers code | Latence moy. | Débit | Mémoire | % GPU | Appels | Erreurs "
        "| Score composite |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]

    for row in rows:
        lines.append(
            "| {model} | {judge} | {pass_rate} | {code} | {latency} | {tps} "
            "| {mem} | {gpu} | {calls} | {errors} | **{composite}** |".format(
                model=f"`{row['model']}`"
                + (
                    f" {RECOMMENDED_MARK}"
                    if row is rows[0] and not row["ineligible"]
                    else ""
                )
                + (" ⚠ *exclu de la recommandation*" if row["ineligible"] else ""),
                judge=_fmt(row["judge_score"], ".2f")
                if row["judge_score"] is not None or not row["judge_errors"]
                else f"erreur ({row['judge_errors']})",
                pass_rate=_fmt(
                    row["judge_pass_rate"] and row["judge_pass_rate"] * 100, ".0f"
                )
                + (" %" if row["judge_pass_rate"] is not None else ""),
                code=_fmt(row["code_mean"] and row["code_mean"] * 100, ".0f")
                + (" %" if row["code_mean"] is not None else ""),
                latency=_fmt(row["latency_mean_s"], ".1f")
                + (" s" if row["latency_mean_s"] is not None else ""),
                tps=_fmt(row["tokens_per_second"], ".1f")
                + (" tok/s" if row["tokens_per_second"] is not None else ""),
                mem=_fmt(row["total_gb"], ".1f")
                + (" Go" if row["total_gb"] is not None else ""),
                gpu=_fmt(row["gpu_fraction"] and row["gpu_fraction"] * 100, ".0f")
                + (" %" if row["gpu_fraction"] is not None else ""),
                calls=row["n_calls"],
                errors=row["n_errors"],
                composite=_fmt(row["composite"], ".3f"),
            )
        )

    lines.append("")

    # Les notes sur la taille/quantification d'un modèle (BENCHMARK_CAVEATS,
    # ex. translategemma/mannix en traduction) ne sont plus affichées, sur
    # demande explicite de l'utilisateur (2026-08-20) — seules les raisons
    # d'exclusion de la recommandation (juge en échec, scorers sous le
    # plancher) restent, plus opérationnelles.
    excluded_reasons = [
        f"- `{r['model']}` : exclu de la recommandation — "
        + (
            "le juge a échoué deux fois de suite (premier essai + retry)"
            if r["judge_errors"]
            else f"scorers code à {r['code_mean']:.0%}, sous le plancher requis "
            f"({CODE_SCORE_FLOOR:.0%}) pour être recommandé malgré la note de juge"
        )
        for r in rows
        if r["ineligible"]
    ]
    if excluded_reasons:
        lines += ["**Réserves sur la comparabilité**", "", *excluded_reasons, ""]

    winner = rows[0]
    if winner["ineligible"]:
        lines += [
            f"> Aucun modèle recommandable sur cette tâche : tous les candidats sont "
            f"exclus (voir réserves ci-dessus). `{winner['model']}` a le meilleur score "
            f"composite ({winner['composite']:.3f}) mais n'est pas mis en avant.",
            "",
        ]
    else:
        lines += [
            f"> {RECOMMENDED_MARK} · `{winner['model']}` — score composite "
            f"{winner['composite']:.3f}.",
            "",
        ]

    lines += _render_best_output(task, winner["model"], document_id=document_id)
    return "\n".join(lines)


def _qa_all_outputs(model: str, document_id: str) -> list[dict[str, Any]]:
    """Les réponses d'un modèle aux 4 questions d'un document, chacune avec sa
    propre note de juge.

    Contrairement aux 3 autres tâches, Q&A produit plusieurs sorties
    indépendantes par (modèle, document) — une par question, même
    `document_id`. N'en montrer qu'une (l'ancien comportement de
    `best_output`, hérité des tâches à sortie unique) cachait 3/4 du travail
    réellement généré et évalué, d'où la confusion signalée par l'utilisateur
    (tableau affichant 4 appels, section "meilleure sortie" n'en montrant
    qu'une).
    """
    judge_name = judges_for(Task.QA)[0].name
    scores: dict[str, float] = {}
    for row in _eval_rows(_load_evaluation(Task.QA, model), document_id):
        values = [
            v
            for key, v in row.items()
            if judge_name in str(key) and isinstance(v, (int, float))
        ]
        if values and row.get("question"):
            scores[row["question"]] = max(values)

    candidates = [
        r
        for r in store.load_all(Task.QA)
        if r["model"] == model
        and not r.get("error")
        and not r.get("smoke")
        and r.get("document_id") == document_id
    ]
    return sorted(
        ({**c, "judge_score": scores.get(c.get("question"))} for c in candidates),
        key=lambda c: c.get("question") or "",
    )


def _render_best_output(
    task: Task, model: str, document_id: str | None = None
) -> list[str]:
    # Rapport par document (le cas courant, y compris le site) : les 4
    # questions, pas une seule. Le rapport combiné (`--all`, document_id=None)
    # garde l'ancien comportement à une seule sortie — il n'y a pas de notion
    # de "toutes les questions" quand on agrège tout le corpus.
    if task is Task.QA and document_id is not None:
        outputs = _qa_all_outputs(model, document_id)
        if not outputs:
            return ["_Aucune sortie exploitable à montrer pour le modèle recommandé._", ""]
        lines = [f"### Réponses de `{model}` aux {len(outputs)} question(s)", ""]
        for o in outputs:
            lines.append(f"**Q. {o.get('question') or '?'}**")
            lines.append("")
            lines += _fenced(normalise_numbers((o.get("output") or "").strip()), "markdown")
            lines.append("")
        return lines

    best = best_output(task, model, document_id=document_id)
    if not best:
        return ["_Aucune sortie exploitable à montrer pour le modèle recommandé._", ""]

    header = (
        f"### Meilleure sortie de `{model}` "
        f"(document `{best['document_id']}`, {best.get('source_lang', '?')})"
    )
    meta = f"_Note du juge : {_fmt(best.get('judge_score'), '.0f')}_"
    if best.get("question"):
        meta += f" · _Question : {best['question']}_"

    if task is Task.EXTRACTION:
        body = _render_extraction(best)
    else:
        body = _fenced(normalise_numbers(best["output"].strip()), "markdown")

    return [header, "", meta, "", *body, ""]


def _fenced(text: str, info: str) -> list[str]:
    """Bloc clôturé, avec assez de backticks pour contenir ceux du modèle.

    Le `.md` garde ainsi la sortie mot pour mot, sans qu'elle puisse casser la
    structure du rapport. C'est le PDF qui la met en forme, en la rendant une
    seconde fois.
    """
    longest = max((len(run) for run in re.findall(r"`+", text)), default=0)
    fence = "`" * max(3, longest + 1)
    return [f"{fence}{info}", text, fence]


def _extraction_value(raw: Any) -> str:
    if raw is None or raw == "":
        return "—"
    if raw == NOT_FOUND_MARKER:
        return "_non trouvé_"
    return normalise_numbers(str(raw))


def _render_extraction(best: dict[str, Any]) -> list[str]:
    """Extraction rendue en tableau champ/valeur, clauses en liste à puces.

    Le JSON brut reste la sortie évaluée, mais il est illisible en PDF. On le
    reproduit tel quel uniquement quand il n'a pas pu être analysé, cas où sa forme
    exacte est justement l'information intéressante.
    """
    parsed = (best.get("metadata") or {}).get("parsed")
    if not isinstance(parsed, dict):
        return [
            "_JSON non exploitable : sortie brute reproduite telle quelle._",
            "",
            "```json",
            best["output"].strip(),
            "```",
        ]

    currency = parsed.get("devise")
    if currency in (None, "", NOT_FOUND_MARKER):
        currency = None

    montant = parsed.get("montant")
    taux = parsed.get("taux")

    lines = [
        "| Champ | Valeur |",
        "|---|---|",
        f"| Client | {_extraction_value(parsed.get('client'))} |",
        f"| Montant | {money(montant, currency) if isinstance(montant, (int, float)) else _extraction_value(montant)} |",
        f"| Taux | {percent(taux) if isinstance(taux, (int, float)) else _extraction_value(taux)} |",
        f"| Durée | {_extraction_value(parsed.get('duree'))} |",
        f"| Date | {_extraction_value(parsed.get('date'))} |",
        "",
    ]

    clauses = parsed.get("clauses_cles") or []
    lines.append("**Clauses clés**")
    lines.append("")

    if not clauses:
        lines += ["_Aucune clause clé extraite._", ""]
        return lines

    for clause in clauses:
        if not isinstance(clause, dict):
            lines.append(f"- {normalise_numbers(str(clause))}")
            continue
        titre = str(clause.get("titre", "")).strip() or "Clause"
        contenu = normalise_numbers(str(clause.get("contenu", "")).strip())
        lines.append(f"- **{titre}** — {contenu}" if contenu else f"- **{titre}**")

    lines.append("")
    return lines


def analysed_documents() -> list[str]:
    """Documents couverts par au moins une sortie d'agent réelle, hors smoke test."""
    found = {
        record["document_id"]
        for task in Task
        for record in store.load_all(task)
        if not record.get("smoke")
    }
    return sorted(found)


def stem_for(documents: list[str]) -> str:
    """Nom de fichier, sans extension, pour un rapport couvrant `documents`.

    Un rapport portant sur un seul document prend son nom. Le rapport global
    (`--all`) garde le nom générique `rapport`.
    """
    return f"{documents[0]}_rapport" if len(documents) == 1 else "rapport"


def report_stem() -> str:
    return stem_for(analysed_documents())


def _load_metrics_evaluation(task: Task, model: str) -> dict[str, Any] | None:
    path = METRICS_EVALUATION_DIR / task.value / f"{store.slug(model)}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text("utf-8"))


def _render_metrics_task(task: Task, document_id: str | None = None) -> str:
    """Tableau du pipeline `metrics` pour une tâche : pas de composite ni de
    "recommandé" — ces métriques n'ont pas été calibrées, contrairement au juge
    principal (voir l'avertissement en tête de `build_metrics_report`). Les modèles
    restent dans l'ordre de `config.MODELS_BY_TASK`, sans classement implicite.
    """
    names = METRIC_NAMES.get(task, ())
    lines = [f"## {_TASK_LABELS[task]}", ""]
    if not names:
        lines += ["_Aucune métrique du pipeline `metrics` définie pour cette tâche._", ""]
        return "\n".join(lines)

    header = ["Modèle", *[_METRIC_LABELS.get(n, n) for n in names], "Appels"]
    lines += [
        "| " + " | ".join(header) + " |",
        "|" + "|".join(["---"] * len(header)) + "|",
    ]

    any_data = False
    for model in MODELS_BY_TASK[task]:
        evaluation = _load_metrics_evaluation(task, model)
        generation = _generation_stats(task, model, document_id=document_id)
        eval_rows = _eval_rows(evaluation, document_id)

        if not eval_rows and not generation["n_calls"]:
            continue
        any_data = True

        cells = [f"`{model}`"]
        for metric_name in names:
            value = _metric_from_rows(eval_rows, metric_name)
            cells.append(_fmt(value, ".2f"))
        cells.append(str(generation["n_calls"]))
        lines.append("| " + " | ".join(cells) + " |")

    if not any_data:
        lines = [f"## {_TASK_LABELS[task]}", "", "_Aucune sortie évaluée en mode `metrics`._", ""]
        return "\n".join(lines)

    lines.append("")
    lines += _render_metrics_best_output(task, document_id=document_id)
    return "\n".join(lines)


def _language_ok(task: Task, model: str, document_id: str) -> bool:
    """La sortie de ce (modèle, document) a-t-elle passé le contrôle de langue ?

    Vient du pipeline `judge` (`results/evaluation/`, scorer déterministe
    `language_conformity`) : le pipeline `metrics` n'a pas de scorer à lui, et
    `faithfulness` peut valider une sortie dans la mauvaise langue cible sans le
    voir — observé concrètement le 2026-08-11 sur `qwen2.5:7b`/traduction/alpha
    (sortie en anglais au lieu du français attendu, `faithfulness`=5.0, justifiée
    par le juge comme « the provided French translation is accurate »).
    Absence de mesure (pas encore évalué en mode `judge`, scorer non concluant)
    n'est pas traitée comme un échec : seul un `False` explicite exclut.
    """
    evaluation = _load_evaluation(task, model)
    return not any(
        row.get("language_conformity/value") is False
        for row in _eval_rows(evaluation, document_id)
    )


def metrics_best_output(
    task: Task, document_id: str | None = None
) -> dict[str, Any] | None:
    """Sortie du modèle le mieux noté sur la tâche, tous critères de
    `_RANKING_METRICS[task]` confondus, parmi celles qui passent le contrôle de
    langue déterministe (`_language_ok`, voir sa docstring).

    Le pipeline `metrics` n'étant pas calibré, ce choix n'est pas une
    recommandation (voir `_render_metrics_task`, qui n'affiche aucun classement
    dans son tableau) — juste un exemple concret à montrer. Quand plusieurs
    métriques de classement existent, chacune est normalisée sur [0, 1] parmi les
    candidats **du même document** (comme le score composite du pipeline `judge`,
    `_min_max`) avant d'être moyennée — une moyenne de valeurs brutes mélangerait
    des échelles incomparables (`faithfulness` 1-5 contre `ari_grade_level` 8-18,
    par exemple, où ARI écraserait tout le reste). Le détail par métrique (valeurs
    brutes, pas normalisées) reste affiché dans le rapport.
    """
    ranking_metrics = _RANKING_METRICS.get(task, ())
    if not ranking_metrics:
        return None

    candidates_raw: list[tuple[str, str, str | None, dict[str, float]]] = []
    # Modèles écartés uniquement par le filtre de langue déterministe (voir
    # _language_ok) — collectés pour que le rapport explique pourquoi un
    # candidat à égalité numérique n'a pas été choisi, plutôt que de le
    # laisser paraître arbitraire (cas réel trouvé le 2026-08-19 : llama3.1:8b
    # à égalité 5.00/5.00 avec le modèle choisi sur `northbridge`, mais une de
    # ses 4 réponses Q&A était en français au lieu de l'anglais attendu).
    language_excluded: dict[str, list[str]] = {}
    for model in MODELS_BY_TASK[task]:
        evaluation = _load_metrics_evaluation(task, model)
        for row in _eval_rows(evaluation, document_id):
            doc_id = row.get("document_id")
            if not doc_id:
                continue
            if not _language_ok(task, model, doc_id):
                language_excluded.setdefault(doc_id, [])
                if model not in language_excluded[doc_id]:
                    language_excluded[doc_id].append(model)
                continue
            per_metric: dict[str, float] = {}
            for name in ranking_metrics:
                for key, raw in row.items():
                    if (
                        isinstance(raw, (int, float))
                        and name in str(key)
                        and str(key).endswith("/value")
                    ):
                        per_metric[name] = float(raw)
                        break
            if per_metric:
                candidates_raw.append((model, doc_id, row.get("question"), per_metric))

    if not candidates_raw:
        return None

    # Normalisation par document : mélanger les échelles ARI de deux documents
    # différents (rapport combiné) n'aurait pas de sens non plus.
    grouped: dict[str, dict[str, list[float]]] = {}
    for _, doc_id, _, per_metric in candidates_raw:
        bucket = grouped.setdefault(doc_id, {})
        for name, value in per_metric.items():
            bucket.setdefault(name, []).append(value)

    def normalised(doc_id: str, name: str, value: float) -> float:
        values = grouped[doc_id][name]
        low, high = min(values), max(values)
        if high == low:
            return 1.0
        if _HIGHER_IS_BETTER.get(name, True):
            return (value - low) / (high - low)
        return (high - value) / (high - low)

    # Le MODÈLE gagnant doit être choisi par sa MOYENNE (comme le tableau de
    # score, `_metric_from_rows`), pas par sa meilleure ligne individuelle.
    # Bug réel trouvé le 2026-08-19 (signalé par l'utilisateur) : Q&A a 4
    # lignes par modèle (une par question) — choisir par ligne isolée pouvait
    # élire un modèle globalement moins bon juste parce qu'une de ses 4
    # réponses avait ponctuellement le score le plus haut, en contradiction
    # avec le tableau affiché juste au-dessus (qui, lui, moyenne correctement).
    # Pour les 3 autres tâches (1 seule ligne par modèle), moyenne d'une seule
    # valeur = cette valeur : aucun changement de comportement pour elles.
    per_row: dict[tuple[str, str], list[tuple[float, str | None, dict[str, float]]]] = {}
    for model, doc_id, question, per_metric in candidates_raw:
        composite = statistics.mean(
            normalised(doc_id, name, value) for name, value in per_metric.items()
        )
        per_row.setdefault((model, doc_id), []).append((composite, question, per_metric))

    best_key: tuple[str, str] | None = None
    best_avg = -1.0
    for key, rows in per_row.items():
        avg = statistics.mean(c for c, _, _ in rows)
        if avg > best_avg:
            best_avg, best_key = avg, key

    model, doc_id = best_key
    rows = per_row[best_key]
    # Ligne représentative pour l'affichage à sortie unique (tâches hors Q&A,
    # ou rapport combiné `--all` où Q&A garde ce même rendu) : la meilleure
    # ligne individuelle DU MODÈLE DÉJÀ ÉLU, jamais celle qui a servi à élire
    # un autre modèle.
    score, question, per_metric = max(rows, key=lambda r: r[0])

    candidates = [
        r
        for r in store.load_all(task)
        if r["model"] == model
        and not r.get("error")
        and not r.get("smoke")
        and r.get("document_id") == doc_id
        and (question is None or r.get("question") == question)
    ]
    if not candidates:
        return None
    return {
        **candidates[0],
        "metric_score": score,
        "metric_breakdown": per_metric,
        "question": question,
        "language_excluded": language_excluded.get(doc_id, []),
    }


def _qa_all_metrics_outputs(model: str, document_id: str) -> list[dict[str, Any]]:
    """Pendant metrics de `_qa_all_outputs` : les réponses d'un modèle aux 4
    questions d'un document, chacune avec son propre détail de métriques.
    """
    ranking_metrics = _RANKING_METRICS.get(Task.QA, ())
    breakdowns: dict[str, dict[str, float]] = {}
    for row in _eval_rows(_load_metrics_evaluation(Task.QA, model), document_id):
        question = row.get("question")
        if not question:
            continue
        per_metric: dict[str, float] = {}
        for name in ranking_metrics:
            for key, raw in row.items():
                if (
                    isinstance(raw, (int, float))
                    and name in str(key)
                    and str(key).endswith("/value")
                ):
                    per_metric[name] = float(raw)
                    break
        if per_metric:
            breakdowns[question] = per_metric

    candidates = [
        r
        for r in store.load_all(Task.QA)
        if r["model"] == model
        and not r.get("error")
        and not r.get("smoke")
        and r.get("document_id") == document_id
    ]
    return sorted(
        (
            {**c, "metric_breakdown": breakdowns.get(c.get("question"), {})}
            for c in candidates
        ),
        key=lambda c: c.get("question") or "",
    )


def _render_metrics_best_output(task: Task, document_id: str | None = None) -> list[str]:
    # Même raisonnement que _render_best_output (pipeline judge) : Q&A a 4
    # sorties indépendantes par (modèle, document), pas une seule — les montrer
    # toutes plutôt qu'une seule choisie arbitrairement.
    if task is Task.QA and document_id is not None:
        best = metrics_best_output(task, document_id=document_id)
        if not best:
            return ["_Aucune sortie évaluée en mode `metrics` à montrer pour cette tâche._", ""]
        outputs = _qa_all_metrics_outputs(best["model"], document_id)
        if not outputs:
            return ["_Aucune sortie évaluée en mode `metrics` à montrer pour cette tâche._", ""]
        lines = [f"### Réponses de `{best['model']}` aux {len(outputs)} question(s)", ""]
        for o in outputs:
            lines.append(f"**Q. {o.get('question') or '?'}**")
            lines.append("")
            lines += _fenced(normalise_numbers((o.get("output") or "").strip()), "markdown")
            lines.append("")
        return lines

    best = metrics_best_output(task, document_id=document_id)
    if not best:
        return ["_Aucune sortie évaluée en mode `metrics` à montrer pour cette tâche._", ""]

    breakdown = " · ".join(
        f"{_METRIC_LABELS.get(name, name)} : {value:.2f}"
        for name, value in best["metric_breakdown"].items()
    )
    header = f"### Sortie de `{best['model']}` la mieux notée"
    meta = f"_{breakdown}_"
    if best.get("question"):
        meta += f" · _Question : {best['question']}_"

    lines = [header, "", meta, ""]
    body = _fenced(normalise_numbers(best["output"].strip()), "markdown")
    return [*lines, *body, ""]


def _render_metrics_extraction(document_id: str | None = None) -> str:
    """Section extraction du rapport `metrics`.

    Aucune métrique `mlflow.evaluate()` n'est définie pour cette tâche (voir
    `evaluation/legacy_metrics.py`) — pas de tableau de score ici, seulement la
    sortie du meilleur modèle selon le pipeline `judge`, pour que ce rapport reste
    lisible seul sans devoir ouvrir l'autre.
    """
    lines = [f"## {_TASK_LABELS[Task.EXTRACTION]}", ""]
    rows = task_table(Task.EXTRACTION, document_id=document_id)
    if not any(r["n_calls"] for r in rows):
        lines += ["_Aucune sortie générée pour cette tâche._", ""]
        return "\n".join(lines)

    lines += [
        "_Pas de métrique du pipeline `metrics` pour l'extraction — sortie du "
        "meilleur modèle selon le pipeline `judge` (voir le rapport `judge` pour "
        "le tableau de score complet)._",
        "",
    ]
    winner = rows[0]
    lines += _render_best_output(Task.EXTRACTION, winner["model"], document_id=document_id)
    return "\n".join(lines)


def build_metrics_report(document_id: str | None = None) -> str:
    """Rapport du second pipeline (`mlflow.evaluate()`, voir
    `evaluation/legacy_metrics.py`). Jamais fusionné avec `build_report()` — le mode
    utilisé doit rester explicite pour quiconque lit le rapport.
    """
    parts = [
        "# Rapport comparatif — métriques complémentaires (mode `metrics`)",
        "",
        "---",
        "",
    ]

    for task in Task:
        if task is Task.EXTRACTION:
            parts.append(_render_metrics_extraction(document_id=document_id))
        elif task in METRIC_NAMES:
            parts.append(_render_metrics_task(task, document_id=document_id))
        else:
            continue
        parts.append("---")
        parts.append("")

    return "\n".join(parts)


def build_report(document_id: str | None = None) -> str:
    parts = [
        "# Rapport comparatif — SLM locaux sur propositions financières",
        "",
        "Score composite = "
        + " + ".join(f"{w:.0%} {k}" for k, w in COMPOSITE_WEIGHTS.items())
        + f", l'axe qualité étant lui-même réparti entre le juge ({QUALITY_MIX['judge']:.0%}) "
        f"et les scorers déterministes ({QUALITY_MIX['code']:.0%}). "
        "Chaque composante est normalisée sur [0, 1] parmi les candidats de la tâche ; "
        "latence et mémoire sont inversées, plus bas valant mieux.",
        "",
        "---",
        "",
    ]

    for task in Task:
        parts.append(_render_task(task, document_id=document_id))
        parts.append("---")
        parts.append("")

    return "\n".join(parts)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--all",
        action="store_true",
        help="Écrire un seul rapport global combiné (rapport.md).",
    )
    parser.add_argument("--out", help="Chemin de sortie (avec --all, ou document unique).")
    parser.add_argument(
        "--eval-mode",
        choices=[m.value for m in EvalMode],
        default=EvalMode.JUDGE.value,
        help="Pipeline à lire : 'judge' (défaut, calibré) ou 'metrics' (voir "
        "evaluation/legacy_metrics.py, non calibré). Jamais fusionnés : fichiers de "
        "sortie distincts (suffixe _metrics en mode metrics).",
    )
    args = parser.parse_args(argv)

    logging_setup.setup()

    from pathlib import Path

    mode = EvalMode(args.eval_mode)
    build = build_metrics_report if mode is EvalMode.METRICS else build_report
    suffix = "_metrics" if mode is EvalMode.METRICS else ""

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    documents = analysed_documents()

    if args.all:
        out = Path(args.out) if args.out else REPORTS_DIR / f"rapport{suffix}.md"
        out.write_text(build(), "utf-8")
        log.info("Rapport global écrit dans %s", out)
        return 0

    if not documents:
        log.error(
            "Aucune sortie d'agent dans results/generation/. "
            "Lancez d'abord run_agents."
        )
        return 1

    if args.out:
        # Un seul fichier demandé explicitement : document unique, sinon global.
        out = Path(args.out)
        doc_id = documents[0] if len(documents) == 1 else None
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(build(document_id=doc_id), "utf-8")
        log.info("Rapport écrit dans %s", out)
        return 0

    for document_id in documents:
        out = REPORTS_DIR / f"{document_id}_rapport{suffix}.md"
        out.write_text(build(document_id=document_id), "utf-8")
        log.info("Rapport écrit dans %s", out)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
