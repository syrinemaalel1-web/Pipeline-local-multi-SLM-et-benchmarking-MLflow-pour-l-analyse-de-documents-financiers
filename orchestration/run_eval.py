"""Phase 2 — évaluation : scorers et juges sur les sorties déjà générées.

    python -m orchestration.run_eval
    python -m orchestration.run_eval --task extraction --no-judges
    python -m orchestration.run_eval --doc proposition_alpha  # un document précis
    python -m orchestration.run_eval --eval-mode metrics --task qa  # second pipeline

L'évaluation lit `results/generation/` plutôt que de rappeler les agents. Deux raisons :
on ne rejoue pas 12 à 20 h d'inférence à chaque ajustement d'un scorer, et un plantage
du juge ne détruit pas les sorties déjà produites.

C'est un écart assumé au `predict_fn` du spec (§3.5) : `mlflow.genai.evaluate` accepte
aussi bien un jeu `inputs`/`outputs` déjà matérialisé, et cette forme est la seule
praticable au vu du coût d'inférence sur cette machine.

Politique d'échec : un appel en erreur ou une sortie vide reste dans le jeu évalué et
compte comme échec. L'exclure avantagerait les modèles instables.

Deux pipelines d'évaluation, sélectionnés par `--eval-mode` (`config.EvalMode`), jamais
fusionnés dans un même run : `judge` (par défaut, `mlflow.genai.evaluate()`, calibré) et
`metrics` (`mlflow.evaluate()`, API dépréciée, non calibré — voir
`evaluation/legacy_metrics.py`). Persistés dans deux dossiers séparés
(`results/evaluation/` et `results/evaluation_metrics/`) ; `reporting.report
--eval-mode` choisit lequel lire, jamais les deux à la fois.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd

from common import logging_setup
from config import (
    MODELS_BY_TASK,
    NOT_FOUND_MARKER,
    REQUEST_TIMEOUT_S,
    RESULTS_DIR,
    TRANSLATION_DIRECTION,
    EvalMode,
    Lang,
    Task,
)
from evaluation import scorers_for
from evaluation.judges import judges_for
from evaluation.legacy_metrics import metrics_for
from orchestration import corpus, mlflow_setup, store

log = logging.getLogger("run_eval")

EVALUATION_DIR = RESULTS_DIR / "evaluation"
METRICS_EVALUATION_DIR = RESULTS_DIR / "evaluation_metrics"


def _expected_output_lang(task: Task, source_lang: Lang) -> Lang:
    """Langue de sortie imposée par la règle de langue du projet.

    Elle suit la langue du document, sauf pour la traduction dont c'est justement le
    rôle d'en changer.
    """
    if task is Task.TRANSLATION:
        return TRANSLATION_DIRECTION[source_lang]
    return source_lang


def _resolve_documents(wanted: list[str], known: list[str]) -> set[str]:
    """Identifiants complets désignés par un identifiant ou un préfixe.

    Même règle que `run_agents --doc` : un préfixe suffit, puisque les identifiants se
    terminent par un hash, mais un préfixe ambigu lève plutôt que de trancher seul.
    """
    resolved: set[str] = set()

    for prefix in wanted:
        matches = [d for d in known if d.startswith(prefix)]
        if not matches:
            available = "\n  ".join(sorted(known))
            raise SystemExit(
                f"Aucun document ne correspond à '{prefix}'. Disponibles :\n  {available}"
            )
        if len(matches) > 1:
            raise SystemExit(f"'{prefix}' est ambigu : {', '.join(sorted(matches))}")
        resolved.add(matches[0])

    return resolved


def build_dataset(
    task: Task, model: str, document_ids: set[str] | None = None
) -> pd.DataFrame:
    documents = {d["document_id"]: d for d in corpus.load_documents()}
    questions = (
        {q["id"]: q for q in corpus.load_qa_questions()} if task is Task.QA else {}
    )

    rows: list[dict[str, Any]] = []

    for record in store.load_all(task):
        if record["model"] != model:
            continue
        if document_ids is not None and record["document_id"] not in document_ids:
            continue

        document = documents.get(record["document_id"])
        if document is None:
            continue

        source_lang = Lang(document["lang"])
        inputs: dict[str, Any] = {
            "task": task.value,
            "document_id": record["document_id"],
            "document": document["text"],
            "lang": source_lang.value,
            "expected_output_lang": _expected_output_lang(task, source_lang).value,
        }

        if task is Task.QA:
            question = questions.get(record.get("item_id") or "")
            inputs["question"] = record.get("question") or (
                question["question"] if question else ""
            )
            inputs["expect_abstention"] = (
                question.get("expect_abstention") if question else None
            )

        rows.append({"inputs": inputs, "outputs": record["output"]})

    return pd.DataFrame(rows)


def _already_evaluated_documents(
    task: Task, model: str, eval_dir: Path | None = None
) -> set[str]:
    """`document_id` déjà présents dans le fichier persisté de ce (tâche, modèle)."""
    path = (eval_dir or EVALUATION_DIR) / task.value / f"{store.slug(model)}.json"
    if not path.exists():
        return set()
    try:
        rows = json.loads(path.read_text("utf-8")).get("rows") or []
    except json.JSONDecodeError:
        return set()
    return {row["document_id"] for row in rows if row.get("document_id")}


def evaluate_task(
    task: Task,
    *,
    models: list[str] | None = None,
    include_judges: bool = True,
    document_ids: set[str] | None = None,
    force: bool = False,
) -> None:
    scorers = scorers_for(task, include_judges=include_judges)

    for model in models or MODELS_BY_TASK[task]:
        # Marqueur consommé par api/jobs.py::progress (barre de progression
        # réelle côté site) — un par (tâche, modèle), que le travail soit
        # réellement effectué ou déjà couvert. Total annoncé une fois dans
        # main(), avant la boucle sur les tâches.
        log.info("PROGRESS_STEP")

        # Reprise : si le process a été interrompu après N modèles sur M pour cette
        # tâche, relancer la même commande ne doit pas rejouer le juge sur les
        # modèles déjà persistés — seul --force le permet explicitement. Ignoré avec
        # --no-judges, dont le but est justement de rejouer les scorers gratuits.
        if not force and include_judges:
            wanted = document_ids or {d["document_id"] for d in corpus.load_documents()}
            already = _already_evaluated_documents(task, model)
            if wanted <= already:
                log.info(
                    "%s / %s : déjà évalué pour %s, rien à refaire (--force pour "
                    "rejouer quand même).",
                    task.value,
                    model,
                    ", ".join(sorted(wanted)) if document_ids else "tout le corpus",
                )
                continue

        data = build_dataset(task, model, document_ids)

        if data.empty:
            log.warning(
                "%s / %s : aucune sortie générée, évaluation ignorée. Lancez d'abord "
                "`python -m orchestration.run_agents --task %s`.",
                task.value,
                model,
                task.value,
            )
            continue

        empty = sum(1 for o in data["outputs"] if not str(o).strip())
        if empty:
            log.warning(
                "%s / %s : %d sortie(s) vide(s) conservée(s) et comptée(s) en échec.",
                task.value,
                model,
                empty,
            )

        log.info("%s / %s : %d ligne(s) à évaluer", task.value, model, len(data))

        with mlflow.start_run(run_name=f"eval:{task.value}:{model}"):
            mlflow.set_tags({"task": task.value, "model_name": model, "phase": "eval"})
            result = mlflow.genai.evaluate(data=data, scorers=scorers)
            log.info("  métriques : %s", result.metrics)

        rows = _tagged_rows(result, data)
        if include_judges:
            _retry_judge_failures(task, model, rows, data)

        _persist(task, model, result.run_id, rows)


def _tagged_rows(result, data: pd.DataFrame) -> list[dict[str, Any]]:
    """Lignes de la table d'évaluation, rattachées à leur document/question source.

    Le nom de colonne des identifiants varie selon le format de table MLflow ; on le
    rattache nous-mêmes, dans l'ordre du jeu évalué, pour que le rapport puisse remonter
    de la meilleure note à la sortie concrète correspondante.
    """
    rows: list[dict[str, Any]] = []
    for name, table in (result.tables or {}).items():
        if hasattr(table, "to_dict"):
            rows = table.to_dict(orient="records")
            log.debug("table d'évaluation retenue : %s", name)
            break

    for row, (_, source) in zip(rows, data.iterrows()):
        row["document_id"] = source["inputs"]["document_id"]
        row["question"] = source["inputs"].get("question")

    return rows


def _retry_judge_failures(
    task: Task, model: str, rows: list[dict[str, Any]], data: pd.DataFrame
) -> None:
    """Un échec de juge (timeout, erreur réseau...) est retenté une fois, en place.

    `mlflow.genai.evaluate` avale un échec de scorer dans un `<juge>/error_message`
    sans `/value` associé (cf. §3 CLAUDE.md — observé sur des documents proches du
    budget de contexte, où l'appel expire avant que le juge ait fini de répondre). Un
    seul essai raté ne doit pas suffire à écarter un modèle : on retente une fois de
    plus avant de considérer l'échec définitif. Seul un second échec pose
    `row["judge_excluded"] = True`, qui exclut le modèle de la recommandation dans
    `reporting.report` sans effacer ses autres métriques (scorers code, latence...).
    """
    (judge,) = judges_for(task)
    error_key = f"{judge.name}/error_message"
    value_key = f"{judge.name}/value"
    rationale_key = f"{judge.name}/rationale"

    for row, (_, source) in zip(rows, data.iterrows()):
        if not row.get(error_key):
            continue

        log.warning(
            "%s / %s : le juge a échoué sur %s (%s), nouvel essai...",
            task.value,
            model,
            row.get("document_id"),
            row[error_key],
        )
        try:
            feedback = judge(inputs=source["inputs"], outputs=source["outputs"])
        except Exception as exc:  # noqa: BLE001 - second échec, on le trace et on continue
            row[error_key] = f"{row[error_key]} | retry: {exc}"
            row["judge_excluded"] = True
            log.warning(
                "  échec persistant après retry, %s exclu de la recommandation sur "
                "%s : %s",
                model,
                task.value,
                exc,
            )
            continue

        row[value_key] = feedback.value
        row[rationale_key] = feedback.rationale
        row.pop(error_key, None)
        log.info("  réussi au second essai : %s", feedback.value)


def _build_metrics_dataset(
    task: Task, model: str, document_ids: set[str] | None = None
) -> pd.DataFrame:
    """Jeu de données pour le pipeline `metrics` (colonnes à plat, pas de dict `inputs`
    imbriqué comme dans `build_dataset`) : c'est ce que `mlflow.evaluate()` attend via
    `predictions=` et `evaluator_config={"col_mapping": ...}`.
    """
    documents = {d["document_id"]: d for d in corpus.load_documents()}
    questions = (
        {q["id"]: q for q in corpus.load_qa_questions()} if task is Task.QA else {}
    )

    rows: list[dict[str, Any]] = []
    for record in store.load_all(task):
        if record["model"] != model:
            continue
        if document_ids is not None and record["document_id"] not in document_ids:
            continue
        document = documents.get(record["document_id"])
        if document is None:
            continue

        row: dict[str, Any] = {
            "document_id": record["document_id"],
            "outputs": record["output"],
            # `context` posé pour toutes les tâches : `faithfulness` en a besoin
            # partout où elle est utilisée (traduction, résumé, qa — voir
            # evaluation/legacy_metrics.py), pas seulement pour qa.
            "context": document["text"],
            # `inputs` structurellement requis par `faithfulness` même si sa propre
            # doc dit l'ignorer ("please ignore the provided input entirely when
            # scoring faithfulness") : mlflow.evaluate() valide sa présence avant de
            # calculer quoi que ce soit. Le document source sert de valeur par défaut
            # pour traduction/résumé (jamais lue) ; qa la remplace par la question,
            # qu'answer_relevance lit réellement.
            "inputs": document["text"],
        }
        if task is Task.QA:
            question = questions.get(record.get("item_id") or "")
            question_text = record.get("question") or (
                question["question"] if question else ""
            )
            row["inputs"] = question_text
            row["question"] = question_text

        rows.append(row)

    return pd.DataFrame(rows)


def _tagged_metrics_rows(result, data: pd.DataFrame) -> list[dict[str, Any]]:
    """Comme `_tagged_rows`, pour la table renvoyée par l'ancienne API `mlflow.evaluate()`.

    `result.tables` contient plusieurs tables (au moins `eval_results_table`, une ligne
    par ligne évaluée, et une table de métadonnées des métriques, une ligne par
    métrique) — prendre "la première table du dict" comme `_tagged_rows` le fait pour
    `mlflow.genai.evaluate()` attrape la mauvaise en pratique (observé : une table à 2
    lignes — une par métrique — au lieu des 4 lignes réellement évaluées). On identifie
    donc la bonne table par son contenu (colonne `outputs`, qu'on a nous-mêmes posée
    dans `data`), pas par son nom ni sa position dans le dict.

    Ses colonnes de score se nomment `<métrique>/v1/score` et `.../justification`,
    pas `.../value` et `.../rationale` comme le pipeline `judge` — renommées ici pour
    que `_aggregate_metrics` et `reporting.report` restent communs aux deux pipelines
    sans code dupliqué.
    """
    rows: list[dict[str, Any]] = []
    for name, table in (result.tables or {}).items():
        if not hasattr(table, "to_dict"):
            continue
        candidate = table.to_dict(orient="records")
        if candidate and "outputs" in candidate[0]:
            rows = candidate
            log.debug("table de métriques retenue : %s (%d ligne(s))", name, len(rows))
            break
        log.debug("table ignorée (pas de colonne 'outputs') : %s", name)

    for row in rows:
        for key in list(row):
            if key.endswith("/score"):
                row[f"{key[: -len('/score')]}/value"] = row.pop(key)
            elif key.endswith("/justification"):
                row[f"{key[: -len('/justification')]}/rationale"] = row.pop(key)

    for row, (_, source) in zip(rows, data.iterrows()):
        row["document_id"] = source["document_id"]
        row["question"] = source.get("question")

    return rows


def evaluate_metrics_task(
    task: Task,
    *,
    models: list[str] | None = None,
    document_ids: set[str] | None = None,
    force: bool = False,
) -> None:
    """Second pipeline d'évaluation (`mlflow.evaluate()`) — voir
    `evaluation/legacy_metrics.py` pour ce qu'il couvre et pourquoi il est séparé.
    """
    # `mlflow.metrics.genai` (faithfulness/answer_relevance) utilise sa propre requête
    # HTTP vers le juge, indépendante de litellm et de JUDGE_INFERENCE_PARAMS, avec un
    # timeout par défaut de 60s (MLFLOW_GENAI_EVAL_LLM_TIMEOUT) — beaucoup trop court
    # pour granite3.3:8b sur ce corpus (appels mesurés à 90-130s). Trouvé en testant le
    # vrai pipeline sur un document réel : sans ce réglage, faithfulness échoue sur
    # 100 % des lignes et answer_relevance sur environ la moitié. `os.environ.setdefault`
    # pour ne jamais écraser un réglage explicite de l'utilisateur.
    os.environ.setdefault("MLFLOW_GENAI_EVAL_LLM_TIMEOUT", str(REQUEST_TIMEOUT_S))

    metrics = metrics_for(task)
    if not metrics:
        log.info(
            "%s [metrics] : aucune métrique définie pour cette tâche, rien à faire.",
            task.value,
        )
        return

    for model in models or MODELS_BY_TASK[task]:
        log.info("PROGRESS_STEP")

        if not force:
            wanted = document_ids or {d["document_id"] for d in corpus.load_documents()}
            already = _already_evaluated_documents(task, model, METRICS_EVALUATION_DIR)
            if wanted <= already:
                log.info(
                    "%s / %s [metrics] : déjà évalué pour %s, rien à refaire "
                    "(--force pour rejouer quand même).",
                    task.value,
                    model,
                    ", ".join(sorted(wanted)) if document_ids else "tout le corpus",
                )
                continue

        data = _build_metrics_dataset(task, model, document_ids)
        if data.empty:
            log.warning(
                "%s / %s [metrics] : aucune sortie générée, évaluation ignorée.",
                task.value,
                model,
            )
            continue

        col_mapping: dict[str, str] = {}
        if "inputs" in data.columns:
            col_mapping["input"] = "inputs"
        if "context" in data.columns:
            col_mapping["context"] = "context"

        log.info(
            "%s / %s [metrics] : %d ligne(s) à évaluer (%s)",
            task.value,
            model,
            len(data),
            ", ".join(m.name for m in metrics),
        )

        with mlflow.start_run(run_name=f"metrics:{task.value}:{model}"):
            mlflow.set_tags({"task": task.value, "model_name": model, "phase": "metrics"})
            result = mlflow.evaluate(
                data=data,
                predictions="outputs",
                extra_metrics=metrics,
                evaluators="default",
                evaluator_config={"col_mapping": col_mapping} if col_mapping else None,
            )
            log.info("  métriques : %s", result.metrics)

        rows = _tagged_metrics_rows(result, data)
        if task is Task.QA:
            _zero_score_on_abstention(rows)
        _persist(task, model, result.run_id, rows, METRICS_EVALUATION_DIR)


def _zero_score_on_abstention(rows: list[dict[str, Any]]) -> None:
    """Force `answer_relevance`/`faithfulness` à 0 quand la sortie Q&A est une
    abstention (`NOT_FOUND_MARKER`), sur demande explicite de l'utilisateur
    (2026-08-19) : ces deux métriques du pipeline `metrics` n'ont aucune notion
    d'abstention correcte/incorrecte (contrairement au juge `qa_groundedness`
    du pipeline `judge`, qui la récompense à raison) — elles notent une
    non-réponse comme une réponse partielle (ex. observé : faithfulness=3/5,
    answer_relevance=1/5 sur une abstention pourtant légitime), ce qui n'a pas
    de sens sur une échelle censée mesurer la qualité d'une réponse : il n'y a
    pas de réponse à mesurer.

    Ne modifie QUE ces deux métriques génériques du pipeline `metrics` — le
    juge `qa_groundedness` du pipeline `judge` reste intouché, il évalue
    correctement le bien-fondé de l'abstention (5/5 si l'info est
    effectivement absente, 1/5 si une valeur est inventée) et le forcer à 0
    casserait précisément le comportement anti-hallucination que ce juge a été
    conçu pour récompenser.
    """
    for row in rows:
        output = str(row.get("outputs") or "").strip()
        if not output.startswith(NOT_FOUND_MARKER):
            continue
        for key in ("answer_relevance/v1/value", "faithfulness/value"):
            if key not in row or row[key] is None:
                continue
            note_key = key.replace("/value", "/rationale")
            row[key] = 0
            row[note_key] = (
                f"{row.get(note_key) or ''}\n\n[Note forcée à 0 : la sortie est une "
                "abstention (aucune réponse à évaluer), pas une réponse partielle.]"
            ).strip()


def _aggregate_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Reconstruit le `metrics` agrégé (moyennes `*/mean`) à partir des lignes.

    Ne pas se fier au `result.metrics` renvoyé par `mlflow.genai.evaluate()` pour ce
    champ une fois les documents fusionnés : il ne couvre que le sous-ensemble évalué
    par ce seul appel, jamais les lignes d'un `--doc` précédent repêchées de disque.
    """
    values: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        for key, raw in row.items():
            if isinstance(key, str) and key.endswith("/value") and isinstance(raw, (int, float)):
                values[f"{key[: -len('/value')]}/mean"].append(float(raw))
    return {key: statistics.mean(vals) for key, vals in values.items()}


def _persist(
    task: Task,
    model: str,
    run_id: str,
    rows: list[dict[str, Any]],
    eval_dir: Path | None = None,
) -> None:
    """Recopie les résultats d'évaluation en JSON à côté des sorties d'agents.

    Le rapport final lit ces fichiers plutôt que d'interroger MLflow : il reste
    reproductible même si le store MLflow est purgé, et se régénère sans dépendre du
    schéma interne des tables d'évaluation.

    Fusionné par `document_id`, pas écrasé : un flux document par document (`--doc`)
    doit accumuler les documents évalués au fil des appels. Un document réévalué
    remplace ses propres lignes (la dernière évaluation gagne), jamais celles des
    autres documents déjà présents dans le fichier.
    """
    path = (eval_dir or EVALUATION_DIR) / task.value / f"{store.slug(model)}.json"
    path.parent.mkdir(parents=True, exist_ok=True)

    evaluated_documents = {row["document_id"] for row in rows}

    existing_rows: list[dict[str, Any]] = []
    if path.exists():
        try:
            existing_rows = json.loads(path.read_text("utf-8")).get("rows") or []
        except json.JSONDecodeError:
            log.warning("%s illisible, remplacé plutôt que fusionné.", path)

    kept = [r for r in existing_rows if r.get("document_id") not in evaluated_documents]
    merged_rows = kept + rows

    payload = {
        "task": task.value,
        "model": model,
        "run_id": run_id,
        "metrics": _aggregate_metrics(merged_rows),
        "rows": merged_rows,
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str), "utf-8"
    )


#: Registre mode -> gestionnaire. Ajouter un mode (ex. un futur "deployment") se limite
#: à un nouveau membre `EvalMode` (config.py) + une entrée ici : le reste (CLI, boucle
#: de dispatch) est déjà générique.
_EVAL_MODE_HANDLERS = {
    EvalMode.JUDGE: lambda task, **kw: evaluate_task(
        task,
        models=kw["models"],
        include_judges=not kw["no_judges"],
        document_ids=kw["document_ids"],
        force=kw["force"],
    ),
    EvalMode.METRICS: lambda task, **kw: evaluate_metrics_task(
        task,
        models=kw["models"],
        document_ids=kw["document_ids"],
        force=kw["force"],
    ),
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=[t.value for t in Task])
    parser.add_argument("--model", action="append", dest="models")
    parser.add_argument(
        "--doc",
        action="append",
        dest="doc_ids",
        help="Restreindre à un document, par identifiant ou préfixe. Les lignes de ce "
        "document remplacent leur ancienne version dans results/evaluation/ ; les "
        "autres documents déjà évalués sont conservés.",
    )
    parser.add_argument(
        "--eval-mode",
        choices=[m.value for m in EvalMode],
        default=EvalMode.JUDGE.value,
        help="Pipeline d'évaluation : 'judge' (défaut, mlflow.genai.evaluate, calibré) "
        "ou 'metrics' (mlflow.evaluate, non calibré — voir evaluation/legacy_metrics.py). "
        "Jamais fusionnés ; persistés dans des dossiers séparés.",
    )
    parser.add_argument(
        "--no-judges",
        action="store_true",
        help="Ne rejouer que les scorers déterministes, qui sont gratuits. Ignoré en "
        "--eval-mode metrics (n'a pas de notion juge/code à part).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Réévaluer même les (tâche, modèle, document) déjà persistés — sinon "
        "un modèle déjà couvert pour tous les documents demandés est sauté, pour "
        "reprendre après une interruption sans rejouer le juge sur du déjà-fait.",
    )
    args = parser.parse_args(argv)

    logging_setup.setup()
    mlflow_setup.init()

    document_ids = None
    if args.doc_ids:
        known = [d["document_id"] for d in corpus.load_documents()]
        document_ids = _resolve_documents(args.doc_ids, known)
        log.warning(
            "Évaluation restreinte à %s. Leurs lignes remplaceront leur ancienne "
            "version dans results/evaluation/ ; les autres documents déjà évalués "
            "sont conservés.",
            ", ".join(sorted(document_ids)),
        )

    mode = EvalMode(args.eval_mode)
    if mode is EvalMode.METRICS and args.no_judges:
        log.warning("--no-judges est ignoré en --eval-mode metrics.")

    tasks = [Task(args.task)] if args.task else list(Task)
    # `metrics` saute une tâche entière (avant la boucle sur les modèles, voir
    # evaluate_metrics_task) quand `metrics_for(task)` est vide — le total doit
    # l'exclure aussi, sinon la barre de progression du site ne finit jamais.
    if mode is EvalMode.METRICS:
        total = sum(len(args.models or MODELS_BY_TASK[t]) for t in tasks if metrics_for(t))
    else:
        total = sum(len(args.models or MODELS_BY_TASK[t]) for t in tasks)
    log.info("PROGRESS_TOTAL=%d", total)

    handler = _EVAL_MODE_HANDLERS[mode]
    for task in tasks:
        handler(
            task,
            models=args.models,
            no_judges=args.no_judges,
            document_ids=document_ids,
            force=args.force,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
