"""Calibration humaine du juge LLM.

Un juge de 8 B qui note d'autres modèles de 8 B n'a aucune autorité intrinsèque. Tant
que ses notes n'ont pas été confrontées à un jugement humain, le rapport final ne
mesure qu'une chose : ce que `aya:8b` pense des sorties. Ce module produit le chiffre
manquant — le taux d'accord entre le juge et un annotateur humain — sur un échantillon
d'une vingtaine de sorties.

Trois commandes, dans cet ordre :

    python -m evaluation.calibration sample      # tire l'échantillon
    python -m evaluation.calibration annotate    # vous annotez, en aveugle
    python -m evaluation.calibration score       # calcule la fiabilité

Deux partis pris méthodologiques :

- **Annotation en aveugle.** Les notes du juge sont écrites dans un fichier distinct de
  celui que l'annotateur remplit, et ne sont jamais affichées pendant l'annotation.
  Montrer la note du juge d'abord suffirait à y aligner l'humain, et le taux d'accord
  mesuré ne vaudrait plus rien.
- **Comparaison binaire.** L'humain répond bon / mauvais, et la note du juge est
  binarisée au seuil `JUDGE_PASS_THRESHOLD`. C'est exactement l'axe qu'utilise le
  pass-rate du rapport ; demander à un humain de reproduire une échelle de 1 à 5
  ajouterait du bruit sans rien mesurer de plus.

L'échantillon est stratifié par tâche et par verdict du juge. Un tirage uniforme serait
trompeur : le juge met 5/5 aux trois quarts des lignes, un échantillon uniforme ne
contiendrait presque aucun cas négatif et l'accord mesuré serait flatteur par
construction.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import textwrap
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from config import JUDGE_MODEL, JUDGE_PASS_THRESHOLD, RESULTS_DIR, Task
from orchestration import corpus, store

CALIBRATION_DIR = RESULTS_DIR / "calibration"

#: Ce que l'annotateur lit et remplit. Ne contient aucune note du juge.
SAMPLE_PATH = CALIBRATION_DIR / "echantillon.jsonl"

#: Notes du juge, mises de côté jusqu'au dépouillement.
JUDGE_PATH = CALIBRATION_DIR / "juge.jsonl"

#: Dossier de lecture : sorties et documents sources, pour annoter hors terminal.
DOSSIER_PATH = CALIBRATION_DIR / "dossier.md"

#: Résultat publiable.
REPORT_PATH = CALIBRATION_DIR / "fiabilite.md"

EVALUATION_DIR = RESULTS_DIR / "evaluation"

#: Suffixe des scorers LLM dans les tables d'évaluation persistées.
_JUDGE_METRICS = ("fidelity", "faithfulness", "groundedness")

_YES = {"o", "oui", "y", "yes", "b", "bon"}
_NO = {"n", "non", "no", "m", "mauvais"}


# --------------------------------------------------------------------------- #
# Lecture des notes du juge déjà produites
# --------------------------------------------------------------------------- #


def _judge_column(row: dict[str, Any]) -> str | None:
    for key in row:
        if key.endswith("/value") and any(m in key for m in _JUDGE_METRICS):
            return key
    return None


def _key(task: str, model: str, document_id: str, question: str | None) -> str:
    return "|".join([task, model, document_id, question or ""])


def judge_scores(evaluation_dir: Path | None = None) -> dict[str, dict[str, Any]]:
    """Notes du juge indexées par (tâche, modèle, document, question)."""
    scores: dict[str, dict[str, Any]] = {}

    for path in sorted((evaluation_dir or EVALUATION_DIR).rglob("*.json")):
        payload = json.loads(path.read_text("utf-8"))
        task, model = payload["task"], payload["model"]

        for row in payload.get("rows") or []:
            column = _judge_column(row)
            if column is None or row.get(column) is None:
                continue

            key = _key(task, model, row.get("document_id", ""), row.get("question"))
            scores[key] = {
                "score": int(row[column]),
                "rationale": str(row.get(column.replace("/value", "/rationale")) or ""),
            }

    return scores


# --------------------------------------------------------------------------- #
# Échantillonnage
# --------------------------------------------------------------------------- #


@dataclass
class Item:
    id: str
    task: str
    model: str
    document_id: str
    question: str | None
    output: str
    judge_score: int = 0
    judge_rationale: str = ""
    human: bool | None = None
    human_note: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


def _candidates(
    evaluation_dir: Path | None = None, document: str | None = None
) -> list[Item]:
    """Sorties générées pour lesquelles une note de juge existe.

    ``document`` restreint le tirage aux sorties d'un seul ``document_id`` — utile pour
    calibrer sur un document à la fois plutôt que sur tout le corpus (voir README, section
    « document par document »).
    """
    scores = judge_scores(evaluation_dir)
    items: list[Item] = []

    for record in store.load_all():
        if document is not None and record["document_id"] != document:
            continue
        question = record.get("question")
        key = _key(
            record["task"], record["model"], record["document_id"], question
        )
        judged = scores.get(key)
        if judged is None:
            continue

        items.append(
            Item(
                id=f"{len(items):03d}",
                task=record["task"],
                model=record["model"],
                document_id=record["document_id"],
                question=question,
                output=record.get("output") or "",
                judge_score=judged["score"],
                judge_rationale=judged["rationale"],
            )
        )

    return items


def _stratified(items: list[Item], n: int, seed: int) -> list[Item]:
    """Tirage réparti entre tâches et entre verdicts du juge.

    Le tour de rôle passe d'une strate à l'autre plutôt que de vider la première :
    avec 76 % de notes maximales, tout autre schéma rendrait un échantillon
    presque exclusivement positif.
    """
    buckets: dict[tuple[str, bool], list[Item]] = defaultdict(list)
    for item in items:
        buckets[(item.task, item.judge_score >= JUDGE_PASS_THRESHOLD)].append(item)

    rng = random.Random(seed)
    for bucket in buckets.values():
        rng.shuffle(bucket)

    order = sorted(buckets)
    picked: list[Item] = []
    while len(picked) < n and any(buckets[k] for k in order):
        for key in order:
            if not buckets[key]:
                continue
            picked.append(buckets[key].pop())
            if len(picked) == n:
                break

    for rank, item in enumerate(picked):
        item.id = f"{rank:03d}"
    return picked


def _prioritized(items: list[Item], n: int, seed: int) -> list[Item]:
    """Tirage ciblé sur les cas les plus informatifs d'un seul document.

    Utilisé quand ``--document`` restreint la calibration à un document : avec un seul
    document, la stratification par tâche a moins de sens que de garantir la présence
    des cas qui ont le plus de chances de trahir le juge — les notes qui ne sont pas déjà
    au plafond, et les questions d'abstention (IBAN), là où l'ancien juge se trompait
    systématiquement (docs/fiabilite-juge.md §3).
    """
    rng = random.Random(seed)
    shuffled = items[:]
    rng.shuffle(shuffled)

    def priority(item: Item) -> tuple[int, int]:
        non_five = 0 if item.judge_score != 5 else 1
        is_abstention_question = item.question is not None and "iban" in item.question.lower()
        return (non_five, 0 if is_abstention_question else 1)

    picked = sorted(shuffled, key=priority)[:n]
    for rank, item in enumerate(picked):
        item.id = f"{rank:03d}"
    return picked


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(row, ensure_ascii=False) for row in rows]
    path.write_text("\n".join(lines) + "\n", "utf-8")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line) for line in path.read_text("utf-8").splitlines() if line.strip()
    ]


def _dossier(items: list[Item]) -> str:
    """Document de travail de l'annotateur : sorties, puis sources en annexe."""
    documents = {d["document_id"]: d for d in corpus.load_documents()}
    used = sorted({item.document_id for item in items})

    lines = [
        "# Dossier de calibration",
        "",
        f"{len(items)} sorties à juger. Pour chacune : la sortie est-elle **bonne sur "
        "le fond**, c'est-à-dire exacte au regard du document source ?",
        "",
        "Ne tenez pas compte de la mise en forme, de la longueur ni de la langue : ces "
        "points sont vérifiés par du code et ne font pas partie de ce que le juge est "
        "censé mesurer. Les documents sources sont en annexe, à la fin du fichier.",
        "",
        "Reportez ensuite vos verdicts avec `python -m evaluation.calibration annotate`.",
        "",
    ]

    for item in items:
        lines += [
            "---",
            "",
            f"## Sortie {item.id} — {item.task}",
            "",
            f"- Modèle : `{item.model}`",
            f"- Document : `{item.document_id}` (annexe plus bas)",
        ]
        if item.question:
            lines.append(f"- Question posée : **{item.question}**")
        lines += ["", "```", item.output.strip() or "(sortie vide)", "```", ""]

    lines += ["", "---", "", "# Annexe — documents sources", ""]
    for document_id in used:
        document = documents.get(document_id)
        lines += [
            f"## `{document_id}`",
            "",
            "```",
            (document or {}).get("text", "(document introuvable)").strip(),
            "```",
            "",
        ]

    return "\n".join(lines)


def cmd_sample(
    n: int,
    seed: int,
    document: str | None = None,
    evaluation_dir: Path | None = None,
) -> int:
    items = _candidates(evaluation_dir, document)
    if not items:
        where = evaluation_dir or EVALUATION_DIR
        print(
            f"Aucune sortie notée par le juge dans {where}"
            + (f" pour le document {document!r}." if document else ".")
            + " Lancez d'abord `python -m orchestration.run_eval`."
        )
        return 1

    picked = _prioritized(items, n, seed) if document else _stratified(items, n, seed)

    _write_jsonl(
        SAMPLE_PATH,
        [
            {
                "id": i.id,
                "task": i.task,
                "model": i.model,
                "document_id": i.document_id,
                "question": i.question,
                "output": i.output,
                "human": None,
                "human_note": "",
            }
            for i in picked
        ],
    )
    _write_jsonl(
        JUDGE_PATH,
        [
            {"id": i.id, "judge_score": i.judge_score, "judge_rationale": i.judge_rationale}
            for i in picked
        ],
    )
    DOSSIER_PATH.parent.mkdir(parents=True, exist_ok=True)
    DOSSIER_PATH.write_text(_dossier(picked), "utf-8")

    spread = defaultdict(int)
    for item in picked:
        spread[item.task] += 1

    print(f"{len(picked)} sorties tirées sur {len(items)} disponibles (seed={seed}).")
    for task, count in sorted(spread.items()):
        print(f"  {task:12s} {count}")
    print()
    print(f"À lire   : {DOSSIER_PATH}")
    print(f"À remplir: {SAMPLE_PATH}  (via `annotate`)")
    print(
        f"Les notes du juge attendent dans {JUDGE_PATH} : ne les ouvrez pas avant "
        "d'avoir annoté, l'accord mesuré n'aurait plus de valeur."
    )
    return 0


# --------------------------------------------------------------------------- #
# Annotation
# --------------------------------------------------------------------------- #


def cmd_annotate(show_full: bool, redo: bool) -> int:
    rows = _read_jsonl(SAMPLE_PATH)
    if not rows:
        print(f"{SAMPLE_PATH} est absent. Lancez d'abord `sample`.")
        return 1

    todo = [r for r in rows if redo or r.get("human") is None]
    if not todo:
        print("Tout est déjà annoté. `--redo` pour reprendre depuis le début.")
        return 0

    print(f"{len(todo)} sortie(s) à juger. La sortie est-elle exacte sur le FOND ?")
    print("  o = bonne   n = mauvaise   s = passer   q = quitter en sauvegardant")
    print(f"Le détail complet est dans {DOSSIER_PATH}.\n")

    for row in todo:
        print("=" * 78)
        print(f"[{row['id']}] {row['task']} — {row['model']}")
        print(f"document : {row['document_id']}")
        if row.get("question"):
            print(f"question : {row['question']}")
        print("-" * 78)

        output = (row.get("output") or "").strip() or "(sortie vide)"
        if not show_full and len(output) > 1200:
            output = output[:1200] + "\n[…] (tronqué, voir le dossier ou --full)"
        print(textwrap.indent(output, "  "))
        print("-" * 78)

        while True:
            answer = input("verdict [o/n/s/q] > ").strip().lower()
            if answer in _YES or answer in _NO:
                row["human"] = answer in _YES
                row["human_note"] = input("remarque (facultatif) > ").strip()
                break
            if answer == "s":
                break
            if answer == "q":
                _write_jsonl(SAMPLE_PATH, rows)
                done = sum(1 for r in rows if r.get("human") is not None)
                print(f"\nSauvegardé : {done}/{len(rows)} annotées.")
                return 0
            print("  réponse attendue : o, n, s ou q")
        print()

    _write_jsonl(SAMPLE_PATH, rows)
    done = sum(1 for r in rows if r.get("human") is not None)
    print(f"Sauvegardé : {done}/{len(rows)} annotées.")
    print("Dépouillement : `python -m evaluation.calibration score`")
    return 0


# --------------------------------------------------------------------------- #
# Dépouillement
# --------------------------------------------------------------------------- #


def _kappa(pairs: list[tuple[bool, bool]]) -> float | None:
    """Kappa de Cohen sur deux séries de verdicts binaires.

    Renvoie ``None`` quand l'accord attendu par hasard vaut 1 : les deux annotateurs
    ont alors donné le même verdict partout, et le kappa n'est pas défini.
    """
    n = len(pairs)
    if not n:
        return None

    agree = sum(1 for a, b in pairs if a == b) / n
    judge_good = sum(1 for a, _ in pairs if a) / n
    human_good = sum(1 for _, b in pairs if b) / n
    chance = judge_good * human_good + (1 - judge_good) * (1 - human_good)

    if math.isclose(chance, 1.0):
        return None
    return (agree - chance) / (1 - chance)


def _wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Intervalle de confiance à 95 % d'une proportion, méthode de Wilson.

    Sur 20 observations, l'intervalle normal sort de [0, 1] dès que le taux dépasse
    0,9 ; Wilson reste correct sur de petits effectifs, ce qui est exactement le cas
    ici et évite d'annoncer une fiabilité plus précise qu'elle ne l'est.
    """
    if not n:
        return (0.0, 0.0)

    p = successes / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z / denom * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    return (max(0.0, center - half), min(1.0, center + half))


def cmd_score() -> int:
    sample = _read_jsonl(SAMPLE_PATH)
    judged = {r["id"]: r for r in _read_jsonl(JUDGE_PATH)}

    if not sample:
        print(f"{SAMPLE_PATH} est absent. Lancez d'abord `sample`.")
        return 1

    annotated = [r for r in sample if r.get("human") is not None and r["id"] in judged]
    missing = len(sample) - len(annotated)

    if not annotated:
        print(
            f"Aucune annotation dans {SAMPLE_PATH}. La fiabilité du juge ne peut pas "
            "être calculée tant que les sorties n'ont pas été jugées à la main : "
            "lancez `python -m evaluation.calibration annotate`."
        )
        return 1

    pairs: list[tuple[bool, bool]] = []
    per_task: dict[str, list[tuple[bool, bool]]] = defaultdict(list)
    disagreements: list[dict[str, Any]] = []

    for row in annotated:
        judge_good = judged[row["id"]]["judge_score"] >= JUDGE_PASS_THRESHOLD
        human_good = bool(row["human"])
        pairs.append((judge_good, human_good))
        per_task[row["task"]].append((judge_good, human_good))

        if judge_good != human_good:
            disagreements.append(
                {
                    **row,
                    "judge_score": judged[row["id"]]["judge_score"],
                    "judge_rationale": judged[row["id"]]["judge_rationale"],
                }
            )

    n = len(pairs)
    agree = sum(1 for a, b in pairs if a == b)
    low, high = _wilson(agree, n)
    kappa = _kappa(pairs)

    false_good = sum(1 for a, b in pairs if a and not b)
    false_bad = sum(1 for a, b in pairs if not a and b)

    lines = [
        "# Fiabilité du juge",
        "",
        f"Juge évalué : `{JUDGE_MODEL}`. Une note est comptée « bonne » à partir de "
        f"{JUDGE_PASS_THRESHOLD}/5, seuil déjà utilisé par le pass-rate du rapport.",
        f"Annotation humaine en aveugle sur {n} sortie(s)"
        + (f", {missing} restant à annoter." if missing else "."),
        "",
        "## Résultat",
        "",
        "| Indicateur | Valeur |",
        "| --- | --- |",
        f"| Accord avec l'humain | **{agree / n:.0%}** ({agree}/{n}) |",
        f"| Intervalle de confiance à 95 % | {low:.0%} – {high:.0%} |",
        f"| Kappa de Cohen | {'non défini' if kappa is None else f'{kappa:.2f}'} |",
        f"| Faux « bon » (juge valide, humain rejette) | {false_good} |",
        f"| Faux « mauvais » (juge rejette, humain valide) | {false_bad} |",
        "",
    ]

    if kappa is None:
        lines += [
            "Le kappa n'est pas défini : l'un des deux annotateurs a rendu le même "
            "verdict sur toutes les lignes, il n'y a donc pas d'accord au-delà du "
            "hasard à mesurer.",
            "",
        ]

    lines += [
        f"L'intervalle reste large parce que l'échantillon est petit : {n} annotations "
        "situent la fiabilité à une vingtaine de points près. Le chiffre sert à écarter "
        "un juge franchement défaillant, pas à départager deux juges proches.",
        "",
        "## Par tâche",
        "",
        "| Tâche | Accord | Lignes |",
        "| --- | --- | --- |",
    ]
    for task, task_pairs in sorted(per_task.items()):
        ok = sum(1 for a, b in task_pairs if a == b)
        lines.append(f"| {task} | {ok / len(task_pairs):.0%} | {len(task_pairs)} |")

    lines += ["", "## Désaccords", ""]
    if not disagreements:
        lines.append("Aucun désaccord sur cet échantillon.")
    else:
        for row in disagreements:
            verdict = "bonne" if row["human"] else "mauvaise"
            lines += [
                f"### {row['id']} — {row['task']} / `{row['model']}`",
                "",
                f"- Juge : {row['judge_score']}/5. Humain : {verdict}.",
                f"- Motif du juge : {' '.join(row['judge_rationale'].split())[:400]}",
            ]
            if row.get("human_note"):
                lines.append(f"- Remarque de l'annotateur : {row['human_note']}")
            lines.append("")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines) + "\n", "utf-8")

    print(f"Accord juge / humain : {agree / n:.0%} ({agree}/{n})")
    print(f"Intervalle 95 %      : {low:.0%} – {high:.0%}")
    print(f"Kappa de Cohen       : {'non défini' if kappa is None else f'{kappa:.2f}'}")
    print(f"Faux « bon »         : {false_good}   faux « mauvais » : {false_bad}")
    if missing:
        print(f"({missing} sortie(s) encore non annotée(s).)")
    print(f"\nRapport écrit : {REPORT_PATH}")
    return 0


# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_sample = sub.add_parser("sample", help="Tirer l'échantillon à annoter.")
    p_sample.add_argument("-n", type=int, default=20, help="Taille de l'échantillon.")
    p_sample.add_argument("--seed", type=int, default=0, help="Graine du tirage.")
    p_sample.add_argument(
        "--document",
        "-d",
        help=(
            "Ne tirer que parmi les sorties de ce document_id. Bascule le tirage sur "
            "les cas prioritaires (notes non-5, questions IBAN) plutôt que la "
            "stratification par tâche, qui suppose plusieurs documents."
        ),
    )
    p_sample.add_argument(
        "--evaluation-dir",
        type=Path,
        default=None,
        help="Dossier de résultats d'évaluation à lire, à la place de results/evaluation.",
    )

    p_annotate = sub.add_parser("annotate", help="Annoter les sorties tirées.")
    p_annotate.add_argument(
        "--full", action="store_true", help="Afficher les sorties longues en entier."
    )
    p_annotate.add_argument(
        "--redo", action="store_true", help="Reprendre toutes les lignes, même annotées."
    )

    sub.add_parser("score", help="Calculer l'accord juge / humain.")

    args = parser.parse_args(argv)

    if args.command == "sample":
        return cmd_sample(args.n, args.seed, args.document, args.evaluation_dir)
    if args.command == "annotate":
        return cmd_annotate(args.full, args.redo)
    return cmd_score()


if __name__ == "__main__":
    raise SystemExit(main())
