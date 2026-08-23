"""Contrôle négatif du juge : sait-il repérer une faute qu'on y a mise exprès ?

    python -m evaluation.judge_probe
    python -m evaluation.judge_probe --judge-model mistral:7b

La calibration humaine mesure l'accord du juge avec un annotateur, mais elle coûte du
temps humain et ne dit pas *pourquoi* le juge se trompe. Ce module répond à une question
plus élémentaire, et entièrement automatique : si l'on prend une sortie correcte et
qu'on y injecte une faute de fond connue, le juge la voit-il ?

Un juge qui échoue ici n'évalue rien. Peu importe alors que ses notes soient stables ou
bien rédigées : elles ne portent aucune information sur l'exactitude, et le classement
des modèles qui en découle n'en porte pas davantage. C'est le test à passer avant de
dépenser la moindre heure d'annotation.

Chaque cas est construit à partir d'une vraie sortie du benchmark, pour que le juge
travaille sur le même matériau qu'en production. Deux mesures en sortent :

- la **sensibilité** : part des sorties corrompues notées sous le seuil de réussite,
  c'est-à-dire les fautes effectivement détectées ;
- la **spécificité** : part des sorties intactes notées au-dessus du seuil, qui vérifie
  que le juge ne se contente pas de tout rejeter.

Un juge utile doit réussir les deux. Les atteindre séparément est trivial : noter 1
partout donne une sensibilité parfaite, noter 5 partout une spécificité parfaite.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Callable

from common import logging_setup
from config import JUDGE_MODEL, JUDGE_PASS_THRESHOLD, Task
from evaluation.judges import judges_for
from orchestration import mlflow_setup, run_eval

log = logging.getLogger("judge_probe")

#: Montant absurde, hors de toute échelle plausible pour ces dossiers.
_ABSURD_AMOUNT = "99999999"

_NUMBER = re.compile(r"\b\d[\d\s\u00a0.,]{3,}\b")

#: Les 3 documents d\u00e9j\u00e0 g\u00e9n\u00e9r\u00e9s+\u00e9valu\u00e9s (un par langue) \u2014 voir CLAUDE.md \u00a74/\u00a75.
#: Un cas dont `Case.document_ids` cible ces 3 IDs est rejou\u00e9 une fois par document,
#: pour distinguer une faute syst\u00e9matique d'un artefact propre \u00e0 un seul document.
_PILOT_DOCUMENTS = (
    "proposition_alpha_construction_ar_23f37ed2",
    "proposal_northbridge_logistics_en_5d39011d",
    "proposition_thalassa_hotellerie_fr_76fc04af",
)


@dataclass
class Case:
    """Une sortie soumise au juge, intacte ou corrompue."""

    task: Task
    label: str
    corrupt: Callable[[str], str] | None
    #: Note attendue : au-dessus du seuil pour une sortie intacte, en dessous sinon.
    should_pass: bool
    #: Si renseigné, le cas est rejoué une fois par document au lieu d'une seule
    #: sortie arbitraire (voir _PILOT_DOCUMENTS) — ajouté le 2026-08-12 après qu'une
    #: substitution de montant ratée sur `translation_fidelity` (document alpha) a
    #: soulevé la question de savoir si le point mort est systématique ou local à ce
    #: document. Voir CLAUDE.md, Journal.
    document_ids: tuple[str, ...] | None = None

    def apply(self, output: str) -> str:
        return output if self.corrupt is None else self.corrupt(output)


# --------------------------------------------------------------------------- #
# Corruptions
# --------------------------------------------------------------------------- #


def _inflate_amount(output: str) -> str:
    """Remplace le premier nombre long par un montant absurde."""
    return _NUMBER.sub(_ABSURD_AMOUNT, output, count=1)


def _falsify_json_amount(output: str) -> str:
    """Porte tout champ de montant d'un JSON d'extraction à une valeur absurde."""
    try:
        payload = json.loads(output)
    except json.JSONDecodeError:
        return _inflate_amount(output)

    if not isinstance(payload, dict):
        return _inflate_amount(output)

    touched = False
    for key in payload:
        if any(word in key.lower() for word in ("montant", "amount", "total")):
            payload[key] = int(_ABSURD_AMOUNT)
            touched = True

    if not touched:
        return _inflate_amount(output)
    return json.dumps(payload, ensure_ascii=False)


def _add_invented_claim(output: str) -> str:
    """Ajoute une conclusion que le document ne contient pas."""
    return (
        f"{output}\n\nLe comité de crédit a définitivement rejeté ce dossier le "
        "3 mars 2024 et la garantie hypothécaire a été saisie."
    )


def _truncate_half(output: str) -> str:
    """Supprime la seconde moitié : simule une omission de clauses."""
    return output[: max(1, len(output) // 2)]


def _fabricate_answer(_: str) -> str:
    """Réponse inventée à une question dont la réponse est absente du document."""
    return (
        "L'IBAN du compte bénéficiaire est TN59 1000 6035 0000 1234 5678, "
        "domicilié à la Banque Centrale de Tunisie."
    )


CASES: list[Case] = [
    Case(Task.EXTRACTION, "extraction intacte", None, True),
    Case(Task.EXTRACTION, "montant remplacé par un montant absurde", _falsify_json_amount, False),
    Case(Task.SUMMARY, "résumé intact", None, True),
    Case(Task.SUMMARY, "conclusion inventée ajoutée", _add_invented_claim, False),
    Case(
        Task.SUMMARY, "montant faussé", _inflate_amount, False, document_ids=_PILOT_DOCUMENTS
    ),
    Case(Task.TRANSLATION, "traduction intacte", None, True),
    Case(Task.TRANSLATION, "seconde moitié supprimée", _truncate_half, False),
    Case(
        Task.TRANSLATION,
        "montant faussé",
        _inflate_amount,
        False,
        document_ids=_PILOT_DOCUMENTS,
    ),
    Case(Task.QA, "réponse ancrée intacte", None, True),
    Case(Task.QA, "réponse fabriquée sur une information absente", _fabricate_answer, False),
]


# --------------------------------------------------------------------------- #


def _first_row(
    task: Task, document_ids: set[str] | None, abstention: bool | None
) -> tuple[dict[str, Any], str] | None:
    """Première sortie non vide parmi `document_ids` (tout le corpus si `None`)."""
    for model in _models_with_outputs(task):
        data = run_eval.build_dataset(task, model, document_ids=document_ids)
        for _, row in data.iterrows():
            if not str(row["outputs"]).strip():
                continue
            if abstention is not None:
                if bool(row["inputs"].get("expect_abstention")) is not abstention:
                    continue
            return row["inputs"], str(row["outputs"])
    return None


def _source_row(
    task: Task, *, abstention: bool | None = None, document_id: str | None = None
) -> tuple[dict[str, Any], str]:
    """Sortie non vide du benchmark pour cette tâche.

    Sans `document_id` explicite, la recherche est restreinte aux 3 documents pilotes
    calibrés (`_PILOT_DOCUMENTS`) — pas « n'importe quel document avec une sortie »
    comme avant le 2026-08-12. Un document hors de cet ensemble (`doc_8dcc7fc6`, proche
    du plafond de contexte `NUM_CTX`) avait produit un artefact de juge — score et
    justification incohérents, probable débordement de contexte — pris pour un vrai
    résultat de contrôle négatif avant ce correctif (voir CLAUDE.md, Journal). Repli
    explicite sur tout le corpus, avec avertissement, seulement si aucun pilote n'a de
    sortie pour cette tâche.
    """
    if document_id:
        result = _first_row(task, {document_id}, abstention)
    else:
        result = _first_row(task, set(_PILOT_DOCUMENTS), abstention)
        if result is None:
            result = _first_row(task, None, abstention)
            if result is not None:
                log.warning(
                    "%s : aucun document pilote disponible, repli hors calibration "
                    "sur document_id=%s.",
                    task.value,
                    result[0].get("document_id"),
                )

    if result is None:
        scope = f" pour {document_id}" if document_id else ""
        raise LookupError(
            f"Aucune sortie exploitable pour {task.value}{scope}. Lancez d'abord "
            "`python -m orchestration.run_agents`."
        )
    return result


def _models_with_outputs(task: Task) -> list[str]:
    from orchestration import store

    return sorted({record["model"] for record in store.load_all(task)})


def run(model_uri: str | None = None) -> int:
    caught = passed = corrupted_total = clean_total = 0
    rows: list[tuple[str, str, str, int, bool]] = []
    #: Résultats groupés par (tâche, cas) pour les cas rejoués sur plusieurs
    #: documents — permet de dire si une faute ratée l'est partout ou seulement sur
    #: un document précis (voir Case.document_ids).
    by_case: dict[tuple[str, str], list[bool]] = {}

    for case in CASES:
        for document_id in case.document_ids or (None,):
            try:
                inputs, output = _source_row(
                    case.task,
                    abstention=(True if case.corrupt is _fabricate_answer else None),
                    document_id=document_id,
                )
            except LookupError as exc:
                log.warning(
                    "%s%s ignoré : %s",
                    case.label,
                    f" ({document_id})" if document_id else "",
                    exc,
                )
                continue

            (judge,) = judges_for(case.task, model_uri=model_uri)
            feedback = judge(inputs=inputs, outputs=case.apply(output))
            score = int(feedback.value)
            above = score >= JUDGE_PASS_THRESHOLD
            correct = above is case.should_pass

            if case.should_pass:
                clean_total += 1
                passed += correct
            else:
                corrupted_total += 1
                caught += correct

            doc_label = document_id or "—"
            rows.append((case.task.value, case.label, doc_label, score, correct))
            by_case.setdefault((case.task.value, case.label), []).append(correct)
            log.info(
                "%-12s %-32s %-45s note=%d  %s",
                case.task.value,
                case.label,
                doc_label,
                score,
                "ok" if correct else "RATÉ",
            )

    print()
    print(f"Juge éprouvé : {model_uri or JUDGE_MODEL}")
    print(f"Seuil de réussite : {JUDGE_PASS_THRESHOLD}/5")
    print()
    print(f"{'tâche':13s} {'cas':32s} {'document':45s} {'note':>4s}  verdict")
    for task, label, doc, score, correct in rows:
        print(f"{task:13s} {label:32s} {doc:45s} {score:>4d}  {'ok' if correct else 'RATÉ'}")

    multi_doc = {k: v for k, v in by_case.items() if len(v) > 1}
    if multi_doc:
        print()
        print("Cas rejoués sur plusieurs documents :")
        for (task, label), results in multi_doc.items():
            n_ok = sum(results)
            if n_ok == 0:
                verdict = "SYSTÉMATIQUE — raté sur les 3 documents"
            elif n_ok == len(results):
                verdict = "détecté sur les 3 documents"
            else:
                verdict = "NON systématique — dépend du document"
            print(f"  {task}/{label} : {n_ok}/{len(results)} détecté(s) — {verdict}")

    print()
    if corrupted_total:
        print(
            f"Sensibilité (fautes injectées détectées) : {caught}/{corrupted_total} "
            f"= {caught / corrupted_total:.0%}"
        )
    if clean_total:
        print(
            f"Spécificité (sorties intactes validées)  : {passed}/{clean_total} "
            f"= {passed / clean_total:.0%}"
        )

    if corrupted_total and caught == 0:
        print()
        print(
            "Le juge n'a détecté AUCUNE des fautes injectées. Ses notes ne mesurent pas "
            "l'exactitude, et le classement des modèles qui en découle est sans valeur "
            "tant que ce point n'est pas corrigé."
        )
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--judge-model",
        help="Éprouver un autre modèle juge, p. ex. `mistral:7b`, sans toucher à config.py.",
    )
    args = parser.parse_args(argv)

    logging_setup.setup()
    mlflow_setup.init()

    model_uri = f"ollama:/{args.judge_model}" if args.judge_model else None
    return run(model_uri)


if __name__ == "__main__":
    raise SystemExit(main())
