"""Second pipeline d'évaluation — `mlflow.evaluate()`, API dépréciée depuis MLflow 3.4.0.

Existe pour une seule raison : `faithfulness`, `answer_relevance`, `ari_grade_level` et
`flesch_kincaid_grade_level` (`mlflow.metrics` / `mlflow.metrics.genai`) n'ont pas
d'équivalent `Scorer` natif dans `mlflow.genai.evaluate()`, le pipeline principal du
projet (`evaluation.judges`, `evaluation.code_scorers`). Vérifié avant d'écrire quoi que
ce soit : ces fonctions renvoient des `mlflow.models.evaluation.base.EvaluationMetric`,
pas des `mlflow.genai.scorers.base.Scorer` — `isinstance(..., Scorer)` est `False`, et
`mlflow.genai.evaluate()` exige `Scorer`. Les y passer directement ne fonctionne pas ;
c'est ce qui justifie ce second pipeline plutôt qu'une intégration au premier.

**Non calibré.** Contrairement au juge principal (`evaluation.judges`, calibré par
annotation humaine — voir CLAUDE.md §3bis/§3ter), ces métriques n'ont fait l'objet
d'aucune vérification contre un jugement humain. `reporting.report --eval-mode metrics`
le rappelle explicitement dans le rapport produit.

`faithfulness`/`answer_relevance` utilisent le même modèle juge que le pipeline
principal (`config.JUDGE_MODEL_URI`), pas un modèle différent.

**`faithfulness` réécrite une seconde fois le 2026-08-12 (voir CLAUDE.md Journal).**
Deux tentatives précédentes, dans l'ordre :

1. Le prompt par défaut de `mlflow.metrics.genai.faithfulness()` (conçu pour
   `openai:/gpt-4`) validait à 5/5 des fautes injectées connues (contrôle négatif :
   0/4 détections).
2. Un `definition`/`grading_prompt` maison via `make_genai_metric`, reprenant la
   discipline de `evaluation/judges.py`, a été essayé — **toujours 0/4**. Diagnostic :
   `make_genai_metric`/`mlflow.evaluate()` envoie un prompt texte libre et parse la
   réponse par regex, sans aucun décodage contraint — contrairement à `make_judge`
   (pipeline `judge`), qui dérive un schéma JSON du type de retour et le transmet à
   Ollama pour contraindre le décodage. Une reformulation du prompt seule ne suffit
   pas sans ce mécanisme.

**Solution retenue : `make_judge` (le même mécanisme que le pipeline `judge`, donc le
même décodage contraint) appelé directement à l'intérieur d'un `eval_fn` maison,
enregistré via `mlflow.metrics.make_metric` pour rester une `EvaluationMetric`
utilisable par `mlflow.evaluate()`.** Ça reste strictement dans le pipeline `metrics` :
aucun résultat n'est partagé ni fusionné avec `results/evaluation/` (le pipeline
`judge`), le juge est juste construit avec la même fonction MLflow. Avantage
supplémentaire par rapport à la tentative 2 : `make_judge` accepte des instructions au
gabarit libre (`{{ inputs }}` / `{{ outputs }}` — seules variables réservées, pas de
notation pointée `inputs.document` : le dict `inputs` est rendu tel quel, exactement
comme dans `evaluation/judges.py`), donc le document source peut maintenant être placé
AVANT la sortie évaluée dans le prompt — impossible avec le template fixe de
`make_genai_metric` (« Output » y précède toujours le contexte).
`_faithfulness_metric` construit un juge par tâche (traduction/résumé/qa), avec la
même discipline anti-complaisance que `evaluation/judges.py` (référence construite
avant lecture de la sortie, justifications de forme irrecevables, barème ancré sur des
écarts constatés) — dupliquée ici plutôt qu'importée, pour que ce module reste
autonome du pipeline `judge`.

**Plafond numérique déterministe ajouté le 2026-08-13, sur demande explicite de
l'utilisateur.** Motif : même avec le décodage contraint (ci-dessus), le juge reste
généreux sur les substitutions de montants noyées dans du texte long (voir CLAUDE.md,
`judge_probe` étendu — 1/3 documents seulement) — et un tableau où `faithfulness`
affiche 5/5 partout, à côté d'ARI/Flesch-Kincaid qui varient normalement, aurait l'air
non discriminant aux yeux d'un lecteur externe. Plutôt que d'ajouter une métrique
séparée (refusé explicitement par l'utilisateur : « une seule colonne »), le contrôle
numérique **plafonne** la note déjà rendue par le juge — il ne peut que la faire
baisser, jamais monter, et seulement quand le juge a effectivement manqué un écart
mesurable. Deux directions différentes selon la tâche, parce que « fidèle » n'a pas le
même sens : la traduction doit tout couvrir (une valeur du document absente de la
sortie est suspecte), le résumé a le droit d'omettre (seule une valeur de la sortie
absente du document est suspecte). Quand le plafond s'active, la justification est
complétée en conséquence — pour que la note affichée et le texte qui l'accompagne ne
se contredisent jamais (l'inverse s'est déjà produit une fois sur ce projet, voir
CLAUDE.md, l'anomalie `doc_8dcc7fc6`).
"""

from __future__ import annotations

import re
import statistics
from typing import Literal

from config import (
    JUDGE_INFERENCE_PARAMS,
    JUDGE_MODEL_URI,
    JUDGE_SCALE_MAX,
    JUDGE_SCALE_MIN,
    Task,
)

#: Tâches couvertes par chaque métrique (répartition corrigée le 2026-08-10 — voir
#: CLAUDE.md Journal). Rien sur l'extraction : pas de métrique candidate dans ce lot.
_TRANSLATION_METRICS = ("faithfulness",)
_SUMMARY_METRICS = ("faithfulness", "ari_grade_level", "flesch_kincaid_grade_level")
_QA_METRICS = ("answer_relevance", "faithfulness")

_RATING = Literal[tuple(range(JUDGE_SCALE_MIN, JUDGE_SCALE_MAX + 1))]  # type: ignore[valid-type]
_SCALE = f"{JUDGE_SCALE_MIN} à {JUDGE_SCALE_MAX}"

#: Dupliqués de `evaluation.code_scorers` (mêmes constantes, même normalisation) —
#: pas importés, pour que ce module reste autonome du pipeline `judge` (voir
#: docstring). Sert au plafond numérique déterministe de `_faithfulness_metric`.
_ARABIC_INDIC = str.maketrans("٠١٢٣٤٥٦٧٨٩٫٬", "0123456789.,")
_NUMBER = re.compile(r"\d[\d\s.,]*\d|\d")


def _numbers(text: str) -> set[str]:
    """Nombres normalisés d'un texte, séparateurs de milliers retirés."""
    normalized = text.translate(_ARABIC_INDIC)
    found: set[str] = set()
    for raw in _NUMBER.findall(normalized):
        cleaned = raw.replace(" ", "").replace(" ", "")
        cleaned = re.sub(r",(?=\d{3}\b)", "", cleaned)
        cleaned = cleaned.replace(",", ".").rstrip(".")
        cleaned = re.sub(r"\.(?=\d{3}\b)", "", cleaned)
        if cleaned:
            found.add(cleaned.lstrip("0") or "0")
    return found


def _translation_cap(document: str, output: str) -> tuple[int, str] | None:
    """(plafond, motif) si au moins un nombre du document est absent de la sortie,
    sinon `None` (rien à plafonner).

    Basé sur un COMPTE, pas un ratio — corrigé le 2026-08-14 après vérification sur
    données réelles : un document de proposition financière contient couramment 80+
    nombres (dates, pourcentages, échéancier), donc la perte d'une seule valeur (le
    cas visé, ex. un montant substitué) ne fait bouger un ratio de couverture que de
    quelques centièmes — mesuré à 0,977 sur un cas réel, largement au-dessus de tout
    seuil utilisable. Compter les absences plutôt que les proportionner évite ce
    problème : une seule valeur manquante compte déjà comme un écart grave, dans le
    même esprit que `evaluation.judges` (« une valeur qui diffère... écart grave, y
    compris quand l'écart chiffré est faible »).
    """
    source_numbers = _numbers(document)
    if not source_numbers:
        return None
    missing = len(source_numbers - _numbers(output))
    if missing == 0:
        return None
    reason = f"{missing} valeur(s) numérique(s) du document absente(s) de la sortie"
    if missing == 1:
        return 3, reason
    if missing <= 3:
        return 2, reason
    return JUDGE_SCALE_MIN, reason


def _summary_cap(document: str, output: str) -> tuple[int, str] | None:
    """(plafond, motif) si des nombres de la sortie n'existent pas dans le document,
    sinon `None`.

    Basé sur un RATIO (contrairement à la traduction) : un résumé est court et
    sélectif, donc une valeur inventée y pèse déjà lourd dans la proportion —
    vérifié en pratique le 2026-08-14, le ratio réagit correctement à un montant
    substitué sur les 3 documents pilotes. L'omission est normale pour un résumé
    (voir `evaluation.legacy_metrics` docstring) : seule une valeur de la sortie
    absente du document compte ici, pas l'inverse.
    """
    output_numbers = _numbers(output)
    if not output_numbers:
        return None
    ratio = len(output_numbers & _numbers(document)) / len(output_numbers)
    if ratio >= 0.95:
        return None
    reason = f"{ratio:.0%} des valeurs numériques de la sortie absentes du document source"
    if ratio >= 0.9:
        return 4, reason
    if ratio >= 0.7:
        return 3, reason
    if ratio >= 0.4:
        return 2, reason
    return JUDGE_SCALE_MIN, reason

#: Rappel commun aux 3 variantes — même discipline que
#: `evaluation.judges._RATING_FOOTER`, dupliquée ici (pas importée : ce module reste
#: autonome du pipeline `judge`, voir docstring).
_RATING_FOOTER = f"""
---

PÉRIMÈTRE. Tu ne notes que le FOND : l'exactitude de ce qui est affirmé, au regard du
document source. Présentation, structure, mise en forme, style, longueur et langue
employée ne relèvent pas de toi et ne valent aucun point — la langue cible est déjà
vérifiée séparément par un scorer déterministe.

RÈGLE ANTI-COMPLAISANCE — la plus importante. Une affirmation n'est pas correcte parce
qu'elle est plausible ou bien formulée. Elle est correcte si, et seulement si, tu l'as
toi-même relevée dans le document source à l'étape 1 ci-dessous. Ton rôle n'est pas de
confirmer la sortie évaluée : c'est de faire ton propre travail de lecture, puis de
constater les écarts.

MÉTHODE. L'ordre importe, ne le raccourcis pas.
1. Lis le document source ci-dessus et établis TA PROPRE RÉFÉRENCE avant de lire la
   sortie évaluée : les valeurs, sections et clauses qui comptent, dans l'ordre.
2. Lis alors la sortie évaluée et compare-la à ta référence, élément par élément.
3. Relève les écarts : une affirmation absente de ta référence est inventée ; une valeur
   qui diffère de ta référence est fausse ; un élément de ta référence absent de la
   sortie est une omission.
4. Applique le barème.

JUSTIFICATIONS IRRECEVABLES. Ne fonde jamais une note sur le fait que la sortie soit
« bien structurée », « claire », « complète » ou « bien rédigée ». Une justification qui
ne cite aucune valeur relevée dans le document source est invalide.

BARÈME. Le nombre d'écarts détermine la note, mais ATTENTION : la note n'est pas ce
nombre. C'est toujours un entier de {_SCALE}, où {JUDGE_SCALE_MAX} est la meilleure
évaluation et {JUDGE_SCALE_MIN} la pire. Zéro écart donne {JUDGE_SCALE_MAX}, jamais 0.
- Note {JUDGE_SCALE_MAX} : aucun écart après comparaison effective avec ta référence.
- Note 4 : une ou deux imprécisions mineures, aucun écart grave.
- Note 3 : un écart grave avéré, ou plusieurs imprécisions cumulées.
- Note 2 : plusieurs écarts graves, ou une omission importante.
- Note {JUDGE_SCALE_MIN} : information inventée, valeur contredite par le document, ou
  sortie inexploitable sur le fond.

FORMAT. Justifie d'abord — ta référence, puis les écarts constatés — et termine par la
phrase « Note attribuée : N/{JUDGE_SCALE_MAX} ». Reporte ce même entier N, et lui seul,
dans le champ `result`, qui reçoit la note et jamais le nombre d'écarts.
""".strip()

_TRANSLATION_INSTRUCTIONS = f"""
Tu évalues la fidélité factuelle (« faithfulness ») d'une traduction de proposition
financière tunisienne.

Le champ `document` de {{{{ inputs }}}} contient le document source. {{{{ outputs }}}}
contient la traduction produite par un autre modèle.

Après avoir établi ta référence (sections, clauses, chiffres clés du document source,
dans l'ordre — note où le document se termine), compare la traduction à ta référence :
- OMISSION : une traduction qui s'arrête avant la fin du document source, ou qui saute
  une clause ou une section, est un écart grave — même si le reste est excellent.
  Vérifie explicitement que la dernière section de ta référence apparaît dans la
  traduction.
- INVENTION : une affirmation de la traduction qui ne découle pas de ta référence
  (chiffre déformé, clause ajoutée) est un écart grave.
- CONTRESENS : un changement de sens sur une clause, une condition ou une obligation est
  un écart grave.
- Une reformulation ou un choix terminologique différent qui préserve intégralement le
  sens de ta référence n'est PAS un écart.
"""

_SUMMARY_INSTRUCTIONS = f"""
Tu évalues la fidélité factuelle (« faithfulness ») d'un résumé exécutif de proposition
financière tunisienne.

Le champ `document` de {{{{ inputs }}}} contient le document source. {{{{ outputs }}}}
contient le résumé produit par un autre modèle.

Ta référence (étape 1) est ce qu'un comité de crédit doit retenir du document source :
le client, le montant et sa devise, la durée, le sens de la recommandation, et les deux
ou trois faits chiffrés déterminants. Compare ensuite le résumé à ta référence :
- FIDÉLITÉ, critère prioritaire : toute affirmation du résumé qui ne découle pas de ta
  référence est un écart grave (chiffre déformé, fait ajouté, conclusion que le document
  ne tire pas).
- COUVERTURE : l'absence du client, du montant, de la devise, de la durée ou du sens de
  la recommandation est une omission importante.
- Le ton ne compte que s'il déforme le fond ; un résumé sobre mais exact ne perd rien à
  sa sécheresse.
"""

_QA_INSTRUCTIONS = f"""
Tu évalues si une réponse à une question posée sur une proposition financière
tunisienne est ancrée dans le document source (« faithfulness »).

Le champ `document` de {{{{ inputs }}}} contient le document source. {{{{ outputs }}}}
contient la réponse produite par un autre modèle.

Ta référence (étape 1) est ta propre réponse à la question, tirée du seul document
source ; si le document ne contient pas l'information, ta référence est qu'aucune
valeur ne doit être avancée. Compare ensuite la réponse évaluée à ta référence :
- Si la réponse avance une valeur absente de ta référence, cette valeur est inventée de
  toutes pièces : c'est l'écart le plus grave possible, quelle que soit sa plausibilité
  apparente ou son format — un identifiant bancaire d'apparence crédible reste une
  invention si le document ne le contient pas.
- Si la réponse donne une valeur différente de celle que tu as toi-même trouvée, c'est
  un écart grave.
- Si la réponse s'abstient alors que le document ne contient effectivement pas
  l'information, il n'y a AUCUN écart : c'est le meilleur résultat possible, ne baisse
  jamais le score pour ce motif.
- Si la réponse donne la même valeur que ta référence, il n'y a aucun écart.
"""


def _faithfulness_judge(instructions: str, description: str):
    """Construit un juge `faithfulness` maison via `make_judge` — même mécanisme de
    décodage contraint que le pipeline `judge` (voir docstring du module), pas une
    référence au juge du pipeline `judge` lui-même.
    """
    from mlflow.genai import make_judge

    return make_judge(
        name="faithfulness",
        instructions=f"{instructions.strip()}\n\n{_RATING_FOOTER}",
        model=JUDGE_MODEL_URI,
        description=description,
        feedback_value_type=_RATING,
        inference_params=JUDGE_INFERENCE_PARAMS,
    )


def _faithfulness_metric(instructions: str, description: str, numeric_check=None):
    """Empaquette le juge `make_judge` ci-dessus en `EvaluationMetric` compatible
    `mlflow.evaluate()`, via un `eval_fn` maison enregistré par `mlflow.metrics.make_metric`.

    `numeric_check` (optionnel, `_translation_cap` ou `_summary_cap`) plafonne la note
    du juge par un contrôle déterministe des nombres — voir docstring du module. Ne
    peut jamais faire monter la note, seulement la baisser ; `None` désactive le
    plafond (utilisé pour le Q&A, non concerné par ce correctif pour l'instant).
    """
    from mlflow.metrics import MetricValue, make_metric

    judge = _faithfulness_judge(instructions, description)

    def eval_fn(predictions, context) -> "MetricValue":
        scores: list[float | None] = []
        justifications: list[str] = []
        for output, document in zip(predictions, context):
            feedback = judge(inputs={"document": document}, outputs=output)
            score = feedback.value
            rationale = feedback.rationale or ""

            if numeric_check is not None and score is not None:
                capped = numeric_check(document, output)
                if capped is not None:
                    cap, reason = capped
                    if cap < score:
                        rationale = (
                            f"{rationale}\n\n[Plafond automatique : {reason} — note "
                            f"ramenée de {score} à {cap}/{JUDGE_SCALE_MAX} "
                            "indépendamment de l'évaluation du juge.]"
                        )
                        score = cap

            scores.append(score)
            justifications.append(rationale)

        valid = [s for s in scores if s is not None]
        aggregates = {"mean": statistics.mean(valid)} if valid else {}
        return MetricValue(scores=scores, justifications=justifications, aggregate_results=aggregates)

    return make_metric(eval_fn=eval_fn, greater_is_better=True, name="faithfulness")


def metrics_for(task: Task) -> list:
    """Métriques `mlflow.evaluate()` applicables à une tâche, instanciées."""
    if task is Task.TRANSLATION:
        return [
            _faithfulness_metric(
                _TRANSLATION_INSTRUCTIONS,
                "Fidélité factuelle de la traduction au document source (pipeline metrics).",
                numeric_check=_translation_cap,
            )
        ]
    if task is Task.SUMMARY:
        from mlflow.metrics import ari_grade_level, flesch_kincaid_grade_level

        return [
            _faithfulness_metric(
                _SUMMARY_INSTRUCTIONS,
                "Fidélité factuelle du résumé au document source (pipeline metrics).",
                numeric_check=_summary_cap,
            ),
            ari_grade_level(),
            flesch_kincaid_grade_level(),
        ]
    if task is Task.QA:
        from mlflow.metrics.genai import answer_relevance

        return [
            answer_relevance(model=JUDGE_MODEL_URI),
            _faithfulness_metric(
                _QA_INSTRUCTIONS,
                "Ancrage de la réponse dans le document source (pipeline metrics).",
            ),
        ]
    return []


#: Nom des métriques par tâche, dans l'ordre — pour construire les colonnes du rapport
#: sans réinstancier les objets `EvaluationMetric` (coûteux : ils embarquent le prompt
#: du juge dans `metric_details`).
METRIC_NAMES: dict[Task, tuple[str, ...]] = {
    Task.TRANSLATION: _TRANSLATION_METRICS,
    Task.SUMMARY: _SUMMARY_METRICS,
    Task.QA: _QA_METRICS,
}
