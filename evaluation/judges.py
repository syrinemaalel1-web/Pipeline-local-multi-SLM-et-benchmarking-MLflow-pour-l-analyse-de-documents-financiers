"""Juges LLM sans référence.

Trois choix structurants, justifiés dans docs/architecture-review.md :

- **`make_judge` plutôt que `Guidelines`** : les juges rendent une note de 1 à 5 et non
  un verdict binaire. Sur une dizaine de documents, un pass-rate binaire a une
  granularité de 11 points et ne départage rien (§4.5).
- **Le document passe par `{{ inputs }}`** : MLflow n'accepte aucune variable de
  gabarit personnalisée, le `{{ document }}` du spec n'existe pas (§3.1).
- **Aucun juge ne fait ce qu'un scorer déterministe fait mieux** : ni comptage de
  mots, ni vérification de champs obligatoires (§4.3).

Le modèle juge est unique, fixe, et absent de toutes les listes de candidats.


Pourquoi ces consignes ont été refaites
---------------------------------------
Le premier jeu de consignes se bornait à dire « n'évalue pas le format ». Le premier run
a montré que cela ne suffit pas : 76 % de notes maximales, extraction et résumé à 100 %
de 5/5, et des justifications qui invoquaient la présentation (« présenté de manière
claire et structurée ») pour motiver la note. Un contrôle négatif — injecter une faute
connue dans une sortie correcte, cf. `evaluation.judge_probe` — a ensuite établi le
diagnostic complet : le juge ne détectait **aucune** des fautes injectées, pas même une
traduction amputée de moitié ou un montant porté à 99 999 999.

La cause n'est pas le modèle. Interrogé en questions ouvertes sur le même document,
`aya:8b` restitue correctement le montant, le taux, la durée, et reconnaît qu'un IBAN
absent est absent. Il sait lire. Ce qu'il ne sait pas faire, c'est contredire : dès
qu'on lui soumet une sortie en lui demandant si elle est correcte, il confirme. Le
défaut est un biais de complaisance, propre au sens de la vérification.

D'où le protocole retenu ici, commun aux quatre juges : le juge établit **d'abord sa
propre référence à partir du seul document**, avant de regarder la sortie évaluée, puis
compare. Il produit un jugement au lieu de ratifier celui d'un autre. S'y ajoutent trois
garde-fous : un barème ancré sur un nombre d'écarts constatés, une note énumérée que le
décodage contraint empêche de sortir du barème, et une liste explicite de justifications
irrecevables — la forme n'étant plus seulement hors périmètre, mais motif nul.

La séparation fond / forme suit la même logique : tout ce qu'un scorer déterministe
tranche mieux qu'un modèle de 8 B lui est retiré. Le juge ne voit ni la longueur, ni la
langue, ni la validité syntaxique, ni — pour le Q&A — le fait qu'une abstention soit
attendue, que `abstention_correct` vérifie exactement.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from mlflow.genai import make_judge

from config import (
    JUDGE_INFERENCE_PARAMS,
    JUDGE_MODEL_URI,
    JUDGE_SCALE_MAX,
    JUDGE_SCALE_MIN,
    MODELS_BY_TASK,
    NOT_FOUND_MARKER,
    Task,
    assert_judge_is_isolated,
)

_SCALE = f"{JUDGE_SCALE_MIN} à {JUDGE_SCALE_MAX}"

#: Type de la note, énuméré plutôt que `int`. MLflow en dérive le schéma JSON transmis
#: à Ollama, qui contraint le décodage : le juge ne *peut* plus émettre de valeur hors
#: barème. Avec un simple `int`, `aya:8b` rendait 0 sur les cas d'abstention — il
#: reportait le nombre d'erreurs constatées au lieu de la note, et aucune reformulation
#: de la consigne ne l'en a dissuadé. L'énumération règle le problème à la source.
_RATING = Literal[tuple(range(JUDGE_SCALE_MIN, JUDGE_SCALE_MAX + 1))]  # type: ignore[valid-type]

#: Rappel commun aux quatre juges, ajouté après la consigne propre à la tâche.
_RATING_FOOTER = f"""
---

PÉRIMÈTRE. Tu ne notes que le FOND : l'exactitude de ce qui est affirmé, au regard du
document source. La forme est vérifiée séparément par du code. Présentation, structure,
mise en forme, clarté, style, concision, longueur, langue employée et validité
syntaxique ne relèvent pas de toi et ne valent aucun point.

RÈGLE ANTI-COMPLAISANCE — la plus importante. Une valeur n'est pas correcte parce
qu'elle est plausible, bien formulée, ou conforme à ce qu'on attend d'un tel dossier.
Elle est correcte si, et seulement si, tu l'as toi-même relevée dans le document à
l'étape 1 ci-dessous. Ton rôle n'est pas de confirmer le travail présenté : c'est de
faire le tien, puis de constater les écarts. Une sortie soignée n'est pas une sortie
exacte, et rien ne t'oblige à trouver la sortie satisfaisante.

MÉTHODE. L'ordre importe, ne le raccourcis pas.
1. Établis TA PROPRE RÉFÉRENCE à partir du seul document, avant de lire la sortie
   évaluée. Relève les valeurs qui comptent (montants, devises, taux, durées, dates,
   noms, clauses) et écris-les au début de ta justification.
2. Lis alors la sortie et compare-la à ta référence, élément par élément.
3. Relève les écarts : une valeur que ta référence ne contient pas est inventée ; une
   valeur qui diffère de ta référence est fausse ; un élément de ta référence absent de
   la sortie est une omission. Sépare les écarts graves des imprécisions mineures.
4. Applique le barème.

JUSTIFICATIONS IRRECEVABLES. Ne fonde jamais une note sur le fait que la sortie soit
« bien structurée », « claire », « complète », « détaillée », « professionnelle » ou
« bien rédigée » : ce sont des observations de forme. Une justification qui ne cite
aucune valeur relevée dans le document est une justification invalide.

BARÈME. Le nombre d'écarts détermine la note, mais ATTENTION : la note n'est pas ce
nombre. C'est toujours un entier de {_SCALE}, où {JUDGE_SCALE_MAX} est la meilleure
évaluation et {JUDGE_SCALE_MIN} la pire. Zéro écart donne {JUDGE_SCALE_MAX}, jamais 0.
N'arrondis pas vers le haut.
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


def _judge(name: str, instructions: str, description: str, model_uri: str):
    # Surtout ne pas passer `base_url` : MLflow y voit une passerelle AI Gateway,
    # sélectionne le GatewayAdapter et poste à la racine du serveur, d'où un 405.
    # L'adresse d'Ollama passe par OLLAMA_API_BASE, posé dans orchestration.mlflow_setup.
    return make_judge(
        name=name,
        instructions=f"{instructions.strip()}\n\n{_RATING_FOOTER}",
        model=model_uri,
        description=description,
        feedback_value_type=_RATING,
        inference_params=JUDGE_INFERENCE_PARAMS,
    )


def _assert_override_is_isolated(model_uri: str) -> None:
    """Un juge de remplacement reste soumis à la règle : jamais un candidat."""
    name = model_uri.split(":/", 1)[-1]
    contaminated = sorted(
        task.value for task, models in MODELS_BY_TASK.items() if name in models
    )
    if contaminated:
        raise AssertionError(
            f"{name!r} est candidat dans {contaminated} : il ne peut pas servir de "
            "juge, il s'auto-évaluerait."
        )


@lru_cache(maxsize=4)
def _build(model_uri: str) -> dict[Task, list]:
    assert_judge_is_isolated()
    _assert_override_is_isolated(model_uri)

    extraction_fidelity = _judge(
        name="extraction_fidelity",
        model_uri=model_uri,
        description="Fidélité des valeurs extraites au document source.",
        instructions=f"""
Tu évalues une extraction de données réalisée sur une proposition financière tunisienne.

Le champ `document` de {{{{ inputs }}}} contient le document source. {{{{ outputs }}}}
contient le JSON produit par un autre modèle.

Commence par extraire toi-même du document, sans regarder le JSON, les valeurs
suivantes lorsqu'elles y figurent : nom du client, montant du financement, devise, taux
d'intérêt, durée, dates, garanties. Écris cette liste : c'est ta référence.

Compare ensuite le JSON à ta référence, champ par champ.

- Une valeur du JSON absente de ta référence est inventée : écart grave, même si elle
  est plausible pour ce type de dossier.
- Un montant, un taux ou une date qui diffère de ta référence est faux : écart grave,
  y compris quand l'écart chiffré est faible.
- Le marqueur "{NOT_FOUND_MARKER}" est le comportement correct sur une information que
  ta référence ne contient pas : ne le sanctionne pas. Il n'est un écart que si
  l'information figure bel et bien dans ta référence.

Le nombre de champs remplis n'entre pas dans la note : un JSON pauvre mais exact vaut
mieux qu'un JSON riche et approximatif.
""",
    )

    summary_faithfulness = _judge(
        name="summary_faithfulness",
        model_uri=model_uri,
        description="Fidélité et couverture factuelle du résumé exécutif.",
        instructions="""
Tu évalues un résumé exécutif de proposition financière tunisienne.

Le champ `document` de {{ inputs }} contient le document source. {{ outputs }} contient
le résumé produit par un autre modèle.

Commence par relever toi-même dans le document, sans lire le résumé, ce qu'un comité de
crédit doit en retenir : le client, le montant et sa devise, la durée, le sens de la
recommandation, et les deux ou trois faits chiffrés déterminants. Écris cette liste :
c'est ta référence.

Compare ensuite le résumé à ta référence.

- FIDÉLITÉ, critère prioritaire. Toute affirmation du résumé qui ne découle pas de ta
  référence est un écart grave : chiffre déformé, fait ajouté, ou conclusion que le
  document ne tire pas. Une décision, un rejet ou un incident que ta référence ne
  mentionne pas est une invention, même énoncé avec assurance.
- COUVERTURE. L'absence dans le résumé du client, du montant, de la devise, de la durée
  ou du sens de la recommandation est une omission importante, pas une imprécision.

Le ton ne compte que s'il déforme le fond : présenter le dossier comme plus sûr que ta
référence ne l'établit est un écart de fond. Un résumé sobre mais exact ne perd rien à
sa sécheresse.
""",
    )

    translation_fidelity = _judge(
        name="translation_fidelity",
        model_uri=model_uri,
        description="Préservation du sens et absence d'omission dans la traduction.",
        instructions="""
Tu évalues la traduction d'une proposition financière tunisienne.

{{ inputs }} contient le document source dans le champ `document`, sa langue dans `lang`,
et la langue cible imposée dans `expected_output_lang`. {{ outputs }} contient la
traduction produite par un autre modèle.

Commence par parcourir le document source et dresser la liste de ses sections et de ses
clauses, dans l'ordre, avec les chiffres clés de chacune. Écris cette liste : c'est ta
référence. Note en particulier où le document se termine.

Compare ensuite la traduction à ta référence.

- OMISSION. Reprends ta liste dans l'ordre et vérifie que chaque section, chaque clause
  et chaque ligne de tableau se retrouve dans la traduction. Une traduction qui s'arrête
  avant la fin du document, ou qui saute une clause, présente une omission grave — même
  si tout ce qu'elle contient par ailleurs est excellent. Vérifie explicitement que la
  dernière section de ta référence figure bien dans la traduction.
- PRÉSERVATION DU SENS. La traduction dit-elle la même chose que ta référence ? Un
  contresens sur une clause, une condition ou une obligation est un écart grave.
- TERMINOLOGIE. Ne compte comme écart qu'un terme dont la traduction change le sens
  juridique ou financier. Une tournure moins idiomatique qu'une autre n'est pas un écart.

Tu ne compares pas la traduction à celle que tu aurais rédigée, mais au sens du source :
une reformulation qui préserve intégralement ce sens n'est pas un écart.
""",
    )

    qa_groundedness = _judge(
        name="qa_groundedness",
        model_uri=model_uri,
        description="Ancrage de la réponse dans le document, invention sanctionnée.",
        instructions=f"""
Tu évalues une réponse à une question portant sur une proposition financière tunisienne.

{{{{ inputs }}}} contient le document source dans le champ `document` et la question dans
`question`. {{{{ outputs }}}} contient la réponse produite par un autre modèle.

COMMENCE PAR RÉPONDRE TOI-MÊME à la question, à partir du seul document, sans lire la
réponse évaluée. Écris ta réponse : c'est ta référence. Si le document ne contient pas
l'information demandée, ta référence est exactement "{NOT_FOUND_MARKER}".

Compare ensuite la réponse évaluée à ta référence.

- Si ta référence est une valeur et que la réponse donne la même : aucun écart.
- Si la réponse donne une valeur différente de ta référence : écart grave.
- Si ta référence est "{NOT_FOUND_MARKER}" et que la réponse signale elle aussi n'avoir
  rien trouvé, les deux concordent : aucun écart, c'est le meilleur résultat possible.
  Ne retire aucun point au motif qu'elle « ne répond pas à la question » — l'exiger
  reviendrait à récompenser l'invention.
- Si ta référence est "{NOT_FOUND_MARKER}" et que la réponse avance malgré tout une
  valeur, cette valeur est inventée de toutes pièces : c'est la faute la plus grave que
  tu puisses relever, et elle mérite la note {JUDGE_SCALE_MIN}. Un identifiant bancaire,
  un montant ou une date d'apparence crédible reste une invention si ta référence ne le
  contient pas ; le fait qu'il respecte un format standard ne prouve rien.

Tu ne juges pas si la réponse est complète ni bien tournée : un scorer séparé vérifie
déjà que le modèle répond quand il le doit et s'abstient quand il le doit. Tu ne mesures
ici que l'écart entre la réponse et ta propre lecture du document.
""",
    )

    return {
        Task.EXTRACTION: [extraction_fidelity],
        Task.SUMMARY: [summary_faithfulness],
        Task.TRANSLATION: [translation_fidelity],
        Task.QA: [qa_groundedness],
    }


def judges_for(task: Task, *, model_uri: str | None = None) -> list:
    """Juges LLM applicables à une tâche.

    `model_uri` sert à éprouver un juge de remplacement sans toucher à la
    configuration ; le benchmark, lui, s'en tient toujours à `JUDGE_MODEL_URI`.
    """
    return _build(model_uri or JUDGE_MODEL_URI)[task]
