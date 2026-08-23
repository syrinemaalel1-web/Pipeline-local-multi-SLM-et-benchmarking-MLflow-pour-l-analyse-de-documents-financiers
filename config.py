"""Source unique de vérité pour le benchmark multi-SLM.

Toute constante qui influence la comparabilité des modèles vit ici, et nulle part
ailleurs : listes de modèles, juge, températures, seed, budgets de tokens,
pondérations du score composite.
"""

from __future__ import annotations

import os
from enum import Enum
from pathlib import Path

# --------------------------------------------------------------------------- #
# Chemins
# --------------------------------------------------------------------------- #

ROOT = Path(__file__).parent

DATA_RAW_DIR = ROOT / "data"
DATA_PROCESSED_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results"
REPORTS_DIR = ROOT / "reports"
PROMPTS_DIR = ROOT / "prompts"

# Jeu de questions Q&A, alimenté par l'utilisateur (voir qa_questions.example.json).
QA_QUESTIONS_PATH = DATA_RAW_DIR / "qa_questions.json"

for _d in (DATA_PROCESSED_DIR, RESULTS_DIR, REPORTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Tâches et langues
# --------------------------------------------------------------------------- #


class Task(str, Enum):
    EXTRACTION = "extraction"
    SUMMARY = "summary"
    TRANSLATION = "translation"
    QA = "qa"


class Lang(str, Enum):
    FR = "fr"
    EN = "en"
    AR = "ar"


class EvalMode(str, Enum):
    """Pipeline d'évaluation à utiliser — un seul actif par appel, jamais fusionnés.

    - JUDGE : pipeline principal, `mlflow.genai.evaluate()` + `evaluation.judges`
      (`make_judge`), calibré par annotation humaine (voir CLAUDE.md §3bis/§3ter).
    - METRICS : second pipeline, `mlflow.evaluate()` (API dépréciée depuis MLflow 3.4.0,
      voir `evaluation/legacy_metrics.py`) — seules `faithfulness`, `answer_relevance`,
      `ari_grade_level` et `flesch_kincaid_grade_level` n'ont pas d'équivalent `Scorer`
      natif dans `mlflow.genai.evaluate()` à ce jour, d'où le second pipeline plutôt
      qu'une intégration au premier. **Non calibré** par annotation humaine.

    Point d'extension pour un futur mode de déploiement (non implémenté) : ajouter un
    membre ici, une entrée dans le registre `_EVAL_MODE_HANDLERS` de
    `orchestration/run_eval.py`, et rien d'autre ne change.
    """

    JUDGE = "judge"
    METRICS = "metrics"


#: Direction de traduction imposée par la langue source (PROJECT.md §1).
#: La langue cible n'est jamais un paramètre libre.
TRANSLATION_DIRECTION: dict[Lang, Lang] = {
    Lang.FR: Lang.EN,
    Lang.EN: Lang.FR,
    Lang.AR: Lang.FR,
}

LANG_NAMES: dict[Lang, str] = {
    Lang.FR: "français",
    Lang.EN: "anglais",
    Lang.AR: "arabe",
}

LANG_NAMES_EN: dict[Lang, str] = {
    Lang.FR: "French",
    Lang.EN: "English",
    Lang.AR: "Arabic",
}


# --------------------------------------------------------------------------- #
# Modèles candidats
# --------------------------------------------------------------------------- #

MODELS_EXTRACTION = [
    "qwen2.5:7b",
    "aya-expanse:8b",
    "phi4-mini:latest",
    "llama3.1:8b",
    "mistral:7b",
]

MODELS_TRANSLATION = [
    "aya-expanse:8b",
    "qwen2.5:7b",
    "translategemma:latest",
    "llama3.1:8b",
    "mannix/llamax3-8b-alpaca",
]

MODELS_SUMMARY = [
    "qwen2.5:7b",
    "llama3.1:8b",
    "phi4-mini:latest",
    "aya-expanse:8b",
    "mistral:7b",
]

MODELS_QA = [
    "llama3.1:8b",
    "qwen2.5:7b",
    "phi4-mini:latest",
    "deepseek-r1:8b",
    "aya-expanse:8b",
]

MODELS_BY_TASK: dict[Task, list[str]] = {
    Task.EXTRACTION: MODELS_EXTRACTION,
    Task.SUMMARY: MODELS_SUMMARY,
    Task.TRANSLATION: MODELS_TRANSLATION,
    Task.QA: MODELS_QA,
}

#: Modèles qui émettent un bloc de raisonnement `<think>` à retirer avant évaluation.
REASONING_MODELS = {"deepseek-r1:8b"}

#: Écarts assumés au principe de comparabilité, repris tels quels dans le rapport final.
BENCHMARK_CAVEATS: dict[str, str] = {
    "translategemma:latest": (
        "4.3B au lieu de 7-8B, et modèle spécialisé traduction (pas généraliste). "
        "Petit mais spécialisé : à lire comme un point de comparaison, pas comme un pair."
    ),
    "mannix/llamax3-8b-alpaca": (
        "Quantifié en Q4_0 et non Q4_K_M comme les autres candidats. "
        "Aucun tag Q4_K_M n'est publié pour ce modèle communautaire."
    ),
    "deepseek-r1:8b": (
        "Modèle de raisonnement : génère 3 à 10x plus de tokens que les autres. "
        "Sa latence n'est pas directement comparable ; le bloc <think> est retiré "
        "avant évaluation."
    ),
}


# --------------------------------------------------------------------------- #
# Juge
# --------------------------------------------------------------------------- #

#: Modèle juge unique et fixe pour tout le projet (PROJECT.md §1).
#: Doit être absent de toutes les listes de candidats : voir assert_judge_is_isolated().
#:
#: `aya:8b` a tenu ce rôle jusqu'au premier contrôle négatif, qu'il a échoué : sur cinq
#: fautes injectées dans de vraies sorties, il n'en détectait qu'une, validant à 5/5 un
#: montant porté à 99 999 999 et un IBAN entièrement fabriqué. `granite3.3:8b` en détecte
#: trois sur cinq à spécificité égale, et n'a de parenté avec aucun candidat — ce qui
#: lève au passage la collision de famille aya / aya-expanse. Il est en revanche environ
#: trois fois plus lent. Toute substitution doit être éprouvée par
#: `python -m evaluation.judge_probe --judge-model <tag>`.
JUDGE_MODEL = "granite3.3:8b"

#: URI attendu par MLflow. LiteLLMAdapter le traduit en "ollama/aya:8b".
JUDGE_MODEL_URI = f"ollama:/{JUDGE_MODEL}"

JUDGE_BASE_URL = os.getenv("OLLAMA_HOST", "http://localhost:11434")

#: Température 0 pour que deux exécutions du juge sur la même sortie concordent.
#: `timeout` aligné sur REQUEST_TIMEOUT_S (déclaré plus bas) : sans lui, litellm
#: retient son propre défaut (~300s), trop court pour un juge qui relit un document
#: proche du budget de contexte — observé en pratique sur doc_8dcc7fc6 (5487 tokens),
#: où l'appel expirait avant que granite3.3:8b n'ait fini de répondre.
JUDGE_INFERENCE_PARAMS = {"temperature": 0.0, "timeout": 1800}

#: Limite connue et assumée du protocole, reprise telle quelle dans le rapport final.
#: Chiffre issu de `evaluation.judge_probe` : voir docs/fiabilite-juge.md.
JUDGE_CAVEAT = (
    "Le juge détecte 3 fautes de fond sur 5 lorsqu'on en injecte volontairement dans de "
    "vraies sorties (contrôle négatif `evaluation.judge_probe`), et valide 4 sorties "
    "intactes sur 4. Il repère les valeurs inventées et les montants faussés, mais lui "
    "échappent encore une conclusion ajoutée à un résumé et une traduction tronquée. "
    "Ses notes départagent donc les modèles sur l'invention, pas sur l'omission : à "
    "lire avec les scorers déterministes, jamais seules."
)


def assert_judge_is_isolated() -> None:
    """Vérifie que le juge n'est candidat dans aucune tâche.

    Appelée au démarrage de tout script qui lance des juges.
    """
    contaminated = {
        task.value: models
        for task, models in MODELS_BY_TASK.items()
        if JUDGE_MODEL in models
    }
    if contaminated:
        raise AssertionError(
            f"JUDGE_MODEL={JUDGE_MODEL!r} apparaît dans {sorted(contaminated)}. "
            "Un modèle ne peut pas s'auto-évaluer : choisissez un juge hors des "
            "listes de candidats."
        )


# --------------------------------------------------------------------------- #
# Ollama
# --------------------------------------------------------------------------- #

OLLAMA_BASE_URL = f"{JUDGE_BASE_URL}/v1"
OLLAMA_API_KEY = "ollama"  # ignoré par Ollama, mais requis par le SDK openai

#: Fenêtre de contexte commune à tous les modèles.
#: Plafonnée par les candidats les plus courts (aya-expanse:8b, llamax3, aya:8b = 8192).
#: L'endpoint compatible OpenAI n'expose pas num_ctx : cette valeur doit être imposée
#: au serveur via OLLAMA_CONTEXT_LENGTH, sinon Ollama tronque l'entrée en silence.
NUM_CTX = 8192

#: Températures par tâche, identiques pour tous les modèles d'une même tâche.
TEMPERATURE_BY_TASK: dict[Task, float] = {
    Task.EXTRACTION: 0.0,  # sortie structurée : aucun intérêt à l'aléatoire
    Task.SUMMARY: 0.2,
    Task.TRANSLATION: 0.2,
    Task.QA: 0.0,
}

#: Plafond de génération par tâche, en tokens.
MAX_TOKENS_BY_TASK: dict[Task, int] = {
    Task.EXTRACTION: 1536,
    Task.SUMMARY: 512,
    Task.TRANSLATION: 4096,
    Task.QA: 512,
}

SEED = 1234

#: Le modèle reste chargé entre deux appels : évite de relire 5 Go de disque
#: à chaque document. Assez long pour couvrir la boucle documents d'un modèle.
OLLAMA_KEEP_ALIVE = "30m"

REQUEST_TIMEOUT_S = 1800


# --------------------------------------------------------------------------- #
# Budgets de tokens
# --------------------------------------------------------------------------- #

#: Marge laissée au prompt système, aux instructions et à l'imprécision du comptage
#: (tiktoken n'est qu'un proxy des tokenizers réels, cf. token_budget.py).
PROMPT_OVERHEAD_TOKENS = 400

#: Nombre maximum de tokens de document tenant dans NUM_CTX pour chaque tâche.
#: Pour la traduction, entrée et sortie partagent la fenêtre, d'où le facteur ~2.
def safe_document_tokens(task: Task) -> int:
    """Tokens de document au-delà desquels Ollama tronquera silencieusement."""
    if task is Task.TRANSLATION:
        # La sortie fait grossièrement la taille de l'entrée.
        return (NUM_CTX - PROMPT_OVERHEAD_TOKENS) // 2
    return NUM_CTX - PROMPT_OVERHEAD_TOKENS - MAX_TOKENS_BY_TASK[task]


#: Le juge reçoit document + sortie de l'agent, plus ses propres instructions.
def safe_document_tokens_for_judge(task: Task) -> int:
    if task is Task.TRANSLATION:
        return (NUM_CTX - PROMPT_OVERHEAD_TOKENS * 2) // 2
    return NUM_CTX - PROMPT_OVERHEAD_TOKENS * 2 - MAX_TOKENS_BY_TASK[task]


# --------------------------------------------------------------------------- #
# Évaluation
# --------------------------------------------------------------------------- #

#: Backend SQLite et non fichier : MLflow 3.14 a placé le file store en mode
#: maintenance et lève une exception à l'ouverture.
MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI", f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}"
)
MLFLOW_EXPERIMENT = "finance-slm-benchmark"

#: Un seul worker : un serveur Ollama unique sur 4 Go de VRAM ne supporte pas
#: le parallélisme, il thrashe à recharger les modèles.
EVAL_MAX_WORKERS = 1

#: Échelle des juges à note. Un juge binaire sur ~9 documents n'a pas assez de
#: granularité pour départager deux modèles.
JUDGE_SCALE_MIN = 1
JUDGE_SCALE_MAX = 5

#: Seuil à partir duquel une note de juge compte comme "réussite" dans le pass-rate.
JUDGE_PASS_THRESHOLD = 4

#: Plancher sur la moyenne des scorers déterministes en dessous duquel un modèle ne
#: peut pas devenir "recommandé", même avec une bonne note de juge. Trouvé nécessaire
#: sur doc_8dcc7fc6 : `mannix/llamax3-8b-alpaca` a produit une traduction quasi vide
#: (arrêt prématuré du modèle, 152 tokens de complétion) que le juge a notée 5/5 (point
#: mort d'omission déjà documenté dans JUDGE_CAVEAT), alors que les scorers code
#: (translation_not_truncated en tête) l'avaient correctement identifiée à 35 % contre
#: 95-100 % pour les autres candidats. Un juge dans l'erreur ne doit pas pouvoir imposer
#: seul un résultat que le code sait déjà disqualifier.
CODE_SCORE_FLOOR = 0.5

#: Fourchette de longueur attendue pour un résumé exécutif, en mots.
SUMMARY_WORD_RANGE = (60, 150)

#: Confiance minimale de py3langid pour qu'une détection de langue soit exploitable.
LANGID_MIN_CONFIDENCE = 0.90

#: Nombre minimum de caractères pour tenter une détection de langue.
LANGID_MIN_CHARS = 40

#: Devises acceptées (ISO 4217). TND est la valeur normale, pas l'exception.
ALLOWED_CURRENCIES = {"TND", "EUR", "USD", "GBP", "CHF", "JPY", "CAD", "AED", "SAR"}

#: Marqueur que les agents doivent produire quand une information est absente.
NOT_FOUND_MARKER = "NON_TROUVE"


# --------------------------------------------------------------------------- #
# Score composite du rapport final
# --------------------------------------------------------------------------- #

#: Pondérations du score composite par tâche. Chaque composante est normalisée
#: sur [0, 1] parmi les candidats de la tâche avant pondération ; la latence est
#: inversée (plus bas = meilleur).
COMPOSITE_WEIGHTS = {
    "quality": 0.70,
    "latency": 0.20,
    "memory": 0.10,
}

#: Répartition interne de l'axe qualité entre le juge LLM et les scorers
#: déterministes. Le juge pèse plus lourd parce qu'il évalue le sens, mais les
#: scorers code gardent un poids réel : ce sont les seuls dont la fiabilité ne
#: dépend pas de celle d'un modèle de 8 B.
QUALITY_MIX = {
    "judge": 0.60,
    "code": 0.40,
}

assert abs(sum(COMPOSITE_WEIGHTS.values()) - 1.0) < 1e-9
assert abs(sum(QUALITY_MIX.values()) - 1.0) < 1e-9
