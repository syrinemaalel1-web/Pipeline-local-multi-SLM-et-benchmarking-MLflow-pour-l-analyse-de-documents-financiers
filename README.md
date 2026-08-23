# Benchmark multi-SLM local — propositions financières tunisiennes

## 📋 Table des matières

- [🎯 Présentation du projet](#-présentation-du-projet)
- [🏢 Contexte](#-contexte)
- [🎯 Objectif et problème traité](#-objectif-et-problème-traité)
- [🚀 Démarrage rapide](#-démarrage-rapide)
- [🏗️ Architecture du système](#️-architecture-du-système)
- [🛠️ Technologies utilisées](#️-technologies-utilisées)
- [⚠️ Limites connues du protocole](#️-limites-connues-du-protocole)
- [✨ Points forts à retenir](#-points-forts-à-retenir)

---

## 🎯 Présentation du projet

Pipeline complet — de l'ingestion d'un document brut jusqu'à un rapport comparatif
consultable dans un navigateur — qui compare plusieurs modèles de langage légers (SLM,
7-8B, quantifiés) sur des tâches réelles d'analyse financière, avec une méthodologie
d'évaluation rigoureuse, reproductible et déployée en application web.

## 🏢 Contexte

Projet réalisé dans le cadre d'un **stage d'ingénieure IA de deux mois**. L'objectif du
stage : évaluer si des modèles de langage légers, exécutés localement, peuvent remplacer
un grand modèle cloud pour l'analyse de documents financiers — en tenant compte des
contraintes réelles d'un tel contexte (confidentialité des documents, coût, ressources
matérielles limitées).

## 🎯 Objectif et problème traité

Les grands modèles (GPT-4, Claude...) donnent d'excellents résultats sur l'analyse de
documents financiers, mais ont un coût et posent un problème de confidentialité pour des
documents sensibles (propositions de financement, données bancaires). La question posée
par ce projet :

> **Peut-on obtenir une qualité suffisante avec des modèles légers (7-8B), exécutés
> 100 % en local, sans jamais envoyer un document à un service externe ?**

Pour y répondre sérieusement, il ne suffit pas de lancer quelques modèles et de regarder
les réponses "à l'œil" — il faut un protocole de comparaison rigoureux, reproductible, et
une méthode d'évaluation elle-même validée. C'est l'objet de ce projet.

---

## 🚀 Démarrage rapide

Deux façons de lancer le projet — **dans les deux cas, Ollama doit tourner nativement sur
la machine hôte** (avec les modèles déjà tirés) : ce n'est pas conteneurisable de façon
fiable, il a besoin d'un accès direct au GPU. Docker ne dispense donc pas de l'installer.

**Ce qu'il faut avoir installé, avant de commencer :**

| Prérequis | Pour quel usage |
|---|---|
| [Ollama](https://ollama.com) natif, modèles tirés, `OLLAMA_CONTEXT_LENGTH=8192` | Dans tous les cas |
| Docker Desktop | Pour l'option A (recommandée pour découvrir le projet) |
| Python 3.12 + Node.js 20 | Pour l'option B (développement, ligne de commande) |

### Option A — Cloner et lancer avec Docker (le plus simple)

```powershell
git clone <url-du-dépôt>
cd finance
docker compose up -d --build
```

Le site est alors sur `http://localhost:3000`, l'API sur `http://localhost:8000`. Rien
d'autre à installer côté Python/Node — tout est packagé dans les 2 images construites par
Docker Compose.

### Option B — Lancer en local, sans Docker (développement)

```powershell
# Backend
python -m venv venv
.\venv\Scripts\python.exe -m pip install -r requirements.txt -r requirements-api.txt
uvicorn api.main:app --reload --port 8000

# Frontend, dans un second terminal
cd web
npm install
npm run dev
```

Le site est alors sur `http://localhost:5173`. Cette option permet aussi d'utiliser le
pipeline en pur ligne de commande, sans passer par le site — voir
[`docs/architecture-review.md`](docs/architecture-review.md) et les modules `orchestration/`.

### Arborescence complète du projet

```
finance/
├── config.py                     # Source unique de vérité (modèles, poids, chemins)
├── requirements.txt
├── requirements-api.txt
├── docker-compose.yml
├── Dockerfile.backend
├── mlflow.db
├── mlruns/
├── CLAUDE.md
├── PROJECT.md
├── README.md
├── docs/
│   ├── architecture-review.md
│   └── fiabilite-juge.md
├── notebooks/
│   └── rapport.ipynb
├── common/
│   ├── language.py
│   ├── tokens.py
│   └── logging_setup.py
├── extraction/
│   ├── docling_loader.py
│   └── ingest.py
├── prompts/
│   ├── extraction.txt
│   ├── summary.txt
│   ├── translation.txt
│   └── qa.txt
├── schemas/
│   └── proposal.py
├── agents/
│   ├── ollama_client.py
│   ├── extraction.py
│   ├── summarize.py
│   ├── translate.py
│   ├── qa.py
│   └── json_utils.py
├── orchestration/
│   ├── corpus.py
│   ├── store.py
│   ├── mlflow_setup.py
│   ├── run_agents.py
│   ├── run_eval.py
│   └── smoke_test.py
├── evaluation/
│   ├── code_scorers.py
│   ├── judges.py
│   ├── legacy_metrics.py
│   └── judge_probe.py
├── reporting/
│   ├── report.py
│   └── pdf.py
├── api/
│   ├── main.py
│   ├── jobs.py
│   └── documents.py
├── web/
│   ├── Dockerfile
│   ├── nginx.conf
│   ├── package.json
│   └── src/
│       ├── App.jsx
│       ├── api.js
│       ├── pages/
│       ├── components/
│       └── hooks/
├── data/
│   ├── qa_questions.json
│   └── processed/
├── results/
│   ├── generation/
│   ├── evaluation/
│   └── evaluation_metrics/
├── reports/
└── tests/
```

---

## 🏗️ Architecture du système

### Flux complet

```
┌────────────────────────────────────────────────────────────────┐
│                     INTERFACE UTILISATEUR                      │
│                    React + Vite (Frontend)                     │
│       Dashboard comparatif · Suivi temps réel · Rapports       │
└────────────────────────────────────────────────────────────────┘
                                │ HTTP (polling)
                                ▼
┌────────────────────────────────────────────────────────────────┐
│                          API BACKEND                           │
│                            FastAPI                              │
│         Orchestration des jobs · Statut des documents          │
└────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌────────────────────────────────────────────────────────────────┐
│                       PIPELINE D'ANALYSE                        │
│                                                                  │
│   Ingestion → Génération (4 agents × N modèles) → Évaluation    │
└────────────────────────────────────────────────────────────────┘
                                │
              ┌─────────────────┴─────────────────┐
              ▼                                    ▼
┌────────────────────────────┐        ┌────────────────────────────┐
│           OLLAMA           │        │           MLFLOW           │
│       8 modèles 7-8B       │        │      Judge + Metrics       │
│      GPU local (hôte)      │        │     Tracking des runs      │
└────────────────────────────┘        └────────────────────────────┘
```

Le pipeline est volontairement **séquentiel et persistant** : chaque étape écrit son
résultat sur disque avant de passer à la suivante. Un ajustement du juge ou d'un scorer
ne rejoue jamais les heures d'inférence déjà effectuées — seule l'évaluation est relancée.

### Étape 1 — Ingestion

- Extraction du texte via **Docling** (PDF et DOCX), y compris les tableaux.
- Détection automatique de la langue (FR / EN / AR) et calcul de la direction de
  traduction attendue.
- Estimation du budget de tokens et avertissement si un document dépasse la fenêtre de
  contexte du modèle — évite qu'Ollama tronque silencieusement l'entrée.
- Résultat mis en cache : Docling n'est plus jamais rappelé ensuite, même si la
  génération est relancée dix fois.

### Étape 2 — Génération (4 agents)

Chaque document passe par **4 agents indépendants**, chacun testé sur plusieurs modèles
candidats :

| Tâche | Description | Exemples de modèles candidats |
|---|---|---|
| **Extraction** | Structuration en JSON (client, montant, devise, taux, durée, clauses) | `llama3.1:8b`, `mistral:7b`, `qwen2.5:7b`, `phi4-mini:latest`, `aya-expanse:8b` |
| **Résumé** | Synthèse exécutive du document | mêmes candidats |
| **Traduction** | FR↔EN↔AR selon la langue source | + `translategemma:latest` (spécialisé), `mannix/llamax3-8b-alpaca` |
| **Question-réponse** | 4 questions par document, avec abstention attendue sur au moins une | + `deepseek-r1:8b` |

**Points d'ingénierie notables :**
- Boucle *modèle-majeure* : chaque modèle (jusqu'à 5-8 Go) est chargé une seule fois et
  enchaîne tous les documents, au lieu de recharger le modèle à chaque appel.
- Pipeline **repris automatiquement** après toute interruption (coupure, redémarrage) :
  chaque appel réussi est persisté immédiatement, au grain (tâche, modèle, document,
  question) — rien n'est jamais recalculé pour rien.
- Optimisé pour tourner sur un poste grand public à ressources limitées (GPU 4 Go,
  offload CPU partiel) — une campagne complète représente 12 à 20 h d'inférence.

### Étape 3 — Évaluation : deux pipelines indépendants

Sans vérité terrain (aucun "corrigé" humain fourni par un annotateur), la qualité d'une
sortie doit être jugée automatiquement. **Plutôt que de se fier à une seule méthode de
notation, le projet en implémente deux, indépendantes et exécutées séparément**, pour
croiser les résultats et ne jamais dépendre d'un seul jugement.

```
┌────────────────────────────────────────────────────────────────┐
│              Sortie d'un agent + document source               │
└────────────────────────────────────────────────────────────────┘
                                │
               ┌────────────────┴────────────────┐
               ▼                                 ▼
┌──────────────────────────────┐    ┌──────────────────────────────┐
│          PIPELINE A          │    │          PIPELINE B          │
│            JUDGE             │    │           METRICS            │
│                              │    │                              │
│    Scorers déterministes     │    │        Faithfulness /        │
│   (JSON, langue, longueur,   │    │       Answer relevance       │
│     chiffres, abstention)    │    │    (mlflow.metrics.genai)    │
│              +               │    │              +               │
│     Juge LLM sur mesure      │    │     ARI / Flesch-Kincaid     │
│    (make_judge, référence    │    │          (textstat,          │
│      avant comparaison)      │    │        sans appel LLM)       │
│              =               │    │                              │
│       Score composite        │    │                              │
│    60% juge + 40% scorers    │    │                              │
└──────────────────────────────┘    └──────────────────────────────┘
               │                                 │
               └────────────────┬────────────────┘
                                ▼
┌────────────────────────────────────────────────────────────────┐
│                       RAPPORT COMPARATIF                       │
└────────────────────────────────────────────────────────────────┘
```

#### Pipeline A — Judge (juge LLM + scorers déterministes)

C'est le pipeline principal, construit autour d'un principe simple : **le code vérifie
la forme, un LLM évalue le fond.**

**Scorers déterministes (code, gratuits, 100 % fiables)** — 8 vérifications réparties
sur les 4 tâches : validité JSON et respect du schéma, champs obligatoires présents,
conformité de la langue de sortie, longueur du résumé dans la plage attendue,
préservation des valeurs numériques en traduction, non-troncature, abstention correcte
en Q&R. Rien de coûteux : aucun appel modèle, résultat instantané et déterministe.

**Juge LLM sur mesure (`make_judge`, MLflow)** — un juge dédié par tâche
(`extraction_fidelity`, `summary_faithfulness`, `translation_fidelity`,
`qa_groundedness`), avec un protocole conçu spécifiquement pour éviter le biais de
complaisance des LLM-juges :

1. Le juge **construit sa propre référence** à partir du seul document source, *avant*
   de lire la sortie à évaluer — il ne ratifie jamais un travail déjà présenté.
2. Il compare ensuite sa référence à la sortie, écart par écart, et applique un barème
   ancré sur le nombre d'écarts constatés (jamais une note "au feeling").
3. La note est un type énuméré (1 à 5), pas un entier libre — le décodage JSON est
   **contraint au niveau du schéma**, empêchant structurellement le modèle de sortir du
   barème.
4. Les justifications de forme ("bien présenté", "clair et structuré") sont explicitement
   déclarées irrecevables : seul un écart de fond, chiffré et vérifiable, compte.

**Validation automatisée du juge (contrôle négatif)** — avant de faire confiance à ce
juge, il a été testé comme un système qu'on veut casser : des fautes connues (montant
falsifié, traduction tronquée de moitié, identifiant bancaire inventé) ont été injectées
dans de vraies sorties déjà correctes, pour vérifier que le juge les détecte. Un premier
juge candidat ne détectait qu'**1 faute injectée sur 5**. Le protocole anti-complaisance
ci-dessus, conçu en réaction à ce résultat, a fait passer la détection à **3 fautes sur
5, avec une spécificité de 4/4** (aucun faux positif sur les sorties correctes) — un gain
mesuré entièrement par du code, en rejouant le même test après chaque itération du
prompt. Détail dans [`docs/fiabilite-juge.md`](docs/fiabilite-juge.md).

**Score composite** : 60 % note du juge + 40 % scorers déterministes, par modèle et par
tâche — le classement final ne repose donc jamais sur le LLM seul.

#### Pipeline B — Metrics (méthode de référence indépendante)

Un second pipeline, basé sur `mlflow.metrics.genai` et `textstat`, exécuté séparément
(`mlflow.evaluate()`) et persisté dans son propre dossier de résultats — **jamais fusionné
avec le pipeline Judge**. Il apporte deux angles que le premier ne couvre pas :

- **Faithfulness** et **Answer relevance** — mesurées par un juge indépendant, avec sa
  propre mécanique de décodage contraint, sur la traduction, le résumé et le Q&R.
- **ARI** et **Flesch-Kincaid** — deux métriques de lisibilité *purement calculatoires*
  (aucun appel LLM, juste de la statistique sur le texte), qui estiment le niveau
  scolaire requis pour comprendre un résumé donné.

**Pourquoi deux pipelines plutôt qu'un seul plus complet ?** Parce qu'un jugement de
qualité obtenu par une seule méthode est un jugement non vérifié. Faire tourner deux
méthodologies indépendantes sur les mêmes sorties permet de repérer les cas où elles
divergent (par exemple une sortie qui obtient un très bon score de fidélité mais un
niveau de lisibilité anormalement élevé) — un signal qu'aucun des deux pipelines,
utilisé seul, n'aurait révélé.

### Étape 4 — Rapport comparatif

Pour chaque document, un rapport Markdown (converti en PDF) est généré automatiquement,
combinant les deux pipelines :

- Un tableau par tâche (modèle, note, latence, débit, mémoire, % GPU, score composite).
- Le modèle recommandé, avec la sortie réelle qu'il a produite — jamais juste un chiffre
  isolé, toujours la preuve concrète derrière la note.
- Les réserves de comparabilité (ex. un modèle spécialisé plus petit, ou un candidat
  exclu de la recommandation faute d'atteindre le seuil minimal de fiabilité de format).
- Un dashboard agrégé, tous documents confondus, pour dégager une tendance générale par
  tâche plutôt que le résultat d'un seul cas.

### Déploiement — Application web

Le pipeline (initialement en ligne de commande) a été packagé en **application web
complète** pour être utilisable sans connaissance technique :

- **Frontend React** — dépôt de document par glisser-déposer, suivi de progression en
  temps réel (0 à 100 %, calculé sur le nombre réel d'unités de travail traitées, pas une
  simulation), tableau de bord comparatif, rapport consultable directement dans le
  navigateur.
- **Backend FastAPI** — orchestre les étapes (ingestion → génération → évaluation) comme
  des jobs suivis par polling, sans jamais dupliquer la logique du pipeline existant :
  l'API appelle le même code, ne le réimplémente pas.

### Déploiement — Docker

L'application est conteneurisée pour être reproductible sur n'importe quelle machine :

```
┌────────────────────────────────────────────────────────────────┐
│                 NAVIGATEUR (poste utilisateur)                 │
└────────────────────────────────────────────────────────────────┘
                                │ localhost:3000 / localhost:8000
                                ▼
               ┌────────────────┴────────────────┐
               ▼                                 ▼
┌──────────────────────────────┐    ┌──────────────────────────────┐
│      CONTENEUR FRONTEND      │    │      CONTENEUR BACKEND       │
│    React (build) + nginx     │    │  FastAPI + pipeline complet  │
└──────────────────────────────┘    └──────────────────────────────┘
               │                                 │
               └────────────────┬────────────────┘
                                ▼
┌────────────────────────────────────────────────────────────────┐
│                  ORCHESTRÉ PAR DOCKER COMPOSE                  │
└────────────────────────────────────────────────────────────────┘
                                │
               ┌────────────────┴────────────────┐
               ▼                                 ▼
┌──────────────────────────────┐    ┌──────────────────────────────┐
│    OLLAMA (machine hôte)     │    │        VOLUMES MONTÉS        │
│     Accès direct au GPU      │    │ data/ · results/ · reports/  │
└──────────────────────────────┘    └──────────────────────────────┘
```

- **2 conteneurs** orchestrés par **Docker Compose** : un pour le backend (Python +
  FastAPI + pipeline), un pour le frontend (React compilé, servi par nginx).
- **Ollama reste sur la machine hôte** (accès direct au GPU), les conteneurs le
  rejoignent via le réseau Docker — pas de virtualisation du GPU, qui resterait fragile
  sous Windows.
- Les données (documents, résultats, rapports) sont montées en volumes : elles
  survivent à tout redémarrage ou reconstruction des conteneurs.

---

## 🛠️ Technologies utilisées

**Modèles & inférence**

| Outil | Rôle |
|---|---|
| Ollama | Serveur d'inférence local pour les 8 modèles candidats (7-8B, quantifiés Q4) |
| Docling | Extraction de texte et de tableaux depuis PDF et DOCX |

**Orchestration & évaluation**

| Outil | Rôle |
|---|---|
| MLflow (`mlflow.genai.evaluate`) | Pipeline Judge — juge LLM + scorers déterministes |
| MLflow (`mlflow.evaluate`) | Pipeline Metrics — méthode de référence indépendante |
| MLflow Tracking | Suivi et persistance de tous les runs d'inférence et d'évaluation |

**Application web**

| Outil | Rôle |
|---|---|
| FastAPI | API backend, orchestration des jobs |
| React + Vite | Interface utilisateur, dashboard, rapports |

**Déploiement**

| Outil | Rôle |
|---|---|
| Docker | Conteneurisation du backend et du frontend |
| Docker Compose | Orchestration des conteneurs, réseau, volumes |

**Langues couvertes**

| Langue | Statut |
|---|---|
| Français | ✅ |
| Anglais | ✅ |
| Arabe | ✅ |

---

## ✨ Points forts à retenir

- **Zéro dépendance cloud** — traitement 100 % local, adapté à des documents financiers
  confidentiels.
- **Méthodologie d'évaluation double et croisée**, pas un score unique aveugle.
- **Juge LLM conçu et testé comme un système à casser**, pas seulement à faire
  confiance — contrôle négatif automatisé, gain de détection mesuré et reproductible.
- **Pipeline résilient** : reprise automatique après interruption, aucune inférence
  perdue sur des campagnes de plusieurs heures.
- **De l'expérimentation en ligne de commande à une application web déployée en
  conteneurs** — un projet complet, pas un notebook isolé.
