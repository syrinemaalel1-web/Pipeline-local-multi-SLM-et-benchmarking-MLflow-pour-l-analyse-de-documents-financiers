"""Étape d'ingestion one-shot : data/*.pdf|docx -> data/processed/*.json.

À lancer une fois, avant tout benchmark :

    python -m extraction.ingest

Le cache produit contient le texte, la langue détectée, la direction de traduction
imposée, le nombre de tokens et les avertissements de dépassement de contexte. Le
benchmark ne relit que ce cache : Docling n'est plus jamais chargé ensuite.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Any

from common import logging_setup
from common.language import detect_language
from common.tokens import context_warnings, count_tokens
from config import (
    DATA_PROCESSED_DIR,
    DATA_RAW_DIR,
    QA_QUESTIONS_PATH,
    TRANSLATION_DIRECTION,
    Lang,
)
from extraction.docling_loader import SUPPORTED_SUFFIXES, extract_document

log = logging.getLogger("ingest")


_NON_ASCII_ID = re.compile(r"[^a-z0-9]+")


def _document_id(path: Path) -> str:
    """Identifiant ASCII stable, utilisé comme nom de dossier dans results/.

    Certains documents ont un nom de fichier en arabe : le translittérer serait
    hasardeux, on garde donc un suffixe de hachage pour rester unique et lisible.
    """
    stem = _NON_ASCII_ID.sub("_", path.stem.lower()).strip("_")
    digest = hashlib.sha1(path.stem.encode("utf-8")).hexdigest()[:8]
    return f"{stem}_{digest}" if stem else f"doc_{digest}"


def find_documents(raw_dir: Path) -> list[Path]:
    return sorted(
        p
        for p in raw_dir.rglob("*")
        if p.is_file()
        and p.suffix.lower() in SUPPORTED_SUFFIXES
        and DATA_PROCESSED_DIR not in p.parents
    )


def ingest_one(path: Path) -> dict[str, Any]:
    doc = extract_document(path)
    text = doc["text"]

    lang, confidence = detect_language(text)
    n_tokens = count_tokens(text)

    record: dict[str, Any] = {
        "document_id": _document_id(path),
        **doc,
        "lang": lang.value if lang else None,
        "lang_confidence": round(confidence, 4),
        "target_lang": TRANSLATION_DIRECTION[lang].value if lang else None,
        "n_tokens_estimated": n_tokens,
        "context_warnings": context_warnings(n_tokens),
    }
    return record


def _report(record: dict[str, Any]) -> None:
    doc_id = record["document_id"]

    if record["lang"] is None:
        log.error(
            "%s : langue indéterminée (confiance %.2f). Un document doit être dans "
            "une seule langue parmi fr/en/ar — ce document sera ignoré par le "
            "benchmark.",
            doc_id,
            record["lang_confidence"],
        )
    else:
        log.info(
            "%s : %s -> traduction vers %s | %s pages, %s tableaux, ~%d tokens",
            doc_id,
            record["lang"],
            record["target_lang"],
            record["n_pages"],
            record["n_tables"],
            record["n_tokens_estimated"],
        )

    if record["rtl_check"]:
        log.warning("%s : %s", doc_id, record["rtl_check"]["note"])

    for scope, message in record["context_warnings"].items():
        log.warning("%s [%s] %s", doc_id, scope, message)


#: Gabarit générique repris tel quel de qa_questions.example.json — c'est aussi,
#: mot pour mot, ce qui a été utilisé à la main pour les 9 premiers documents du
#: corpus (voir data/qa_questions.json). Générique par construction (montant,
#: conditions financières, destinataire, IBAN absent) : s'applique à n'importe
#: quelle proposition de financement, mais PAS à un document d'un autre genre
#: (rapport annuel, etc.) — aucune détection de genre ici, limite acceptée
#: sciemment (voir CLAUDE.md, décision utilisateur du 2026-08-19).
#:
#: `q_duree_taux` : reformulée le 2026-08-19 après un cas réel — la question
#: d'origine ("Sur quelle durée s'étale le financement, et quel taux ou quelle
#: marge s'y applique ?") suppose un PRÊT bancaire (durée + taux d'intérêt).
#: Le corpus contient en réalité au moins trois genres de "proposition
#: financière", pas un seul : prêt bancaire (durée + taux d'intérêt), projet
#: d'investissement autofinancé (payback + TRI/VAN, aucun prêt), contrat de
#: service à un client (échéancier de paiement, "marge" = marge commerciale,
#: pas un taux de prêt). Une détection de genre par mots-clés a été essayée et
#: écartée : "marge" et "taux" apparaissent dans les trois genres avec des sens
#: différents, aucun mot-clé simple ne les sépare fiablement (vérifié sur le
#: corpus réel). La question est donc élargie pour rester répondable dans les
#: trois cas plutôt que de deviner le genre — un prêt y répond par sa durée et
#: son taux d'intérêt, un projet d'investissement par son TRI/délai de retour,
#: un contrat de service par son échéancier ; aucun des trois ne doit produire
#: "NON_TROUVE" de façon injustifiée.
_QA_TEMPLATE = [
    ("q_montant", "Quel est le montant total du financement demandé, et dans quelle devise ?", False),
    (
        "q_duree_taux",
        "Quelles sont les conditions financières du dossier — durée et taux d'intérêt d'un "
        "financement, taux de rentabilité (TRI) ou délai de retour sur investissement d'un "
        "projet, ou échéancier de paiement d'un contrat de service ?",
        False,
    ),
    ("q_destinataire", "À quel établissement financier cette proposition est-elle adressée ?", False),
    ("q_iban_absent", "Quel est le numéro IBAN du compte bancaire du bénéficiaire ?", True),
]


def _seed_qa_questions(document_id: str, text: str) -> int:
    """Ajoute le gabarit Q&A générique pour un document tout juste ingéré.

    N'écrase jamais une entrée existante : si `document_id` a déjà des
    questions (curées à la main, ou déjà seedées par un run précédent), ne
    touche à rien. C'est ce qui permet d'exclure délibérément un document du
    Q&A (ex. rapport_zeitouna_2025, qui n'est pas une proposition de
    financement) — il suffit de ne jamais l'y laisser rentrer, y compris après
    un `--force` (voir l'appelant : le seeding n'est déclenché que pour un
    document réellement nouveau, jamais sur une réingestion).

    Garde-fou sur l'abstention : si le mot "IBAN" apparaît réellement dans le
    texte, `expect_abstention=True` serait faux pour ce document précis
    (pénaliserait à tort un modèle qui le rapporte correctement) — la question
    est alors omise plutôt que posée avec la mauvaise attente.
    """
    existing: list[dict[str, Any]] = []
    if QA_QUESTIONS_PATH.exists():
        try:
            existing = json.loads(QA_QUESTIONS_PATH.read_text("utf-8"))
        except json.JSONDecodeError:
            log.warning(
                "%s illisible : questions Q&A non générées automatiquement pour %s.",
                QA_QUESTIONS_PATH,
                document_id,
            )
            return 0

    if any(q.get("document_id") == document_id for q in existing):
        return 0

    has_iban = "iban" in text.lower()
    if has_iban:
        log.warning(
            "%s : IBAN mentionné dans le document — question d'abstention Q&A "
            "omise (l'attente 'absent du document' serait fausse pour ce cas).",
            document_id,
        )

    added = 0
    for qid, question, abstention in _QA_TEMPLATE:
        if abstention and has_iban:
            continue
        existing.append(
            {
                "id": qid,
                "document_id": document_id,
                "question": question,
                "expect_abstention": abstention,
            }
        )
        added += 1

    QA_QUESTIONS_PATH.write_text(
        json.dumps(existing, ensure_ascii=False, indent=2) + "\n", "utf-8"
    )
    return added


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=DATA_RAW_DIR)
    parser.add_argument("--out-dir", type=Path, default=DATA_PROCESSED_DIR)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Ré-ingérer les documents déjà présents dans le cache.",
    )
    args = parser.parse_args(argv)

    logging_setup.setup()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    documents = find_documents(args.raw_dir)

    if not documents:
        log.error(
            "Aucun PDF ou DOCX trouvé dans %s. Déposez-y vos documents avant de "
            "lancer l'ingestion.",
            args.raw_dir.resolve(),
        )
        return 1

    log.info("%d document(s) à traiter depuis %s", len(documents), args.raw_dir)
    # Marqueurs consommés par api/jobs.py::progress pour une barre de
    # progression réelle côté site (voir ce module pour le format attendu) —
    # sans effet sur l'usage CLI, juste deux lignes de log en plus.
    log.info("PROGRESS_TOTAL=%d", len(documents))

    warned = 0
    by_lang: dict[str, int] = {}

    for path in documents:
        log.info("PROGRESS_STEP")
        out_path = args.out_dir / f"{_document_id(path)}.json"
        # Capturé avant le write : distingue un document réellement nouveau
        # d'une réingestion --force d'un document déjà connu — seul le premier
        # cas déclenche le seeding Q&A ci-dessous (voir _seed_qa_questions).
        is_new_document = not out_path.exists()
        if out_path.exists() and not args.force:
            log.info("%s : déjà en cache, ignoré (--force pour refaire)", out_path.name)
            continue

        try:
            record = ingest_one(path)
        except Exception:
            log.exception("Échec de l'ingestion de %s", path)
            continue

        _report(record)
        warned += bool(record["context_warnings"])
        by_lang[record["lang"] or "inconnu"] = by_lang.get(record["lang"] or "inconnu", 0) + 1

        out_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), "utf-8")

        if is_new_document:
            added = _seed_qa_questions(record["document_id"], record["text"])
            if added:
                log.info(
                    "%s : %d question(s) Q&A générées automatiquement dans %s",
                    record["document_id"],
                    added,
                    QA_QUESTIONS_PATH.name,
                )

    log.info("Répartition par langue : %s", by_lang or "aucun nouveau document")

    missing_langs = {lang.value for lang in Lang} - set(by_lang)
    if missing_langs and by_lang:
        log.warning(
            "Aucun document en %s. Le benchmark ne pourra pas comparer les modèles "
            "sur ces langues, alors que la contrainte de langue est un critère "
            "central du protocole.",
            sorted(missing_langs),
        )

    if warned:
        log.warning(
            "%d document(s) dépassent le budget de contexte. Ollama les tronquerait "
            "silencieusement : réduisez-les ou acceptez que les scores de ces "
            "documents soient ininterprétables.",
            warned,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
