"""PDF et DOCX vers texte structuré unifié, via Docling.

Ce module n'est jamais importé pendant le benchmark : Docling charge des modèles de
layout et de structure de tableaux qui se disputeraient la RAM avec Ollama. Il tourne
une fois, à l'ingestion, et écrit un cache sur disque (voir ingest.py).
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Any

SUPPORTED_SUFFIXES = {".pdf", ".docx"}

_ARABIC = re.compile(r"[\u0600-\u06FF\u0750-\u077F]")


@lru_cache(maxsize=1)
def _converter():
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    pipeline_options = PdfPipelineOptions()
    # Les documents sont garantis en texte natif (PROJECT.md §1). L'OCR ne ferait
    # que ralentir et introduire du bruit.
    pipeline_options.do_ocr = False
    pipeline_options.do_table_structure = True

    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )


def _rtl_health(text: str) -> dict[str, Any] | None:
    """Indice de texte arabe restitué à l'envers par l'extracteur PDF.

    Docling peut inverser l'ordre des caractères sur du RTL. Le symptôme est un texte
    où l'arabe est présent mais où les mots ne se recomposent pas. On ne peut pas le
    détecter de façon certaine sans dictionnaire ; on signale plutôt la présence
    d'arabe pour que le premier document arabe soit relu à l'oeil avant de conclure
    que les modèles sont mauvais.
    """
    arabic_chars = len(_ARABIC.findall(text))
    if arabic_chars == 0:
        return None
    return {
        "arabic_char_ratio": round(arabic_chars / max(len(text), 1), 3),
        "note": (
            "Document contenant de l'arabe : vérifier visuellement l'ordre des "
            "caractères (risque d'inversion RTL à l'extraction PDF) avant "
            "d'interpréter les scores."
        ),
    }


def extract_document(path: str | Path) -> dict[str, Any]:
    """Extrait un PDF ou un DOCX vers texte + métadonnées.

    Renvoie ``{"text", "n_pages", "n_tables", "n_chars", "source_path", "format",
    "rtl_check"}``. Le texte est du Markdown : il conserve titres et tableaux, ce qui
    aide les modèles d'extraction à localiser les montants sans coûter beaucoup de
    tokens.
    """
    path = Path(path)
    if path.suffix.lower() not in SUPPORTED_SUFFIXES:
        raise ValueError(
            f"Format non supporté : {path.suffix!r}. Attendu : "
            f"{sorted(SUPPORTED_SUFFIXES)}"
        )

    result = _converter().convert(path)
    doc = result.document
    text = doc.export_to_markdown()

    return {
        "source_path": str(path),
        "format": path.suffix.lower().lstrip("."),
        "text": text,
        "n_pages": len(doc.pages) or None,
        "n_tables": len(doc.tables),
        "n_chars": len(text),
        "rtl_check": _rtl_health(text),
    }
