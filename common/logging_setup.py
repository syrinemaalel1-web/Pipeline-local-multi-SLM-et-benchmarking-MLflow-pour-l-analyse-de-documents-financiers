"""Configuration de journalisation commune aux scripts en ligne de commande.

La console Windows est en cp1252 : sans reconfiguration, la moindre trace contenant
de l'arabe fait échouer le handler de logging. Le corpus étant trilingue, ce n'est pas
un cas marginal.
"""

from __future__ import annotations

import logging
import sys


def setup(level: int = logging.INFO) -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
        force=True,
    )

    # Docling journalise une ligne par étape de pipeline et par document.
    logging.getLogger("docling").setLevel(logging.WARNING)
    logging.getLogger("docling_core").setLevel(logging.WARNING)
