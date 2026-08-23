"""Rendu PDF des documents Markdown du projet.

    python -m reporting.pdf                      # un PDF par document + revue
    python -m reporting.pdf --all                # rapport global combiné + revue
    python -m reporting.pdf --source chemin.md   # un fichier précis

Le rendu passe par un navigateur en mode headless plutôt que par une bibliothèque
PDF pure Python. La raison est le corpus : les meilleures sorties reproduites dans le
rapport sont parfois en arabe, et seul un vrai moteur de rendu applique correctement
la ligature et le sens d'écriture. `unicode-bidi: plaintext` laisse ensuite chaque
bloc déduire sa direction de son premier caractère fort, sans annoter le HTML.
"""

from __future__ import annotations

import argparse
import datetime as dt
import html
import logging
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from common import logging_setup
from config import JUDGE_MODEL, REPORTS_DIR, ROOT

log = logging.getLogger("pdf")

BROWSER_CANDIDATES = [
    Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
    Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
    Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
    Path(r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"),
]

STYLESHEET = """
@page {
  size: A4;
  margin: 18mm 16mm 20mm 16mm;
}

:root {
  --ink: #0e1b26;
  --body: #22323e;
  --muted: #61737f;

  --rule: #d7e1e7;
  --rule-soft: #e9eff3;
  --surface: #f5f8fa;
  --code-bg: #f2f6f8;

  /* Accent principal : pastilles, puces, liens, encarts. */
  --accent: #0e6b74;
  --accent-soft: #e3f0f1;

  /* Tête de tableau volontairement neutre. La couleur de section vit dans les
     titres ; des tableaux multicolores rendraient le document bavard. */
  --table-head: #1b3a45;

  /* Section courante, réassignée titre par titre plus bas. */
  --section: var(--accent);
  --section-soft: var(--accent-soft);
}

* { box-sizing: border-box; }

body {
  margin: 0;
  color: var(--body);
  font-family: "Segoe UI", "Calibri", "Noto Sans", system-ui, sans-serif;
  font-size: 10.5pt;
  line-height: 1.62;
  -webkit-print-color-adjust: exact;
  print-color-adjust: exact;
}

/* Chaque bloc déduit son sens d'écriture de son contenu : indispensable pour
   les sorties arabes reproduites dans le rapport. */
p, li, td, th, pre, blockquote, h1, h2, h3, h4, dd {
  unicode-bidi: plaintext;
  text-align: start;
}

/* ---------- Couverture ---------- */

/* Ni flex, ni `position: absolute` : les deux font déborder ou tronquer le contenu
   à l'impression Chrome sur une page paginée (vérifié : une ligne du tableau méta
   disparaissait silencieusement avec `position: absolute` + une `min-height`
   supérieure à la page). `min-height` seule, en flux normal (pas de flex, pas de
   position calculée), n'a pas ce problème — vérifié par rendu réel après coup :
   sert juste à porter le filet de pied de page en bas de la zone imprimable
   (297mm A4 - 18mm - 20mm de marges @page = 259mm), sans jamais forcer le
   contenu à s'y étirer. */
.cover {
  page-break-after: always;
  border-top: 6px solid var(--accent);
  /* Le bloc titre+méta ne remplit jamais une page A4 (259mm imprimables) : plutôt
     que de le laisser collé en haut avec tout le vide qui en découle en bas (essayé
     à 30mm : ~57% de la page restait blanche, uniquement sous le tableau), le
     padding est calé pour centrer verticalement ce bloc dans la page (valeur
     calée au rendu réel via reporting.pdf --preview, comme le reste de cette
     section — le mapping mm->page imprimée de Chrome n'est pas fiable en calcul
     théorique seul). */
  padding-top: 88mm;
}

.cover .eyebrow {
  font-size: 9pt;
  letter-spacing: .18em;
  text-transform: uppercase;
  color: var(--accent);
  font-weight: 600;
  margin-bottom: 10mm;
}

.cover h1 {
  font-size: 30pt;
  line-height: 1.18;
  margin: 0 0 6mm 0;
  border: 0;
  padding: 0;
  font-weight: 650;
  letter-spacing: -.015em;
  /* Sans cette annulation, la règle générale des h1 casse la couverture en deux. */
  page-break-before: avoid;
}

.cover .subtitle {
  font-size: 13pt;
  color: var(--muted);
  font-weight: 400;
  margin: 0 0 14mm 0;
  max-width: 130mm;
  line-height: 1.5;
}

.cover table.meta {
  /* Écart modeste sous le sous-titre, en flux normal (voir la note sur .cover) :
     valeur calée au rendu réel, pas au calcul théorique. Une valeur plus grande
     (90mm, essayée avant) ouvrait un second vide entre le sous-titre et le
     tableau en plus de celui déjà inévitable sous le tableau (la couverture
     occupe toujours une page entière à cause de page-break-after) — deux vides
     valent pire qu'un seul. */
  margin: 22mm 0 0 0;
  border-collapse: collapse;
  border: 0;
  border-top: 2.5px solid var(--accent);
  background: var(--surface);
  width: 142mm;
  font-size: 9.5pt;
}

.cover table.meta th {
  background: none;
  color: var(--muted);
  text-transform: uppercase;
  letter-spacing: .07em;
  font-size: 7.6pt;
  font-weight: 600;
  width: 46mm;
  padding: 2.4mm 6mm 2.4mm 4mm;
  vertical-align: top;
  border: 0;
}

.cover table.meta td {
  border: 0;
  padding: 2.4mm 4mm 2.4mm 0;
  font-weight: 550;
  color: var(--ink);
}

.cover table.meta tbody tr:nth-child(even) { background: none; }

/* ---------- Titres ---------- */

h1 {
  font-size: 19pt;
  font-weight: 650;
  color: var(--ink);
  letter-spacing: -.01em;
  margin: 0 0 6mm 0;
  padding-top: 4mm;
  border-top: 3px solid var(--accent);
  page-break-before: always;
  page-break-after: avoid;
}

/* Bandeau plein plutôt qu'un simple filet : à taille de police proche du corps
   de texte, seule une masse colorée sépare vraiment les sections. */
/* Bandeau plein (texte blanc) : à distance de lecture PDF, un titre teinté sur
   fond pâle se confond encore avec le corps. La masse colorée tranche. */
h2 {
  font-size: 13.5pt;
  font-weight: 650;
  letter-spacing: .01em;
  color: #fff;
  background: var(--section);
  border-radius: 3px;
  margin: 11mm 0 5mm 0;
  padding: 3.2mm 4.5mm;
  page-break-after: avoid;
}

/* Une teinte par section, en repère de navigation : le rapport répète les mêmes
   rubriques quatre fois. Les quatre nuances partagent valeur et saturation, pour
   rester une palette et non un arc-en-ciel. */
.content h2:nth-of-type(4n+1) { --section: #0e6b74; --section-soft: #e3f0f1; }
.content h2:nth-of-type(4n+2) { --section: #2f5b8c; --section-soft: #e7edf5; }
.content h2:nth-of-type(4n+3) { --section: #6a4677; --section-soft: #f0e9f3; }
.content h2:nth-of-type(4n+4) { --section: #8a552c; --section-soft: #f7ece2; }

h3 {
  font-size: 11.5pt;
  font-weight: 650;
  color: var(--ink);
  margin: 7mm 0 2.5mm 0;
  padding-left: 3mm;
  border-left: 2.5px solid var(--rule);
  page-break-after: avoid;
}

h4 {
  font-size: 9.4pt;
  font-weight: 650;
  color: var(--muted);
  text-transform: uppercase;
  letter-spacing: .06em;
  margin: 5mm 0 1.5mm 0;
  page-break-after: avoid;
}

/* La première section ne doit pas provoquer une page blanche après la couverture. */
.content > h1:first-child { page-break-before: avoid; }

p { margin: 0 0 3.2mm 0; }

strong { font-weight: 640; }

/* ---------- Listes ---------- */

ul, ol { margin: 0 0 3.5mm 0; padding-left: 6mm; }
li { margin-bottom: 1.4mm; }
li::marker { color: var(--accent); }

/* ---------- Tableaux ---------- */

table {
  width: 100%;
  border-collapse: collapse;
  margin: 4mm 0 6mm 0;
  font-size: 8.8pt;
  border: 1px solid var(--rule);
  page-break-inside: auto;
}

thead { display: table-header-group; }
tr { page-break-inside: avoid; }

th {
  background: var(--table-head);
  color: #fff;
  font-weight: 600;
  text-align: start;
  padding: 2.4mm 2.8mm;
  border: 0;
  border-right: 1px solid rgba(255, 255, 255, .16);
  font-size: 8.2pt;
  letter-spacing: .03em;
}

td {
  padding: 2.1mm 2.8mm;
  border-bottom: 1px solid var(--rule);
  border-right: 1px solid var(--rule);
  vertical-align: top;
}

th:last-child, td:last-child { border-right: 0; }
tbody tr:last-child td { border-bottom: 0; }
tbody tr:nth-child(even) { background: var(--surface); }
tbody td:first-child { font-weight: 560; color: var(--ink); }

/* Les noms de modèles sont des identifiants : les couper les rend illisibles. */
td:first-child code { white-space: nowrap; }

/* ---------- Modèle recommandé ---------- */

.badge {
  display: inline-block;
  background: var(--accent);
  color: #fff;
  font-size: 6.8pt;
  font-weight: 650;
  letter-spacing: .07em;
  text-transform: uppercase;
  padding: .3mm 1.4mm;
  border-radius: 2px;
  white-space: nowrap;
}

/* Déclaré après la règle de zébrage, qui a la même spécificité. */
tbody tr.recommended,
tbody tr.recommended:nth-child(even) { background: var(--accent-soft); }
tr.recommended td { font-weight: 600; }
tr.recommended td:first-child { border-left: 2.5px solid var(--accent); }

/* ---------- Citations / encarts ---------- */

/* Fond neutre bordé, et non teinté à l'accent : les encarts voisinent avec les
   bandeaux de section colorés, qui doivent rester les seules masses de couleur. */
blockquote {
  margin: 4mm 0;
  padding: 3mm 4mm;
  background: var(--surface);
  border: 1px solid var(--rule);
  border-left: 3.5px solid var(--accent);
  border-radius: 0 3px 3px 0;
  page-break-inside: avoid;
}

blockquote p:last-child { margin-bottom: 0; }

/* ---------- Code ---------- */

code {
  font-family: "Cascadia Mono", "Consolas", monospace;
  font-size: .88em;
  background: var(--code-bg);
  padding: .4mm 1.2mm;
  border-radius: 2px;
  color: #0b3d47;
}

pre {
  background: var(--code-bg);
  border-left: 3px solid var(--rule);
  padding: 3mm 4mm;
  margin: 3mm 0 5mm 0;
  border-radius: 0 3px 3px 0;
  font-size: 8.4pt;
  line-height: 1.5;
  /* Les sorties de modèles contiennent de très longues lignes : sans repli,
     le PDF les tronque au lieu de les afficher. */
  white-space: pre-wrap;
  word-break: break-word;
  overflow-wrap: anywhere;
  /* Pas d'anti-coupure ici : une sortie de modèle dépasse souvent la page, et
     l'interdire ne fait que la repousser en laissant un grand blanc avant. */
}

pre code { background: none; padding: 0; font-size: inherit; color: inherit; }

hr {
  border: 0;
  border-top: 1px solid var(--rule);
  margin: 7mm 0;
}

/* ---------- Sortie de modèle reproduite ---------- */

.model-output {
  margin: 4mm 0 6mm 0;
  padding: 4mm 5mm;
  background: var(--surface);
  border: 1px solid var(--rule);
  border-left: 3.5px solid var(--rule);
  border-radius: 0 3px 3px 0;
  font-size: 9.4pt;
}

.model-output > *:first-child { margin-top: 0; }
.model-output > *:last-child { margin-bottom: 0; }
.model-output hr { margin: 4mm 0; }

/* Titres du document reproduit : hiérarchisés entre eux, mais visiblement d'un
   autre ordre que ceux du rapport, qu'ils ne doivent pas concurrencer. */
.model-output .mo-h {
  font-weight: 650;
  color: var(--ink);
  line-height: 1.35;
  margin: 4.5mm 0 1.8mm 0;
  unicode-bidi: plaintext;
  page-break-after: avoid;
}

.model-output .mo-h1 { font-size: 12pt; }
.model-output .mo-h2 {
  font-size: 11pt;
  color: var(--accent);
  border-bottom: 1px solid var(--rule);
  padding-bottom: 1.2mm;
}
.model-output .mo-h3 { font-size: 10pt; }
.model-output .mo-h4,
.model-output .mo-h5,
.model-output .mo-h6 {
  font-size: 9.2pt;
  color: var(--muted);
  text-transform: uppercase;
  letter-spacing: .04em;
}

/* Mêmes bordures et zébrage que les tableaux du rapport : sans ça, les tableaux
   du document traduit restent lisibles mais paraissent « moins finis » que
   l'extraction. */
.model-output table {
  font-size: 8.2pt;
  background: #fff;
  margin: 3mm 0 4mm 0;
  border: 1px solid var(--rule);
}
.model-output th {
  background: var(--table-head);
  color: #fff;
  border-right: 1px solid rgba(255, 255, 255, .16);
}
.model-output td {
  border-bottom: 1px solid var(--rule);
  border-right: 1px solid var(--rule);
}
.model-output th:last-child,
.model-output td:last-child { border-right: 0; }
.model-output tbody tr:nth-child(even) { background: var(--surface); }
.model-output pre { background: #fff; }

a { color: var(--accent); text-decoration: none; }

/* ---------- Pied de document ---------- */

.colophon {
  margin-top: 10mm;
  padding-top: 3mm;
  border-top: 1px solid var(--rule);
  font-size: 8pt;
  color: var(--muted);
}
"""


def _find_browser() -> Path:
    for path in BROWSER_CANDIDATES:
        if path.exists():
            return path
    for name in ("msedge", "chrome", "chromium"):
        found = shutil.which(name)
        if found:
            return Path(found)
    raise RuntimeError(
        "Aucun navigateur Chromium trouvé (Edge ou Chrome). Le rendu PDF en dépend "
        "pour la mise en forme de l'arabe."
    )


def _markdown_to_html(text: str) -> str:
    from markdown_it import MarkdownIt

    parser = MarkdownIt("commonmark").enable(["table", "strikethrough"])
    return _apply_badges(_expand_model_outputs(parser.render(text), parser.render))


#: Sorties de modèle, isolées par `reporting.report` dans un bloc ```markdown
#: (ou ```text pour les rapports antérieurs).
_MODEL_OUTPUT = re.compile(
    r'<pre><code class="language-(?:markdown|text)">(.*?)</code></pre>', re.S
)

_HEADING_TAG = re.compile(r"<(?P<closing>/?)h(?P<level>[1-6])>")


def _demote_headings(html_text: str) -> str:
    """Sort les titres d'un document reproduit de la hiérarchie du rapport.

    Un « ## » du modèle deviendrait sinon une section du rapport : il hériterait
    du bandeau de section et décalerait la palette de toutes les suivantes, qui
    est assignée par rang.
    """

    def replace(match: re.Match[str]) -> str:
        if match.group("closing"):
            return "</div>"
        return f'<div class="mo-h mo-h{match.group("level")}">'

    return _HEADING_TAG.sub(replace, html_text)


#: Caractères de contrôle C1 (U+0080–U+009F) parfois émis par un tokenizer comme
#: séparateur de milliers erroné (observé : U+0085 "NEL" à la place d'une espace
#: insécable, ex. "2\x85561\x85500"). `str.splitlines()` et certains analyseurs
#: markdown les traitent comme des sauts de ligne : une seule ligne de tableau se
#: retrouve alors coupée en plusieurs lignes, ce qui casse le tableau produit.
_C1_CONTROL = re.compile("[-]")


def _prepare_model_markdown(text: str) -> str:
    """Normalise typographie et séparateurs avant le second passage Markdown.

    Les modèles écrivent souvent des lignes de métadonnées avec ` | ` qui ne sont
    pas des tableaux : laissés tels quels, ces pipes restent visibles dans le PDF
    alors que ceux des vrais tableaux ont disparu. On les remplace par un point
    médian, et on uniformise les montants.
    """
    from reporting.report import NBSP, normalise_numbers

    text = _C1_CONTROL.sub(NBSP, text)

    lines: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        is_table_row = stripped.startswith("|") and stripped.count("|") >= 2
        if not is_table_row and " | " in line:
            line = line.replace(" | ", " · ")
        lines.append(line)
    return normalise_numbers("\n".join(lines))


def _expand_model_outputs(html_text: str, render) -> str:
    """Rend réellement les sorties de modèle au lieu de les montrer en brut.

    Les documents traduits sont du Markdown : laissés dans un bloc de code, ils
    affichent leurs `**` et leurs `|` au lieu de gras et de tableaux. On les rend
    donc une seconde fois, dans un encart qui signale qu'il s'agit de contenu
    reproduit et non du texte du rapport.
    """

    def replace(match: re.Match[str]) -> str:
        source = _prepare_model_markdown(html.unescape(match.group(1)))
        return f'<div class="model-output">{_demote_headings(render(source))}</div>'

    return _MODEL_OUTPUT.sub(replace, html_text)


def _apply_badges(html_text: str) -> str:
    """Transforme le marqueur du modèle recommandé en pastille, et teinte sa ligne.

    Le rapport écrit un simple `**recommandé**` : le Markdown reste lisible seul, et
    la mise en couleur n'existe que dans le PDF.
    """
    from reporting.report import RECOMMENDED_MARK

    label = RECOMMENDED_MARK.strip("*")
    marker = f"<strong>{label}</strong>"

    def tag_row(match: re.Match[str]) -> str:
        row = match.group(0)
        if marker not in row:
            return row
        return row.replace("<tr>", '<tr class="recommended">', 1)

    html_text = re.sub(r"<tr>.*?</tr>", tag_row, html_text, flags=re.S)
    return html_text.replace(marker, f'<span class="badge">{label}</span>')


def _split_title(markdown: str) -> tuple[str, str]:
    """Sépare le titre de niveau 1 du reste, pour le porter sur la couverture."""
    match = re.match(r"\A\s*#\s+(.+?)\s*\n", markdown)
    if not match:
        return "Rapport", markdown
    return match.group(1), markdown[match.end() :]


def _cover(title: str, subtitle: str, meta: dict[str, str]) -> str:
    rows = "\n".join(
        f"<tr><th>{html.escape(k)}</th><td>{html.escape(v)}</td></tr>"
        for k, v in meta.items()
    )
    return f"""
<section class="cover">
  <div class="eyebrow">Benchmark multi-SLM local</div>
  <h1>{html.escape(title)}</h1>
  <p class="subtitle">{html.escape(subtitle)}</p>
  <table class="meta">{rows}</table>
</section>
"""


def build_html(
    markdown: str, *, subtitle: str, meta: dict[str, str], title: str | None = None
) -> str:
    extracted_title, body = _split_title(markdown)
    return f"""<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="utf-8">
<title>{html.escape(title or extracted_title)}</title>
<style>{STYLESHEET}</style>
</head>
<body>
{_cover(title or extracted_title, subtitle, meta)}
<main class="content">
{_markdown_to_html(body)}
<p class="colophon">Document généré automatiquement depuis les artefacts du benchmark
(<code>results/</code>). Il se régénère sans relancer d'inférence.</p>
</main>
</body>
</html>
"""


def html_to_pdf(html_text: str, pdf_path: Path) -> Path:
    browser = _find_browser()

    # Le navigateur résout --print-to-pdf dans son propre répertoire courant : un
    # chemin relatif écrirait ailleurs, silencieusement.
    pdf_path = pdf_path.resolve()
    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    # Sans cette suppression, un PDF déjà présent ferait passer le contrôle de
    # succès ci-dessous même si le navigateur n'a rien produit.
    pdf_path.unlink(missing_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        source = tmp_dir / "document.html"
        source.write_text(html_text, encoding="utf-8")

        command = [
            str(browser),
            "--headless=new",
            "--disable-gpu",
            # Profil jetable : sans lui, la commande échoue si le navigateur
            # de l'utilisateur est déjà ouvert.
            f"--user-data-dir={tmp_dir / 'profile'}",
            "--no-first-run",
            "--no-pdf-header-footer",
            "--run-all-compositor-stages-before-draw",
            "--virtual-time-budget=8000",
            f"--print-to-pdf={pdf_path}",
            source.as_uri(),
        ]

        result = subprocess.run(command, capture_output=True, text=True, timeout=180)

    if not pdf_path.exists() or pdf_path.stat().st_size == 0:
        raise RuntimeError(
            f"Le navigateur n'a produit aucun PDF (code {result.returncode}).\n"
            f"{result.stdout}\n{result.stderr}"
        )

    return pdf_path


def render(
    source: Path, pdf_path: Path, *, subtitle: str, meta: dict[str, str]
) -> Path:
    markdown = source.read_text(encoding="utf-8")
    html_to_pdf(build_html(markdown, subtitle=subtitle, meta=meta), pdf_path)
    log.info("%s -> %s (%.0f Ko)", source.name, pdf_path, pdf_path.stat().st_size / 1024)
    return pdf_path


def preview(pdf_path: Path, out_dir: Path, *, pages: int = 2, scale: float = 2.0):
    """Rasterise les premières pages en PNG, pour vérifier le rendu à l'oeil."""
    import pypdfium2 as pdfium

    out_dir.mkdir(parents=True, exist_ok=True)
    document = pdfium.PdfDocument(pdf_path)
    written = []

    for index in range(min(pages, len(document))):
        image = document[index].render(scale=scale).to_pil()
        target = out_dir / f"{pdf_path.stem}-p{index + 1}.png"
        image.save(target)
        written.append(target)

    return written


# --------------------------------------------------------------------------- #


def _parse_report_stem(stem: str):
    """(document_id ou `None` si combiné, `EvalMode`) si `stem` suit la convention de
    nommage des rapports (`reporting.report`), sinon `None`."""
    from config import EvalMode

    if stem == "rapport_metrics":
        return None, EvalMode.METRICS
    if stem == "rapport":
        return None, EvalMode.JUDGE
    if stem.endswith("_rapport_metrics"):
        return stem[: -len("_rapport_metrics")], EvalMode.METRICS
    if stem.endswith("_rapport"):
        return stem[: -len("_rapport")], EvalMode.JUDGE
    return None


def _regenerate_report(source: Path, document_id: str | None) -> None:
    """Réécrit `source` depuis `results/` si son nom suit la convention des rapports,
    juste avant sa conversion en PDF.

    Corrige un incident réel (2026-08-13, document northbridge) : `run_eval` avait
    tourné avec succès, mais `reporting.report` n'avait jamais été relancé entre les
    deux — `reporting.pdf` a alors reconverti tel quel un `.md` vieux de deux jours,
    silencieusement, sans qu'aucune commande n'ait échoué. Avec ce correctif, l'ordre
    des commandes ne peut plus produire un PDF périmé : le Markdown est toujours
    reconstruit depuis les résultats actuels avant d'être rendu, quel que soit l'appel
    (`--source` explicite, `--all`, ou la liste par défaut). Sans effet sur un fichier
    hors convention (ex. `docs/architecture-review.md`) — le nom du fichier fait foi
    pour déterminer le mode et le document ; `document_id` ne sert que de repli si le
    nom seul ne le précise pas (ex. `--out` personnalisé sur un fichier déjà nommé
    `rapport_metrics.md`).
    """
    parsed = _parse_report_stem(source.stem)
    if parsed is None:
        return
    doc_id, mode = parsed

    from config import EvalMode
    from reporting.report import build_metrics_report, build_report

    build = build_metrics_report if mode is EvalMode.METRICS else build_report
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text(build(document_id=doc_id or document_id), "utf-8")
    log.info("%s régénéré depuis results/ avant conversion.", source.name)


def _targets(*, combined: bool) -> list[tuple[Path, Path, str, str | None]]:
    """Documents Markdown à convertir, avec document_id optionnel pour les métadonnées."""
    from reporting.report import analysed_documents

    subtitle = (
        "Comparaison de petits modèles de langage locaux sur quatre tâches "
        "appliquées à des propositions financières tunisiennes."
    )
    revue = (
        ROOT / "docs" / "architecture-review.md",
        REPORTS_DIR / "revue-architecture.pdf",
        "Revue technique menée avant implémentation : risques matériels, "
        "vérification des API MLflow, et arbitrages retenus.",
        None,
    )

    if combined:
        return [
            (REPORTS_DIR / "rapport.md", REPORTS_DIR / "rapport.pdf", subtitle, None),
            revue,
        ]

    documents = analysed_documents()
    targets = [
        (
            REPORTS_DIR / f"{document_id}_rapport.md",
            REPORTS_DIR / f"{document_id}_rapport.pdf",
            subtitle,
            document_id,
        )
        for document_id in documents
    ]
    targets.append(revue)
    return targets


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, help="Markdown à convertir.")
    parser.add_argument("--out", type=Path, help="PDF de sortie.")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Convertir le rapport global combiné (rapport.md).",
    )
    parser.add_argument("--preview", action="store_true", help="Générer des aperçus PNG.")
    parser.add_argument("--preview-pages", type=int, default=2)
    args = parser.parse_args(argv)

    logging_setup.setup()

    base_meta = {
        "Projet": "Pipeline local multi-SLM",
        "Modèle juge": JUDGE_MODEL,
        "Généré le": dt.datetime.now().strftime("%d/%m/%Y à %H:%M"),
    }

    if args.source:
        targets = [
            (
                args.source,
                args.out or args.source.with_suffix(".pdf"),
                "Document du benchmark multi-SLM local.",
                None,
            )
        ]
    else:
        targets = _targets(combined=args.all)

    for source, pdf_path, subtitle, document_id in targets:
        _regenerate_report(source, document_id)
        if not source.exists():
            log.warning("%s est absent, ignoré.", source)
            continue

        meta = dict(base_meta)
        if document_id:
            meta["Document analysé"] = document_id

        render(source, pdf_path, subtitle=subtitle, meta=meta)

        if args.preview:
            for image in preview(
                pdf_path, REPORTS_DIR / "preview", pages=args.preview_pages
            ):
                log.info("  aperçu : %s", image)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
