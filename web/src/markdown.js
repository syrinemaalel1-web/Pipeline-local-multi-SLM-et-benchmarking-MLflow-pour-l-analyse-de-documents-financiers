import { marked } from "marked";
import { colorFor } from "./colors.js";

// Les sorties de modèle (traduction, résumé, Q&R) sont enveloppées dans un bloc
// de code ```markdown``` côté backend (reporting/report.py::_fenced) — pour de
// bonnes raisons dans le fichier .md/PDF (fidélité mot pour mot du texte brut,
// insensible à tout caractère markdown accidentel dans la sortie du modèle).
// Mais ça empêche `marked` de rendre les tableaux qu'elles contiennent : sans
// second passage, un tableau de traduction s'affiche comme du texte brut dans
// un bloc de code au lieu d'un vrai tableau HTML. Le rendu PDF fait déjà ce
// second passage (reporting/pdf.py::_prepare_model_markdown) — on reproduit la
// même logique ici. Les blocs de langage différent (ex. ```json``` du fallback
// d'extraction non parseable) ne sont pas concernés : ils restent du code brut,
// à raison.
function renderNestedMarkdown(container) {
  container.querySelectorAll("pre > code.language-markdown").forEach((code) => {
    const wrapper = document.createElement("div");
    wrapper.className = "nested-markdown";
    wrapper.innerHTML = marked.parse(code.textContent);
    code.parentElement.replaceWith(wrapper);
  });
}

// Une sortie de modèle peut être en arabe (résumé/traduction sur un document
// source arabe) — sans direction explicite, le texte reste aligné à gauche par
// défaut du navigateur, contraire au sens de lecture RTL. On détecte la
// direction bloc par bloc (premier caractère "fort" rencontré, même principe
// que l'attribut HTML natif dir="auto") plutôt qu'au niveau de tout l'encart,
// pour ne pas casser un tableau qui mélange une colonne arabe et une colonne
// de nombres/latin. Plages Unicode en \u pour éviter tout risque de
// corruption de caractères littéraux dans le code source : arabe de base
// (؀-ۿ), supplément arabe (ݐ-ݿ), arabe étendu-A
// (ࢠ-ࣿ), formes de présentation arabes A/B (ﭐ-﷿,
// ﹰ-﻿) — les seules langues RTL réellement présentes dans ce projet.
const _RTL_RANGE = /[؀-ۿݐ-ݿࢠ-ࣿﭐ-﷿ﹰ-﻿]/;
const _LTR_RANGE = /[A-Za-zÀ-ɏ]/;

function firstStrongDirection(text) {
  for (const ch of text) {
    if (_RTL_RANGE.test(ch)) return "rtl";
    if (_LTR_RANGE.test(ch)) return "ltr";
  }
  return null;
}

function applyTextDirection(container) {
  const tags = ["p", "li", "blockquote", "h1", "h2", "h3", "h4", "h5", "h6", "td", "th"];
  const selector = tags.map((tag) => `.nested-markdown ${tag}`).join(", ");
  container.querySelectorAll(selector).forEach((el) => {
    const direct = [...el.childNodes]
      .filter((n) => n.nodeType === Node.TEXT_NODE || n.nodeName === "STRONG" || n.nodeName === "EM")
      .map((n) => n.textContent)
      .join(" ");
    const dir = firstStrongDirection(direct || el.textContent);
    if (dir) el.setAttribute("dir", dir);
  });
}

// Chaque identifiant en code inline (nom de modèle, id de document) reçoit une
// pastille de couleur cohérente — le même modèle garde la même couleur partout
// dans le rapport (voir colors.js).
function colorizeInlineCode(container) {
  container.querySelectorAll("code").forEach((el) => {
    if (el.parentElement.tagName === "PRE") return;
    el.classList.add("tag-badge", `tag-${colorFor(el.textContent)}`);
  });
}

// Les modèles n'émettent pas de vrais titres markdown (#/##) dans leurs sorties
// — les "titres" de section sont des paragraphes entièrement en gras (ex.
// "**Aperçu de l'entreprise**"). On les repère (un <strong> qui couvre tout le
// paragraphe, rien d'autre autour) pour les distinguer du gras normal au fil du
// texte (ex. "**5 200 000** TND"), qui doit lui rester en ligne, sans couleur.
function colorizeNestedTitles(container) {
  container.querySelectorAll(".nested-markdown p").forEach((p) => {
    const strong = p.querySelector(":scope > strong");
    if (strong && strong.textContent.trim() === p.textContent.trim() && strong.textContent.trim()) {
      p.classList.add("nested-title");
    }
  });
}

// `reporting/report.py` marque déjà textuellement la ligne/le paragraphe
// recommandé (RECOMMENDED_MARK = "**recommandé**") et les lignes exclues
// ("exclu de la recommandation") — on s'appuie sur ces marqueurs déjà présents
// plutôt que de redériver un classement côté frontend.
function highlightRecommended(container) {
  container.querySelectorAll("blockquote").forEach((bq) => {
    if (bq.textContent.includes("recommandé")) bq.classList.add("winner-callout");
  });
  container.querySelectorAll("table tbody tr").forEach((tr) => {
    if (tr.textContent.includes("recommandé")) tr.classList.add("row-recommended");
    else if (tr.textContent.includes("exclu de la recommandation")) tr.classList.add("row-excluded");
  });
}

export function renderReport(markdown) {
  const container = document.createElement("div");
  container.innerHTML = marked.parse(markdown);

  renderNestedMarkdown(container);
  applyTextDirection(container);
  highlightRecommended(container);
  colorizeNestedTitles(container);
  colorizeInlineCode(container);

  return container.innerHTML;
}
