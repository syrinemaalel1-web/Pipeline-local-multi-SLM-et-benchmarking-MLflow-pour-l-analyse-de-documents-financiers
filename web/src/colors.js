// Palette pastel partagée (mêmes teintes que les cartes de stats du Dashboard)
// — un modèle donné garde toujours la même couleur, partout dans l'app.
const PALETTE = ["primary", "teal", "amber", "purple", "rose"];

function hash(text) {
  let h = 0;
  for (let i = 0; i < text.length; i++) h = (h * 31 + text.charCodeAt(i)) | 0;
  return Math.abs(h);
}

export function colorFor(name) {
  return PALETTE[hash(name) % PALETTE.length];
}

// Échelle rouge -> ambre -> vert pour une note, sur [min, max] (plus haut = mieux).
export function scoreColor(value, min, max) {
  if (typeof value !== "number") return "ink-soft";
  const ratio = (value - min) / (max - min || 1);
  if (ratio >= 0.75) return "teal";
  if (ratio >= 0.45) return "amber";
  return "rose";
}
