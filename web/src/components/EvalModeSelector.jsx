const MODES = [
  {
    id: "judge",
    icon: "🤖",
    title: "Judge",
    desc: "Pipeline Judge, orchestré via mlflow.genai.evaluate() : chaque juge est un Scorer MLflow construit par mlflow.genai.judges.make_judge.",
    metrics: [
      "extraction_fidelity, summary_faithfulness, translation_fidelity, qa_groundedness — note 1 à 5, le juge construit sa propre référence avant de comparer",
      "Scorers code — JSON/schéma valides, champs requis, langue, longueur, chiffres préservés, non-troncature, abstention correcte",
      "Composite : 60 % juge + 40 % scorers",
    ],
  },
  {
    id: "metrics",
    icon: "📐",
    title: "Metrics",
    desc: "Deuxième méthode d'évaluation, basée sur mlflow.metrics.genai et textstat, exécutée via mlflow.evaluate().",
    metrics: [
      "Faithfulness (traduction, résumé, Q&R) — même juge que le pipeline Judge",
      "Answer relevance (Q&R)",
      "ARI — niveau de lisibilité (résumé)",
      "Flesch-Kincaid — niveau de lisibilité (résumé)",
    ],
  },
];

export function EvalModeSelector({ value, onChange }) {
  return (
    <div className="mode-selector">
      {MODES.map((mode) => (
        <button
          key={mode.id}
          type="button"
          className={`mode-card ${value === mode.id ? "mode-card-active" : ""}`}
          onClick={() => onChange(mode.id)}
        >
          <span className="mode-card-icon">{mode.icon}</span>
          <span className="mode-card-body">
            <span className="mode-card-title">{mode.title}</span>
            <span className="mode-card-desc">{mode.desc}</span>
            <ul className="mode-card-metrics">
              {mode.metrics.map((m) => (
                <li key={m}>{m}</li>
              ))}
            </ul>
          </span>
          <span className="mode-card-check">✓</span>
        </button>
      ))}
    </div>
  );
}
