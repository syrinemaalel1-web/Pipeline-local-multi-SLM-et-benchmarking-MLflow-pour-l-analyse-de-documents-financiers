import { useEffect, useState } from "react";
import { getDashboard } from "../api.js";
import { EvalModeSelector } from "../components/EvalModeSelector.jsx";
import { colorFor, scoreColor } from "../colors.js";

const JUDGE_COLUMNS = [
  ["model", "Modèle"],
  ["judge_score", "Note juge"],
  ["code_mean", "Scorers code"],
  ["composite", "Composite"],
  ["latency_mean_s", "Latence (s)"],
  ["ineligible", "Exclu ?"],
];

const MEDALS = ["🥇", "🥈", "🥉"];

// Échelle de la note pour la teinte de couleur — pas la même borne pour un
// score composite [0,1], une note juge [1,5], ou une métrique de lecture sans
// borne fixe (ARI/Flesch) : celles-ci ne sont pas colorées, faute d'échelle
// de référence fiable (voir CLAUDE.md — pas de direction "meilleur" universelle).
const SCALES = {
  judge_score: [1, 5],
  composite: [0, 1],
  code_mean: [0, 1],
  faithfulness: [1, 5],
  answer_relevance: [1, 5],
};

function fmt(value) {
  if (value === null || value === undefined) return "—";
  if (typeof value === "number") return value.toFixed(2);
  if (typeof value === "boolean") return value ? "oui" : "non";
  return String(value);
}

function metricsColumns(rows) {
  const keys = new Set();
  for (const row of rows) {
    for (const key of Object.keys(row)) {
      if (key !== "model") keys.add(key);
    }
  }
  return [["model", "Modèle"], ...[...keys].map((k) => [k, k])];
}

function chartKey(evalMode, rows) {
  if (evalMode === "judge") return "composite";
  if (rows.some((r) => "faithfulness" in r)) return "faithfulness";
  const numeric = Object.keys(rows[0] || {}).find(
    (k) => k !== "model" && typeof rows[0][k] === "number"
  );
  return numeric || null;
}

function Cell({ colKey, value }) {
  if (colKey === "model") {
    return (
      <td>
        <span className={`tag-badge tag-${colorFor(value)}`}>{value}</span>
      </td>
    );
  }
  const scale = SCALES[colKey];
  if (scale && typeof value === "number") {
    const color = scoreColor(value, scale[0], scale[1]);
    return (
      <td>
        <span className={`tag-badge tag-${color}`}>{fmt(value)}</span>
      </td>
    );
  }
  return <td>{fmt(value)}</td>;
}

function BarChart({ rows, valueKey }) {
  if (!valueKey) return null;
  const values = rows.map((r) => r[valueKey]).filter((v) => typeof v === "number");
  const max = Math.max(...values, 0.0001);
  return (
    <div className="bar-chart">
      <div className="bar-chart-label">{valueKey}</div>
      {rows.map((row) => {
        const value = row[valueKey];
        const width = typeof value === "number" ? (value / max) * 100 : 0;
        return (
          <div className="bar-row" key={row.model}>
            <span className={`tag-badge tag-${colorFor(row.model)} bar-model`}>{row.model}</span>
            <div className="bar-track">
              <div className={`bar-fill bar-fill-${colorFor(row.model)}`} style={{ width: `${width}%` }} />
            </div>
            <span className="bar-value">{fmt(value)}</span>
          </div>
        );
      })}
    </div>
  );
}

function TaskCard({ task, rows, evalMode }) {
  if (!rows || rows.length === 0) {
    return (
      <div className="card dashboard-task">
        <h2>{task}</h2>
        <p className="hint">Aucune donnée pour cette tâche.</p>
      </div>
    );
  }
  const columns = evalMode === "judge" ? JUDGE_COLUMNS : metricsColumns(rows);
  const valueKey = chartKey(evalMode, rows);

  return (
    <div className="card dashboard-task">
      <h2>{task}</h2>
      <table className="data-table">
        <thead>
          <tr>
            <th></th>
            {columns.map(([key, label]) => (
              <th key={key}>{label}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, i) => (
            <tr
              key={row.model}
              className={
                evalMode === "judge" && i === 0 && !row.ineligible ? "row-recommended" : ""
              }
            >
              <td className="rank-cell">
                {evalMode === "judge" && !row.ineligible && MEDALS[i] ? MEDALS[i] : ""}
              </td>
              {columns.map(([key]) => (
                <Cell key={key} colKey={key} value={row[key]} />
              ))}
            </tr>
          ))}
        </tbody>
      </table>
      <BarChart rows={rows} valueKey={valueKey} />
    </div>
  );
}

export function ComparisonPage() {
  const [evalMode, setEvalMode] = useState("judge");
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    setLoading(true);
    setError(null);
    getDashboard(evalMode)
      .then(setData)
      .catch((err) => setError(String(err)))
      .finally(() => setLoading(false));
  }, [evalMode]);

  return (
    <>
      <div className="page-header">
        <div>
          <h1>Comparaison des modèles</h1>
          <p className="subtitle">Vue agrégée, tous documents confondus, par tâche</p>
        </div>
      </div>

      <EvalModeSelector value={evalMode} onChange={setEvalMode} />

      {loading && <p>Chargement...</p>}
      {error && <p className="error">{error}</p>}

      {data &&
        Object.entries(data).map(([task, rows]) => (
          <TaskCard key={task} task={task} rows={rows} evalMode={evalMode} />
        ))}
    </>
  );
}
