import { useEffect, useState } from "react";
import { getDocuments, getReport } from "../api.js";
import { EvalModeSelector } from "../components/EvalModeSelector.jsx";
import { BoolBadge } from "../components/StatusBadge.jsx";
import { renderReport } from "../markdown.js";

export function ResultsPage({ onReanalyze }) {
  const [documents, setDocuments] = useState([]);
  const [loading, setLoading] = useState(true);
  const [selected, setSelected] = useState(null);
  const [evalMode, setEvalMode] = useState("judge");
  const [markdown, setMarkdown] = useState(null);
  const [reportLoading, setReportLoading] = useState(false);
  const [error, setError] = useState(null);

  function reload() {
    setLoading(true);
    getDocuments()
      .then(setDocuments)
      .catch((err) => setError(String(err)))
      .finally(() => setLoading(false));
  }

  useEffect(reload, []);

  function openReport(doc) {
    setSelected(doc.document_id);
    setMarkdown(null);
    const mode = doc.report_judge ? "judge" : "metrics";
    setEvalMode(mode);
    loadReport(doc.document_id, mode);
  }

  function loadReport(documentId, mode) {
    setReportLoading(true);
    setError(null);
    getReport(documentId, mode)
      .then((data) => setMarkdown(data.markdown))
      .catch((err) => setError(String(err)))
      .finally(() => setReportLoading(false));
  }

  return (
    <>
      <div className="page-header">
        <div>
          <h1>Résultats</h1>
          <p className="subtitle">Historique des documents traités et leurs rapports</p>
        </div>
        <div className="page-header-actions">
          <button className="btn-secondary" onClick={reload}>Actualiser</button>
        </div>
      </div>

      {error && <p className="error">{error}</p>}

      <div className="card">
        <table className="data-table">
          <thead>
            <tr>
              <th>Document</th>
              <th>Fichier</th>
              <th>Ingéré</th>
              <th>Généré</th>
              <th>Évalué</th>
              <th>Rapport</th>
              <th></th>
            </tr>
          </thead>
          <tbody>
            {documents.map((doc) => (
              <tr
                key={doc.document_id}
                className={`row-clickable ${selected === doc.document_id ? "row-selected" : ""}`}
                onClick={() => (doc.report_judge || doc.report_metrics) && openReport(doc)}
              >
                <td className="mono">{doc.document_id}</td>
                <td>{doc.source_file}</td>
                <td><BoolBadge ok={doc.ingested} labelYes="oui" labelNo="non" /></td>
                <td><BoolBadge ok={doc.generated} labelYes="oui" labelNo="non" /></td>
                <td><BoolBadge ok={doc.evaluated_judge || doc.evaluated_metrics} labelYes="oui" labelNo="non" /></td>
                <td><BoolBadge ok={doc.report_judge || doc.report_metrics} labelYes="voir" labelNo="—" /></td>
                <td>
                  <button
                    className="btn-secondary"
                    onClick={(e) => {
                      e.stopPropagation();
                      onReanalyze(doc.document_id);
                    }}
                  >
                    Relancer
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
        {!loading && documents.length === 0 && <p className="hint">Aucun document pour l'instant.</p>}
      </div>

      {selected && (
        <div className="card">
          <h2>Rapport — <span className="mono">{selected}</span></h2>
          <EvalModeSelector
            value={evalMode}
            onChange={(mode) => {
              setEvalMode(mode);
              loadReport(selected, mode);
            }}
          />
          {reportLoading && <p>Chargement...</p>}
          {markdown && (
            <div
              className="report-content"
              dangerouslySetInnerHTML={{ __html: renderReport(markdown) }}
            />
          )}
        </div>
      )}
    </>
  );
}
