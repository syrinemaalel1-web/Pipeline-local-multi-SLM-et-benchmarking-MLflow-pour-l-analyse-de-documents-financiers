import { useRef, useState } from "react";
import { startEvaluate, startGenerate, uploadDocument, waitForJob } from "../api.js";
import { EvalModeSelector } from "../components/EvalModeSelector.jsx";
import { Modal } from "../components/Modal.jsx";
import { PipelineStepper } from "../components/PipelineStepper.jsx";

const STEP_DEFS = [
  { id: "ingest", icon: "🧩", title: "Ingestion", desc: "Extraction du texte (Docling)" },
  { id: "generate", icon: "🤖", title: "Génération", desc: "4 tâches × modèles candidats (Ollama)" },
  { id: "evaluate", icon: "🔎", title: "Évaluation", desc: "Notation des sorties" },
];

export function AnalysisPage({ initialDocumentId, onDone }) {
  const [documentId, setDocumentId] = useState(initialDocumentId || null);
  const [filename, setFilename] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [running, setRunning] = useState(false);
  const [jobs, setJobs] = useState({ ingest: null, generate: null, evaluate: null });
  const [evalMode, setEvalMode] = useState("judge");
  const [showEvalModal, setShowEvalModal] = useState(false);
  const [error, setError] = useState(null);
  const [finished, setFinished] = useState(false);
  const fileInput = useRef(null);

  async function handleFile(file) {
    if (!file) return;
    setUploading(true);
    setError(null);
    setFinished(false);
    try {
      const result = await uploadDocument(file);
      setDocumentId(result.document_id);
      setFilename(result.filename);
      setJobs({ ingest: result.job_id, generate: null, evaluate: null });
    } catch (err) {
      setError(String(err));
    } finally {
      setUploading(false);
    }
  }

  // "Lancer l'analyse" ne fait qu'ingestion + génération. L'évaluation attend
  // le choix du moteur dans la fenêtre modale, une fois la génération terminée.
  async function runAnalysis() {
    setError(null);
    setFinished(false);
    setRunning(true);
    try {
      if (jobs.ingest) {
        const status = await waitForJob(jobs.ingest);
        if (status === "error") throw new Error("L'ingestion a échoué — voir le journal ci-contre.");
      }

      const gen = await startGenerate(documentId);
      setJobs((prev) => ({ ...prev, generate: gen.job_id }));
      const genStatus = await waitForJob(gen.job_id);
      if (genStatus === "error") throw new Error("La génération a échoué — voir le journal ci-contre.");

      setShowEvalModal(true);
    } catch (err) {
      setError(String(err));
    } finally {
      setRunning(false);
    }
  }

  async function confirmEvaluate() {
    setShowEvalModal(false);
    setError(null);
    setRunning(true);
    try {
      const ev = await startEvaluate(documentId, evalMode);
      setJobs((prev) => ({ ...prev, evaluate: ev.job_id }));
      const evStatus = await waitForJob(ev.job_id);
      if (evStatus === "error") throw new Error("L'évaluation a échoué — voir le journal ci-contre.");
      setFinished(true);
    } catch (err) {
      setError(String(err));
    } finally {
      setRunning(false);
    }
  }

  const steps = STEP_DEFS.map((def) => ({ ...def, jobId: jobs[def.id] }));

  return (
    <>
      <div className="page-header">
        <div>
          <h1>Nouvelle analyse</h1>
          <p className="subtitle">Déposez une proposition financière et lancez le pipeline complet</p>
        </div>
      </div>

      {error && <p className="error">{error}</p>}

      <div className="analysis-grid">
        {/* -------------------------------------------------- Left column */}
        <div className="card">
          <h2>📎 Déposer le document</h2>
          {!documentId ? (
            <div className="dropzone" onClick={() => fileInput.current?.click()}>
              <div className="dropzone-icon">📁</div>
              <div style={{ fontWeight: 700, color: "var(--primary)" }}>
                {uploading ? "Envoi en cours..." : "Déposer la proposition financière"}
              </div>
              <p className="hint" style={{ margin: "0.3rem 0 0.7rem" }}>Ou cliquez pour parcourir vos fichiers</p>
              <span className="format-pill">PDF</span>
              <span className="format-pill">DOCX</span>
              <input
                ref={fileInput}
                type="file"
                accept=".pdf,.docx"
                onChange={(e) => handleFile(e.target.files?.[0])}
              />
            </div>
          ) : (
            <p className="hint">
              📄 <strong>{filename || documentId}</strong> —{" "}
              <span className="mono">{documentId}</span>
            </p>
          )}

          <button
            className="btn-primary"
            style={{ width: "100%", padding: "0.8rem", fontSize: "0.98rem", marginTop: "1.4rem" }}
            disabled={!documentId || running || jobs.generate}
            onClick={runAnalysis}
          >
            {running && !showEvalModal ? "Analyse en cours..." : "▶ Lancer l'analyse"}
          </button>

          {jobs.generate && !finished && !showEvalModal && !running && (
            <button
              className="btn-secondary"
              style={{ width: "100%", marginTop: "0.7rem" }}
              onClick={() => setShowEvalModal(true)}
            >
              Choisir le moteur d'évaluation →
            </button>
          )}

          {finished && (
            <button
              className="btn-secondary"
              style={{ width: "100%", marginTop: "0.7rem" }}
              onClick={() => onDone(documentId)}
            >
              Voir les résultats →
            </button>
          )}
        </div>

        {/* ------------------------------------------------- Right column */}
        <div className="card">
          <h2>⚙️ Statut du pipeline</h2>
          {!documentId ? (
            <div className="pipeline-idle">
              <div className="pipeline-idle-icon">⏳</div>
              <p>Déposez un document et cliquez sur « Lancer l'analyse » pour commencer.</p>
            </div>
          ) : (
            <PipelineStepper steps={steps} />
          )}
        </div>
      </div>

      <Modal open={showEvalModal} onClose={() => setShowEvalModal(false)}>
        <div className="modal-header">
          <h2>Choisir le moteur d'évaluation</h2>
          <p className="subtitle">
            Document : <strong>{filename || documentId}</strong>
          </p>
        </div>
        <div className="modal-body">
          <EvalModeSelector value={evalMode} onChange={setEvalMode} />
        </div>
        <div className="modal-footer">
          <button className="btn-secondary" onClick={() => setShowEvalModal(false)}>
            Annuler
          </button>
          <button className="btn-primary" onClick={confirmEvaluate}>
            ▶ Lancer l'évaluation
          </button>
        </div>
      </Modal>
    </>
  );
}
