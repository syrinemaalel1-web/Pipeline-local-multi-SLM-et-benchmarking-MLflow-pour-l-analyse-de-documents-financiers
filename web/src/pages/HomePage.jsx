import { useEffect, useState } from "react";
import { getDocuments } from "../api.js";
import { BoolBadge } from "../components/StatusBadge.jsx";

const PIPELINE_NODES = [
  { icon: "📤", title: "Upload", tag: "PDF / DOCX", color: "primary" },
  { icon: "🧩", title: "Ingestion", tag: "Docling", color: "teal" },
  { icon: "🤖", title: "Génération", tag: "Ollama (5 SLM)", color: "amber" },
  { icon: "🔎", title: "Évaluation", tag: "MLflow judge", color: "purple" },
  { icon: "📊", title: "Rapport", tag: "Markdown + PDF", color: "rose" },
  { icon: "🐳", title: "Déploiement", tag: "Docker", color: "teal" },
];

const AGENTS = [
  { icon: "🔍", name: "Extraction", desc: "Structure les données clés en JSON — montant, taux, durée, clauses" },
  { icon: "📝", name: "Résumé", desc: "Synthèse exécutive du dossier pour un comité de crédit" },
  { icon: "🌍", name: "Traduction", desc: "FR ↔ EN ↔ AR, direction imposée par la langue source" },
  { icon: "💬", name: "Q&R", desc: "Réponse ancrée au document, abstention si l'information est absente" },
];

function ActiveAgents() {
  return (
    <div className="card">
      <h2>🤖 Agents actifs</h2>
      <p className="subtitle">Les 4 agents du pipeline, chacun testé sur 5 modèles candidats</p>
      <div className="agent-grid">
        {AGENTS.map((agent) => (
          <div className="agent-card" key={agent.name}>
            <span className="agent-icon">{agent.icon}</span>
            <div>
              <div className="agent-name">{agent.name}</div>
              <div className="agent-desc">{agent.desc}</div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function StatCard({ icon, color, label, value, foot }) {
  return (
    <div className="stat-card">
      <div className="stat-card-top">
        <span className="stat-card-label">{label}</span>
        <span className="stat-icon" style={{ background: `var(--${color}-soft)`, color: `var(--${color})` }}>
          {icon}
        </span>
      </div>
      <div className="stat-value">{value}</div>
      <div className="stat-foot">{foot}</div>
    </div>
  );
}

function RecentDocuments({ documents, loading, onNavigate }) {
  const recent = documents.slice(-6).reverse();
  return (
    <div className="card">
      <div className="section-header">
        <div>
          <h2>Documents récents</h2>
          <p className="subtitle">Derniers documents déposés et leur avancement</p>
        </div>
        <button className="btn-secondary" onClick={() => onNavigate("results")}>
          Voir tout →
        </button>
      </div>
      {loading && <p>Chargement...</p>}
      {!loading && recent.length === 0 && (
        <p className="hint">
          Aucun document pour l'instant — déposez-en un depuis « + Nouvelle analyse ».
        </p>
      )}
      {!loading && recent.length > 0 && (
        <div className="recent-table-wrap">
          <table className="data-table">
            <thead>
              <tr>
                <th>Document</th>
                <th>Fichier</th>
                <th>Ingéré</th>
                <th>Généré</th>
                <th>Évalué</th>
                <th>Rapport</th>
              </tr>
            </thead>
            <tbody>
              {recent.map((doc) => (
                <tr key={doc.document_id}>
                  <td className="mono">{doc.document_id}</td>
                  <td>{doc.source_file}</td>
                  <td><BoolBadge ok={doc.ingested} labelYes="oui" labelNo="non" /></td>
                  <td><BoolBadge ok={doc.generated} labelYes="oui" labelNo="non" /></td>
                  <td><BoolBadge ok={doc.evaluated_judge || doc.evaluated_metrics} labelYes="oui" labelNo="non" /></td>
                  <td><BoolBadge ok={doc.report_judge || doc.report_metrics} labelYes="voir" labelNo="—" /></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

export function HomePage({ onNavigate }) {
  const [documents, setDocuments] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    getDocuments()
      .then(setDocuments)
      .finally(() => setLoading(false));
  }, []);

  const ingested = documents.filter((d) => d.ingested).length;
  const generated = documents.filter((d) => d.generated).length;
  const evaluated = documents.filter((d) => d.evaluated_judge || d.evaluated_metrics).length;
  const reports = documents.filter((d) => d.report_judge || d.report_metrics).length;

  return (
    <>
      <div className="page-header">
        <div>
          <h1>Dashboard</h1>
          <p className="subtitle">Vue d'ensemble du pipeline d'analyse de propositions financières</p>
        </div>
        <div className="page-header-actions">
          <button className="btn-primary" onClick={() => onNavigate("analysis")}>
            + Nouvelle analyse
          </button>
        </div>
      </div>

      <div className="stat-grid">
        <StatCard icon="🧩" color="teal" label="Documents ingérés" value={loading ? "…" : ingested} foot="Texte extrait (Docling)" />
        <StatCard icon="🤖" color="amber" label="Documents générés" value={loading ? "…" : generated} foot="4 tâches × 5 modèles" />
        <StatCard icon="🔎" color="purple" label="Documents évalués" value={loading ? "…" : evaluated} foot="Judge et/ou metrics" />
        <StatCard icon="📊" color="rose" label="Rapports disponibles" value={loading ? "…" : reports} foot="Markdown + PDF" />
      </div>

      <div className="card">
        <h2>Pipeline de traitement</h2>
        <p className="subtitle">Du dépôt d'un document à son rapport comparatif</p>
        <div className="pipeline-flow">
          {PIPELINE_NODES.map((node, i) => (
            <div key={node.title} style={{ display: "contents" }}>
              <div className="pipeline-node">
                <div
                  className="pipeline-node-icon"
                  style={{
                    background: `var(--${node.color}-soft)`,
                    borderColor: `var(--${node.color})`,
                  }}
                >
                  {node.icon}
                </div>
                <div className="pipeline-node-title">{node.title}</div>
                <div className="pipeline-node-tag">{node.tag}</div>
              </div>
              {i < PIPELINE_NODES.length - 1 && <div className="pipeline-arrow">→</div>}
            </div>
          ))}
        </div>
      </div>

      <ActiveAgents />

      <RecentDocuments documents={documents} loading={loading} onNavigate={onNavigate} />
    </>
  );
}
