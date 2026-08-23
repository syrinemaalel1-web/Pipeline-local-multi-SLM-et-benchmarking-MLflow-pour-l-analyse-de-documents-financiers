import { useEffect, useRef } from "react";
import { useJob } from "../hooks/useJob.js";
import { StatusBadge } from "./StatusBadge.jsx";

// Vue de log en direct pour un job — affiche les lignes au fur et à mesure
// qu'elles arrivent (polling), défile automatiquement vers le bas.
export function LogView({ jobId }) {
  const { lines, status, error } = useJob(jobId);
  const boxRef = useRef(null);

  useEffect(() => {
    if (boxRef.current) {
      boxRef.current.scrollTop = boxRef.current.scrollHeight;
    }
  }, [lines]);

  if (!jobId) return null;

  return (
    <div className="log-view">
      <div className="log-view-header">
        <StatusBadge status={status} />
        {error && <span className="log-error">Erreur de connexion : {error}</span>}
      </div>
      <pre className="log-box" ref={boxRef}>
        {lines.length ? lines.join("\n") : "En attente des premières lignes..."}
      </pre>
    </div>
  );
}
