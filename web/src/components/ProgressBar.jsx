// `progress` (si fourni, voir api/jobs.py::progress) vient du décompte réel de
// lignes déjà écrites par le script CLI — pas une estimation. Certaines phases
// (ingestion d'un seul document, tout début d'un job avant le premier marqueur)
// n'ont pas encore ce chiffre : on retombe alors sur une barre animée plutôt
// que d'inventer un pourcentage qu'on ne peut pas mesurer honnêtement.
export function ProgressBar({ status, progress }) {
  if (!status) return null;
  const cls =
    status === "done" ? "progress-done" : status === "error" ? "progress-error" : "";

  if (!progress || progress.total === 0) {
    return (
      <div className="progress-bar">
        <div className={`progress-bar-fill ${cls}`} />
      </div>
    );
  }

  const percent =
    status === "done"
      ? 100
      : Math.min(100, Math.round((progress.completed / progress.total) * 100));

  return (
    <div className="progress-numeric">
      <div className="progress-bar progress-bar-determinate">
        <div className={`progress-bar-fill ${cls}`} style={{ width: `${percent}%` }} />
      </div>
      <span className="progress-percent">{percent}%</span>
    </div>
  );
}
