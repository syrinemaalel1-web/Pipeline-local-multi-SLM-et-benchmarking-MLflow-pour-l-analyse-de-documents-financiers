import { useEffect, useRef, useState } from "react";
import { getJobLogs, getJobStatus } from "../api.js";

// Suivi d'un job par polling (pas de WebSocket, voir api/jobs.py pour le
// choix) : toutes les `intervalMs`, on demande les nouvelles lignes de log
// depuis la dernière reçue (`since`), et le statut du job. S'arrête tout seul
// une fois le job terminé ou en erreur.
export function useJob(jobId, intervalMs = 3000) {
  const [lines, setLines] = useState([]);
  const [status, setStatus] = useState(jobId ? "running" : null);
  const [progress, setProgress] = useState(null);
  const [error, setError] = useState(null);
  const sinceRef = useRef(0);

  useEffect(() => {
    setLines([]);
    setStatus(jobId ? "running" : null);
    setProgress(null);
    setError(null);
    sinceRef.current = 0;
    if (!jobId) return;

    let cancelled = false;
    let timer;

    async function poll() {
      try {
        const logsData = await getJobLogs(jobId, sinceRef.current);
        if (cancelled) return;
        if (logsData.lines.length) {
          setLines((prev) => [...prev, ...logsData.lines]);
          sinceRef.current = logsData.next_since;
        }

        const statusData = await getJobStatus(jobId);
        if (cancelled) return;
        setStatus(statusData.status);
        if (statusData.progress) setProgress(statusData.progress);

        if (statusData.status === "running") {
          timer = setTimeout(poll, intervalMs);
        }
      } catch (err) {
        if (!cancelled) {
          setError(String(err));
          timer = setTimeout(poll, intervalMs);
        }
      }
    }

    poll();
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [jobId, intervalMs]);

  return { lines, status, progress, error };
}
