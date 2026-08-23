// Client HTTP vers le backend FastAPI (api/main.py). Aucune logique métier ici :
// juste des appels REST, tels que le backend les expose.

export const API_BASE = "http://localhost:8000";

// "TypeError: Failed to fetch" est l'erreur générique du navigateur quand la
// requête n'a même pas pu partir (serveur éteint, mauvais port, réseau) — sans
// ce filtre, l'utilisateur voit ce message technique sans savoir quoi en faire.
async function apiFetch(path, options) {
  let res;
  try {
    res = await fetch(`${API_BASE}${path}`, options);
  } catch {
    throw new Error(
      `Impossible de joindre le serveur API sur ${API_BASE}. Vérifiez qu'il est ` +
        `bien lancé : uvicorn api.main:app --reload --port 8000`
    );
  }
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`${res.status} ${res.statusText} : ${body}`);
  }
  return res.json();
}

export function getDocuments() {
  return apiFetch("/documents");
}

export function uploadDocument(file) {
  const body = new FormData();
  body.append("file", file);
  return apiFetch("/documents/upload", { method: "POST", body });
}

export function startIngest() {
  return apiFetch("/ingest", { method: "POST" });
}

export function startGenerate(documentId) {
  return apiFetch("/generate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ document_id: documentId }),
  });
}

export function startEvaluate(documentId, evalMode) {
  return apiFetch("/evaluate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ document_id: documentId, eval_mode: evalMode }),
  });
}

export function getJobStatus(jobId) {
  return apiFetch(`/jobs/${jobId}`);
}

// Attend qu'un job se termine (poll), pour enchaîner les étapes côté frontend
// (le backend reste sans état entre les requêtes, voir api/jobs.py).
export async function waitForJob(jobId, intervalMs = 3000) {
  for (;;) {
    const { status } = await getJobStatus(jobId);
    if (status !== "running") return status;
    await new Promise((resolve) => setTimeout(resolve, intervalMs));
  }
}

export function getJobLogs(jobId, since) {
  return apiFetch(`/jobs/${jobId}/logs?since=${since}`);
}

export function getReport(documentId, evalMode) {
  return apiFetch(`/reports/${documentId}?eval_mode=${evalMode}`);
}

export function getDashboard(evalMode) {
  return apiFetch(`/dashboard?eval_mode=${evalMode}`);
}
