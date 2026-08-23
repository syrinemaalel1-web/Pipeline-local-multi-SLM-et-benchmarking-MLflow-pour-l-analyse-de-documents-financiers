const LABELS = {
  running: "en cours",
  done: "terminé",
  error: "erreur",
};

export function StatusBadge({ status }) {
  if (!status) return null;
  return <span className={`badge badge-${status}`}>{LABELS[status] || status}</span>;
}

export function BoolBadge({ ok, labelYes, labelNo }) {
  return (
    <span className={`badge ${ok ? "badge-done" : "badge-pending"}`}>
      {ok ? labelYes : labelNo}
    </span>
  );
}
