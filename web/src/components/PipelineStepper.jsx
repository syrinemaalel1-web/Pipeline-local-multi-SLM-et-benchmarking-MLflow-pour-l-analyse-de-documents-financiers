import { useState } from "react";
import { useJob } from "../hooks/useJob.js";
import { ProgressBar } from "./ProgressBar.jsx";
import { StatusBadge } from "./StatusBadge.jsx";

function StepRow({ step, isLast }) {
  const { lines, status, progress } = useJob(step.jobId);
  const [expanded, setExpanded] = useState(false);
  const effectiveStatus = step.jobId ? status : "pending";
  const lastLine = lines[lines.length - 1];

  return (
    <div className="stepper-item">
      {!isLast && <div className="stepper-line" />}
      <div className={`stepper-icon step-${effectiveStatus}`}>
        {effectiveStatus === "done" ? "✓" : effectiveStatus === "error" ? "✕" : step.icon}
      </div>
      <div className="stepper-body">
        <div className="stepper-title">
          {step.title}
          <StatusBadge status={step.jobId ? status : null} />
        </div>
        <div className="stepper-desc">{step.desc}</div>
        {effectiveStatus === "running" && <ProgressBar status={status} progress={progress} />}
        {lastLine && (
          <>
            <div className="stepper-log-line" title={lastLine}>{lastLine}</div>
            <button className="stepper-toggle" onClick={() => setExpanded((v) => !v)}>
              {expanded ? "Masquer le journal" : "Voir le journal complet"}
            </button>
            {expanded && <pre className="log-box" style={{ marginTop: "0.4rem" }}>{lines.join("\n")}</pre>}
          </>
        )}
      </div>
    </div>
  );
}

export function PipelineStepper({ steps }) {
  return (
    <div className="stepper">
      {steps.map((step, i) => (
        <StepRow key={step.id} step={step} isLast={i === steps.length - 1} />
      ))}
    </div>
  );
}
