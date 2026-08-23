import { useState } from "react";
import { Sidebar } from "./components/Sidebar.jsx";
import { HomePage } from "./pages/HomePage.jsx";
import { AnalysisPage } from "./pages/AnalysisPage.jsx";
import { ResultsPage } from "./pages/ResultsPage.jsx";
import { ComparisonPage } from "./pages/ComparisonPage.jsx";

export function App() {
  const [page, setPage] = useState("home");
  const [pendingDocumentId, setPendingDocumentId] = useState(null);

  function goToAnalysis(documentId) {
    setPendingDocumentId(documentId || null);
    setPage("analysis");
  }

  function goToResults() {
    setPage("results");
  }

  return (
    <div className="layout">
      <Sidebar page={page} onNavigate={setPage} />
      <main className="main">
        {page === "home" && <HomePage onNavigate={setPage} />}
        {page === "analysis" && (
          <AnalysisPage
            key={pendingDocumentId || "new"}
            initialDocumentId={pendingDocumentId}
            onDone={goToResults}
          />
        )}
        {page === "results" && <ResultsPage onReanalyze={goToAnalysis} />}
        {page === "comparison" && <ComparisonPage />}
      </main>
    </div>
  );
}
