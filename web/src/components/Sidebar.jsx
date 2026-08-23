const NAV_GROUPS = [
  {
    label: "Principal",
    items: [
      { id: "home", icon: "🏠", label: "Dashboard" },
      { id: "analysis", icon: "📤", label: "Nouvelle analyse" },
      { id: "results", icon: "📁", label: "Résultats" },
    ],
  },
  {
    label: "Benchmark",
    items: [{ id: "comparison", icon: "📊", label: "Comparaison modèles" }],
  },
];

export function Sidebar({ page, onNavigate }) {
  return (
    <aside className="sidebar">
      <div className="brand">
        <div className="brand-icon">🗂️</div>
        <div>
          <div className="brand-name">Finance SLM</div>
          <div className="brand-sub">Pipeline local multi-agents</div>
        </div>
      </div>

      {NAV_GROUPS.map((group) => (
        <div className="nav-group" key={group.label}>
          <div className="nav-group-label">{group.label}</div>
          {group.items.map((item) => (
            <button
              key={item.id}
              className={page === item.id ? "nav-item active" : "nav-item"}
              onClick={() => onNavigate(item.id)}
            >
              <span className="nav-icon">{item.icon}</span>
              {item.label}
            </button>
          ))}
        </div>
      ))}
    </aside>
  );
}
