"""Tests de non-régression pour la persistance et la reprise de `orchestration.run_eval`.

Deux sujets couverts, tous deux vérifiés sans appeler Ollama ni mlflow.genai.evaluate pour de
vrai (mocké) :

1. **Fusion** — `_persist` écrivait auparavant `results/evaluation/<tâche>/<modèle>.json` en
   écrasant tout le fichier à chaque appel : un flux `run_eval --doc X` suivi de
   `run_eval --doc Y` effaçait les lignes de X au lieu de les accumuler avec celles de Y (voir
   CLAUDE.md §5). Prouvé par `test_second_doc_does_not_erase_first` et consorts.
2. **Reprise** — `evaluate_task` rejouait tous les modèles demandés à chaque invocation, y
   compris ceux déjà persistés : après une interruption au milieu d'une tâche (N modèles sur M
   déjà évalués), relancer la même commande rejouait le juge sur les N déjà faits avant
   d'atteindre le reste. Prouvé par `test_skip_already_evaluated_model_without_force` et
   consorts (CLAUDE.md, point sur le checkpointing).

    python -m pytest tests/test_run_eval_persist.py -q
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import mlflow
import pandas as pd
import pytest

from config import Task
from orchestration import run_eval


class _FakeTable:
    """Imite l'objet table renvoyé par `result.tables` de mlflow.genai.evaluate()."""

    def __init__(self, records: list[dict[str, Any]]):
        self._records = records

    def to_dict(self, orient: str = "records") -> list[dict[str, Any]]:
        return list(self._records)


def _fake_result(run_id: str, records: list[dict[str, Any]]) -> SimpleNamespace:
    return SimpleNamespace(
        run_id=run_id, tables={"eval_results": _FakeTable(records)}, metrics={}
    )


def _row(document_id: str, score: float) -> dict[str, Any]:
    """Ligne déjà tagguée, telle que `_tagged_rows` la produirait."""
    return {
        "document_id": document_id,
        "question": None,
        "summary_faithfulness/value": score,
    }


@pytest.fixture(autouse=True)
def _isolated_evaluation_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(run_eval, "EVALUATION_DIR", tmp_path)
    return tmp_path


def _persisted_path(tmp_path) -> Any:
    return tmp_path / "summary" / "qwen2.5_7b.json"


def test_second_doc_does_not_erase_first(tmp_path):
    run_eval._persist(Task.SUMMARY, "qwen2.5:7b", "run-a", [_row("doc_a", 5)])
    run_eval._persist(Task.SUMMARY, "qwen2.5:7b", "run-b", [_row("doc_b", 3)])

    payload = json.loads(_persisted_path(tmp_path).read_text("utf-8"))
    doc_ids = {row["document_id"] for row in payload["rows"]}

    assert doc_ids == {"doc_a", "doc_b"}
    assert len(payload["rows"]) == 2


def test_re_evaluating_a_doc_replaces_only_its_own_rows(tmp_path):
    run_eval._persist(Task.SUMMARY, "qwen2.5:7b", "run-a", [_row("doc_a", 5)])
    run_eval._persist(Task.SUMMARY, "qwen2.5:7b", "run-b", [_row("doc_b", 3)])
    run_eval._persist(Task.SUMMARY, "qwen2.5:7b", "run-a2", [_row("doc_a", 4)])

    payload = json.loads(_persisted_path(tmp_path).read_text("utf-8"))
    rows_by_doc = {row["document_id"]: row for row in payload["rows"]}

    assert set(rows_by_doc) == {"doc_a", "doc_b"}
    assert rows_by_doc["doc_a"]["summary_faithfulness/value"] == 4
    assert rows_by_doc["doc_b"]["summary_faithfulness/value"] == 3


def test_aggregate_metrics_reflect_merged_rows_not_last_call_only(tmp_path):
    run_eval._persist(Task.SUMMARY, "qwen2.5:7b", "run-a", [_row("doc_a", 5)])
    # Deuxième document, note différente : la moyenne agrégée doit désormais porter sur
    # les deux lignes (5 et 3 -> 4.0), pas seulement sur celle de ce second appel (3.0),
    # sans quoi le tableau du rapport final (document_id=None) lirait un chiffre qui ne
    # correspond à aucun sous-ensemble réel.
    run_eval._persist(Task.SUMMARY, "qwen2.5:7b", "run-b", [_row("doc_b", 3)])

    payload = json.loads(_persisted_path(tmp_path).read_text("utf-8"))

    assert payload["metrics"]["summary_faithfulness/mean"] == pytest.approx(4.0)


# --------------------------------------------------------------------------- #
# Reprise : un modèle déjà persisté n'est pas rejoué sans --force
# --------------------------------------------------------------------------- #


def _seed_persisted(tmp_path, task: Task, model: str, document_id: str) -> None:
    path = tmp_path / task.value / f"{run_eval.store.slug(model)}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "task": task.value,
                "model": model,
                "rows": [{"document_id": document_id, "summary_faithfulness/value": 5}],
                "metrics": {"summary_faithfulness/mean": 5.0},
            }
        ),
        "utf-8",
    )


@pytest.fixture
def _no_real_mlflow_tracking(monkeypatch):
    """`evaluate_task` ouvre un run et pose des tags : neutralisé, hors sujet ici."""
    monkeypatch.setattr(mlflow, "start_run", lambda **_: _NullContext())
    monkeypatch.setattr(mlflow, "set_tags", lambda *_a, **_k: None)


class _NullContext:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _fake_generation_record(document_id: str) -> dict[str, Any]:
    return {
        "task": "summary",
        "model": "qwen2.5:7b",
        "document_id": document_id,
        "item_id": None,
        "output": f"résumé de {document_id}",
        "error": None,
        "question": None,
    }


def _fake_document(document_id: str) -> dict[str, Any]:
    return {"document_id": document_id, "lang": "fr", "text": f"texte source {document_id}"}


def test_skip_already_evaluated_model_without_force(tmp_path, monkeypatch, _no_real_mlflow_tracking):
    monkeypatch.setattr(run_eval, "EVALUATION_DIR", tmp_path)
    _seed_persisted(tmp_path, Task.SUMMARY, "qwen2.5:7b", "doc_a")

    def _must_not_be_called(*_a, **_k):
        raise AssertionError("mlflow.genai.evaluate appelé alors que le modèle était déjà évalué")

    monkeypatch.setattr(mlflow.genai, "evaluate", _must_not_be_called)

    # document_ids couvre exactement ce qui est déjà persisté : aucune évaluation attendue,
    # donc build_dataset (qui lirait le vrai corpus) ne doit même pas être atteint.
    run_eval.evaluate_task(
        Task.SUMMARY, models=["qwen2.5:7b"], document_ids={"doc_a"}
    )


def test_force_reevaluates_even_if_already_persisted(tmp_path, monkeypatch, _no_real_mlflow_tracking):
    monkeypatch.setattr(run_eval, "EVALUATION_DIR", tmp_path)
    _seed_persisted(tmp_path, Task.SUMMARY, "qwen2.5:7b", "doc_a")
    monkeypatch.setattr(
        run_eval.store, "load_all", lambda task=None: iter([_fake_generation_record("doc_a")])
    )
    monkeypatch.setattr(
        run_eval.corpus, "load_documents", lambda: [_fake_document("doc_a")]
    )

    calls = []

    def _fake_evaluate(data, scorers):
        calls.append(len(data))
        return _fake_result("run-forced", [{"summary_faithfulness/value": 4}])

    monkeypatch.setattr(mlflow.genai, "evaluate", _fake_evaluate)

    run_eval.evaluate_task(
        Task.SUMMARY, models=["qwen2.5:7b"], document_ids={"doc_a"}, force=True
    )

    assert calls == [1]


def test_missing_document_is_not_skipped(tmp_path, monkeypatch, _no_real_mlflow_tracking):
    monkeypatch.setattr(run_eval, "EVALUATION_DIR", tmp_path)
    _seed_persisted(tmp_path, Task.SUMMARY, "qwen2.5:7b", "doc_a")
    # doc_b n'a jamais été évalué : le sous-ensemble demandé n'est pas couvert, il ne
    # doit donc pas être sauté même sans --force.
    monkeypatch.setattr(
        run_eval.store,
        "load_all",
        lambda task=None: iter(
            [_fake_generation_record("doc_a"), _fake_generation_record("doc_b")]
        ),
    )
    monkeypatch.setattr(
        run_eval.corpus,
        "load_documents",
        lambda: [_fake_document("doc_a"), _fake_document("doc_b")],
    )

    calls = []

    def _fake_evaluate(data, scorers):
        calls.append(sorted(data["inputs"].apply(lambda i: i["document_id"])))
        return _fake_result(
            "run-mixed",
            [{"summary_faithfulness/value": 5}, {"summary_faithfulness/value": 3}],
        )

    monkeypatch.setattr(mlflow.genai, "evaluate", _fake_evaluate)

    run_eval.evaluate_task(
        Task.SUMMARY, models=["qwen2.5:7b"], document_ids={"doc_a", "doc_b"}
    )

    assert calls == [["doc_a", "doc_b"]]


# --------------------------------------------------------------------------- #
# Retry : un échec de juge isolé est retenté une fois avant d'être définitif
# --------------------------------------------------------------------------- #


class _FakeJudge:
    """Un seul appel `judge(inputs=..., outputs=...)`, comme le vrai objet Judge."""

    name = "summary_faithfulness"

    def __init__(self, outcome):
        self._outcome = outcome  # SimpleNamespace(value=..., rationale=...) ou Exception
        self.calls = 0

    def __call__(self, *, inputs, outputs):
        self.calls += 1
        if isinstance(self._outcome, Exception):
            raise self._outcome
        return self._outcome


def _dataset_row(document_id: str) -> pd.DataFrame:
    return pd.DataFrame(
        [{"inputs": {"document_id": document_id}, "outputs": "sortie du modèle"}]
    )


def test_retry_succeeds_clears_error_and_fills_value(monkeypatch):
    judge = _FakeJudge(SimpleNamespace(value=4, rationale="ok au second essai"))
    monkeypatch.setattr(run_eval, "judges_for", lambda task: (judge,))

    rows = [
        {
            "document_id": "doc_a",
            "summary_faithfulness/error_message": "timed out after 300s",
        }
    ]
    run_eval._retry_judge_failures(Task.SUMMARY, "qwen2.5:7b", rows, _dataset_row("doc_a"))

    assert judge.calls == 1
    assert rows[0]["summary_faithfulness/value"] == 4
    assert rows[0]["summary_faithfulness/rationale"] == "ok au second essai"
    assert "summary_faithfulness/error_message" not in rows[0]
    assert "judge_excluded" not in rows[0]


def test_retry_fails_again_marks_excluded(monkeypatch):
    judge = _FakeJudge(TimeoutError("timed out after 1800s"))
    monkeypatch.setattr(run_eval, "judges_for", lambda task: (judge,))

    rows = [
        {
            "document_id": "doc_a",
            "summary_faithfulness/error_message": "timed out after 300s",
        }
    ]
    run_eval._retry_judge_failures(Task.SUMMARY, "qwen2.5:7b", rows, _dataset_row("doc_a"))

    assert judge.calls == 1
    assert rows[0]["judge_excluded"] is True
    assert "summary_faithfulness/value" not in rows[0]
    assert "timed out after 300s" in rows[0]["summary_faithfulness/error_message"]


def test_retry_does_not_touch_rows_without_error(monkeypatch):
    judge = _FakeJudge(SimpleNamespace(value=5, rationale="ne devrait pas être appelé"))
    monkeypatch.setattr(run_eval, "judges_for", lambda task: (judge,))

    rows = [{"document_id": "doc_a", "summary_faithfulness/value": 5}]
    run_eval._retry_judge_failures(Task.SUMMARY, "qwen2.5:7b", rows, _dataset_row("doc_a"))

    assert judge.calls == 0
    assert rows[0]["summary_faithfulness/value"] == 5
