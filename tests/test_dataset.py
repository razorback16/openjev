"""Conversion invariants that protect labels, licensing, and split isolation."""
import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("pyarrow")
pytest.importorskip("yaml")
pytest.importorskip("datasketch")

spec = importlib.util.spec_from_file_location("compiler", Path(__file__).resolve().parents[1] / "scripts/build_public_dataset.py")
compiler = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compiler)


def test_tasksource_period_is_not_part_of_label():
    row = {"inputs": 'With no explanation, label the following with either "reverted_card_payment?" or "card_arrival".\nMy payment came back.',
           "targets": "reverted_card_payment?."}
    state, options, target, _ = compiler.parse_tasksource(row)
    assert target == "reverted_card_payment?"
    assert target in options
    assert state == "My payment came back."


def test_ambiguous_target_rejected():
    row = {"inputs": 'With no explanation, label the following with either "yes" or "no".\nA state.', "targets": "maybe."}
    with pytest.raises(ValueError, match="unmatched_target"):
        compiler.parse_tasksource(row)


def test_option_shuffle_preserves_semantic_targets():
    positive_positions = set()
    for i in range(100):
        r = {"id": str(i), "questions": {}, "targets": {}}
        compiler.categorical(r, ["one", "two", "three"], "two", "Choose a number.", 1)
        assert r["targets"]["q1"] == "two"
        assert "two" in r["questions"]["q1"]["criteria"]
        assert sorted(r["targets"][k] for k in ("q2", "q3")) == ["no", "yes"]
        for qid in ("q2", "q3"):
            if r["targets"][qid] == "yes":
                assert '"two"' in r["questions"][qid]["instructions"]
                positive_positions.add(qid)
    assert positive_positions == {"q2", "q3"}


def test_original_heldout_context_blocks_all_questions():
    index = compiler.NativeIndex()
    index.add("source", "same premise", "train question", "yes", "train:1", True)
    index.add("source", "same premise", "test question", "no", "test:1", False)
    assert index.get("source", "same premise", "train question") is None


def test_conflicting_original_labels_rejected():
    index = compiler.NativeIndex()
    index.add("source", "a", "b", "yes", "train:1", True)
    index.add("source", "a", "b", "no", "train:2", True)
    assert index.get("source", "a", "b") is None


def test_allowlist_and_benchmark_exclusion():
    import yaml
    root = Path(__file__).resolve().parents[1] / "dataset"
    registry = yaml.safe_load((root / "training_sources.yaml").read_text())
    evaluation = yaml.safe_load((root / "evaluation_sources.yaml").read_text())
    assert compiler.source_for("WANLI", "tasksource", registry, evaluation) == "wanli"
    for task in ("anli_r1_10templates", "snli_10templates", "bigbench/logic", "mmlu/math", "arc_easy_10templates", "unknown"):
        assert compiler.source_for(task, "flan", registry, evaluation) is None
    registry["sources"]["wanli"]["license"] = "CC-BY-NC-4.0"
    assert compiler.source_for("WANLI", "tasksource", registry, evaluation) is None


def test_split_assignment_is_deterministic():
    assert compiler.split_for("shared original") == compiler.split_for("shared original")
    assert {compiler.split_for(str(i)) for i in range(1000)} == {"train", "validation", "calibration"}


def test_near_duplicate_removes_whole_lower_priority_group():
    def rec(rid, state, split, group):
        return {"id": rid, "state": state, "split": split, "group_id": group, "provenance": {"source": "test"}}
    data = [rec("v", "A customer has requested a refund for their recent purchase.", "validation", "v"),
            rec("t", "A customer has requested a refund for their recent purchase.", "train", "shared"),
            rec("t2", "A different question from the same original context.", "train", "shared"),
            rec("c", "The server is offline because the power supply failed.", "calibration", "c")]
    kept, audit = compiler.near_duplicate_filter(data)
    assert {r["id"] for r in kept} == {"v", "c"}
    assert audit["removed_records"] == 2
