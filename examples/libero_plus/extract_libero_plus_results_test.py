import copy
import json
import sys

import pytest

from examples.libero_plus import extract_libero_plus_results as extractor


def result_data():
    return {
        "meta": {
            "task_suite_name": "libero_spatial",
            "selected_task_ids": {"libero_spatial": [0, 1]},
            "num_trials_per_task": 1,
        },
        "success": [{"task_id": 0, "episode_index": 0, "extra": {"suite": "libero_spatial"}}],
        "failure": [{"task_id": 1, "episode_index": 0, "extra": {"suite": "libero_spatial"}}],
        "running_counts": {"total_episodes": 100, "total_successes": 100},
    }


@pytest.mark.parametrize("completed", [None, True])
def test_complete_results_use_episode_evidence_including_legacy(completed):
    data = result_data()
    if completed is not None:
        data["meta"]["completed"] = completed
    assert extractor.extract_rate(data) == 0.5


@pytest.mark.parametrize(
    "problem",
    ["error", "missing", "duplicate", "unexpected", "empty", "metadata", "counters_only", "unfinished"],
)
def test_invalid_results_cannot_produce_final_scores(problem):
    data = result_data()
    if problem == "error":
        data["failure"][0]["error"] = "ConnectionClosed"
    elif problem == "missing":
        data["failure"] = []
    elif problem == "duplicate":
        data["failure"] = copy.deepcopy(data["success"])
    elif problem == "unexpected":
        data["failure"][0]["task_id"] = 2
    elif problem == "empty":
        data["success"], data["failure"] = [], []
    elif problem == "metadata":
        del data["meta"]["selected_task_ids"]
    elif problem == "counters_only":
        del data["success"], data["failure"]
    else:
        data["meta"]["completed"] = False
    expected_message = {
        "error": "infrastructure error",
        "missing": "incomplete",
        "duplicate": "duplicate",
        "unexpected": "unexpected",
        "empty": "incomplete",
        "metadata": "cannot verify completion",
        "counters_only": "counters alone",
        "unfinished": "final validation",
    }
    with pytest.raises(ValueError, match=expected_message[problem]):
        extractor.extract_rate(data)


@pytest.mark.parametrize("bad_json", [False, True])
def test_cli_fails_without_overwriting_output_when_one_input_is_invalid(tmp_path, monkeypatch, capsys, bad_json):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    (inputs / "good.json").write_text(json.dumps(result_data()))
    data = result_data()
    data["failure"][0]["error"] = "ConnectionClosed"
    bad_path = inputs / "bad.json"
    bad_path.write_text("broken JSON" if bad_json else json.dumps(data))
    output = tmp_path / "review.tsv"
    output.write_text("previous evidence")
    monkeypatch.setattr(sys, "argv", ["extract", str(inputs), "--output", str(output)])
    assert extractor.main() == 1
    assert output.read_text() == "previous evidence"
    assert str(bad_path) in capsys.readouterr().err


def test_cli_accepts_complete_results_alongside_manifest(tmp_path, monkeypatch, capsys):
    suite = tmp_path / "libero_spatial"
    suite.mkdir()
    (suite / "results_Sensor_Noise.json").write_text(json.dumps(result_data()))
    (tmp_path / "eval_manifest.json").write_text('{"identity": {}}')
    monkeypatch.setattr(sys, "argv", ["extract", str(tmp_path)])
    assert extractor.main() == 0
    assert "50" in capsys.readouterr().out


@pytest.mark.parametrize("content", ["{}", "[]"])
def test_empty_result_file_is_not_silently_skipped(tmp_path, content):
    path = tmp_path / "results_Sensor_Noise.json"
    path.write_text(content)
    with pytest.raises(ValueError, match="evidence|JSON object"):
        extractor.load_scores([path])
