import json
import pathlib

import pytest

from examples.libero_plus import eval_libero_plus


def test_build_task_list_sorts_largest_categories_first(tmp_path: pathlib.Path):
    classification_path = tmp_path / "libero" / "libero" / "benchmark" / "task_classification.json"
    classification_path.parent.mkdir(parents=True)
    classification_path.write_text(
        json.dumps(
            {
                "suite_a": [
                    {"category": "small"},
                    {"category": "large"},
                    {"category": "large"},
                ],
                "suite_b": [
                    {"category": "small"},
                    {"category": "small"},
                    {"category": "large"},
                    {"category": "large"},
                    {"category": "large"},
                ],
            }
        )
    )
    args = eval_libero_plus.Args(
        libero_plus_path=str(tmp_path),
        task_suites="suite_a,suite_b",
        categories="small,large",
    )

    tasks = eval_libero_plus._build_task_list(args)

    assert [(task["suite"], task["category"]) for task in tasks] == [
        ("suite_b", "large"),
        ("suite_a", "large"),
        ("suite_b", "small"),
        ("suite_a", "small"),
    ]


def test_output_dir_guard_allows_missing_or_empty_directory(tmp_path: pathlib.Path):
    output_dir = tmp_path / "new-run"
    eval_libero_plus._ensure_output_dir_unused(output_dir)
    output_dir.mkdir()
    eval_libero_plus._ensure_output_dir_unused(output_dir)


def test_output_dir_guard_rejects_prior_outputs(tmp_path: pathlib.Path):
    output_dir = tmp_path / "old-run"
    output_dir.mkdir()
    (output_dir / "results.json").write_text("{}")

    with pytest.raises(FileExistsError, match="Refusing to reuse"):
        eval_libero_plus._ensure_output_dir_unused(output_dir)
