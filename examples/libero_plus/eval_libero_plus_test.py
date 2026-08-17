import pathlib

import pytest

from examples.libero_plus import eval_libero_plus


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
