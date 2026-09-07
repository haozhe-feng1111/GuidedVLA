"""CPU regression tests; simulator imports are stubbed, evaluation code is real."""
# ruff: noqa: SLF001 -- regression tests exercise the existing private helpers

import dataclasses
import hashlib
import importlib.util
import json
import pathlib
import queue
import sys
import types
from unittest import mock

import numpy as np
import pytest

from examples.libero_plus import eval_libero_plus as runner


@pytest.fixture(autouse=True)
def reset_runner_state():
    runner.shutdown_event.clear()
    runner.shutdown_in_progress.clear()
    runner.task_failures.clear()
    runner.active_processes.clear()
    runner.gpu_reservations.clear()
    runner.gpu_first_ready.clear()
    runner.task_queue = queue.Queue()
    yield
    runner.shutdown_event.clear()


@pytest.fixture
def client_modules(monkeypatch):
    # Import client algorithms without installing MuJoCo, torch or either
    # benchmark. No policy/model implementation is replaced in production.
    for name in ("imageio", "libero", "libero.libero", "libero.libero.benchmark", "libero.libero.envs"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    benchmark = sys.modules["libero.libero.benchmark"]
    monkeypatch.setattr(sys.modules["libero.libero"], "benchmark", benchmark, raising=False)
    monkeypatch.setattr(sys.modules["libero.libero"], "get_libero_path", lambda name: name, raising=False)
    monkeypatch.setattr(sys.modules["libero.libero.envs"], "OffScreenRenderEnv", object, raising=False)
    monkeypatch.setattr(sys.modules["imageio"], "mimwrite", mock.Mock(), raising=False)
    image_tools = types.ModuleType("openpi_client.image_tools")
    image_tools.convert_to_uint8 = lambda value: value
    image_tools.resize_with_pad = lambda value, *_: value
    monkeypatch.setitem(sys.modules, "openpi_client.image_tools", image_tools)

    modules = {}
    for name in ("libero", "libero_plus"):
        module_name = f"_eval_test_{name}"
        path = runner.REPO_ROOT / "examples" / name / "main.py"
        spec = importlib.util.spec_from_file_location(module_name, path)
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, module_name, module)
        spec.loader.exec_module(module)
        modules[name] = module
    return modules


def test_client_environment_does_not_inherit_server_python_paths(monkeypatch, tmp_path):
    monkeypatch.setenv("PYTHONPATH", "/wrong/python3.11/site-packages:/wrong/client/src")
    monkeypatch.setenv("PYTHONHOME", "/wrong/python")
    monkeypatch.setenv("MAGICK_HOME", "/native/imagemagick")
    args = runner.Args(libero_plus_path=str(tmp_path), client_python="/client/bin/python")
    env = runner._build_client_env(args)
    assert env["PYTHONPATH"].split(":") == [str(tmp_path), str(runner.REPO_ROOT / "packages/openpi-client/src")]
    assert "PYTHONHOME" not in env
    assert env["PYTHONNOUSERSITE"] == "1"
    assert env["MAGICK_HOME"] == "/native/imagemagick"


def test_resume_command_is_accepted_by_the_actual_client_parser(client_modules, tmp_path):
    import tyro

    module = client_modules["libero_plus"]
    args = runner.Args(resume=True, libero_plus_path=str(tmp_path))
    command, _ = runner._build_client_cmd(
        args, {"suite": "libero_10", "category": "Sensor Noise", "slug": "sensor"}, 18080
    )
    parsed = tyro.cli(module.Args, args=module._normalize_cli_args(command[2:]))
    assert parsed.resume is True
    assert parsed.category == "Sensor Noise"
    assert parsed.port == 18080


@pytest.mark.parametrize("launch_error", [False, True])
def test_server_start_failure_terminates_instead_of_respawning(monkeypatch, tmp_path, launch_error):
    args = runner.Args(
        log_dir=str(tmp_path),
        spawn_cooldown_sec=0,
        check_interval_sec=0.01,
        enable_conservative_cuda_mem_cap=False,
    )
    monkeypatch.setattr(runner, "_validate_args", lambda _: "checkpoint")
    monkeypatch.setattr(
        runner, "_run_client_preflight", lambda *_: {"executable": "python", "prefix": "env", "imageio": "ok"}
    )
    monkeypatch.setattr(runner, "_discover_gpu_ids", lambda _: [0])
    monkeypatch.setattr(runner, "get_gpu_vram_stats", lambda _: (0, 80000))
    monkeypatch.setattr(
        runner, "_build_task_list", lambda _: [{"suite": "libero_10", "category": None, "slug": "libero_10"}]
    )
    start = mock.Mock(side_effect=FileNotFoundError("server dependency") if launch_error else None)
    start.return_value.poll.return_value = 1
    monkeypatch.setattr(runner, "_start_logged_process", start)
    monkeypatch.setattr(runner, "_wait_for_server", lambda *_: False)
    monkeypatch.setattr(runner.subprocess, "check_output", lambda *_args, **_kwargs: "missing tokenizer")
    with pytest.raises(RuntimeError, match="failed"):
        runner.main(args)
    assert start.call_count == 1
    assert runner.shutdown_event.is_set()


def _run_args(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "model.safetensors").write_bytes(b"weights")
    assets = checkpoint / "assets/ybwowen/libero"
    assets.mkdir(parents=True)
    (assets / "norm_stats.json").write_text("{}")
    return runner.Args(
        checkpoint_dir=str(checkpoint),
        categories="",
        libero_plus_path=str(tmp_path),
        server_python=sys.executable,
        client_python=sys.executable,
        results_base_dir=str(tmp_path / "results"),
        video_base_dir=str(tmp_path / "videos"),
        log_dir=str(tmp_path / "logs"),
    )


def test_resume_checks_identity_and_allows_resource_changes(tmp_path):
    args = _run_args(tmp_path)
    runner._validate_args(args)
    runner._validate_args(dataclasses.replace(args, resume=True, gpu_ids="3", start_port=12000))
    with pytest.raises(ValueError, match="identity"):
        runner._validate_args(dataclasses.replace(args, resume=True, replan_steps=10))
    with pytest.raises(ValueError, match="identity"):
        runner._validate_args(dataclasses.replace(args, resume=True, policy_config="another_model"))
    with pytest.raises(FileExistsError):
        runner._validate_args(args)


def test_resume_does_not_bypass_legacy_output_guard(tmp_path):
    args = _run_args(tmp_path)
    with pytest.raises(FileNotFoundError, match="eval_manifest"):
        runner._validate_args(dataclasses.replace(args, resume=True))


def test_resume_rejects_replaced_weights_at_the_same_path(tmp_path):
    args = _run_args(tmp_path)
    runner._validate_args(args)
    (pathlib.Path(args.checkpoint_dir) / "model.safetensors").write_bytes(b"another checkpoint")
    with pytest.raises(ValueError, match="identity"):
        runner._validate_args(dataclasses.replace(args, resume=True))


def test_checkpoint_digest_matches_sha256_without_file_digest(monkeypatch, tmp_path):
    args = _run_args(tmp_path)
    content = b"weights" * 400000  # Cross multiple read chunks.
    (pathlib.Path(args.checkpoint_dir) / "model.safetensors").write_bytes(content)
    monkeypatch.delattr(hashlib, "file_digest", raising=False)
    identity = runner._evaluation_identity(args)
    assert identity["checkpoint_files"]["model.safetensors"] == hashlib.sha256(content).hexdigest()
    runner._validate_args(args)
    runner._validate_args(dataclasses.replace(args, resume=True))


def test_missing_norm_stats_fail_before_outputs_or_workers(tmp_path):
    args = _run_args(tmp_path)
    (pathlib.Path(args.checkpoint_dir) / "assets/ybwowen/libero/norm_stats.json").unlink()
    with pytest.raises(FileNotFoundError, match="norm_stats"):
        runner._validate_args(args)
    assert not pathlib.Path(args.results_base_dir).exists()


def _store(client_modules, tmp_path):
    module = client_modules["libero_plus"]
    args = module.Args(results_json_path=str(tmp_path / "results.json"))
    store = module.ResultsStore(args)
    store.initialize({"libero_spatial": [0]})
    return module, args, store


def test_failed_episode_is_retried_and_replaced_across_buckets(client_modules, tmp_path):
    _, args, store = _store(client_modules, tmp_path)
    kwargs = {
        "task_id": 0,
        "task_description": "task",
        "episode_index": 0,
        "steps_taken": 1,
        "video_path": tmp_path / "failure.mp4",
        "extra": {"suite": "libero_spatial"},
    }
    store.record_episode(**kwargs, success=False, error="ConnectionClosed")
    data = json.loads(store.path.read_text())
    assert data["running_counts"] == {
        "total_episodes": 0,
        "total_successes": 0,
        "total_errors": 1,
        "success_rate": None,
    }
    assert data["meta"]["completed"] is False
    assert store.completed_episode(0, 0, "libero_spatial") is None
    with pytest.raises(RuntimeError, match="errored"):
        store.require_complete({"libero_spatial": [0]})
    store.record_episode(**kwargs, success=True)
    data = json.loads(pathlib.Path(args.results_json_path).read_text())
    assert len(data["success"]) == 1
    assert data["failure"] == []
    assert data["running_counts"]["total_episodes"] == 1
    assert store.completed_episode(0, 0, "libero_spatial")["success"] is True
    store.require_complete({"libero_spatial": [0]})
    data = json.loads(store.path.read_text())
    assert data["running_counts"]["total_errors"] == 0
    assert data["meta"]["completed"] is True
    from examples.libero_plus import extract_libero_plus_results as extractor

    assert extractor.extract_rate(data) == 1.0
    with pytest.raises(RuntimeError, match="missing"):
        store.require_complete({"libero_spatial": [0, 1]})


def test_results_reject_protocol_changes_and_corrupt_json(client_modules, tmp_path):
    module, args, _ = _store(client_modules, tmp_path)
    with pytest.raises(FileExistsError):
        module.ResultsStore(args).initialize({"libero_spatial": [0]})
    with pytest.raises(ValueError, match="protocol"):
        module.ResultsStore(dataclasses.replace(args, resume=True, seed=8)).initialize({"libero_spatial": [0]})
    path = pathlib.Path(args.results_json_path)
    path.write_text("broken evidence")
    with pytest.raises(RuntimeError, match="losing evidence"):
        module.ResultsStore(args).initialize({"libero_spatial": [0]})
    assert path.read_text() == "broken evidence"


def test_resume_recomputes_valid_counts_without_counting_infrastructure_errors(client_modules, tmp_path):
    module, args, store = _store(client_modules, tmp_path)
    kwargs = {
        "task_id": 0,
        "task_description": "task",
        "steps_taken": 1,
        "video_path": tmp_path / "video.mp4",
        "extra": {"suite": "libero_spatial"},
    }
    store.record_episode(**kwargs, episode_index=0, success=True)
    store.record_episode(**kwargs, episode_index=1, success=False)
    store.record_episode(**kwargs, episode_index=2, success=False, error="ConnectionClosed")
    expected = {"total_episodes": 2, "total_successes": 1, "total_errors": 1, "success_rate": 0.5}
    data = json.loads(store.path.read_text())
    assert data["running_counts"] == expected
    data["running_counts"] = {"total_episodes": 3, "total_successes": 1, "success_rate": 1 / 3}
    store.path.write_text(json.dumps(data))
    module.ResultsStore(dataclasses.replace(args, resume=True)).initialize({"libero_spatial": [0]})
    assert json.loads(store.path.read_text())["running_counts"] == expected


@pytest.mark.parametrize("name", ["libero", "libero_plus"])
@pytest.mark.parametrize("failure", [None, "model", "inference", "video"])
def test_episode_error_is_reported_and_environments_are_closed(client_modules, monkeypatch, tmp_path, name, failure):
    module = client_modules[name]
    args = module.Args(
        results_json_path=str(tmp_path / "results.json"),
        video_out_path=str(tmp_path / "videos"),
        num_trials_per_task=1,
        num_steps_wait=0,
    )
    task = types.SimpleNamespace(language="test task")
    suite = types.SimpleNamespace(n_tasks=2, get_task=lambda _: task, get_task_init_states=lambda _: [np.zeros(1)])
    monkeypatch.setattr(
        module.benchmark, "get_benchmark_dict", lambda: {"libero_spatial": lambda: suite}, raising=False
    )
    obs = {
        "agentview_image": np.zeros((2, 2, 3)),
        "robot0_eye_in_hand_image": np.zeros((2, 2, 3)),
        "robot0_eef_pos": np.zeros(3),
        "robot0_eef_quat": np.array([0.0, 0.0, 0.0, 1.0]),
        "robot0_gripper_qpos": np.zeros(2),
    }
    envs = []

    def make_env(*_):
        env = mock.Mock()
        env.reset.return_value = obs
        env.set_init_state.return_value = obs
        env.step.return_value = (obs, 0, failure != "model", {})
        envs.append(env)
        return env, "test task"

    monkeypatch.setattr(module, "_get_libero_env", make_env)
    client = mock.Mock()
    if failure == "inference":
        client.infer.side_effect = ConnectionError("broken inference")
    else:
        client.infer.return_value = {"actions": np.zeros((5, 7))}
    monkeypatch.setattr(module._websocket_client_policy, "WebsocketClientPolicy", lambda *_: client)
    if failure == "video":
        monkeypatch.setattr(module.imageio, "mimwrite", mock.Mock(side_effect=OSError("disk full")))
        with pytest.raises(OSError, match="disk full"):
            module.eval_libero(args)
    elif failure == "inference":
        with pytest.raises(RuntimeError, match="infrastructure"):
            module.eval_libero(args)
        data = json.loads(pathlib.Path(args.results_json_path).read_text())
        assert "ConnectionError" in data["failure"][0]["error"]
        assert client.infer.call_count == 1
    else:
        module.eval_libero(args)
        data = json.loads(pathlib.Path(args.results_json_path).read_text())
        assert len(data["failure"] if failure == "model" else data["success"]) == 2
        assert all(not row.get("error") for row in data["success"] + data["failure"])
        if name == "libero_plus":
            # No real video files were written. JSON alone must prevent reruns.
            calls = client.infer.call_count
            module.eval_libero(dataclasses.replace(args, resume=True))
            assert client.infer.call_count == calls
    assert all(env.close.call_count == 1 for env in envs)


def test_websocket_cold_inference_does_not_use_keepalive_timeout(monkeypatch):
    from openpi_client import msgpack_numpy
    from openpi_client import websocket_client_policy

    connection = mock.Mock()
    connection.recv.return_value = msgpack_numpy.packb({})
    connect = mock.create_autospec(websocket_client_policy.websockets.sync.client.connect, return_value=connection)
    monkeypatch.setattr(websocket_client_policy.websockets.sync.client, "connect", connect)
    websocket_client_policy.WebsocketClientPolicy("127.0.0.1", 8000)
    if websocket_client_policy.websockets.__version__.startswith("13."):
        assert "ping_timeout" not in connect.call_args.kwargs
    else:
        assert "ping_timeout" in connect.call_args.kwargs
        assert connect.call_args.kwargs["ping_timeout"] is None
