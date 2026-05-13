import dataclasses
import importlib
import json
import logging
import pathlib
import random
import subprocess
import sys
import traceback
from typing import Any
from typing import Literal

import numpy as np
from openpi_client import action_chunk_broker
from openpi_client import websocket_client_policy
import tyro


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]


def _default_robotwin_path() -> pathlib.Path:
    return (REPO_ROOT / "third_party" / "RoboTwin").resolve()


def _prepend_path(path: pathlib.Path) -> None:
    path_str = str(path.resolve())
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


@dataclasses.dataclass(frozen=True)
class RobotwinBindings:
    root: pathlib.Path
    configs_path: pathlib.Path
    unstable_error: type[Exception]
    generate_episode_descriptions: Any


def _load_robotwin_bindings(robotwin_root: pathlib.Path) -> RobotwinBindings:
    _prepend_path(robotwin_root)
    _prepend_path(robotwin_root / "description" / "utils")

    from envs import CONFIGS_PATH  # type: ignore
    from envs.utils.create_actor import UnStableError  # type: ignore
    from generate_episode_instructions import generate_episode_descriptions  # type: ignore

    return RobotwinBindings(
        root=robotwin_root,
        configs_path=pathlib.Path(CONFIGS_PATH),
        unstable_error=UnStableError,
        generate_episode_descriptions=generate_episode_descriptions,
    )


def _load_yaml(path: pathlib.Path) -> dict[str, Any]:
    import yaml

    with path.open("r", encoding="utf-8") as f:
        return yaml.load(f.read(), Loader=yaml.FullLoader)


def _resolve_robot_path(robotwin_root: pathlib.Path, value: str) -> pathlib.Path:
    path = pathlib.Path(value)
    if not path.is_absolute():
        path = robotwin_root / path
    return path.resolve()


def _make_task_env(task_name: str):
    envs_module = importlib.import_module(f"envs.{task_name}")
    try:
        env_class = getattr(envs_module, task_name)
    except AttributeError as exc:
        raise SystemExit(f"Robotwin task '{task_name}' not found") from exc
    return env_class()


def _to_chw_uint8(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.dtype != np.uint8:
        if np.issubdtype(image.dtype, np.floating):
            image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)
    return np.transpose(image, (2, 0, 1))


def _encode_observation(observation: dict[str, Any], instruction: str) -> dict[str, Any]:
    return {
        "state": np.asarray(observation["joint_action"]["vector"], dtype=np.float32),
        "images": {
            "cam_high": _to_chw_uint8(observation["observation"]["head_camera"]["rgb"]),
            "cam_left_wrist": _to_chw_uint8(observation["observation"]["left_camera"]["rgb"]),
            "cam_right_wrist": _to_chw_uint8(observation["observation"]["right_camera"]["rgb"]),
        },
        "prompt": instruction,
    }


def _choose_instruction(
    task_name: str,
    instruction_type: Literal["seen", "unseen"],
    episode_info: dict[str, Any],
    generate_episode_descriptions_fn,
) -> str:
    descriptions = generate_episode_descriptions_fn(task_name, [episode_info], 100)
    if not descriptions:
        return task_name.replace("_", " ")

    candidates = descriptions[0].get(instruction_type) or descriptions[0].get("seen") or descriptions[0].get("unseen")
    if not candidates:
        return task_name.replace("_", " ")
    return random.choice(candidates)


def _start_video_recorder(task_env, video_size: str, episode_index: int) -> None:
    if getattr(task_env, "eval_video_path", None) is None:
        return

    video_path = pathlib.Path(task_env.eval_video_path) / f"episode{episode_index}.mp4"
    ffmpeg = subprocess.Popen(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pixel_format",
            "rgb24",
            "-video_size",
            video_size,
            "-framerate",
            "10",
            "-i",
            "-",
            "-pix_fmt",
            "yuv420p",
            "-vcodec",
            "libx264",
            "-crf",
            "23",
            str(video_path),
        ],
        stdin=subprocess.PIPE,
    )
    task_env._set_eval_video_ffmpeg(ffmpeg)
    logging.info("Recording rollout video to %s", video_path)


def _write_summary(
    run_dir: pathlib.Path,
    *,
    task_name: str,
    task_config: str,
    instruction_type: str,
    num_trials: int,
    num_success: int,
    results: list[dict[str, Any]],
) -> pathlib.Path:
    summary = {
        "task_name": task_name,
        "task_config": task_config,
        "instruction_type": instruction_type,
        "num_trials": num_trials,
        "num_success": num_success,
        "success_rate": (num_success / num_trials) if num_trials else 0.0,
        "results": results,
    }
    results_path = run_dir / f"{task_name}_{task_config}_{instruction_type}.json"
    results_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return results_path


@dataclasses.dataclass
class Args:
    robotwin_path: pathlib.Path = dataclasses.field(default_factory=_default_robotwin_path)

    task_name: str = "grab_roller"
    task_config: str = "demo_randomized"
    instruction_type: Literal["seen", "unseen"] = "unseen"

    host: str = "0.0.0.0"
    port: int = 8000
    action_horizon: int = 50

    seed: int = 0
    num_trials: int = 100

    out_dir: pathlib.Path = dataclasses.field(default_factory=lambda: REPO_ROOT / "data" / "robotwin" / "eval")
    record_videos: bool = True


def main(args: Args) -> None:
    bindings = _load_robotwin_bindings(args.robotwin_path.resolve())

    task_config_path = bindings.configs_path / f"{args.task_config}.yml"
    if not task_config_path.exists():
        raise FileNotFoundError(f"Robotwin task config not found: {task_config_path}")
    task_kwargs = _load_yaml(task_config_path)

    embodiment_config = _load_yaml(bindings.configs_path / "_embodiment_config.yml")
    camera_config = _load_yaml(bindings.configs_path / "_camera_config.yml")

    head_camera_type = task_kwargs["camera"]["head_camera_type"]
    head_camera = camera_config[head_camera_type]
    task_kwargs["head_camera_h"] = head_camera["h"]
    task_kwargs["head_camera_w"] = head_camera["w"]
    video_size = f"{head_camera['w']}x{head_camera['h']}"

    embodiment_type = task_kwargs["embodiment"]
    if len(embodiment_type) == 1:
        left_robot_file = _resolve_robot_path(bindings.root, embodiment_config[embodiment_type[0]]["file_path"])
        right_robot_file = left_robot_file
        task_kwargs["dual_arm_embodied"] = True
    elif len(embodiment_type) == 3:
        left_robot_file = _resolve_robot_path(bindings.root, embodiment_config[embodiment_type[0]]["file_path"])
        right_robot_file = _resolve_robot_path(bindings.root, embodiment_config[embodiment_type[1]]["file_path"])
        task_kwargs["embodiment_dis"] = embodiment_type[2]
        task_kwargs["dual_arm_embodied"] = False
    else:
        raise ValueError("Robotwin embodiment config must contain either 1 or 3 items.")

    task_kwargs["task_name"] = args.task_name
    task_kwargs["task_config"] = args.task_config
    task_kwargs["left_robot_file"] = str(left_robot_file)
    task_kwargs["right_robot_file"] = str(right_robot_file)
    task_kwargs["left_embodiment_config"] = _load_yaml(left_robot_file / "config.yml")
    task_kwargs["right_embodiment_config"] = _load_yaml(right_robot_file / "config.yml")
    task_kwargs["eval_mode"] = True
    task_kwargs["save_data"] = False

    run_dir = (args.out_dir / args.task_name / args.task_config).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    logging.info("Robotwin eval outputs will be written under %s", run_dir)
    if args.record_videos and task_kwargs.get("eval_video_log", False):
        task_kwargs["eval_video_save_dir"] = str(run_dir)
        logging.info("Per-rollout videos enabled")
    else:
        logging.info("Per-rollout videos disabled")

    ws_policy = websocket_client_policy.WebsocketClientPolicy(host=args.host, port=args.port)
    logging.info("Server metadata: %s", ws_policy.get_server_metadata())
    policy = action_chunk_broker.ActionChunkBroker(ws_policy, action_horizon=args.action_horizon)

    task_env = _make_task_env(args.task_name)
    task_env.suc = 0
    task_env.test_num = 0

    next_seed = 100000 * (1 + args.seed)
    episode_index = 0
    results: list[dict[str, Any]] = []

    while task_env.test_num < args.num_trials:
        render_freq = task_kwargs["render_freq"]
        task_kwargs["render_freq"] = 0

        try:
            task_env.setup_demo(now_ep_num=episode_index, seed=next_seed, is_test=True, **task_kwargs)
            episode_info = task_env.play_once()
            expert_success = bool(task_env.plan_success and task_env.check_success())
            task_env.close_env()
        except bindings.unstable_error:
            task_env.close_env()
            next_seed += 1
            task_kwargs["render_freq"] = render_freq
            continue
        except Exception:
            logging.warning("Robotwin expert rollout failed for seed %s:\n%s", next_seed, traceback.format_exc())
            try:
                task_env.close_env()
            except Exception:
                pass
            next_seed += 1
            task_kwargs["render_freq"] = render_freq
            continue

        task_kwargs["render_freq"] = render_freq
        if not expert_success:
            next_seed += 1
            continue

        task_env.setup_demo(now_ep_num=episode_index, seed=next_seed, is_test=True, **task_kwargs)
        instruction = _choose_instruction(
            args.task_name,
            args.instruction_type,
            episode_info["info"],
            bindings.generate_episode_descriptions,
        )
        task_env.set_instruction(instruction)

        if getattr(task_env, "eval_video_path", None) is not None:
            _start_video_recorder(task_env, video_size, task_env.test_num)

        success = False
        policy.reset()
        while task_env.take_action_cnt < task_env.step_lim:
            raw_observation = task_env.get_obs()
            encoded_observation = _encode_observation(raw_observation, instruction)
            action = policy.infer(encoded_observation)
            task_env.take_action(action["actions"], action_type="qpos")
            if task_env.eval_success:
                success = True
                break

        if getattr(task_env, "eval_video_path", None) is not None:
            task_env._del_eval_video_ffmpeg()

        if success:
            task_env.suc += 1

        task_env.test_num += 1
        results.append(
            {
                "episode_index": episode_index,
                "seed": next_seed,
                "instruction": instruction,
                "success": success,
            }
        )
        results_path = _write_summary(
            run_dir,
            task_name=args.task_name,
            task_config=args.task_config,
            instruction_type=args.instruction_type,
            num_trials=task_env.test_num,
            num_success=task_env.suc,
            results=results,
        )
        logging.info(
            "Task=%s config=%s success=%s rate=%.2f%% seed=%s",
            args.task_name,
            args.task_config,
            success,
            100.0 * task_env.suc / task_env.test_num,
            next_seed,
        )
        logging.info("Updated results file: %s", results_path)

        clear_cache_freq = int(task_kwargs.get("clear_cache_freq", 1))
        task_env.close_env(clear_cache=(task_env.test_num % clear_cache_freq == 0))
        episode_index += 1
        next_seed += 1

    results_path = _write_summary(
        run_dir,
        task_name=args.task_name,
        task_config=args.task_config,
        instruction_type=args.instruction_type,
        num_trials=task_env.test_num,
        num_success=task_env.suc,
        results=results,
    )
    logging.info("Wrote results to %s", results_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    tyro.cli(main)
