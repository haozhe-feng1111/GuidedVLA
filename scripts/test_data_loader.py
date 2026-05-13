"""Test script to verify data loader functionality and check loaded data format."""

import argparse
import dataclasses
import logging
import sys

import torch

import openpi.training.config as _config
from openpi.training.data_loader import create_data_loader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
    force=True,
)
logger = logging.getLogger(__name__)


def inspect_data_loader(
    config_name: str,
    *,
    num_batches: int = 3,
    framework: str = "pytorch",
    split: str = "train",
    num_workers: int = 0,
):
    """Test data loader and print information about loaded batches.

    Args:
        config_name: Name of the training config to use.
        num_batches: Number of batches to test.
        framework: Framework to use ("jax" or "pytorch").
        split: Data split to use ("train", "val", or "all").
        num_workers: Number of worker processes (0 = single process, recommended for testing).
    """
    logger.info(f"Loading config: {config_name}")
    config = _config._CONFIGS_DICT[config_name]  # noqa: SLF001

    # Override num_workers for testing to avoid multiprocessing issues
    original_num_workers = config.num_workers
    if num_workers is not None:
        config = dataclasses.replace(config, num_workers=num_workers)
        logger.info(f"Using num_workers={num_workers} (original: {original_num_workers})")

    logger.info(f"Creating data loader (framework={framework}, split={split})...")
    data_loader = create_data_loader(
        config,
        framework=framework,
        split=split,
        num_batches=num_batches,
        shuffle=False,  # Don't shuffle for testing
    )

    logger.info(f"Data loader created. Testing {num_batches} batches...")
    logger.info("=" * 80)

    batch_count = 0
    for batch_idx, (observation, actions, object_targets) in enumerate(data_loader):
        batch_count += 1
        logger.info(f"\n{'=' * 80}")
        logger.info(f"Batch {batch_idx + 1}/{num_batches}")
        logger.info(f"{'=' * 80}")

        # Check observation
        logger.info("\n[Observation]")
        logger.info(f"  Type: {type(observation)}")
        if hasattr(observation, "images"):
            logger.info(f"  Images keys: {list(observation.images.keys())}")
            for key, img in observation.images.items():
                if isinstance(img, torch.Tensor):
                    logger.info(f"    {key}: shape={img.shape}, dtype={img.dtype}")
                else:
                    logger.info(f"    {key}: type={type(img)}")

        if hasattr(observation, "state"):
            state = observation.state
            if isinstance(state, torch.Tensor):
                logger.info(f"  State: shape={state.shape}, dtype={state.dtype}")
            else:
                logger.info(f"  State: type={type(state)}")

        if hasattr(observation, "skill_id"):
            skill_id = observation.skill_id
            if skill_id is not None:
                if isinstance(skill_id, torch.Tensor):
                    logger.info(
                        "  Skill ID: shape=%s, dtype=%s, sample_values=%s",
                        skill_id.shape,
                        skill_id.dtype,
                        skill_id[:3].tolist() if len(skill_id) >= 3 else skill_id.tolist(),
                    )
                else:
                    logger.info(f"  Skill ID: type={type(skill_id)}, value={skill_id}")

        if hasattr(observation, "skill_soft"):
            skill_soft = observation.skill_soft
            if skill_soft is not None and isinstance(skill_soft, torch.Tensor):
                logger.info(f"  Skill Soft: shape={skill_soft.shape}, dtype={skill_soft.dtype}")
                logger.info(f"    Sample distribution (first sample): {skill_soft[0].tolist()[:5]}...")

        # Check actions
        logger.info("\n[Actions]")
        if isinstance(actions, torch.Tensor):
            logger.info(f"  Shape: {actions.shape}, dtype={actions.dtype}")
            logger.info(f"  Value range: [{actions.min().item():.3f}, {actions.max().item():.3f}]")
        else:
            logger.info(f"  Type: {type(actions)}")

        # Check object targets
        logger.info("\n[Object Targets]")
        if object_targets is None:
            logger.info("  No object targets found.")
        else:
            logger.info(f"  Keys: {list(object_targets.keys())}")

            if "object_maps" in object_targets:
                obj_maps = object_targets["object_maps"]
                obj_masks = object_targets.get("object_masks")
                if isinstance(obj_maps, torch.Tensor):
                    logger.info(f"  Object maps: shape={obj_maps.shape}, dtype={obj_maps.dtype}")
                    logger.info(f"    Value range: [{obj_maps.min().item():.3f}, {obj_maps.max().item():.3f}]")
                    if obj_masks is not None:
                        logger.info(f"  Object masks: shape={obj_masks.shape}, dtype={obj_masks.dtype}")
                        logger.info(f"    Valid views per sample: {obj_masks.sum(dim=1).tolist()[:5]}...")

        if batch_count >= num_batches:
            break

    logger.info(f"\n{'=' * 80}")
    logger.info(f"✓ Successfully loaded and tested {batch_count} batches")
    logger.info(f"{'=' * 80}")


def main():
    parser = argparse.ArgumentParser(
        description="Test data loader and verify loaded data format",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config-name",
        type=str,
        required=True,
        help="Name of the training config to use (e.g., 'pi0_base_aloha_robotwin_full_skill')",
    )
    parser.add_argument(
        "--num-batches",
        type=int,
        default=3,
        help="Number of batches to test",
    )
    parser.add_argument(
        "--framework",
        type=str,
        default="pytorch",
        choices=["jax", "pytorch"],
        help="Framework to use",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "val", "all"],
        help="Data split to use",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of worker processes (0 = single process, recommended for testing to avoid multiprocessing issues)",
    )

    args = parser.parse_args()

    inspect_data_loader(
        config_name=args.config_name,
        num_batches=args.num_batches,
        framework=args.framework,
        split=args.split,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
