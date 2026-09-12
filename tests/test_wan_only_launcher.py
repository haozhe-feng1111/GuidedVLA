"""CPU-only shell contract checks; never start training or import model code."""
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "manifests/train_libero_stage2_wan_only_b24_4gpu_company.sh"


class WanOnlyLauncherTest(unittest.TestCase):
    def test_print_mode_is_portable_and_has_no_writes(self):
        with tempfile.TemporaryDirectory(prefix="wan base ") as base:
            result = subprocess.run(["bash", str(SCRIPT)], cwd="/tmp", text=True,
                capture_output=True, check=True,
                env=dict(os.environ, GUIDEDVLA_BASE=base, GUIDEDVLA_PRINT_ONLY="1"))
            args = shlex.split(result.stdout)
            self.assertEqual(args[args.index("--batch-size") + 1], "24")
            self.assertEqual(args[args.index("--gradient-accumulation-steps") + 1], "1")
            self.assertIn("--nproc-per-node=4", args)
            self.assertIn("pi0_libero_wan22_vae_only_b24", args)
            self.assertIn("--model.disable-image-augmentation", args)
            self.assertIn("--use-gradient-checkpointing", args)
            self.assertEqual(args[args.index("--pytorch-weight-path") + 1],
                base + "/outputs/guidedvla_libero_stage1_4gpu_30k/pi0_libero/guidedvla_libero_stage1_4gpu_30k/30000")
            self.assertEqual(list(Path(base).iterdir()), [])

    def test_existing_run_is_rejected_before_import_or_gpu(self):
        with tempfile.TemporaryDirectory() as base:
            (Path(base) / "logs" / "existing").mkdir(parents=True)
            result = subprocess.run(["bash", str(SCRIPT)], capture_output=True, text=True,
                env=dict(os.environ, GUIDEDVLA_BASE=base, GUIDEDVLA_RUN_ID="existing",
                         GUIDEDVLA_PRINT_ONLY="0", GUIDEDVLA_CHECK_ONLY="1"))
            self.assertEqual(result.returncode, 2)
            self.assertIn("Refusing existing run directory", result.stderr)

    def test_missing_assets_fail_before_training(self):
        with tempfile.TemporaryDirectory() as base:
            result = subprocess.run(["bash", str(SCRIPT)], capture_output=True, text=True,
                env=dict(os.environ, GUIDEDVLA_BASE=base,
                         GUIDEDVLA_PRINT_ONLY="0", GUIDEDVLA_CHECK_ONLY="1"))
            self.assertEqual(result.returncode, 2)
            self.assertIn("Missing required asset", result.stderr)
            self.assertEqual(list(Path(base).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
