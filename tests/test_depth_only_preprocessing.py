"""Exercise the real forward preprocessing decision without loading model weights."""
import ast
import os
import socket
import subprocess
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]


class PreprocessingPolicyTest(unittest.TestCase):
    def test_launcher_temp_socket_with_long_asset_root(self):
        # Exercise the real shell assignments, stopping before any preflight/write.
        launcher = ROOT / "manifests/train_libero_stage2_depth_only_b24_4gpu.sh"
        prefix = launcher.read_text().split("TRAIN_CMD=(", 1)[0]
        env = dict(os.environ, GUIDEDVLA_BASE="/tmp/" + "long-asset-root-" * 12)
        env.pop("GUIDEDVLA_TMP_ROOT", None)
        tmp = Path(subprocess.check_output(
            ["bash", "-c", prefix + '\nprintf "%s" "$TMP_ROOT"'],
            cwd=ROOT, env=env, text=True,
        ))
        # A long BASE/RUN_ID must not leak into the multiprocessing socket path.
        self.assertLess(len(str(tmp).encode()), 60)
        tmp.mkdir(mode=0o700)
        try:
            with tempfile.TemporaryDirectory(prefix="pymp-", dir=tmp) as child:
                with socket.socket(socket.AF_UNIX) as sock:
                    sock.bind(str(Path(child) / "listener-12345678"))
        finally:
            tmp.rmdir()

    def test_training_and_validation_policy(self):
        tree = ast.parse((ROOT / "src/openpi/models_pytorch/pi0_pytorch.py").read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "PI0Pytorch")
        forward = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "forward")
        forward.decorator_list = []
        module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), forward], type_ignores=[])
        ns = {}
        exec(compile(ast.fix_missing_locations(module), "forward-policy", "exec"), ns)

        class ReachedPreprocessing(Exception):
            pass

        for training, object_loss, disabled, expected in [
            (True, True, False, False),  # unchanged full-guidance baseline
            (True, False, False, True),  # unchanged legacy depth-only behavior
            (True, False, True, False), # new controlled depth-only arm
            (False, False, False, False),
            (False, False, True, False),
        ]:
            with self.subTest(training=training, object_loss=object_loss, disabled=disabled):
                captured = []
                def preprocess(observation, *, train):
                    captured.append(train)
                    raise ReachedPreprocessing
                model = SimpleNamespace(training=training, config=SimpleNamespace(disable_image_augmentation=disabled), _preprocess_observation=preprocess)
                with self.assertRaises(ReachedPreprocessing):
                    ns["forward"](model, SimpleNamespace(), None, use_object_loss=object_loss, object_targets=object())
                self.assertEqual(captured, [expected])


if __name__ == "__main__":
    unittest.main()
