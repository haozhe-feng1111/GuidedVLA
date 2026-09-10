"""Exercise the real forward preprocessing decision without loading model weights."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]


class PreprocessingPolicyTest(unittest.TestCase):
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
