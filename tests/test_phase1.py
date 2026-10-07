from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest

from agri.cli import main
from agri.config import default_config, load_config, validate_config
from agri.workspace import DIRECTORIES, PROJECT_ROOT, setup, status


class ConfigurationTests(unittest.TestCase):
    def test_safe_defaults(self):
        config = validate_config(default_config())
        self.assertTrue(config["offline"])
        self.assertEqual(config["device"], "cpu")
        self.assertFalse(config["model"]["pretrained"])
        self.assertTrue(config["training"]["dry_run"])
        self.assertFalse(config["training"]["allow_large_runs"])

    def test_defaults_are_independent(self):
        config = default_config()
        config["model"]["layers"] = 99
        self.assertEqual(default_config()["model"]["layers"], 4)

    def test_invalid_configs(self):
        cases = [
            (None, "schema_version", True), (None, "schema_version", 2),
            (None, "seed", -1), (None, "offline", "false"),
            (None, "device", "invalid"), (None, "project", "other"),
            ("model", "layers", 0), ("model", "layers", True),
            ("model", "attention_heads", 3), ("model", "pretrained", True),
            ("model", "dropout", float("nan")), ("model", "dropout", 1),
            ("training", "learning_rate", float("inf")),
            ("training", "batch_size", -1), ("training", "epochs", 1.5),
            ("training", "dry_run", 1), ("training", "allow_large_runs", "yes"),
            ("training", "checkpoint_every_steps", 21),
        ]
        for section, key, value in cases:
            with self.subTest(section=section, key=key, value=value):
                config = default_config()
                target = config if section is None else config[section]
                target[key] = value
                with self.assertRaises(ValueError):
                    validate_config(config)
        for config in [None, [], {}, {**default_config(), "typo": 1}]:
            with self.assertRaises(ValueError):
                validate_config(config)
        for section in ["model", "training"]:
            config = default_config()
            config[section] = []
            with self.assertRaises(ValueError):
                validate_config(config)

    def test_project_config_matches_defaults(self):
        self.assertEqual(load_config(PROJECT_ROOT / "configs" / "agri-mini.json"),
                         default_config())


class WorkspaceTests(unittest.TestCase):
    def test_setup_is_idempotent_and_preserves_user_data(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertEqual(setup(root)["phase1_status"], "GREEN")
            for name in DIRECTORIES:
                self.assertTrue(root.joinpath(*name.split("/")).is_dir())
            raw = root / "data" / "raw" / "existing.txt"
            raw.write_bytes(b"unaltered source data")
            config_path = root / "configs" / "agri-mini.json"
            config = default_config()
            config["seed"] = 7
            config_path.write_text(json.dumps(config), encoding="utf-8")
            before = config_path.read_bytes()
            setup(root)
            self.assertEqual(config_path.read_bytes(), before)
            self.assertEqual(raw.read_bytes(), b"unaltered source data")
            self.assertTrue((root / "data" / "source_registry.yaml").exists())
            self.assertIsNone(status(root)["metrics"])

    def test_status_does_not_create_directories(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "not-created"
            self.assertEqual(status(root)["status"], "RED")
            self.assertFalse(root.exists())

    def test_setup_does_not_replace_invalid_config(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "configs").mkdir()
            path = root / "configs" / "agri-mini.json"
            path.write_text("{invalid", encoding="utf-8")
            with self.assertRaises(ValueError):
                setup(root)
            self.assertEqual(path.read_text(encoding="utf-8"), "{invalid")
            self.assertFalse((root / "data").exists())

    def test_setup_rejects_directory_collision(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "training").write_text("keep", encoding="utf-8")
            with self.assertRaises(ValueError):
                setup(root)
            self.assertEqual((root / "training").read_text(encoding="utf-8"), "keep")
            self.assertFalse((root / "app").exists())

    def test_cli_setup_and_status(self):
        with tempfile.TemporaryDirectory() as directory:
            for command, expected in [("status", 1), ("setup", 0), ("status", 0)]:
                output = io.StringIO()
                with redirect_stdout(output):
                    result = main([command, "--root", directory])
                self.assertEqual(result, expected)
                self.assertEqual(json.loads(output.getvalue())["phase"], 1 if expected else 2)

    def test_cli_reports_invalid_config(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            setup(root)
            (root / "configs" / "agri-mini.json").write_text("null", encoding="utf-8")
            output = io.StringIO()
            with redirect_stdout(output):
                result = main(["status", "--root", directory])
            self.assertEqual(result, 1)
            self.assertEqual(json.loads(output.getvalue())["status"], "RED")


if __name__ == "__main__":
    unittest.main()
