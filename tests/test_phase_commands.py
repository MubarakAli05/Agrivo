from contextlib import redirect_stdout
import io
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from agri.cli import main


class PhaseCommandTests(unittest.TestCase):
    def test_commands_are_lazy_and_forward_only_declared_arguments(self):
        observed = []

        def search(root, question, limit, split):
            observed.append((root, question, limit, split))
            return {"phase": 8, "status": "GREEN"}

        with patch("agri.phase_commands.import_module", return_value=SimpleNamespace(search=search)) as importer:
            output = io.StringIO()
            with redirect_stdout(output):
                code = main(["search", "--root", str(Path("example-workspace")), "--question", "pH?", "--limit", "2"])
        self.assertEqual(code, 0)
        self.assertEqual(observed, [(Path("example-workspace"), "pH?", 2, "train")])
        importer.assert_called_once_with("retrieval.index")
        self.assertEqual(json.loads(output.getvalue())["phase"], 8)

    def test_unavailable_optional_backend_is_a_phase_specific_json_error(self):
        with patch("agri.phase_commands.import_module", side_effect=ImportError("Optional dependency unavailable")):
            output = io.StringIO()
            with redirect_stdout(output):
                code = main(["build-index"])
        report = json.loads(output.getvalue())
        self.assertEqual(code, 1)
        self.assertEqual(report["phase"], 8)
        self.assertEqual(report["status"], "RED")
        self.assertIn("unavailable", report["error"])


if __name__ == "__main__":
    unittest.main()
