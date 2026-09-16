import subprocess
import sys
import unittest


class TestLazyPerthNetImports(unittest.TestCase):
    def test_import_perth_net_without_librosa(self):
        code = (
            "import sys\n"
            "from perth.perth_net.perth_net_implicit.model.perth_net import PerthNet\n"
            "assert PerthNet is not None\n"
            "assert 'librosa' not in sys.modules, sys.modules.keys()\n"
            "assert 'soxr' not in sys.modules\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_perth_package_import_without_librosa(self):
        code = (
            "import sys\n"
            "import perth\n"
            "assert 'librosa' not in sys.modules, sys.modules.keys()\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_core_dependencies_do_not_declare_librosa(self):
        from pathlib import Path

        text = Path(__file__).resolve().parents[1].joinpath("pyproject.toml").read_text()
        core = text.split("[project.optional-dependencies]", 1)[0]
        self.assertNotIn("librosa", core)
        self.assertIn("librosa>=0.11.0", text)
