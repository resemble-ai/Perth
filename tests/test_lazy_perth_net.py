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
