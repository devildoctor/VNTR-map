from __future__ import annotations

import importlib.util
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER_PATH = ROOT / "程序" / "20260826_VNTR功能统一调用器.py"
REGISTRY_PATH = ROOT / "程序" / "20260826_VNTR功能注册表.json"
INPUT_GFA = ROOT / "测试数据" / "IRF2BPL.HPRC.fa.3034e58.11fba48.904d69e.smooth.final.gfa"


def load_launcher():
    spec = importlib.util.spec_from_file_location("vntrmap_launcher", LAUNCHER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import launcher: {LAUNCHER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class UnifiedLauncherTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.launcher = load_launcher()

    def test_registry_dependency_order(self):
        registry, order = self.launcher.load_registry(REGISTRY_PATH)
        self.assertEqual(order, ["P1", "P2", "P3", "P4", "P5"])
        self.assertEqual(len(registry["stages"]), 5)

    def test_dry_run_resolves_all_stage_artifacts(self):
        with tempfile.TemporaryDirectory(prefix="vntrmap-launcher-") as temporary:
            outdir = Path(temporary) / "dryrun"
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                code = self.launcher.main(
                    [
                        "--registry",
                        str(REGISTRY_PATH),
                        "pipeline",
                        "--input",
                        str(INPUT_GFA),
                        "--outdir",
                        str(outdir),
                        "--prefix",
                        "test",
                        "--dry-run",
                    ]
                )
            self.assertEqual(code, 0, stdout.getvalue())
            manifest = json.loads((outdir / "run_manifest.json").read_text(encoding="utf-8"))
            self.assertTrue(manifest["success"])
            self.assertTrue((outdir / "run_attempts" / f"{manifest['attempt_id']}.json").is_file())
            self.assertEqual([row["stage"] for row in manifest["stage_results"]], ["P1", "P2", "P3", "P4", "P5"])
            self.assertTrue(all(row["status"] == "planned" for row in manifest["stage_results"]))
            p5_command = manifest["stage_results"][-1]["command"]
            self.assertIn(str(outdir / "P4" / "test_P4_P4变异证据标注图.gfa"), p5_command)

    def test_zero_exit_with_missing_required_output_fails(self):
        fake_registry = {
            "schema_version": 1,
            "stages": [
                {
                    "id": "X1",
                    "name": "fake",
                    "description": "zero exit without output",
                    "depends_on": [],
                    "command": [sys.executable, "-c", "print('completed without artifact')"],
                    "arguments": [],
                    "outputs": {"validation": "${stage_dir}/${stage_prefix}_validation.tsv"},
                    "required_outputs": ["validation"],
                }
            ],
        }
        with tempfile.TemporaryDirectory(prefix="vntrmap-launcher-") as temporary:
            temporary_path = Path(temporary)
            registry_path = temporary_path / "registry.json"
            outdir = temporary_path / "run"
            registry_path.write_text(json.dumps(fake_registry), encoding="utf-8")
            stdout = io.StringIO()
            stderr = io.StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                code = self.launcher.main(
                    [
                        "--registry",
                        str(registry_path),
                        "pipeline",
                        "--input",
                        str(INPUT_GFA),
                        "--outdir",
                        str(outdir),
                        "--prefix",
                        "fake",
                    ]
                )
            self.assertEqual(code, 1, stdout.getvalue() + stderr.getvalue())
            manifest = json.loads((outdir / "run_manifest.json").read_text(encoding="utf-8"))
            self.assertFalse(manifest["success"])
            self.assertEqual(manifest["stage_results"][0]["status"], "missing_output")
            self.assertEqual(manifest["stage_results"][0]["missing_outputs"], ["validation"])

    def test_registry_rejects_dependency_cycle(self):
        cyclic = {
            "schema_version": 1,
            "stages": [
                {
                    "id": stage_id,
                    "name": stage_id,
                    "depends_on": dependencies,
                    "command": [sys.executable],
                    "arguments": [],
                    "outputs": {"result": "${stage_dir}/result.txt"},
                    "required_outputs": ["result"],
                }
                for stage_id, dependencies in [("A", ["B"]), ("B", ["A"])]
            ],
        }
        with tempfile.TemporaryDirectory(prefix="vntrmap-launcher-") as temporary:
            path = Path(temporary) / "cyclic.json"
            path.write_text(json.dumps(cyclic), encoding="utf-8")
            with self.assertRaises(self.launcher.LauncherError):
                self.launcher.load_registry(path)


if __name__ == "__main__":
    unittest.main(verbosity=2)
