"""narrator must import and run without narrator_tune (the offline tuning
tool is not part of the app, its service, or the PyInstaller build)."""
import re
import subprocess
import sys
import textwrap
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"


def test_narrator_runs_without_narrator_tune(tmp_path):
    script = textwrap.dedent(
        """
        import importlib.abc
        import sys
        from pathlib import Path


        class _BlockTune(importlib.abc.MetaPathFinder):
            def find_spec(self, name, path=None, target=None):
                if name == "narrator_tune" or name.startswith("narrator_tune."):
                    raise ImportError(f"{name} is not installed (blocked by test)")
                return None


        sys.meta_path.insert(0, _BlockTune())

        import narrator.main
        from narrator.app_controller import AppController
        from narrator.asr import bench, stream_sim, streaming_decoder, worker
        from narrator.config.loader import ensure_config_exists, load_config

        config_path = Path(sys.argv[1])
        ensure_config_exists(config_path)
        config = load_config(config_path)
        assert config.asr.model
        assert not any(name.startswith("narrator_tune") for name in sys.modules)
        print("OK")
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path / "config.yaml")],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().endswith("OK")


def test_narrator_sources_never_import_narrator_tune():
    pattern = re.compile(r"^\s*(from|import)\s+narrator_tune\b|import_module\(\s*[\"']narrator_tune", re.MULTILINE)
    offenders = [
        str(path.relative_to(SRC))
        for path in (SRC / "narrator").rglob("*.py")
        if pattern.search(path.read_text(encoding="utf-8"))
    ]
    assert offenders == []
