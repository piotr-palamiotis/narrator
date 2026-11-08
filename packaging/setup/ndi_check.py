#!/usr/bin/env python3
"""NDI checks for the setup scripts. Run with the venv's python.

  ndi_check.py runtime   Is narrator's own NDI sender able to load its runtime?
                         Reports whether the installed cyndilib wheel bundles
                         libndi (it does on PyPI manylinux wheels: a vendored
                         libndi under site-packages/cyndilib.libs/) and opens
                         an NDI Finder, which loads the library for real.
                         Exit 0 = works, 1 = cyndilib present but no runtime,
                         2 = cyndilib not installed.
  ndi_check.py find [--timeout S] [--config PATH]
                         mDNS discovery of narrator's NDI source through
                         narrator's own helper (narrator.audio.ndi_capture),
                         which is what `narrator --list-devices` uses. Exit 0 =
                         found, 1 = not found / discovery unavailable.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def bundled_libndi() -> list[str]:
    import importlib.util

    spec = importlib.util.find_spec("cyndilib")
    if spec is None or not spec.origin:
        return []
    pkg = Path(spec.origin).parent
    found = []
    for pattern_dir in (pkg.parent / "cyndilib.libs", pkg / "wrapper" / "bin"):
        if pattern_dir.is_dir():
            found += sorted(str(p) for p in pattern_dir.rglob("libndi*.so*"))
    return found


def cmd_runtime() -> int:
    try:
        import cyndilib
    except ImportError as exc:
        print(f"cyndilib not importable: {exc}")
        return 2
    from importlib.metadata import version

    libs = bundled_libndi()
    print(f"cyndilib {version('cyndilib')}: bundled libndi: {libs[0] if libs else 'none'}")
    try:
        from cyndilib import Finder

        finder = Finder()
        finder.open()
        finder.close()
    except Exception as exc:  # NDI runtime missing / unsupported CPU
        print(f"NDI Finder failed to open: {exc}")
        return 1
    print("NDI runtime loads (Finder opened)")
    return 0


def _configured_source_name(config: Path | None) -> str:
    try:
        import yaml
        from narrator.paths import repo_config_path

        path = config or repo_config_path()
        if path and path.is_file():
            return str((yaml.safe_load(path.read_text()) or {}).get("ndi", {}).get("source_name") or "LiveCaption")
    except Exception:
        pass
    return "LiveCaption"


def cmd_find(timeout: float, config: Path | None) -> int:
    from narrator.audio.ndi_capture import is_own_source, list_ndi_sources

    own = _configured_source_name(config)
    names = list_ndi_sources(timeout_s=timeout, own_stream_name=None)
    for n in names:
        print(f"  discovered: {n}")
    if any(is_own_source(n, own) for n in names):
        print(f"FOUND narrator's source (stream name {own!r})")
        return 0
    print(f"narrator's source (stream name {own!r}) not discovered"
          + (" (no NDI sources at all: NDI runtime or Avahi/mDNS problem?)" if not names else ""))
    return 1


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("runtime")
    f = sub.add_parser("find")
    f.add_argument("--timeout", type=float, default=5.0)
    f.add_argument("--config", type=Path, default=None)
    a = ap.parse_args()
    return cmd_runtime() if a.cmd == "runtime" else cmd_find(a.timeout, a.config)


if __name__ == "__main__":
    sys.exit(main())
