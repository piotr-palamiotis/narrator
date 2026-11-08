"""Repo-anchored layout (narrator.paths): model lookup across models/ and the
legacy pywhispercpp cache, the shared repo root, config seeding."""
from pathlib import Path

import narrator.paths as paths
from narrator.config.loader import ensure_config_exists


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x")
    return path


def test_repo_root_is_this_checkout():
    assert paths.REPO_ROOT == Path(__file__).resolve().parents[1]
    assert paths.models_dir() == paths.REPO_ROOT / "models"
    assert paths.var_dir() == paths.REPO_ROOT / "var"


def test_narrator_tune_shares_the_repo_root():
    from narrator_tune import paths as tune_paths

    assert tune_paths.REPO_ROOT == paths.REPO_ROOT
    assert tune_paths.VAR_DIR == paths.var_dir() / "tune"


def test_model_prefers_repo_models_dir_over_legacy_cache(tmp_path, monkeypatch):
    repo_models, legacy = tmp_path / "models", tmp_path / "legacy"
    in_repo = _touch(repo_models / "ggml-small.bin")
    _touch(legacy / "ggml-small.bin")
    _touch(legacy / "ggml-large-v3.bin")
    monkeypatch.setattr(paths, "models_dir", lambda: repo_models)
    monkeypatch.setattr(paths, "_legacy_models_dir", lambda: legacy)

    assert paths.find_model_file("small") == in_repo.resolve()
    # Only in the legacy cache: reused, not re-downloaded.
    assert paths.find_model_file("large-v3") == (legacy / "ggml-large-v3.bin").resolve()
    assert paths.model_load_args("large-v3") == (str((legacy / "ggml-large-v3.bin").resolve()), None)
    assert paths.cached_model_names() == ["large-v3", "small"]


def test_missing_model_downloads_into_repo_models_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "models_dir", lambda: tmp_path / "models")
    monkeypatch.setattr(paths, "_legacy_models_dir", lambda: tmp_path / "legacy")

    assert paths.find_model_file("medium") is None
    assert paths.model_load_args("medium") == ("medium", str(tmp_path / "models"))


def test_direct_model_path_is_used_as_is(tmp_path, monkeypatch):
    custom = _touch(tmp_path / "elsewhere" / "custom.bin")
    monkeypatch.setattr(paths, "models_dir", lambda: tmp_path / "models")
    assert paths.model_load_args(str(custom)) == (str(custom.resolve()), None)


def test_outside_a_checkout_models_fall_back_to_pywhispercpp_default(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "REPO_ROOT", None)
    monkeypatch.setattr(paths, "_legacy_models_dir", lambda: tmp_path / "legacy")
    assert paths.models_dir() is None
    assert paths.model_load_args("medium") == ("medium", None)
    assert paths.var_dir() != Path(paths.__file__).resolve().parents[2] / "var"


def test_ensure_config_creates_missing_parent_dirs_from_defaults(tmp_path):
    target = tmp_path / "config" / "config.yaml"
    ensure_config_exists(target)
    assert target.read_text(encoding="utf-8").startswith("app:")
    target.write_text("app:\n  log_level: DEBUG\n", encoding="utf-8")
    ensure_config_exists(target)  # never overwrites
    assert "DEBUG" in target.read_text(encoding="utf-8")
