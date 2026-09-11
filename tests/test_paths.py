"""Tests for paths.py — cohort resolution and env-var precedence.

Uses monkeypatch on os.environ so tests do not depend on the real
SLEEPFM_COHORT_ROOT the developer might have exported.
"""

from pathlib import Path

import pytest

from paths import CohortPaths, resolve_cohort


class TestResolveCohort:
    def test_explicit_root_wins(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SLEEPFM_COHORT_ROOT", str(tmp_path / "env_root"))
        explicit = tmp_path / "explicit_root"
        cohort = resolve_cohort(root=explicit)
        assert cohort.root == explicit.resolve()
        assert cohort.embedding_dir == (explicit / "embeddings").resolve()

    def test_env_used_when_no_arg(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SLEEPFM_COHORT_ROOT", str(tmp_path / "env_root"))
        cohort = resolve_cohort()
        assert cohort.root == (tmp_path / "env_root").resolve()

    def test_legacy_fallback_when_env_empty(self, monkeypatch, caplog):
        monkeypatch.delenv("SLEEPFM_COHORT_ROOT", raising=False)
        with caplog.at_level("WARNING"):
            cohort = resolve_cohort()
        assert "smoke_run" in str(cohort.root).lower()
        assert any("contaminated" in r.message.lower() for r in caplog.records), (
            "must warn when falling back to contaminated cohort"
        )

    def test_metadata_csv_env_overrides_root_layout(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SLEEPFM_COHORT_ROOT", str(tmp_path))
        custom_csv = tmp_path / "custom.csv"
        monkeypatch.setenv("SLEEPFM_METADATA_CSV", str(custom_csv))
        cohort = resolve_cohort()
        assert cohort.metadata_csv == custom_csv.resolve()

    def test_returns_cohort_paths_dataclass(self, tmp_path):
        cohort = resolve_cohort(root=tmp_path)
        assert isinstance(cohort, CohortPaths)
        assert cohort.label  # non-empty
        assert cohort.embedding_dir.name == "embeddings"
