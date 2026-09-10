"""Tests for clinical_analysis.py — cluster + statistical test pipeline.

Uses synthetic embedding matrices to test the UMAP+HDBSCAN wrapper and
statistical test dispatcher without depending on the real MESA metadata CSV.
"""

import numpy as np
import pandas as pd
import pytest

from clinical_analysis import (
    fit_umap_hdbscan,
    cluster_summary,
    run_statistical_tests,
    CONTINUOUS_VARS,
)


def _synthetic_metadata(n: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "subject_id": [f"{i:04d}" for i in range(n)],
        "age": rng.integers(50, 90, size=n),
        "bmi": rng.normal(28, 5, size=n),
        "ahi": rng.exponential(15, size=n),
        "ahi_obs": rng.exponential(12, size=n),
        "odi3": rng.exponential(18, size=n),
        "sex": rng.integers(0, 2, size=n),
        "race": rng.integers(1, 5, size=n),
    })


class TestFitUmapHdbscan:
    def test_returns_2d_coords_and_labels(self):
        X = np.random.RandomState(42).normal(size=(30, 128))
        coords, labels = fit_umap_hdbscan(X, seed=42)
        assert coords.shape == (30, 2)
        assert labels.shape == (30,)

    def test_labels_are_integer_valued(self):
        X = np.random.RandomState(42).normal(size=(30, 128))
        _, labels = fit_umap_hdbscan(X, seed=42)
        assert np.issubdtype(labels.dtype, np.integer)

    def test_n_neighbors_clamped_when_small_n(self):
        """UMAP n_neighbors must not exceed n_samples - 1."""
        X = np.random.RandomState(0).normal(size=(4, 128))
        coords, labels = fit_umap_hdbscan(X, umap_n_neighbors=10, seed=42)
        assert coords.shape == (4, 2)


class TestClusterSummary:
    def test_returns_int_keyed_dict(self):
        labels = np.array([0, 0, 1, 1, 1, -1, 2])
        summary = cluster_summary(labels)
        assert summary == {-1: 1, 0: 2, 1: 3, 2: 1}
        assert all(isinstance(k, int) for k in summary.keys())

    def test_all_noise(self):
        labels = np.array([-1, -1, -1])
        assert cluster_summary(labels) == {-1: 3}


class TestRunStatisticalTests:
    def test_kruskal_wallis_columns_present(self):
        labels = np.array([0, 0, 0, 1, 1, 1, 2, 2])
        meta = _synthetic_metadata(8)
        tests = run_statistical_tests(labels, meta)
        assert "variable" in tests.columns
        assert "test" in tests.columns
        assert "p_value" in tests.columns
        continuous_rows = tests[tests["test"] == "Kruskal-Wallis"]
        assert set(continuous_rows["variable"]) <= set(CONTINUOUS_VARS)

    def test_sex_gets_chi_square_or_fisher(self):
        labels = np.array([0, 0, 0, 1, 1, 1, 2, 2])
        meta = _synthetic_metadata(8)
        tests = run_statistical_tests(labels, meta)
        sex_row = tests[tests["variable"] == "sex"]
        assert len(sex_row) == 1
        assert sex_row.iloc[0]["test"] in {"chi-square", "Fisher's exact", "chi-square/fisher"}

    def test_single_cluster_returns_note(self):
        labels = np.zeros(10, dtype=int)
        meta = _synthetic_metadata(10)
        tests = run_statistical_tests(labels, meta)
        assert "note" in tests.columns
        assert tests.shape[0] == 1
        assert "one" in tests.iloc[0]["note"].lower() or "cluster" in tests.iloc[0]["note"].lower()

    def test_all_noise_returns_note(self):
        labels = np.full(10, -1, dtype=int)
        meta = _synthetic_metadata(10)
        tests = run_statistical_tests(labels, meta)
        assert tests.shape[0] == 1
        assert "insufficient" in tests.iloc[0]["note"].lower()

    def test_noise_points_excluded_from_tests(self):
        """Adding noise points should not change test p-values on core clusters."""
        core_labels = np.array([0, 0, 0, 1, 1, 1])
        core_meta = _synthetic_metadata(6, seed=1)
        noisy_labels = np.concatenate([core_labels, np.full(4, -1)])
        noisy_meta = pd.concat([core_meta, _synthetic_metadata(4, seed=99).assign(
            subject_id=[f"noise_{i}" for i in range(4)]
        )], ignore_index=True)

        core_tests = run_statistical_tests(core_labels, core_meta).set_index("variable")
        noisy_tests = run_statistical_tests(noisy_labels, noisy_meta).set_index("variable")

        for var in ["age", "ahi"]:
            if var in core_tests.index and var in noisy_tests.index:
                assert np.isclose(core_tests.loc[var, "p_value"],
                                  noisy_tests.loc[var, "p_value"], atol=1e-9), (
                    f"noise points changed {var} p-value")

    def test_significant_signal_recovered(self):
        """A planted cluster-vs-AHI signal should give low p-value."""
        rng = np.random.default_rng(42)
        n_per = 15
        labels = np.array([0] * n_per + [1] * n_per)
        meta = pd.DataFrame({
            "subject_id": [f"{i:04d}" for i in range(2 * n_per)],
            "age": rng.integers(50, 90, size=2 * n_per),
            "bmi": rng.normal(28, 5, size=2 * n_per),
            "ahi": np.concatenate([
                rng.normal(5, 2, size=n_per),   # cluster 0: low AHI
                rng.normal(40, 5, size=n_per),  # cluster 1: high AHI
            ]),
            "ahi_obs": rng.normal(15, 5, size=2 * n_per),
            "odi3": rng.normal(20, 8, size=2 * n_per),
            "sex": rng.integers(0, 2, size=2 * n_per),
        })
        tests = run_statistical_tests(labels, meta)
        ahi_p = float(tests[tests["variable"] == "ahi"]["p_value"].iloc[0])
        assert ahi_p < 0.001, f"planted signal should give p<0.001, got {ahi_p}"
