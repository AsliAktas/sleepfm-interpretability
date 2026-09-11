"""Tests for rigor_analysis.py — multiple testing, permutation, stability, sweep.

Uses statsmodels as an oracle for BH-FDR (which is easy to get wrong in the
step-up procedure). Uses synthetic labels+values to check that permutation
tests recover planted signals and that the Phipson-Smyth bound is respected.
"""

import numpy as np
import pandas as pd
import pytest
from statsmodels.stats.multitest import multipletests

from rigor_analysis import (
    _phipson_smyth_p,
    bonferroni_fdr_correction,
    bonferroni_fdr_family,
    bootstrap_stability,
    hdbscan_sweep,
    kruskal_p,
    multiseed_stability,
    permutation_test_familywise,
    permutation_test_pvalue,
)


class TestBonferroniFdrCorrection:
    def test_bonferroni_matches_manual(self):
        ps = [0.01, 0.04, 0.03, 0.005, 0.20]
        out = bonferroni_fdr_correction(ps)
        expected = np.minimum(np.array(ps) * 5, 1.0)
        np.testing.assert_allclose(out["bonferroni_p"].values, expected)

    def test_bh_fdr_matches_statsmodels(self):
        ps = [0.01, 0.04, 0.03, 0.005, 0.20]
        _, ref, _, _ = multipletests(ps, method="fdr_bh")
        got = bonferroni_fdr_correction(ps)["bh_fdr_p"].values
        np.testing.assert_allclose(got, ref, rtol=1e-9)

    def test_bh_fdr_monotone_after_sort(self):
        """BH-adjusted p-values must be monotone non-decreasing in sorted order."""
        rng = np.random.default_rng(0)
        ps = rng.uniform(0, 1, size=20).tolist()
        out = bonferroni_fdr_correction(ps)
        sorted_adj = out.assign(raw=ps).sort_values("raw")["bh_fdr_p"].values
        assert (np.diff(sorted_adj) >= -1e-9).all()

    def test_nan_p_values_pass_through(self):
        ps = [0.01, np.nan, 0.20]
        out = bonferroni_fdr_correction(ps)
        assert np.isnan(out["bonferroni_p"].iloc[1])
        assert np.isnan(out["bh_fdr_p"].iloc[1])
        assert not out["reject_bonferroni"].iloc[1]

    def test_n_planned_overrides_len(self):
        """When n_planned > len(p_values), Bonferroni divides by larger n."""
        ps = [0.01, 0.04]
        out_default = bonferroni_fdr_correction(ps)
        out_planned = bonferroni_fdr_correction(ps, n_planned=10)
        assert out_planned["bonferroni_p"].iloc[0] > out_default["bonferroni_p"].iloc[0]
        np.testing.assert_allclose(out_planned["bonferroni_p"].iloc[0], 0.01 * 10)

    def test_bh_uses_same_family_size_as_bonferroni(self):
        """Audit finding 4: BH must use n_planned (not valid.sum()) so a
        family with NaN entries does not silently relax the correction."""
        ps = [0.01, 0.02, 0.03, np.nan]
        # With n_planned=4, the smallest BH-adjusted p is 0.01 * 4/1 = 0.04
        # (further pooled by the step-down min-cummax); if BH used m=3 it
        # would give 0.01 * 3/1 = 0.03 — a false-positive-friendly value.
        out = bonferroni_fdr_correction(ps, n_planned=4)
        assert out["bh_fdr_p"].iloc[0] >= 0.04 - 1e-9

    def test_bh_fdr_matches_statsmodels_no_nan(self):
        """Backward-compat: with no NaN entries our BH still equals statsmodels."""
        ps = [0.01, 0.04, 0.03, 0.005, 0.20]
        _, ref, _, _ = multipletests(ps, method="fdr_bh")
        got = bonferroni_fdr_correction(ps)["bh_fdr_p"].values
        np.testing.assert_allclose(got, ref, rtol=1e-9)

    def test_reject_threshold(self):
        ps = [0.001, 0.5]
        out = bonferroni_fdr_correction(ps, alpha=0.05)
        assert bool(out["reject_bonferroni"].iloc[0])
        assert not bool(out["reject_bonferroni"].iloc[1])


class TestBonferroniFdrFamily:
    def test_joint_family_size(self):
        ps = {
            ("BAS", "age"): 0.01,
            ("BAS", "ahi"): 0.20,
            ("RESP", "age"): 0.02,
            ("RESP", "ahi"): 0.30,
        }
        out = bonferroni_fdr_family(ps)
        np.testing.assert_allclose(out["bonferroni_p"].iloc[0], 0.04)
        assert "modality" in out.columns and "variable" in out.columns


class TestPhipsonSmyth:
    def test_never_zero(self):
        assert _phipson_smyth_p(0, 1000) == 1 / 1001

    def test_never_greater_than_one(self):
        assert _phipson_smyth_p(1000, 1000) == 1001 / 1001

    def test_ordering_preserved(self):
        assert _phipson_smyth_p(5, 100) < _phipson_smyth_p(50, 100)


class TestPermutationPvalue:
    def test_p_bounded_positive(self):
        rng = np.random.default_rng(0)
        labels = np.array([0, 0, 1, 1, 0, 1])
        vals = rng.normal(size=6)
        _, emp_p, _ = permutation_test_pvalue(labels, vals, kruskal_p, n_permutations=100)
        assert emp_p > 0.0, "Phipson-Smyth must prevent empirical_p == 0"
        assert emp_p <= 1.0

    def test_deterministic_with_same_seed(self):
        rng = np.random.default_rng(0)
        labels = np.array([0, 0, 1, 1, 0, 1])
        vals = rng.normal(size=6)
        obs1, emp1, _ = permutation_test_pvalue(labels, vals, kruskal_p, n_permutations=200, seed=42)
        obs2, emp2, _ = permutation_test_pvalue(labels, vals, kruskal_p, n_permutations=200, seed=42)
        assert obs1 == obs2
        assert emp1 == emp2

    def test_planted_signal_recovered(self):
        labels = np.array([0] * 15 + [1] * 15)
        vals = np.concatenate([
            np.random.default_rng(42).normal(loc=0, scale=1, size=15),
            np.random.default_rng(43).normal(loc=5, scale=1, size=15),
        ])
        _, emp_p, _ = permutation_test_pvalue(labels, vals, kruskal_p, n_permutations=200)
        assert emp_p < 0.05, f"strong signal should yield low empirical_p, got {emp_p}"


class TestPermutationFamilywise:
    def test_familywise_adjusted_p_is_stricter_than_per_var(self):
        """Westfall-Young step-down: familywise_adjusted_p >= per_var_p for every variable."""
        rng = np.random.default_rng(0)
        n = 30
        labels = np.array([0] * 15 + [1] * 15)
        meta = pd.DataFrame({
            "v1": np.concatenate([rng.normal(0, 1, 15), rng.normal(3, 1, 15)]),  # planted
            "v2": rng.normal(0, 1, n),  # null
            "v3": rng.normal(0, 1, n),  # null
        })
        out = permutation_test_familywise(labels, meta, ["v1", "v2", "v3"],
                                          kruskal_p, n_permutations=200)
        assert (out["familywise_adjusted_p"] >= out["per_var_empirical_p"] - 1e-9).all()

    def test_preserves_row_correlations(self):
        """Joint row shuffle keeps (v1, v2) correlation structure intact."""
        rng = np.random.default_rng(0)
        n = 40
        labels = rng.integers(0, 2, size=n)
        v1 = rng.normal(size=n)
        v2 = v1 + rng.normal(scale=0.1, size=n)  # highly correlated with v1
        meta = pd.DataFrame({"v1": v1, "v2": v2})
        # If we shuffled independently, joint distribution corr would break; the
        # test itself doesn't observe that directly, but with true row-shuffle
        # both variables move together — so their per-var null distributions
        # should be identical up to noise.
        out = permutation_test_familywise(labels, meta, ["v1", "v2"], kruskal_p,
                                          n_permutations=100, seed=7)
        # v1 and v2 (near-identical) should get near-identical observed_p
        assert abs(out.iloc[0]["observed_p"] - out.iloc[1]["observed_p"]) < 0.1


class TestKruskalP:
    def test_returns_one_when_insufficient_data(self):
        assert kruskal_p(np.array([0]), np.array([1.0])) == 1.0

    def test_returns_low_p_for_strong_signal(self):
        labels = np.array([0] * 10 + [1] * 10)
        vals = np.concatenate([np.zeros(10), np.ones(10) * 10])
        assert kruskal_p(labels, vals) < 0.001

    def test_nan_safe(self):
        labels = np.array([0, 0, 0, 1, 1, 1])
        vals = np.array([1.0, np.nan, 2.0, 5.0, 6.0, np.nan])
        assert 0.0 < kruskal_p(labels, vals) < 1.0


class TestMultiseedStability:
    def test_diagonal_is_one(self):
        X = np.random.RandomState(0).normal(size=(30, 128))
        stab = multiseed_stability(X, n_seeds=3)
        np.testing.assert_allclose(np.diag(stab.pairwise_ari), 1.0)

    def test_symmetric(self):
        X = np.random.RandomState(0).normal(size=(30, 128))
        stab = multiseed_stability(X, n_seeds=3)
        np.testing.assert_allclose(stab.pairwise_ari, stab.pairwise_ari.T)

    def test_n_seeds_matches_labels(self):
        X = np.random.RandomState(0).normal(size=(30, 128))
        stab = multiseed_stability(X, n_seeds=4)
        assert len(stab.labels_per_seed) == 4
        assert stab.pairwise_ari.shape == (4, 4)


class TestBootstrapStability:
    def test_returns_dataframe_of_correct_shape(self):
        X = np.random.RandomState(0).normal(size=(30, 128))
        df = bootstrap_stability(X, n_bootstraps=4, subsample_frac=0.8)
        assert list(df.columns) == ["bootstrap", "subsample_size", "n_clusters",
                                    "n_noise", "noise_fraction"]
        assert len(df) == 4
        # 80% of 30 = 24
        assert (df["subsample_size"] == 24).all()

    def test_noise_fraction_bounded(self):
        X = np.random.RandomState(0).normal(size=(30, 128))
        df = bootstrap_stability(X, n_bootstraps=3)
        assert (df["noise_fraction"] >= 0).all()
        assert (df["noise_fraction"] <= 1).all()

    def test_deterministic_with_same_seed(self):
        X = np.random.RandomState(0).normal(size=(30, 128))
        a = bootstrap_stability(X, n_bootstraps=3, seed=7)
        b = bootstrap_stability(X, n_bootstraps=3, seed=7)
        pd.testing.assert_frame_equal(a, b)


class TestHdbscanSweep:
    def test_columns_and_length(self):
        X = np.random.RandomState(0).normal(size=(30, 128))
        out = hdbscan_sweep(X, min_cluster_sizes=(2, 3, 5))
        assert list(out.columns) == ["min_cluster_size", "n_clusters", "n_noise", "silhouette"]
        assert len(out) == 3

    def test_n_noise_nonnegative(self):
        X = np.random.RandomState(0).normal(size=(30, 128))
        out = hdbscan_sweep(X)
        assert (out["n_noise"] >= 0).all()
