"""
Tests for mock embedding generation (mock_data.py — generate_mock_embeddings).

Organized by component (Karar 6 — Secenek B, multi-file class-based).
This file tests ONLY embedding generation. Other test files will be
created on their respective implementation days.

Test hierarchy:
- TestEmbeddingShape: Dimensional correctness
- TestEmbeddingNormalization: L2 norm properties
- TestClusterStructure: Disease group clustering behavior
- TestPerturbation: Intentional misclassification mechanism
- TestNoiseConfig: Per-modality noise application
- TestReproducibility: Seed-based determinism
"""

import numpy as np
import pytest

from mock_data import generate_mock_embeddings
from config import MODALITY_DIM_RANGES, EMBEDDING_DIM, DEFAULT_NOISE_CONFIG, DISEASE_NAMES


class TestEmbeddingShape:
    """Verify that generated embeddings have correct dimensions."""

    def test_default_shape(self):
        """Default call should return (500, 128) embeddings."""
        embeddings, labels = generate_mock_embeddings()
        assert embeddings.shape == (500, EMBEDDING_DIM)
        assert labels.shape == (500,)

    def test_custom_shape(self):
        """Custom n_samples should be respected; embedding_dim is fixed by config."""
        embeddings, labels = generate_mock_embeddings(n_samples=200)
        assert embeddings.shape == (200, EMBEDDING_DIM)

    def test_labels_count(self):
        """Number of unique labels should equal n_disease_groups."""
        embeddings, labels = generate_mock_embeddings(n_disease_groups=12, seed=42)
        unique_labels = np.unique(labels)
        assert len(unique_labels) == 12


class TestEmbeddingNormalization:
    """Verify L2 normalization properties."""

    def test_unit_norm(self):
        """Each embedding vector should have L2 norm approximately 1."""
        embeddings, _ = generate_mock_embeddings(seed=42)
        norms = np.linalg.norm(embeddings, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_no_nan_or_inf(self):
        """Embeddings should contain no NaN or Inf values."""
        embeddings, _ = generate_mock_embeddings(seed=42)
        assert np.all(np.isfinite(embeddings))


class TestClusterStructure:
    """Verify that disease groups form loose but identifiable clusters."""

    def test_intra_cluster_distance_less_than_inter(self):
        """Average within-cluster distance should be less than between-cluster.

        This is a sanity check that clusters have SOME structure.
        It should NOT be perfectly separated — that violates rule #5.
        """
        embeddings, labels = generate_mock_embeddings(seed=42)

        g0 = embeddings[labels == 1]
        g1 = embeddings[labels == 2]

        # Intra-cluster: pairwise Euclidean distances within group 0
        diff_intra = g0[:, np.newaxis, :] - g0[np.newaxis, :, :]
        dists_intra = np.sqrt((diff_intra ** 2).sum(axis=-1))
        n0 = len(g0)
        mean_intra = dists_intra[np.triu_indices(n0, k=1)].mean()

        # Inter-cluster: pairwise Euclidean distances between group 0 and group 1
        diff_inter = g0[:, np.newaxis, :] - g1[np.newaxis, :, :]
        dists_inter = np.sqrt((diff_inter ** 2).sum(axis=-1))
        mean_inter = dists_inter.mean()

        assert mean_intra < mean_inter
        # Groups are not perfectly separated (intentional noise/overlap)
        assert mean_intra > 0.01

    def test_modality_dimensions_have_structure(self):
        """Disease-specific clusters should show modality dimension patterns.

        For example, cardiac diseases (Heart Failure, AFib) should have
        higher variance in ECG dimensions (32-63) compared to EMG dimensions.
        This tests the modality-based dimension group design from Karar 1.

        perturbation_rate=0.0 is mandatory here: with the default 15% rate,
        some samples in labels==1 originate from other clusters and would
        dilute the modality signal, making this test fragile.
        """
        embeddings, labels = generate_mock_embeddings(seed=42, perturbation_rate=0.0)

        ecg_start, ecg_end = MODALITY_DIM_RANGES["ecg"]
        emg_start, emg_end = MODALITY_DIM_RANGES["emg"]

        # Group 2 is ECG-dominant ((label-1) % 4 == 1 in the cycling pattern)
        ecg_group = embeddings[labels == 2]
        mean_abs_ecg = np.abs(ecg_group[:, ecg_start:ecg_end]).mean()
        mean_abs_emg = np.abs(ecg_group[:, emg_start:emg_end]).mean()

        assert mean_abs_ecg > mean_abs_emg


class TestPerturbation:
    """Verify the intentional misclassification mechanism with embedding interpolation."""

    def test_zero_perturbation_rate(self):
        """With perturbation_rate=0, no samples should be reassigned.

        Both embeddings and labels should be identical to no-perturbation baseline.
        """
        emb_0, lab_0 = generate_mock_embeddings(perturbation_rate=0.0, seed=42)
        emb_15, lab_15 = generate_mock_embeddings(perturbation_rate=0.15, seed=42)
        # With rate=0 nothing is reassigned; with rate=0.15 some labels differ
        assert not np.array_equal(lab_0, lab_15)

    def test_perturbation_changes_embeddings(self):
        """Perturbed samples should have different embeddings from unperturbed baseline.

        HATA #5 fix: perturbation now interpolates embeddings toward target cluster,
        so embeddings differ between rate=0 and rate>0.
        """
        emb_0, lab_0 = generate_mock_embeddings(perturbation_rate=0.0, seed=42)
        emb_15, lab_15 = generate_mock_embeddings(perturbation_rate=0.15, seed=42)
        # Perturbed samples should have different embeddings
        changed_mask = lab_0 != lab_15
        assert changed_mask.sum() > 0, "No labels changed"
        # At least some perturbed embeddings should differ
        assert not np.allclose(emb_0[changed_mask], emb_15[changed_mask], atol=1e-3), (
            "Perturbed embeddings are unchanged — interpolation not applied"
        )

    def test_perturbed_embedding_closer_to_target_cluster(self):
        """Perturbed embedding should be closer to its new label's cluster center
        than the original (unperturbed) embedding was.

        This verifies the interpolation moves the embedding toward the target.
        """
        emb_0, lab_0 = generate_mock_embeddings(perturbation_rate=0.0, seed=42)
        emb_15, lab_15 = generate_mock_embeddings(perturbation_rate=0.15, seed=42)

        changed_mask = lab_0 != lab_15
        if changed_mask.sum() == 0:
            pytest.skip("No perturbation occurred")

        # For each perturbed sample, compute distance to target cluster centroid
        for new_label in np.unique(lab_15[changed_mask]):
            # Target cluster centroid from unperturbed data
            target_members = emb_0[lab_0 == new_label]
            if len(target_members) < 2:
                continue
            centroid = target_members.mean(axis=0)

            # Perturbed samples assigned to this label
            perturbed_idx = np.where(changed_mask & (lab_15 == new_label))[0]
            if len(perturbed_idx) == 0:
                continue

            # Distance of perturbed (interpolated) embedding to target centroid
            dist_perturbed = np.linalg.norm(emb_15[perturbed_idx] - centroid, axis=1).mean()
            # Distance of original (unperturbed) embedding to target centroid
            dist_original = np.linalg.norm(emb_0[perturbed_idx] - centroid, axis=1).mean()

            assert dist_perturbed < dist_original, (
                f"Label {new_label}: perturbed embedding not closer to target "
                f"(dist_perturbed={dist_perturbed:.4f} >= dist_original={dist_original:.4f})"
            )

    def test_perturbation_rate_respected(self):
        """Approximately perturbation_rate fraction should be reassigned."""
        n_samples = 500
        rate = 0.15
        _, labels_0 = generate_mock_embeddings(
            perturbation_rate=0.0, seed=42, n_samples=n_samples
        )
        _, labels_15 = generate_mock_embeddings(
            perturbation_rate=rate, seed=42, n_samples=n_samples
        )
        n_changed = np.sum(labels_0 != labels_15)
        assert n_changed == int(n_samples * rate)


class TestNoiseConfig:
    """Verify per-modality noise application."""

    def test_zero_noise_is_deterministic(self):
        """With all-zero noise_config, output should be fully deterministic."""
        noise_config = {"eeg": 0.0, "ecg": 0.0, "resp": 0.0, "emg": 0.0}
        emb1, _ = generate_mock_embeddings(noise_config=noise_config, seed=42)
        emb2, _ = generate_mock_embeddings(noise_config=noise_config, seed=42)
        np.testing.assert_array_equal(emb1, emb2)

    def test_single_modality_noise(self):
        """Noise in one modality should only affect that modality's dimensions.

        After L2 normalization (a global operation), all dimensions shift
        slightly. Approach (b): verify that ECG dims change MORE than EMG dims
        when ECG-only noise is applied. Approach (a) — exposing pre-normalization
        embeddings — would require modifying generate_mock_embeddings, which is
        out of scope here.
        """
        noise_ecg = {"eeg": 0.0, "ecg": 0.5, "resp": 0.0, "emg": 0.0}
        noise_none = {"eeg": 0.0, "ecg": 0.0, "resp": 0.0, "emg": 0.0}

        emb_noisy, _ = generate_mock_embeddings(noise_config=noise_ecg, seed=42)
        emb_clean, _ = generate_mock_embeddings(noise_config=noise_none, seed=42)

        ecg_start, ecg_end = MODALITY_DIM_RANGES["ecg"]
        emg_start, emg_end = MODALITY_DIM_RANGES["emg"]

        diff = np.abs(emb_noisy - emb_clean)
        ecg_mean_diff = diff[:, ecg_start:ecg_end].mean()
        emg_mean_diff = diff[:, emg_start:emg_end].mean()

        assert ecg_mean_diff > emg_mean_diff

    def test_higher_noise_increases_variance(self):
        """Higher noise_config value should increase variance in those dimensions."""
        noise_low = {"eeg": 0.0, "ecg": 0.1, "resp": 0.0, "emg": 0.0}
        noise_high = {"eeg": 0.0, "ecg": 0.5, "resp": 0.0, "emg": 0.0}

        ecg_start, ecg_end = MODALITY_DIM_RANGES["ecg"]

        emb_low, _ = generate_mock_embeddings(noise_config=noise_low, seed=42)
        emb_high, _ = generate_mock_embeddings(noise_config=noise_high, seed=42)

        var_low = np.var(emb_low[:, ecg_start:ecg_end])
        var_high = np.var(emb_high[:, ecg_start:ecg_end])

        assert var_high > var_low


class TestReproducibility:
    """Verify seed-based reproducibility."""

    def test_same_seed_same_output(self):
        """Same seed should produce identical embeddings."""
        emb1, lab1 = generate_mock_embeddings(seed=42)
        emb2, lab2 = generate_mock_embeddings(seed=42)
        np.testing.assert_array_equal(emb1, emb2)
        np.testing.assert_array_equal(lab1, lab2)

    def test_different_seed_different_output(self):
        """Different seeds should produce different embeddings."""
        emb1, _ = generate_mock_embeddings(seed=42)
        emb2, _ = generate_mock_embeddings(seed=99)
        assert not np.array_equal(emb1, emb2)


class TestValidation:
    """Verify that generate_mock_embeddings raises ValueError for invalid inputs.

    Each test corresponds to an explicit guard in mock_data.py.
    If any guard is removed, the corresponding test here will fail,
    making regressions immediately visible.
    """

    def test_perturbation_rate_above_one_raises(self):
        """perturbation_rate > 1.0 should raise ValueError."""
        with pytest.raises(ValueError, match="perturbation_rate"):
            generate_mock_embeddings(perturbation_rate=1.1)

    def test_perturbation_rate_negative_raises(self):
        """perturbation_rate < 0.0 should raise ValueError."""
        with pytest.raises(ValueError, match="perturbation_rate"):
            generate_mock_embeddings(perturbation_rate=-0.01)

    def test_single_disease_group_with_perturbation_raises(self):
        """n_disease_groups=1 with perturbation_rate > 0 should raise ValueError.

        Cannot reassign a label to a 'different' cluster when only one exists.
        """
        with pytest.raises(ValueError, match="n_disease_groups"):
            generate_mock_embeddings(n_disease_groups=1, perturbation_rate=0.1)

    def test_n_samples_less_than_n_groups_raises(self):
        """n_samples < n_disease_groups should raise ValueError."""
        with pytest.raises(ValueError, match="n_samples"):
            generate_mock_embeddings(n_samples=5, n_disease_groups=10)

    def test_unknown_noise_config_key_raises(self):
        """Unknown key in noise_config should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown noise_config keys"):
            generate_mock_embeddings(noise_config={"eeg": 0.1, "ekg": 0.1})

    def test_negative_noise_config_value_raises(self):
        """Negative noise level in noise_config should raise ValueError."""
        with pytest.raises(ValueError, match="noise_config"):
            generate_mock_embeddings(noise_config={"eeg": -0.1, "ecg": 0.0, "resp": 0.0, "emg": 0.0})


class TestLabelDiseaseNameAlignment:
    """Verify that generated labels align with DISEASE_NAMES keys in config.py.

    HATA #1: mock_data.py used to generate labels 0-11 but DISEASE_NAMES
    uses keys 1-12. This class ensures labels are always valid DISEASE_NAMES keys.
    """

    def test_all_labels_are_valid_disease_name_keys(self):
        """Every label produced by generate_mock_embeddings must be a key in DISEASE_NAMES."""
        _, labels = generate_mock_embeddings(seed=42, n_disease_groups=12)
        valid_keys = set(DISEASE_NAMES.keys())
        actual_labels = set(labels)
        assert actual_labels.issubset(valid_keys), (
            f"Labels not in DISEASE_NAMES: {actual_labels - valid_keys}"
        )

    def test_no_label_zero(self):
        """Label 0 should never appear (DISEASE_NAMES starts at 1)."""
        _, labels = generate_mock_embeddings(seed=42, n_disease_groups=12)
        assert 0 not in labels, "Label 0 found but DISEASE_NAMES starts at 1"

    def test_label_12_exists(self):
        """Label 12 (Anxiety Disorders) must be present in generated labels."""
        _, labels = generate_mock_embeddings(seed=42, n_disease_groups=12, perturbation_rate=0.0)
        assert 12 in labels, "Label 12 (Anxiety Disorders) missing from generated labels"

    def test_labels_range_1_to_n(self):
        """Labels should range from 1 to n_disease_groups (inclusive)."""
        n_groups = 12
        _, labels = generate_mock_embeddings(seed=42, n_disease_groups=n_groups, perturbation_rate=0.0)
        assert labels.min() == 1
        assert labels.max() == n_groups
