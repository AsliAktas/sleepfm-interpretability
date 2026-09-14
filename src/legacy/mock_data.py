"""
Mock data generation for SleepFM Interpretability Project.

Generates realistic but synthetic embeddings, CoxPH scores,
and modality-specific embeddings for pipeline development and testing.

Design decisions (see KARAR_NOKTALARI.md):
- Karar 1: Hybrid clustering (cluster + random perturbation) with
  modality-based dimension groups (4x32 equal split)
- Karar 2: Linear projection for CoxPH scores with structured weight vectors
- Karar 3: Correlated modality embeddings with shared_ratio parameter
- Noise follows per-modality config, default all zeros (deterministic baseline)

All mock data includes intentional noise and variance —
perfect separation or correlation would be scientifically meaningless.
"""

import numpy as np
from typing import Dict, Tuple, Optional

from config import DEFAULT_NOISE_CONFIG, EMBEDDING_DIM, MODALITY_DIM_RANGES, DISEASE_NAMES
from utils import normalize_l2

_DOMINANT = 2.0
_SECONDARY = 1.0
_BACKGROUND = 0.3


def generate_mock_embeddings(
    n_samples: int = 500,
    n_disease_groups: int = 12,
    noise_config: Optional[Dict[str, float]] = None,
    perturbation_rate: float = 0.15,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate mock patient embeddings with disease-group structure.

    Uses hybrid strategy (Karar 1 — Secenek C):
    1. Create cluster centers with modality-based dimension weighting
    2. Generate samples around centers with per-modality noise
    3. Randomly reassign perturbation_rate fraction of samples
       to different clusters (intentional misclassification)

    Modality-based dimension groups:
    - Dimensions 0-31:   EEG-dominant features
    - Dimensions 32-63:  ECG-dominant features
    - Dimensions 64-95:  Respiratory-dominant features
    - Dimensions 96-127: EMG-dominant features

    Each disease cluster has different activation patterns across
    modality dimension groups. For example, a cardiac disease cluster
    has higher magnitude in ECG dimensions (32-63).

    Args:
        n_samples: Total number of patient embeddings to generate.
        n_disease_groups: Number of disease groups to simulate.
        noise_config: Per-modality noise levels. Keys: "eeg", "ecg", "resp", "emg".
            Values: noise standard deviation for that modality's dimensions.
            Default: all zeros (deterministic baseline).
        perturbation_rate: Fraction of samples to randomly reassign to
            different clusters (0.0 to 1.0). Simulates misclassification.
        seed: Random seed for reproducibility.

    Returns:
        embeddings: np.ndarray of shape (n_samples, embedding_dim), L2-normalized.
        labels: np.ndarray of shape (n_samples,), integer disease group labels.
            Labels range from 1 to n_disease_groups (1-based, matching DISEASE_NAMES keys).
            After perturbation, some labels may not match their cluster.

    Notes:
        - Embeddings are L2-normalized (unit vectors on hypersphere)
        - Groups are NOT perfectly separable — this is intentional (rule #5)
        - perturbation_rate=0.15 means ~15% of samples are in "wrong" clusters
        - noise_config allows ablation-style testing: set one modality's noise
          high while keeping others at zero
    """
    if not 0.0 <= perturbation_rate <= 1.0:
        raise ValueError(f"perturbation_rate must be in [0, 1], got {perturbation_rate}")
    if perturbation_rate > 0.0 and n_disease_groups < 2:
        raise ValueError(f"n_disease_groups must be >= 2 when perturbation_rate > 0, got {n_disease_groups}")
    if n_samples < n_disease_groups:
        raise ValueError(
            f"n_samples ({n_samples}) must be >= n_disease_groups ({n_disease_groups})"
        )

    if noise_config is None:
        noise_config = DEFAULT_NOISE_CONFIG.copy()
    unknown_keys = set(noise_config) - set(MODALITY_DIM_RANGES)
    if unknown_keys:
        raise ValueError(f"Unknown noise_config keys: {unknown_keys}. Valid keys: {set(MODALITY_DIM_RANGES)}")
    for mod, val in noise_config.items():
        if val < 0.0:
            raise ValueError(f"noise_config values must be >= 0, got noise_config['{mod}'] = {val}")

    rng = np.random.default_rng(seed)
    modality_list = list(MODALITY_DIM_RANGES.keys())
    n_modalities = len(modality_list)

    # --- Step 1: Cluster centers with modality-based dimension weighting ---
    # Uses _DISEASE_MODALITY_PROFILE (built from _DISEASE_PROFILES) for clinically
    # motivated dominant/secondary modality assignments when available.
    # Falls back to round-robin for unknown disease_no values.
    # Labels are 1-based to match DISEASE_NAMES keys in config.py.
    # Dominant dims: weight 2.0 | Secondary (next modality): 1.0 | Background: 0.3
    centers = np.zeros((n_disease_groups, EMBEDDING_DIM))
    for g in range(1, n_disease_groups + 1):
        if g in _DISEASE_MODALITY_PROFILE:
            dominant_mod, secondary_mod = _DISEASE_MODALITY_PROFILE[g]
        else:
            dominant_mod = modality_list[(g - 1) % n_modalities]
            secondary_mod = modality_list[g % n_modalities]
        for mod, (start, end) in MODALITY_DIM_RANGES.items():
            if mod == dominant_mod:
                w = _DOMINANT
            elif mod == secondary_mod:
                w = _SECONDARY
            else:
                w = _BACKGROUND
            centers[g - 1, start:end] = w * rng.standard_normal(end - start)

    # --- Step 2: Assign balanced labels; generate samples around cluster centers ---
    # Labels are 1-based: [1, 2, ..., n_disease_groups]
    base_labels = np.tile(np.arange(1, n_disease_groups + 1), n_samples // n_disease_groups + 1)[:n_samples]
    base_labels = rng.permutation(base_labels)

    # Baseline per-dim cluster spread (ensures groups are not perfectly separable)
    _CLUSTER_SPREAD = 0.5

    embeddings = np.zeros((n_samples, EMBEDDING_DIM))
    for g in range(1, n_disease_groups + 1):
        mask = base_labels == g
        n_in_group = int(mask.sum())
        if n_in_group == 0:
            continue
        group_emb = np.broadcast_to(centers[g - 1], (n_in_group, EMBEDDING_DIM)).copy()
        for mod, (start, end) in MODALITY_DIM_RANGES.items():
            total_std = _CLUSTER_SPREAD + noise_config.get(mod, 0.0)
            group_emb[:, start:end] += total_std * rng.standard_normal((n_in_group, end - start))
        embeddings[mask] = group_emb

    # --- Step 3: Perturbation — reassign labels AND interpolate embeddings ---
    # Simulates comorbidity: perturbed patient's embedding is shifted toward
    # the target cluster center, so both label and embedding reflect the new disease.
    # Interpolation: emb = alpha * original_emb + (1-alpha) * target_center
    # alpha=0.5 gives equal weight to original and target (realistic comorbidity).
    final_labels = base_labels.copy()
    n_perturb = int(n_samples * perturbation_rate)
    _PERTURB_ALPHA = 0.5
    if n_perturb > 0:
        perturb_idxs = rng.choice(n_samples, size=n_perturb, replace=False)
        # Vectorized: draw offset from [1, n_disease_groups-1], shift cyclically.
        # ((label-1 + offset) % n) + 1 guarantees a different label in [1, n], uniformly.
        offsets = rng.integers(1, n_disease_groups, size=n_perturb)
        new_labels = (final_labels[perturb_idxs] - 1 + offsets) % n_disease_groups + 1
        final_labels[perturb_idxs] = new_labels

        # Interpolate embeddings toward target cluster center
        for i, idx in enumerate(perturb_idxs):
            target_label = new_labels[i]
            target_center = centers[target_label - 1]  # centers is 0-indexed
            embeddings[idx] = _PERTURB_ALPHA * embeddings[idx] + (1.0 - _PERTURB_ALPHA) * target_center

    # --- L2-normalize to unit vectors on hypersphere ---
    embeddings = normalize_l2(embeddings)

    return embeddings, final_labels


def generate_mock_coxph_scores(
    embeddings: np.ndarray,
    n_diseases: int = 12,
    weight_vectors: Optional[np.ndarray] = None,
    noise_config: Optional[Dict[str, float]] = None,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate mock CoxPH hazard scores via linear projection (Karar 2 — Secenek B).

    Each disease has a weight vector structured by modality dimension groups.
    Risk score = dot(embedding, weight_vector) + noise.

    Weight vectors are NOT random — they are structured so that:
    - Cardiac diseases have high weights in ECG dimensions (32-63)
    - Neurological diseases have high weights in EEG dimensions (0-31)
    - Respiratory diseases have high weights in respiratory dims (64-95)
    - etc.

    This makes the mock data interpretable: you can read which modality
    drives each disease's risk score directly from the weight vector.

    Args:
        embeddings: Patient embeddings, shape (n_samples, embedding_dim).
        n_diseases: Number of diseases to generate scores for.
        weight_vectors: Optional pre-defined weight matrix of shape
            (n_diseases, embedding_dim). If None, generates structured
            weight vectors based on expected modality-disease relationships.
        noise_config: Per-modality noise to add to scores. Same format as
            generate_mock_embeddings. Applied after projection.
            Default: all zeros (deterministic baseline).
        seed: Random seed for reproducibility.

    Returns:
        scores: np.ndarray of shape (n_samples, n_diseases).
            Values represent log-hazard ratios (can be negative).
        weight_vectors: np.ndarray of shape (n_diseases, embedding_dim).
            The weight vectors used (returned for inspection/testing).

    Notes:
        - At zero noise, scores are deterministic given embeddings + weights
        - Weight vectors encode modality importance per disease
        - noise_config allows ablation: "add noise to ECG dims only,
          observe cardiac disease score degradation"
        - When n_diseases > n_modalities (4), dominant-secondary modality pairs
          repeat cyclically (e.g., disease 0 and disease 4 share the same
          dominant modality). Scores for those diseases will be correlated.
        - n_diseases need not equal n_disease_groups from generate_mock_embeddings.
          If they differ, score column i does not correspond to label i. Use
          profile_dict_to_weight_matrix with explicit weight_vectors to avoid
          silent index mismatches.
    """
    n_samples, embedding_dim = embeddings.shape
    if embedding_dim != EMBEDDING_DIM:
        raise ValueError(f"embedding_dim must be {EMBEDDING_DIM}, got {embedding_dim}")

    if noise_config is None:
        noise_config = DEFAULT_NOISE_CONFIG.copy()
    unknown_keys = set(noise_config) - set(MODALITY_DIM_RANGES)
    if unknown_keys:
        raise ValueError(f"Unknown noise_config keys: {unknown_keys}. Valid keys: {set(MODALITY_DIM_RANGES)}")
    for mod, val in noise_config.items():
        if val < 0.0:
            raise ValueError(f"noise_config values must be >= 0, got noise_config['{mod}'] = {val}")

    if weight_vectors is not None:
        if weight_vectors.shape != (n_diseases, embedding_dim):
            raise ValueError(
                f"weight_vectors must have shape ({n_diseases}, {embedding_dim}), "
                f"got {weight_vectors.shape}"
            )

    rng = np.random.default_rng(seed)
    modality_list = list(MODALITY_DIM_RANGES.keys())
    n_modalities = len(modality_list)

    # --- Generate structured weight vectors if not provided ---
    # Uses _DISEASE_MODALITY_PROFILE for clinically motivated assignments.
    # CoxPH weight index d maps to disease_no (d+1) in DISEASE_NAMES.
    # Falls back to round-robin for unknown disease_no values.
    if weight_vectors is None:
        weight_vectors = np.zeros((n_diseases, embedding_dim))
        for d in range(n_diseases):
            disease_no = d + 1
            if disease_no in _DISEASE_MODALITY_PROFILE:
                dominant_mod, secondary_mod = _DISEASE_MODALITY_PROFILE[disease_no]
            else:
                dominant_mod = modality_list[d % n_modalities]
                secondary_mod = modality_list[(d + 1) % n_modalities]
            for mod, (start, end) in MODALITY_DIM_RANGES.items():
                if mod == dominant_mod:
                    w = _DOMINANT
                elif mod == secondary_mod:
                    w = _SECONDARY
                else:
                    w = _BACKGROUND
                weight_vectors[d, start:end] = w * rng.standard_normal(end - start)

    # --- Compute scores per modality partial contribution, add noise after projection ---
    # Splitting into modality partials allows per-modality noise (ablation axis):
    #   noise_config["ecg"] > 0  →  ECG contribution to all diseases gets noise
    #   cardiac diseases (high ECG weights) degrade more in relative terms
    scores = np.zeros((n_samples, n_diseases))
    for mod, (start, end) in MODALITY_DIM_RANGES.items():
        partial = embeddings[:, start:end] @ weight_vectors[:, start:end].T  # (n_samples, n_diseases)
        noise_std = noise_config.get(mod, 0.0)
        if noise_std > 0.0:
            partial = partial + noise_std * rng.standard_normal((n_samples, n_diseases))
        scores += partial

    return scores, weight_vectors.copy()


def generate_mock_modality_embeddings(
    n_samples: int = 500,
    embedding_dim: int = 128,
    modalities: Tuple[str, ...] = ("eeg", "ecg", "resp", "emg"),
    shared_ratio: float = 0.3,
    noise_config: Optional[Dict[str, float]] = None,
    seed: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """Generate correlated modality embeddings (Karar 3 — Secenek B).

    Strategy:
    1. Generate a "combined" base embedding for each patient
    2. Project to each modality using orthogonal projection matrices (QR)
    3. Each modality embedding = shared_component + unique_component
       where shared_ratio controls the balance

    shared_ratio=1.0: modalities carry identical information
    shared_ratio=0.0: modalities are completely independent
    Realistic range: 0.2-0.5

    Args:
        n_samples: Number of patients.
        embedding_dim: Dimensionality per modality embedding. Must be EMBEDDING_DIM (128).
        modalities: Names of modalities to generate. Must be non-empty.
        shared_ratio: Fraction of variance shared across modalities (0.0 to 1.0).
            Start with a fixed value, then vary as an ablation axis.
        noise_config: Per-modality noise levels. Same format as other functions.
            Default: all zeros (deterministic baseline).
        seed: Random seed for reproducibility.

    Returns:
        Dictionary with the following keys:
        - "combined": np.ndarray of shape (n_samples, embedding_dim), L2-normalized base.
        - Each modality name: np.ndarray of shape (n_samples, embedding_dim), L2-normalized.
        - "proj_{mod}" for each modality: np.ndarray of shape (embedding_dim, embedding_dim),
          orthogonal projection matrix used. Included for test inspection.

    Notes:
        - Combined embedding is the base from which modalities are derived
        - All modality embeddings are L2-normalized
        - Projection matrices are orthogonal (QR), so combined_norm @ Q preserves
          norm distribution — shared_component and unique_component have comparable
          scale before mixing, making shared_ratio a genuine variance-split control
        - Both components are L2-normalized before mixing:
            shared_ratio=1.0 → modality ≈ shared_component (fully correlated)
            shared_ratio=0.0 → modality ≈ unique_component (fully independent)
        - shared_ratio is itself an interpretable parameter: varying it
          answers "how much does cross-modality information sharing affect
          downstream predictions?"
        - At noise_config all zeros, output is deterministic
    """
    if not 0.0 <= shared_ratio <= 1.0:
        raise ValueError(f"shared_ratio must be in [0, 1], got {shared_ratio}")
    if len(modalities) == 0:
        raise ValueError("modalities must not be empty")
    if embedding_dim != EMBEDDING_DIM:
        raise ValueError(f"embedding_dim must be {EMBEDDING_DIM}, got {embedding_dim}")

    if noise_config is None:
        noise_config = {m: 0.0 for m in modalities}
    unknown_keys = set(noise_config) - set(modalities)
    if unknown_keys:
        raise ValueError(f"Unknown noise_config keys: {unknown_keys}. Valid keys: {set(modalities)}")
    for mod, val in noise_config.items():
        if val < 0.0:
            raise ValueError(f"noise_config values must be >= 0, got noise_config['{mod}'] = {val}")

    rng = np.random.default_rng(seed)

    # --- Step 1: Generate combined base embedding (L2-normalized) ---
    combined_raw = rng.standard_normal((n_samples, embedding_dim))
    combined_norm = normalize_l2(combined_raw)

    result: Dict[str, np.ndarray] = {"combined": combined_norm}

    # --- Steps 2 & 3: Per-modality orthogonal projection + variance-controlled mixing ---
    # Q is orthogonal (QR decomposition) so combined_norm @ Q preserves norm distribution.
    # Both shared_component and unique_component are L2-normalized before mixing,
    # ensuring shared_ratio genuinely controls the variance split.
    for mod in modalities:
        # Orthogonal projection matrix via QR decomposition
        Q, _ = np.linalg.qr(rng.standard_normal((embedding_dim, embedding_dim)))
        result[f"proj_{mod}"] = Q  # returned for test inspection

        shared_component = combined_norm @ Q  # (n_samples, embedding_dim), norm preserved

        unique_raw = rng.standard_normal((n_samples, embedding_dim))
        unique_component = normalize_l2(unique_raw)

        modality_emb = shared_ratio * shared_component + (1.0 - shared_ratio) * unique_component

        noise_std = noise_config.get(mod, 0.0)
        if noise_std > 0.0:
            modality_emb = modality_emb + noise_std * rng.standard_normal((n_samples, embedding_dim))

        # L2-normalize final embedding
        result[mod] = normalize_l2(modality_emb)

    return result


def generate_mock_survival_data(
    n_samples: int = 500,
    n_diseases: int = 12,
    max_followup_years: float = 6.0,
    event_rate: float = 0.15,
    scores: Optional[np.ndarray] = None,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate mock survival time and event indicator data.

    Creates time-to-event data compatible with CoxPH analysis.
    Some patients experience the event, others are censored.

    Args:
        n_samples: Number of patients.
        n_diseases: Number of diseases.
        max_followup_years: Maximum follow-up time.
        event_rate: Approximate proportion of patients who experience each event.
        scores: Optional CoxPH log-hazard scores from generate_mock_coxph_scores,
            shape (n_samples, n_diseases). When provided, each patient's hazard
            rate is modulated as lam_i = lam_base * exp(score_i - mean(scores_col)),
            so higher-risk patients have shorter expected survival times.
            Scores are mean-centered per disease before exponentiation to preserve
            the event_rate calibration. Default: None (uniform hazard).
        seed: Random seed for reproducibility.

    Returns:
        event_times: np.ndarray of shape (n_samples, n_diseases).
            Time in years until event or censoring.
        event_indicators: np.ndarray of shape (n_samples, n_diseases).
            1 = event occurred, 0 = censored.

    Notes:
        - event_rate is approximate — actual observed rates will be lower than event_rate
          because uniform censoring competes with the event process. This is expected
          behaviour, not a bug. With the default event_rate=0.15, the observed rate
          is approximately 7-8% (roughly half) due to competing uniform censoring.
        - When scores is provided, patient-level hazard varies but the population-average
          hazard remains lam_base (scores are mean-centered per disease), so the
          approximate event_rate interpretation is preserved at the population level.
        - Diseases are independent across patients: comorbidity structure is not modelled.
        - Times are positive and <= max_followup_years
        - Follows exponential distribution (standard in survival analysis)
    """
    if n_samples <= 0:
        raise ValueError(f"n_samples must be > 0, got {n_samples}")
    if n_diseases <= 0:
        raise ValueError(f"n_diseases must be > 0, got {n_diseases}")
    if max_followup_years <= 0.0:
        raise ValueError(f"max_followup_years must be > 0, got {max_followup_years}")
    if not 0.0 < event_rate < 1.0:
        raise ValueError(f"event_rate must be in (0, 1), got {event_rate}")
    if scores is not None and scores.shape != (n_samples, n_diseases):
        raise ValueError(
            f"scores must have shape ({n_samples}, {n_diseases}), "
            f"got {scores.shape}"
        )

    rng = np.random.default_rng(seed)

    # Calibrate exponential hazard rate so that P(T <= max_followup_years) ≈ event_rate
    # in a no-censoring scenario. With competing uniform censoring, the observed rate
    # will be lower — this is documented in Notes above.
    # From 1 - exp(-λ * L) = event_rate  →  λ = -log(1 - event_rate) / L
    lam = -np.log(1.0 - event_rate) / max_followup_years

    # Event times: T_event ~ Exp(λ_i), shape (n_samples, n_diseases).
    # When scores provided: lam_i = lam * exp(score_i - mean(scores_col)).
    # Centering per disease preserves population-mean hazard = lam.
    if scores is not None:
        scores_centered = scores - scores.mean(axis=0)   # (n_samples, n_diseases)
        lam_matrix = lam * np.exp(scores_centered)       # (n_samples, n_diseases)
        event_time_raw = rng.exponential(scale=1.0 / lam_matrix)
    else:
        event_time_raw = rng.exponential(scale=1.0 / lam, size=(n_samples, n_diseases))

    # Censoring times: T_censor ~ Uniform(0, max_followup_years), independent per patient-disease.
    censor_time = rng.uniform(0.0, max_followup_years, size=(n_samples, n_diseases))

    # Observed time = min(T_event, T_censor).
    # No clip needed: both inputs are already in [0, max_followup_years].
    event_times = np.minimum(event_time_raw, censor_time)

    # Event occurred iff T_event <= T_censor (event happened before censoring).
    # int64 for downstream compatibility with lifelines / scikit-survival.
    event_indicators = (event_time_raw <= censor_time).astype(np.int64)

    return event_times, event_indicators


# Clinically motivated (dominant, secondary) modality pairs per disease.
# Defined at module level to avoid reconstruction on every call.
# NOT ground truth — test scenario only; will be validated against literature in Day 11-12.
# All 12 unique (dominant, secondary) pairs across 4 modalities are used (4×3 = 12),
# so every disease has a distinct profile — no duplicates by construction.
_DISEASE_PROFILES: Tuple[Tuple[str, str, str], ...] = (
    ("Depression",                        "eeg",  "ecg"),   # sleep EEG abnormalities + HRV changes
    ("Dementia",                          "eeg",  "resp"),  # sleep EEG changes + disordered breathing
    ("Stroke",                            "eeg",  "emg"),   # sleep EEG changes + motor/muscle involvement
    ("Hypertension",                      "ecg",  "eeg"),   # cardiovascular + autonomic nervous system
    ("Heart Failure",                     "ecg",  "resp"),  # cardiac dysfunction + Cheyne-Stokes breathing
    ("Atrial Fibrillation and Flutter",   "ecg",  "emg"),   # arrhythmia + muscle artifacts
    ("Obesity",                           "resp", "eeg"),   # hypoventilation + cortical arousal
    ("Obstructive Sleep Apnea",           "resp", "ecg"),   # upper airway obstruction + autonomic responses
    ("Chronic Airway Obstruction (COPD)", "resp", "emg"),   # respiratory dysfunction + respiratory muscle
    ("Chronic Kidney Disease",            "emg",  "eeg"),   # restless legs (common in CKD) + cortical arousal
    ("Type 2 Diabetes",                   "emg",  "ecg"),   # peripheral neuropathy + cardiovascular autonomic
    ("Anxiety Disorders",                 "emg",  "resp"),  # psychomotor tension + respiratory dysregulation
)

# Validate modality keys and name uniqueness at import time, not per call.
_profile_names = [d for d, _, _ in _DISEASE_PROFILES]
if len(_profile_names) != len(set(_profile_names)):
    raise ValueError("Duplicate disease names in _DISEASE_PROFILES")
_valid_modalities = set(MODALITY_DIM_RANGES)
for _d, _dom, _sec in _DISEASE_PROFILES:
    if _dom not in _valid_modalities:
        raise ValueError(
            f"Unknown dominant modality '{_dom}' for disease '{_d}'. "
            f"Valid: {_valid_modalities}"
        )
    if _sec not in _valid_modalities:
        raise ValueError(
            f"Unknown secondary modality '{_sec}' for disease '{_d}'. "
            f"Valid: {_valid_modalities}"
        )
del _profile_names, _valid_modalities, _d, _dom, _sec

# Build lookup: disease_no -> (dominant_modality, secondary_modality)
# Maps DISEASE_NAMES (config.py) to _DISEASE_PROFILES via disease name matching.
_PROFILE_BY_NAME: Dict[str, Tuple[str, str]] = {
    name: (dom, sec) for name, dom, sec in _DISEASE_PROFILES
}
_DISEASE_MODALITY_PROFILE: Dict[int, Tuple[str, str]] = {
    dno: _PROFILE_BY_NAME[dname]
    for dno, dname in DISEASE_NAMES.items()
    if dname in _PROFILE_BY_NAME
}


def build_disease_weight_profiles(
    embedding_dim: int = 128,
) -> Dict[str, np.ndarray]:
    """Build structured weight profiles for each disease.

    Maps each of the 12 diseases to a weight vector where modality
    dimension groups have clinically motivated relative magnitudes.

    Example:
        Heart Failure -> high ECG weights, moderate respiratory weights
        Dementia -> high EEG weights, low other weights
        OSA -> high respiratory weights, moderate EEG weights

    This function encodes the "expected dominant modality" as a
    test scenario, NOT as a ground truth claim. Different profiles
    can be tested to see how the pipeline behaves.

    Args:
        embedding_dim: Must be 128 (4 modalities x 32 dims each).

    Returns:
        Dictionary mapping disease name (English) to weight vector
        of shape (embedding_dim,). Vectors are read-only (not writeable).

    Notes:
        - Weight magnitudes represent relative importance, not absolute
        - Weight values are uniform per modality group (all dims in a group share
          the same scalar), unlike generate_mock_coxph_scores which uses
          Gaussian-scaled vectors — the two are not directly comparable.
        - These profiles will be validated against literature in Day 11-12
        - Profiles are deterministic (no random component)
        - Random component comes from noise_config in score generation
    """
    if embedding_dim != EMBEDDING_DIM:
        raise ValueError(f"embedding_dim must be {EMBEDDING_DIM}, got {embedding_dim}")

    profiles: Dict[str, np.ndarray] = {}
    for disease, dominant_mod, secondary_mod in _DISEASE_PROFILES:
        vec = np.zeros(embedding_dim)
        for mod, (start, end) in MODALITY_DIM_RANGES.items():
            if mod == dominant_mod:
                w = _DOMINANT
            elif mod == secondary_mod:
                w = _SECONDARY
            else:
                w = _BACKGROUND
            vec[start:end] = w
        vec.flags.writeable = False  # prevent accidental mutation by caller
        profiles[disease] = vec

    return profiles


def profile_dict_to_weight_matrix(
    profiles: Dict[str, np.ndarray],
    disease_order: Tuple[str, ...],
) -> np.ndarray:
    """Convert a disease-name-keyed weight dict to an ordered weight matrix.

    Bridge between build_disease_weight_profiles (returns a dict keyed by
    disease name) and generate_mock_coxph_scores (expects a matrix indexed
    by integer position). Passing weight_vectors explicitly with this function
    makes the row order unambiguous and avoids silent semantic mismatches with
    the default cyclic assignment (index % n_modalities).

    Args:
        profiles: Dict mapping disease name to weight vector, as returned
            by build_disease_weight_profiles(). Vectors may be read-only.
        disease_order: Sequence of disease names defining row order.
            Row i of the output corresponds to disease_order[i].

    Returns:
        weight_matrix: np.ndarray of shape (len(disease_order), EMBEDDING_DIM).
            Row i = profiles[disease_order[i]]. Writeable copy.

    Raises:
        KeyError: If any name in disease_order is not in profiles.
        ValueError: If profile vectors have inconsistent shapes.

    Example:
        profiles = build_disease_weight_profiles()
        order = tuple(name for name, _, _ in _DISEASE_PROFILES)
        W = profile_dict_to_weight_matrix(profiles, order)
        scores, _ = generate_mock_coxph_scores(
            embeddings, n_diseases=len(order), weight_vectors=W
        )
    """
    vectors = []
    expected_shape: Optional[Tuple[int, ...]] = None
    for name in disease_order:
        if name not in profiles:
            raise KeyError(
                f"Disease '{name}' not found in profiles. "
                f"Available: {sorted(profiles)}"
            )
        vec = profiles[name]
        if expected_shape is None:
            expected_shape = vec.shape
        elif vec.shape != expected_shape:
            raise ValueError(
                f"Inconsistent vector shapes in profiles: "
                f"expected {expected_shape}, got {vec.shape} for '{name}'"
            )
        vectors.append(vec.copy())  # copy because source vectors may be read-only
    return np.stack(vectors, axis=0)
