"""
Central configuration for SleepFM Interpretability Project.

Contains disease definitions, phecode mappings, modality info,
expected dominant signals, and project-wide constants.

This is the single source of truth for all disease/modality metadata.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Literal, Tuple


# =============================================================================
# Modality Definitions
# =============================================================================

MODALITIES: Tuple[str, ...] = ("eeg", "ecg", "resp", "emg")

MODALITY_FULL_NAMES: Dict[str, str] = {
    "eeg": "Electroencephalography",
    "ecg": "Electrocardiography",
    "resp": "Respiratory",
    "emg": "Electromyography",
}

# Equal dimension split: 4 modalities x 32 dimensions = 128 total
# (Karar 1: equal split to avoid implicit weighting bias)
MODALITY_DIM_RANGES: Dict[str, Tuple[int, int]] = {
    "eeg": (0, 32),
    "ecg": (32, 64),
    "resp": (64, 96),
    "emg": (96, 128),
}

# =============================================================================
# Embedding Configuration
# =============================================================================

EMBEDDING_DIM: int = 128
DIMS_PER_MODALITY: int = 32

# Validate scalar constants are consistent with MODALITY_DIM_RANGES
_expected_dim_ranges = {
    m: (i * DIMS_PER_MODALITY, (i + 1) * DIMS_PER_MODALITY)
    for i, m in enumerate(MODALITIES)
}
assert EMBEDDING_DIM == len(MODALITIES) * DIMS_PER_MODALITY, (
    f"EMBEDDING_DIM ({EMBEDDING_DIM}) != len(MODALITIES) * DIMS_PER_MODALITY "
    f"({len(MODALITIES)} * {DIMS_PER_MODALITY})"
)
assert MODALITY_DIM_RANGES == _expected_dim_ranges, (
    f"MODALITY_DIM_RANGES inconsistent with MODALITIES/DIMS_PER_MODALITY. "
    f"Expected: {_expected_dim_ranges}"
)
del _expected_dim_ranges

N_SLEEP_STAGES: int = 4  # N1, N2, N3, REM (Wake excluded from analysis)

SLEEP_STAGES: Dict[int, str] = {
    1: "N1",
    2: "N2",
    3: "N3",
    4: "REM",
}

# =============================================================================
# Disease Configuration
# =============================================================================

@dataclass
class DiseaseConfig:
    """Configuration for a single disease in the analysis.

    Attributes:
        disease_no: Sequential number (1-12).
        name_en: English disease name.
        name_tr: Turkish disease name.
        primary_label_idx: Primary index in SleepFM's 1065-condition output.
        primary_phecode: Phecode string or aggregate label.
        primary_phenotype: Human-readable phenotype description.
        alternative_label_idxs: Other relevant label indices.
        all_label_idxs: All label indices (primary + alternatives).
        label_type: "aggregate" or "specific_phecode".
        expected_dominant_modality: Expected primary signal from literature.
            To be filled on Day 11-12. Empty string until then.
        expected_dominant_stage: Expected sleep stage from literature.
            To be filled on Day 11-12. Empty string until then.
        literature_reference: PubMed reference supporting expectations.
            To be filled on Day 11-12. Empty string until then.
    """
    disease_no: int
    name_en: str
    name_tr: str
    primary_label_idx: int
    primary_phecode: str
    primary_phenotype: str
    alternative_label_idxs: List[int]
    all_label_idxs: List[int]
    label_type: Literal["aggregate", "specific_phecode"]
    expected_dominant_modality: str = ""
    expected_dominant_stage: str = ""
    literature_reference: str = ""

    def __post_init__(self) -> None:
        valid_label_types = ("aggregate", "specific_phecode")
        if self.label_type not in valid_label_types:
            raise ValueError(
                f"label_type must be one of {valid_label_types}, got {self.label_type!r}"
            )
        expected = [self.primary_label_idx] + self.alternative_label_idxs
        if self.all_label_idxs != expected:
            raise ValueError(
                f"all_label_idxs {self.all_label_idxs} does not match "
                f"[primary_label_idx] + alternative_label_idxs = {expected}"
            )


# 12 diseases selected for the interpretability analysis
# Source: label_mapping.csv from SleepFM preprocessing
DISEASES: Dict[int, DiseaseConfig] = {
    1: DiseaseConfig(
        disease_no=1,
        name_en="Heart Failure",
        name_tr="Kalp Yetmezliği",
        primary_label_idx=1043,
        primary_phecode="Heart Failure",
        primary_phenotype="Heart Failure (aggregate)",
        alternative_label_idxs=[458, 459, 460, 461, 462],
        all_label_idxs=[1043, 458, 459, 460, 461, 462],
        label_type="aggregate",
    ),
    2: DiseaseConfig(
        disease_no=2,
        name_en="Atrial Fibrillation and Flutter",
        name_tr="Atriyal Fibrilasyon ve Flutter",
        primary_label_idx=1044,
        primary_phecode="Atrial Fibrillation and Flutter",
        primary_phenotype="Atrial Fibrillation and Flutter (aggregate)",
        alternative_label_idxs=[447, 448],
        all_label_idxs=[1044, 447, 448],
        label_type="aggregate",
    ),
    3: DiseaseConfig(
        disease_no=3,
        name_en="Hypertension",
        name_tr="Hipertansiyon",
        primary_label_idx=1049,
        primary_phecode="Hypertension",
        primary_phenotype="Hypertension (aggregate)",
        alternative_label_idxs=[401, 402],
        all_label_idxs=[1049, 401, 402],
        label_type="aggregate",
    ),
    4: DiseaseConfig(
        disease_no=4,
        name_en="Dementia",
        name_tr="Demans",
        primary_label_idx=1057,
        primary_phecode="Dementia",
        primary_phenotype="Dementia (aggregate)",
        alternative_label_idxs=[210, 212, 213, 214],
        all_label_idxs=[1057, 210, 212, 213, 214],
        label_type="aggregate",
    ),
    5: DiseaseConfig(
        disease_no=5,
        name_en="Stroke",
        name_tr="İnme",
        primary_label_idx=1064,
        primary_phecode="Stroke",
        primary_phenotype="Stroke (aggregate)",
        alternative_label_idxs=[],
        all_label_idxs=[1064],
        label_type="aggregate",
    ),
    6: DiseaseConfig(
        disease_no=6,
        name_en="Depression",
        name_tr="Depresyon",
        primary_label_idx=1060,
        primary_phecode="Depression",
        primary_phenotype="Depression (aggregate)",
        alternative_label_idxs=[233, 234],
        all_label_idxs=[1060, 233, 234],
        label_type="aggregate",
    ),
    7: DiseaseConfig(
        disease_no=7,
        name_en="Type 2 Diabetes",
        name_tr="Tip 2 Diyabet",
        primary_label_idx=1056,
        primary_phecode="Type 2 Diabetes",
        primary_phenotype="Type 2 Diabetes (aggregate)",
        alternative_label_idxs=[97, 98, 99, 100, 101],
        all_label_idxs=[1056, 97, 98, 99, 100, 101],
        label_type="aggregate",
    ),
    8: DiseaseConfig(
        disease_no=8,
        name_en="Obstructive Sleep Apnea",
        name_tr="Obstrüktif Uyku Apnesi",
        primary_label_idx=272,
        primary_phecode="327.32",
        primary_phenotype="Obstructive sleep apnea",
        alternative_label_idxs=[270, 271],
        all_label_idxs=[272, 270, 271],
        label_type="specific_phecode",
    ),
    9: DiseaseConfig(
        disease_no=9,
        name_en="Obesity",
        name_tr="Obezite",
        primary_label_idx=170,
        primary_phecode="278.1",
        primary_phenotype="Obesity",
        alternative_label_idxs=[169, 171],
        all_label_idxs=[170, 169, 171],
        label_type="specific_phecode",
    ),
    10: DiseaseConfig(
        disease_no=10,
        name_en="Chronic Airway Obstruction (COPD)",
        name_tr="Kronik Hava Yolu Obstrüksiyonu (KOAH)",
        primary_label_idx=535,
        primary_phecode="496.0",
        primary_phenotype="Chronic airway obstruction",
        alternative_label_idxs=[536, 537, 538],
        all_label_idxs=[535, 536, 537, 538],
        label_type="specific_phecode",
    ),
    11: DiseaseConfig(
        disease_no=11,
        name_en="Chronic Kidney Disease",
        name_tr="Kronik Böbrek Hastalığı",
        primary_label_idx=1055,
        primary_phecode="Chronic Kidney Disease",
        primary_phenotype="Chronic Kidney Disease (aggregate)",
        alternative_label_idxs=[689, 690, 691],
        all_label_idxs=[1055, 689, 690, 691],
        label_type="aggregate",
    ),
    12: DiseaseConfig(
        disease_no=12,
        name_en="Anxiety Disorders",
        name_tr="Anksiyete Bozuklukları",
        primary_label_idx=1061,
        primary_phecode="Anxiety Disorders",
        primary_phenotype="Anxiety Disorders (aggregate)",
        alternative_label_idxs=[237, 238, 239],
        all_label_idxs=[1061, 237, 238, 239],
        label_type="aggregate",
    ),
}

# Validate that each DISEASES key matches the embedded disease_no
for _key, _d in DISEASES.items():
    if _key != _d.disease_no:
        raise ValueError(
            f"DISEASES key {_key!r} does not match disease_no {_d.disease_no!r} "
            f"for disease '{_d.name_en}'"
        )
del _key, _d


# Convenience: disease number -> English name mapping
DISEASE_NAMES: Dict[int, str] = {
    d.disease_no: d.name_en for d in DISEASES.values()
}

# Convenience: English name -> disease number mapping
DISEASE_NAME_TO_NO: Dict[str, int] = {
    d.name_en: d.disease_no for d in DISEASES.values()
}

# =============================================================================
# Mock Data Defaults
# =============================================================================

DEFAULT_NOISE_CONFIG: Dict[str, float] = {
    "eeg": 0.0,
    "ecg": 0.0,
    "resp": 0.0,
    "emg": 0.0,
}

DEFAULT_PERTURBATION_RATE: float = 0.15
DEFAULT_SHARED_RATIO: float = 0.3

# =============================================================================
# Statistical Constants
# =============================================================================

DEFAULT_SEED: int = 42
BOOTSTRAP_N_RESAMPLES: int = 1000
BOOTSTRAP_CONFIDENCE_LEVEL: float = 0.95
PERMUTATION_N: int = 1000
SIGNIFICANCE_LEVEL: float = 0.05

# =============================================================================
# Paths (adjust to your local environment)
# =============================================================================

DATA_DIR: Path = Path(os.environ.get("SLEEPFM_DATA_DIR", "data"))
OUTPUT_DIR: Path = Path(os.environ.get("SLEEPFM_OUTPUT_DIR", "outputs"))
FIGURES_DIR: Path = Path(os.environ.get("SLEEPFM_FIGURES_DIR", "outputs/figures"))
