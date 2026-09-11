"""Cohort-aware path resolution — single source of truth for data location.

Prevents the "clean cohort default" trap: `real_embeddings.py`,
`clinical_analysis.py`, and `chunk_level_analysis.py` used to each carry
their own DEFAULT_* constants pointing at `smoke_run/` (the contaminated
cohort). A user who downloaded a fresh cohort into `mesa_test_clean/`
would still silently read the old data unless every callsite passed
explicit paths.

This module centralises the resolution so switching cohorts is a single
env-var change:

    $env:SLEEPFM_COHORT_ROOT = "C:/.../mesa_test_clean_run"
    # embeddings expected at $SLEEPFM_COHORT_ROOT/embeddings/
    # XML annotations at   $SLEEPFM_COHORT_ROOT/xml/
    # metadata CSV at      $SLEEPFM_COHORT_ROOT/metadata.csv (or CLEAN_METADATA_CSV override)

Callers must pass the resolved paths explicitly to the loader functions;
the loaders no longer have magic defaults pointing at a specific cohort.
Legacy fallbacks (unset env var) point at the contaminated smoke_run so
existing scripts do not break on this refactor, but a WARNING is emitted
so the choice is loud in every run.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_LEGACY_ROOT = _PROJECT_ROOT / "data" / "smoke_run"
_LEGACY_MESA_XML = Path("C:/Users/User/Desktop/Projeler/SleepFM/mesa")
_LEGACY_METADATA_CSV = Path(
    "C:/Users/User/Desktop/Projeler/SleepFM/mesa/.csv/mesa-sleep-dataset-0.8.0.csv"
)

_COHORT_ENV = "SLEEPFM_COHORT_ROOT"
_METADATA_ENV = "SLEEPFM_METADATA_CSV"
_XML_ENV = "SLEEPFM_MESA_XML_DIR"


@dataclass(frozen=True)
class CohortPaths:
    """Resolved cohort locations. `label` names the cohort in log lines."""
    label: str
    root: Path
    embedding_dir: Path
    xml_dir: Path
    metadata_csv: Path


def resolve_cohort(
    root: Optional[Path] = None,
    metadata_csv: Optional[Path] = None,
    xml_dir: Optional[Path] = None,
) -> CohortPaths:
    """Return a CohortPaths for the requested layout.

    Precedence (highest first): explicit arg -> env var -> legacy fallback
    (with a warning). Callers that want deterministic behaviour should
    always pass `root` explicitly; the env-var and legacy paths exist for
    convenience in scripts and are noisy on purpose.
    """
    env_root = os.environ.get(_COHORT_ENV, "").strip()
    env_metadata = os.environ.get(_METADATA_ENV, "").strip()
    env_xml = os.environ.get(_XML_ENV, "").strip()

    if root is not None:
        cohort_root = Path(root).resolve()
        source = "explicit"
    elif env_root:
        cohort_root = Path(env_root).resolve()
        source = f"${_COHORT_ENV}"
    else:
        cohort_root = _LEGACY_ROOT.resolve()
        source = "legacy fallback (contaminated smoke_run)"
        logger.warning(
            "resolve_cohort: no root passed and $%s unset — falling back to "
            "the contaminated smoke_run cohort. Set $%s to point at the "
            "clean cohort or pass root=... explicitly.",
            _COHORT_ENV, _COHORT_ENV,
        )

    embedding_dir = cohort_root / "embeddings"
    xml_resolved = Path(xml_dir or env_xml).resolve() if (xml_dir or env_xml) else _LEGACY_MESA_XML
    if metadata_csv is not None:
        metadata_path = Path(metadata_csv).resolve()
    elif env_metadata:
        metadata_path = Path(env_metadata).resolve()
    else:
        candidate = cohort_root / "metadata.csv"
        metadata_path = candidate if candidate.exists() else _LEGACY_METADATA_CSV

    return CohortPaths(
        label=f"{cohort_root.name} ({source})",
        root=cohort_root,
        embedding_dir=embedding_dir,
        xml_dir=xml_resolved,
        metadata_csv=metadata_path,
    )
