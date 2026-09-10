"""Sleep-stage-aware chunk labelling for SleepFM 5-minute embeddings.

Parses MESA NSRR XML annotations (`mesa-sleep-XXXX-nsrr.xml`) and assigns
each SleepFM 5-minute embedding chunk a dominant sleep stage. Downstream
uses:

- Cluster-vs-stage confusion matrices (does the embedding respect sleep
  architecture?)
- Wake / artifact chunk exclusion for disease-focused clustering
- Stage-specific analyses (e.g. apnea burden in REM chunks)

MESA XML stage codes (AASM):
    0 = Wake, 1 = N1, 2 = N2, 3 = N3 (also 4 legacy -> N3), 5 = REM, "Unsure"

Design notes on audit hardening
-------------------------------
- `label_chunks` accepts a `recording_offset_sec` — SleepFM preprocessing may
  drop a few seconds of calibration/artifact at the start of a recording, and
  if we ignore that offset every stage label is shifted (audit #4).
- `ChunkLabel.coverage_fraction` reports how much of the 5-min window was
  actually scored (audit #1, #6). `dominant_fraction` uses the covered
  window as denominator so an end-of-recording chunk with 100 s of Wake
  reports Wake at 1.0 (not 0.33).
- `parse_stages` wraps `ET.parse` and returns `[]` on a corrupt XML instead
  of crashing the entire cohort load (audit #3).
- `load_chunk_labels_for_cohort` returns a tuple `(labels, skipped)` so
  callers can surface how many subjects lacked XMLs (audit #5).
- Default `min_dominant_fraction` lowered to 0.4 (audit #2): N1 rarely fills
  half a 5-min chunk, so 0.5 systematically drops all N1 chunks.
"""

from __future__ import annotations

import logging
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_STAGE_MAP: Dict[str, str] = {
    "0": "Wake",
    "1": "N1",
    "2": "N2",
    "3": "N3",
    "4": "N3",
    "5": "REM",
}
CANONICAL_STAGES: Tuple[str, ...] = ("Wake", "N1", "N2", "N3", "REM", "Unsure")
SLEEP_STAGES: Tuple[str, ...] = ("N1", "N2", "N3", "REM")

DEFAULT_CHUNK_SEC: int = 300
DEFAULT_DOMINANT_FRACTION: float = 0.4  # N1-friendly (audit #2)


@dataclass
class StageEvent:
    start_sec: float
    duration_sec: float
    stage: str

    @property
    def end_sec(self) -> float:
        return self.start_sec + self.duration_sec


@dataclass
class ChunkLabel:
    chunk_idx: int
    chunk_start_sec: float
    chunk_end_sec: float
    dominant_stage: str
    stage_fractions: Dict[str, float] = field(default_factory=dict)
    coverage_fraction: float = 0.0

    @property
    def dominant_fraction(self) -> float:
        """Fraction of the *covered* window that the dominant stage occupies.

        Denominator is `coverage_fraction * chunk_sec`, not the full chunk,
        so a chunk that overlaps only the last 100 s of a recording (Wake
        for all 100 s) reports 1.0 rather than 0.33.
        """
        return self.stage_fractions.get(self.dominant_stage, 0.0)


def parse_stages(xml_path: Path) -> List[StageEvent]:
    """Extract sleep-stage events from a NSRR annotation XML.

    Returns [] on corrupt / missing XML (audit #3). Callers that need to
    distinguish 'no stages in recording' from 'file broken' should check the
    file separately before calling.
    """
    try:
        tree = ET.parse(str(xml_path))
    except (ET.ParseError, OSError) as e:
        logger.warning("parse_stages: failed to parse %s (%s)", xml_path, type(e).__name__)
        return []
    root = tree.getroot()
    events: List[StageEvent] = []
    for scored in root.iter("ScoredEvent"):
        event_type = (scored.findtext("EventType") or "").strip()
        if not event_type.startswith("Stages"):
            continue
        concept = (scored.findtext("EventConcept") or "").strip()
        code = concept.split("|")[-1] if "|" in concept else concept
        stage = _STAGE_MAP.get(code, "Unsure")
        start_txt = scored.findtext("Start")
        dur_txt = scored.findtext("Duration")
        if start_txt is None or dur_txt is None:
            continue
        try:
            start = float(start_txt)
            duration = float(dur_txt)
        except ValueError:
            continue
        if duration <= 0:
            continue
        events.append(StageEvent(start, duration, stage))
    return events


def label_chunks(
    stage_events: List[StageEvent],
    n_chunks: int,
    chunk_sec: int = DEFAULT_CHUNK_SEC,
    recording_offset_sec: float = 0.0,
) -> List[ChunkLabel]:
    """Assign each chunk a dominant stage from overlap-weighted event durations.

    recording_offset_sec: seconds trimmed from the start of the EDF before
        SleepFM's chunking (calibration, front-of-recording artifacts). The
        first embedding chunk covers XML absolute time
        [recording_offset_sec, recording_offset_sec + chunk_sec).
    """
    labels: List[ChunkLabel] = []
    for i in range(n_chunks):
        chunk_start = recording_offset_sec + i * chunk_sec
        chunk_end = chunk_start + chunk_sec
        stage_seconds: Dict[str, float] = {}
        for ev in stage_events:
            overlap_start = max(ev.start_sec, chunk_start)
            overlap_end = min(ev.end_sec, chunk_end)
            overlap = overlap_end - overlap_start
            if overlap > 0:
                stage_seconds[ev.stage] = stage_seconds.get(ev.stage, 0.0) + overlap
        covered = sum(stage_seconds.values())
        coverage_fraction = covered / chunk_sec
        if covered <= 0:
            labels.append(ChunkLabel(i, chunk_start, chunk_end, "Unknown",
                                     stage_fractions={}, coverage_fraction=0.0))
            continue
        dominant = max(stage_seconds, key=stage_seconds.get)
        # Divide by *covered* so partial coverage doesn't systematically
        # penalise end-of-recording chunks (audit #1).
        fractions = {s: sec / covered for s, sec in stage_seconds.items()}
        labels.append(ChunkLabel(i, chunk_start, chunk_end, dominant,
                                 stage_fractions=fractions,
                                 coverage_fraction=coverage_fraction))
    return labels


def filter_chunk_indices(
    labels: List[ChunkLabel],
    include_stages: Optional[Tuple[str, ...]] = None,
    exclude_stages: Optional[Tuple[str, ...]] = ("Wake", "Unsure", "Unknown"),
    min_dominant_fraction: float = DEFAULT_DOMINANT_FRACTION,
    min_coverage_fraction: float = 0.5,
) -> List[int]:
    """Return chunk indices that pass stage filters.

    Default excludes Wake+Unsure+Unknown — appropriate for disease-focused
    clustering. For sleep-architecture analyses (WASO, sleep-onset latency,
    Wake-vs-sleep contrast) pass `exclude_stages=None`. "Unsure" and
    "Unknown" are distinct: Unsure = scorer marked ambiguous, Unknown = XML
    did not cover this window; drop both by default because neither
    contributes clean stage-conditioned analysis.
    """
    kept: List[int] = []
    for lbl in labels:
        if lbl.coverage_fraction < min_coverage_fraction:
            continue
        if include_stages is not None and lbl.dominant_stage not in include_stages:
            continue
        if exclude_stages is not None and lbl.dominant_stage in exclude_stages:
            continue
        if lbl.dominant_fraction < min_dominant_fraction:
            continue
        kept.append(lbl.chunk_idx)
    return kept


def stage_summary(labels: List[ChunkLabel]) -> Dict[str, int]:
    counts: Dict[str, int] = {s: 0 for s in CANONICAL_STAGES}
    counts["Unknown"] = 0
    for lbl in labels:
        counts[lbl.dominant_stage] = counts.get(lbl.dominant_stage, 0) + 1
    return counts


def load_chunk_labels_for_cohort(
    xml_dir: Path,
    n_chunks_by_subject: Dict[str, int],
    chunk_sec: int = DEFAULT_CHUNK_SEC,
    recording_offset_by_subject: Optional[Dict[str, float]] = None,
) -> Tuple[Dict[str, List[ChunkLabel]], List[str]]:
    """Parse XML + label chunks for every subject with a known chunk count.

    Returns (labels, skipped) — the skipped list carries subject ids whose
    XML was missing or unparseable, so cross-cohort sizes are auditable
    (audit #5). recording_offset_by_subject lets callers pass per-subject
    EDF-start offsets when SleepFM preprocessing dropped a calibration
    prefix (audit #4).
    """
    result: Dict[str, List[ChunkLabel]] = {}
    skipped: List[str] = []
    offsets = recording_offset_by_subject or {}
    for sid, n_chunks in n_chunks_by_subject.items():
        xml_path = xml_dir / f"mesa-sleep-{sid}-nsrr.xml"
        if not xml_path.exists():
            skipped.append(sid)
            continue
        events = parse_stages(xml_path)
        if not events:
            skipped.append(sid)
            continue
        result[sid] = label_chunks(
            events, n_chunks=n_chunks, chunk_sec=chunk_sec,
            recording_offset_sec=offsets.get(sid, 0.0),
        )
    if skipped:
        logger.warning("load_chunk_labels_for_cohort: skipped %d subject(s): %s",
                       len(skipped), skipped)
    return result, skipped
