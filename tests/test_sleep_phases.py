"""Tests for sleep_phases.py — MESA XML parsing and chunk-level stage labelling.

Uses synthetic XML fragments in tmp_path plus one integration test against a
real MESA XML if available on the developer machine (skipped otherwise).
"""

from pathlib import Path

import pytest

from sleep_phases import (
    CANONICAL_STAGES,
    ChunkLabel,
    DEFAULT_CHUNK_SEC,
    DEFAULT_DOMINANT_FRACTION,
    SLEEP_STAGES,
    StageEvent,
    filter_chunk_indices,
    label_chunks,
    load_chunk_labels_for_cohort,
    parse_stages,
    stage_summary,
)


MESA_XML_ROOT = Path("C:/Users/User/Desktop/Projeler/SleepFM/mesa")
REAL_MESA_XML = MESA_XML_ROOT / "mesa-sleep-REDACTED-nsrr.xml"


def _write_min_xml(path: Path, stage_events: list) -> None:
    """Write a minimal NSRR-style annotation XML with the given stage events."""
    events_xml = "\n".join(
        f"""<ScoredEvent>
<EventType>Stages|Stages</EventType>
<EventConcept>{concept}</EventConcept>
<Start>{start}</Start>
<Duration>{duration}</Duration>
</ScoredEvent>"""
        for concept, start, duration in stage_events
    )
    path.write_text(
        f"""<?xml version="1.0" encoding="UTF-8"?>
<PSGAnnotation>
<EpochLength>30</EpochLength>
<ScoredEvents>
{events_xml}
</ScoredEvents>
</PSGAnnotation>""",
        encoding="utf-8",
    )


class TestParseStages:
    def test_parses_canonical_stage_codes(self, tmp_path):
        xml = tmp_path / "sub.xml"
        _write_min_xml(xml, [
            ("Wake|0", 0.0, 300.0),
            ("Stage 1 sleep|1", 300.0, 60.0),
            ("Stage 2 sleep|2", 360.0, 240.0),
            ("Stage 3 sleep|3", 600.0, 300.0),
            ("REM sleep|5", 900.0, 600.0),
        ])
        events = parse_stages(xml)
        stages = [e.stage for e in events]
        assert stages == ["Wake", "N1", "N2", "N3", "REM"]

    def test_folds_legacy_stage_4_into_n3(self, tmp_path):
        xml = tmp_path / "sub.xml"
        _write_min_xml(xml, [("Stage 4 sleep|4", 0.0, 100.0)])
        events = parse_stages(xml)
        assert events[0].stage == "N3"

    def test_unknown_code_becomes_unsure(self, tmp_path):
        xml = tmp_path / "sub.xml"
        _write_min_xml(xml, [("Unsure|Unsure", 0.0, 30.0)])
        events = parse_stages(xml)
        assert events[0].stage == "Unsure"

    def test_ignores_non_stage_events(self, tmp_path):
        xml = tmp_path / "sub.xml"
        # Manually build XML with a mixed event type
        xml.write_text("""<?xml version="1.0"?>
<PSGAnnotation>
<ScoredEvents>
<ScoredEvent>
<EventType>Respiratory|Respiratory</EventType>
<EventConcept>SpO2 desaturation|SpO2 desaturation</EventConcept>
<Start>10</Start><Duration>20</Duration>
</ScoredEvent>
<ScoredEvent>
<EventType>Stages|Stages</EventType>
<EventConcept>Wake|0</EventConcept>
<Start>0</Start><Duration>300</Duration>
</ScoredEvent>
</ScoredEvents>
</PSGAnnotation>""", encoding="utf-8")
        events = parse_stages(xml)
        assert len(events) == 1
        assert events[0].stage == "Wake"

    def test_ignores_zero_or_negative_duration(self, tmp_path):
        xml = tmp_path / "sub.xml"
        _write_min_xml(xml, [
            ("Wake|0", 0.0, 0.0),
            ("Stage 2 sleep|2", 100.0, -5.0),
            ("REM sleep|5", 200.0, 60.0),
        ])
        events = parse_stages(xml)
        assert len(events) == 1
        assert events[0].stage == "REM"


class TestLabelChunks:
    def test_chunk_fully_inside_single_stage(self):
        events = [StageEvent(0.0, 900.0, "N2")]
        labels = label_chunks(events, n_chunks=2, chunk_sec=300)
        assert labels[0].dominant_stage == "N2"
        assert labels[0].dominant_fraction == 1.0
        assert labels[0].coverage_fraction == 1.0
        assert labels[1].dominant_stage == "N2"

    def test_chunk_crosses_transition(self):
        events = [
            StageEvent(0.0, 200.0, "Wake"),
            StageEvent(200.0, 100.0, "N1"),
        ]
        labels = label_chunks(events, n_chunks=1, chunk_sec=300)
        assert labels[0].dominant_stage == "Wake"
        # With fixed denominator = covered (not chunk_sec), fractions sum to 1
        assert labels[0].stage_fractions["Wake"] == pytest.approx(200 / 300)
        assert labels[0].stage_fractions["N1"] == pytest.approx(100 / 300)
        assert labels[0].coverage_fraction == pytest.approx(1.0)

    def test_partial_coverage_chunk_reports_full_dominant(self):
        """Audit #1 fix: a chunk covering only 100 s of Wake should report
        Wake at fraction 1.0 (not 0.33) with coverage_fraction=0.33."""
        events = [StageEvent(0.0, 100.0, "Wake")]
        labels = label_chunks(events, n_chunks=1, chunk_sec=300)
        assert labels[0].dominant_stage == "Wake"
        assert labels[0].dominant_fraction == pytest.approx(1.0)
        assert labels[0].coverage_fraction == pytest.approx(100 / 300)

    def test_chunk_beyond_recording_end_marked_unknown(self):
        events = [StageEvent(0.0, 100.0, "Wake")]
        labels = label_chunks(events, n_chunks=3, chunk_sec=300)
        assert labels[0].dominant_stage == "Wake"
        assert labels[1].dominant_stage == "Unknown"
        assert labels[1].coverage_fraction == 0.0
        assert labels[2].dominant_stage == "Unknown"

    def test_chunk_boundary_edge_case(self):
        events = [
            StageEvent(0.0, 300.0, "N3"),
            StageEvent(300.0, 300.0, "REM"),
        ]
        labels = label_chunks(events, n_chunks=2, chunk_sec=300)
        assert labels[0].dominant_stage == "N3"
        assert labels[1].dominant_stage == "REM"

    def test_recording_offset_shifts_alignment(self):
        """Audit #4 fix: if SleepFM dropped the first 180 s (calibration),
        chunk 0 should map to XML absolute time [180, 480)."""
        events = [
            StageEvent(0.0, 180.0, "Wake"),
            StageEvent(180.0, 300.0, "N2"),
        ]
        labels_no_offset = label_chunks(events, n_chunks=1, chunk_sec=300)
        labels_with_offset = label_chunks(events, n_chunks=1, chunk_sec=300,
                                          recording_offset_sec=180.0)
        # Without offset: chunk 0 = 180 s Wake + 120 s N2 -> Wake dominant.
        # With correct offset: chunk 0 = 300 s N2 fully.
        assert labels_no_offset[0].dominant_stage == "Wake"
        assert labels_no_offset[0].stage_fractions["Wake"] == pytest.approx(180 / 300)
        assert labels_with_offset[0].dominant_stage == "N2"
        assert labels_with_offset[0].dominant_fraction == pytest.approx(1.0)


class TestParseStagesErrors:
    def test_corrupt_xml_returns_empty(self, tmp_path):
        """Audit #3 fix: broken XML must not crash the whole cohort load."""
        broken = tmp_path / "broken.xml"
        broken.write_text("<PSGAnnotation><ScoredEvent>truncated",
                          encoding="utf-8")
        assert parse_stages(broken) == []

    def test_missing_file_returns_empty(self, tmp_path):
        assert parse_stages(tmp_path / "nope.xml") == []


class TestFilterChunkIndices:
    def _labels(self, dominant_by_chunk):
        return [
            ChunkLabel(i, i * 300, (i + 1) * 300, d,
                       stage_fractions={d: 1.0}, coverage_fraction=1.0)
            for i, d in enumerate(dominant_by_chunk)
        ]

    def test_default_drops_wake_unsure_unknown(self):
        labels = self._labels(["Wake", "N2", "Unknown", "REM", "Unsure", "N3"])
        kept = filter_chunk_indices(labels)
        assert kept == [1, 3, 5]

    def test_include_stages_whitelist(self):
        labels = self._labels(["Wake", "N2", "REM", "N3"])
        kept = filter_chunk_indices(labels, include_stages=("REM",))
        assert kept == [2]

    def test_low_dominant_fraction_dropped(self):
        labels = [
            ChunkLabel(0, 0, 300, "N2",
                       stage_fractions={"N2": 0.4, "N1": 0.35, "Wake": 0.25},
                       coverage_fraction=1.0),
            ChunkLabel(1, 300, 600, "N2",
                       stage_fractions={"N2": 0.9, "N1": 0.1},
                       coverage_fraction=1.0),
        ]
        kept = filter_chunk_indices(labels, min_dominant_fraction=0.5)
        assert kept == [1]

    def test_low_coverage_dropped(self):
        """Audit #1 followup: a chunk with only 30 % coverage should be
        dropped by min_coverage_fraction default (0.5) even if its dominant
        stage was N2 at 1.0 within the covered region."""
        labels = [
            ChunkLabel(0, 0, 300, "N2",
                       stage_fractions={"N2": 1.0}, coverage_fraction=0.3),
            ChunkLabel(1, 300, 600, "N2",
                       stage_fractions={"N2": 1.0}, coverage_fraction=0.9),
        ]
        kept = filter_chunk_indices(labels)
        assert kept == [1]

    def test_n1_friendly_default_threshold(self):
        """Audit #2 fix: default 0.4 keeps N1-dominant chunks that 0.5
        would have dropped (N1 rarely fills half a 5-min window)."""
        n1_chunk = ChunkLabel(
            0, 0, 300, "N1",
            stage_fractions={"N1": 0.42, "N2": 0.35, "Wake": 0.23},
            coverage_fraction=1.0,
        )
        assert filter_chunk_indices([n1_chunk]) == [0]
        assert DEFAULT_DOMINANT_FRACTION <= 0.45


class TestStageSummary:
    def test_counts_all_canonical_keys(self):
        labels = [ChunkLabel(i, 0, 300, s, {s: 1.0}, coverage_fraction=1.0)
                  for i, s in enumerate(["Wake", "N2", "N2", "REM", "Unknown"])]
        summary = stage_summary(labels)
        assert summary["Wake"] == 1
        assert summary["N2"] == 2
        assert summary["REM"] == 1
        assert summary["Unknown"] == 1
        for s in CANONICAL_STAGES:
            assert s in summary


class TestLoadChunkLabelsForCohort:
    """Audit #5 fix + #8 test coverage."""

    def _write_valid_xml(self, path, events):
        events_xml = "\n".join(
            f"""<ScoredEvent>
<EventType>Stages|Stages</EventType>
<EventConcept>{concept}</EventConcept>
<Start>{start}</Start>
<Duration>{duration}</Duration>
</ScoredEvent>""" for concept, start, duration in events
        )
        path.write_text(
            f"""<?xml version="1.0"?>
<PSGAnnotation><ScoredEvents>
{events_xml}
</ScoredEvents></PSGAnnotation>""",
            encoding="utf-8",
        )

    def test_returns_skipped_for_missing_xml(self, tmp_path):
        self._write_valid_xml(tmp_path / "mesa-sleep-REDACTED-nsrr.xml",
                              [("Wake|0", 0.0, 300.0)])
        labels, skipped = load_chunk_labels_for_cohort(
            tmp_path, {"REDACTED": 1, "REDACTED": 1, "0003": 1},
        )
        assert set(labels.keys()) == {"REDACTED"}
        assert set(skipped) == {"REDACTED", "0003"}

    def test_returns_skipped_for_corrupt_xml(self, tmp_path):
        self._write_valid_xml(tmp_path / "mesa-sleep-REDACTED-nsrr.xml",
                              [("Wake|0", 0.0, 300.0)])
        (tmp_path / "mesa-sleep-REDACTED-nsrr.xml").write_text(
            "<PSGAnnotation><ScoredEvent>truncated", encoding="utf-8",
        )
        labels, skipped = load_chunk_labels_for_cohort(
            tmp_path, {"REDACTED": 1, "REDACTED": 1},
        )
        assert set(labels.keys()) == {"REDACTED"}
        assert skipped == ["REDACTED"]

    def test_recording_offset_propagated(self, tmp_path):
        self._write_valid_xml(tmp_path / "mesa-sleep-REDACTED-nsrr.xml", [
            ("Wake|0", 0.0, 120.0),
            ("Stage 2 sleep|2", 120.0, 300.0),
        ])
        labels, _ = load_chunk_labels_for_cohort(
            tmp_path, {"REDACTED": 1},
            recording_offset_by_subject={"REDACTED": 120.0},
        )
        assert labels["REDACTED"][0].dominant_stage == "N2"


@pytest.mark.skipif(not REAL_MESA_XML.exists(),
                    reason=f"real MESA XML not available at {REAL_MESA_XML}")
class TestRealMesaXml:
    """Integration smoke: parsing a real MESA XML must yield sensible counts."""

    def test_parses_all_expected_stages(self):
        events = parse_stages(REAL_MESA_XML)
        stages = {e.stage for e in events}
        # A full-night PSG will always contain at least Wake and one sleep stage
        assert "Wake" in stages
        assert stages & set(SLEEP_STAGES), "no sleep stage found in real recording"

    def test_epoch_durations_multiples_of_thirty(self):
        """MESA uses 30-second epochs; durations should be multiples of 30
        within IEEE-754 tolerance (audit #8: exact modulo unreliable on floats).
        """
        import math
        events = parse_stages(REAL_MESA_XML)
        offenders = [e for e in events
                     if not math.isclose(e.duration_sec % 30, 0, abs_tol=1e-6)
                     and not math.isclose(e.duration_sec % 30, 30, abs_tol=1e-6)]
        assert len(offenders) == 0, (
            f"non-30s-multiple stage durations: {offenders[:3]}"
        )

    def test_chunk_labeling_covers_full_night(self):
        events = parse_stages(REAL_MESA_XML)
        # Cover the full recording with 5-min chunks
        total = max(e.end_sec for e in events)
        n_chunks = int(total // DEFAULT_CHUNK_SEC)
        labels = label_chunks(events, n_chunks=n_chunks)
        unknown_ratio = sum(1 for lbl in labels if lbl.dominant_stage == "Unknown") / len(labels)
        assert unknown_ratio < 0.1, (
            f"too many chunks labeled Unknown ({unknown_ratio:.0%}); "
            "chunk boundaries may not match XML coverage"
        )
