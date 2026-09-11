"""Post-render bus leveling.

Lands every section on a known loudness, with the vocal a fixed margin above
the instrumental bed inside it. Runs on the rendered buses (what the listener
hears) rather than raw stems, so neither the arrangement's role gains nor a
sparse source recording can leave a section quiet or a vocal hot.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np
import pyloudnorm

from musicmixer.models import Section, VOCAL_BUS_STEMS
from musicmixer.services.processor import LUFS_FLOOR, auto_level
from musicmixer.services.renderer import build_section_curve, section_sample_bounds

logger = logging.getLogger(__name__)

SECTION_TARGET_LUFS = -18.0
ENERGY_OFFSET_DB: dict[str, float] = {"low": -3.0, "medium": -1.5, "high": 0.0, "peak": 1.0}
VOCAL_OVER_BED_DB = 2.0
BED_GAIN_RANGE_DB = (-6.0, 12.0)
VOCAL_GAIN_RANGE_DB = (-12.0, 12.0)
MIN_MEASURE_SEC = 3.0


@dataclass
class SectionLevel:
    label: str
    bed_lufs: float        # before leveling; NaN when unmeasurable
    bed_gain_db: float
    vocal_lufs: float      # before leveling; NaN when vocal absent/unmeasurable
    vocal_gain_db: float

    def as_dict(self) -> dict:
        def _r(v: float) -> float | None:
            return None if math.isnan(v) else round(v, 1)
        return {
            "label": self.label,
            "bed_lufs": _r(self.bed_lufs),
            "bed_gain_db": round(self.bed_gain_db, 1),
            "vocal_lufs": _r(self.vocal_lufs),
            "vocal_gain_db": round(self.vocal_gain_db, 1),
        }


def _measure(meter: pyloudnorm.Meter, audio: np.ndarray, sr: int) -> float:
    if len(audio) < MIN_MEASURE_SEC * sr:
        return math.nan
    lufs = meter.integrated_loudness(audio)
    return lufs if lufs > LUFS_FLOOR else math.nan


def _vocal_planned(section: Section) -> bool:
    return any(section.stem_gains.get(s, 0.0) > 0.0 for s in VOCAL_BUS_STEMS)


def _fill_unmeasured(gains: list[float | None]) -> list[float]:
    """Unmeasurable sections inherit the previous section's gain (or the next
    one's at the start) so the curve stays flat across their boundaries."""
    filled: list[float | None] = list(gains)
    for i in range(1, len(filled)):
        if filled[i] is None:
            filled[i] = filled[i - 1]
    for i in range(len(filled) - 2, -1, -1):
        if filled[i] is None:
            filled[i] = filled[i + 1]
    return [0.0 if g is None else g for g in filled]


def _db_to_lin(db: float) -> float:
    return 10.0 ** (db / 20.0)


def level_buses(
    vocal_bus: np.ndarray,
    instrumental_bus: np.ndarray,
    sections: list[Section],
    beat_frames: np.ndarray,
    sr: int,
    target_bpm: float | None = None,
    hop_length: int = 512,
) -> tuple[np.ndarray, np.ndarray, list[SectionLevel]]:
    """Level the rendered buses section by section.

    1. Smooth intra-section swings on the bed alone (slow leveler; the vocal
       is not in the detector, so it cannot pump the bed).
    2. Put the bed at the section target (``SECTION_TARGET_LUFS`` plus the
       energy offset).
    3. Where the plan has a vocal, set it ``VOCAL_OVER_BED_DB`` above the bed,
       then trim both so the summed section lands on the target: the bed makes
       room under the vocal and comes back up when it leaves.

    Gain changes ramp over section transitions via ``build_section_curve``.
    """
    total = len(instrumental_bus)
    if not sections or total == 0:
        return vocal_bus, instrumental_bus, []

    last_beat = max(s.end_beat for s in sections)
    bounds = section_sample_bounds(sections, total, beat_frames, sr, hop_length, target_bpm)
    meter = pyloudnorm.Meter(sr)
    targets = [
        SECTION_TARGET_LUFS + ENERGY_OFFSET_DB.get(s.energy, ENERGY_OFFSET_DB["medium"])
        for s in sections
    ]

    instrumental_bus = auto_level(
        instrumental_bus, sr,
        window_sec=4.0, max_boost_db=4.0, max_cut_db=3.0,
        target_percentile=50.0, active_floor_db=-50.0,
    )

    bed_lufs = [_measure(meter, instrumental_bus[s:e], sr) for s, e in bounds]
    bed_gains_db = _fill_unmeasured([
        None if math.isnan(lufs) else float(np.clip(target - lufs, *BED_GAIN_RANGE_DB))
        for lufs, target in zip(bed_lufs, targets)
    ])

    # The plan decides where the vocal is. Measuring instead would read the
    # renderer's ramp tail in a silent section as "active" and boost it.
    vocal_lufs = [
        _measure(meter, vocal_bus[s:e], sr) if _vocal_planned(section) else math.nan
        for section, (s, e) in zip(sections, bounds)
    ]
    vocal_gains: list[float | None] = [None] * len(sections)
    for i, (s, e) in enumerate(bounds):
        if math.isnan(vocal_lufs[i]):
            continue
        leveled_bed = targets[i] if math.isnan(bed_lufs[i]) else bed_lufs[i] + bed_gains_db[i]
        vocal_gain = float(np.clip(leveled_bed + VOCAL_OVER_BED_DB - vocal_lufs[i], *VOCAL_GAIN_RANGE_DB))

        summed = (
            instrumental_bus[s:e] * _db_to_lin(bed_gains_db[i])
            + vocal_bus[s:e] * _db_to_lin(vocal_gain)
        )
        trim = targets[i] - _measure(meter, summed, sr)
        if not math.isnan(trim):
            bed_gains_db[i] = float(np.clip(bed_gains_db[i] + trim, *BED_GAIN_RANGE_DB))
            vocal_gain = float(np.clip(vocal_gain + trim, *VOCAL_GAIN_RANGE_DB))
        vocal_gains[i] = vocal_gain
    vocal_gains_db = _fill_unmeasured(vocal_gains)

    bed_curve = build_section_curve(
        sections, [_db_to_lin(g) for g in bed_gains_db],
        total, beat_frames, sr, hop_length, last_beat, target_bpm,
    )
    vocal_curve = build_section_curve(
        sections, [_db_to_lin(g) for g in vocal_gains_db],
        total, beat_frames, sr, hop_length, last_beat, target_bpm,
    )
    instrumental_bus = instrumental_bus * bed_curve[:, np.newaxis]
    vocal_bus = vocal_bus * vocal_curve[:, np.newaxis]

    levels = [
        SectionLevel(section.label, b_lufs, b_gain, v_lufs, v_gain)
        for section, b_lufs, b_gain, v_lufs, v_gain
        in zip(sections, bed_lufs, bed_gains_db, vocal_lufs, vocal_gains_db)
    ]
    for i, lv in enumerate(levels):
        logger.info(
            "Section %d %s (%s, target %.1f): bed %.1f LUFS %+.1f dB | vocal %.1f LUFS %+.1f dB",
            i, lv.label, sections[i].energy, targets[i],
            lv.bed_lufs, lv.bed_gain_db, lv.vocal_lufs, lv.vocal_gain_db,
        )
    return vocal_bus, instrumental_bus, levels
