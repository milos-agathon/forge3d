"""Curved-label glyph spacing measured from rendered ink columns."""
import numpy as np
import pytest

from _terrain_runtime import terrain_rendering_available
from _curved_label_fixture import render_curved_label


pytestmark = pytest.mark.skipif(not terrain_rendering_available(),
                              reason="needs a terrain-capable GPU adapter")


def _ink_runs(pixels, base):
    # White text over the fixture's black base; the black halo is invisible.
    ink = np.abs(pixels[..., :3].astype(int) - base[..., :3]).max(axis=2) > 128
    columns = np.flatnonzero(ink.any(axis=0))
    breaks = np.flatnonzero(np.diff(columns) > 1)
    starts = np.concatenate(([columns[0]], columns[breaks + 1]))
    ends = np.concatenate((columns[breaks], [columns[-1]]))
    return list(zip(starts.tolist(), ends.tolist()))


def test_horizontal_curved_label_glyph_gaps_follow_advances():
    pixels, base, _ = render_curved_label(0)
    runs = _ink_runs(pixels, base)
    # One ink run per visible glyph of "Silver River": no glyph collides with
    # its neighbour (the defect overlapped "R" and "i").
    assert len(runs) == len("SilverRiver"), runs
    gaps = [right[0] - left[1] - 1 for left, right in zip(runs, runs[1:])]
    word_gap = gaps[5]  # between "Silver" and "River"
    intra = gaps[:5] + gaps[6:]
    # The defect opened a gap after each narrow "i" and crowded the rest.
    after_i = [gaps[1], gaps[7]]
    others = [gap for index, gap in enumerate(gaps) if index not in (1, 5, 7)]
    assert max(after_i) <= max(others) + 1, gaps
    assert word_gap >= 2 * max(intra), gaps
