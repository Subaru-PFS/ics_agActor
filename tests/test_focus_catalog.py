"""acquire_field estimates focus from matches against the catalog the autoguide loop uses.

Its offsets still come from the stricter acquisition catalog; only the detections
handed to the focus estimate come from a second match.
"""

import warnings
from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest
from agActor import field_acquisition
from agActor.utils.data import GuideCatalog, GuideOffsets, filter_guide_objects
from pfs.utils.datamodel.ag import SourceMatchingFlags as M

with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    from pfs.utils.datamodel.ag import AutoGuiderStarMask as S

# What the acquisition filter asks of a star; the loop's filter only removes galaxies.
CLEAN = S.NON_BINARY | S.GAIA | S.PHOTO_SIG | S.ASTROMETRIC | S.PMRA | S.PMDEC | S.PARA


def _detected(n: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "agc_camera_id": 0,
            "spot_id": range(n),
            "centroid_x_pix": 100.0 * (1 + np.arange(n)),
            "centroid_y_pix": 100.0 * (1 + np.arange(n)),
            "central_image_moment_11_pix": 0.1,
            "central_image_moment_20_pix": 2.5,
            "central_image_moment_02_pix": 2.3,
            "peak_intensity": 280.0,
            "image_moment_00_pix": 13400.0,
            "flags": 0,
        }
    )


def _catalog(star_flags: list[int], is_guide: bool) -> GuideCatalog:
    """Star ``i`` sits under detection ``i``; filtered as get_guide_objects filters it."""
    n = len(star_flags)
    stars = pd.DataFrame(
        {
            "source_id": range(1, n + 1),
            "ra": 0.0,
            "dec": 0.0,
            "mag": 12.0,
            "agc_camera_id": 0,
            "x": 100.0 * (1 + np.arange(n)),
            "y": 100.0 * (1 + np.arange(n)),
            "flags": [int(f) for f in star_flags],
        }
    )
    return GuideCatalog(
        guide_objects=filter_guide_objects(stars, is_guide=is_guide),
        ra=0.0, dec=0.0, inr=0.0, inst_pa=0.0, m2_pos3=6.0, adc=0.0,
        taken_at=datetime(2026, 11, 15, tzinfo=UTC),
    )  # fmt: skip


def _run_acquire_field(monkeypatch, star_flags: list[int], **kwargs) -> tuple[GuideOffsets, list[int]]:
    """Run acquire_field with a matcher that matches detection ``i`` to star ``i`` when the
    catalog it is given keeps that star. Returns the offsets and the catalog size of each match."""
    catalog_sizes = []

    def fake_calculate_offsets(*, guide_objects, detected_array, **_):
        kept = guide_objects.query("filtered_by == 0")
        catalog_sizes.append(len(kept))
        n = len(detected_array)
        result = np.zeros((n, 12))
        result[:, 10] = 1
        for i in range(n):
            if i in kept.index:
                result[i, 9] = kept.index.get_loc(i)
                result[i, 11] = M.GOOD_MATCH
            else:
                result[i, 11] = M.NO_MATCH
        return 0.0, 0.0, np.nan, np.nan, result, 0.0, np.ones(n, dtype=bool), kept

    monkeypatch.setattr(field_acquisition, "get_detected_objects", lambda frame_id: _detected(len(star_flags)))
    monkeypatch.setattr(
        field_acquisition, "get_guide_objects", lambda *, is_guide, **_: _catalog(star_flags, is_guide)
    )
    monkeypatch.setattr(field_acquisition, "calculate_offsets", fake_calculate_offsets)
    offsets = field_acquisition.acquire_field(design_id=1, frame_id=1, altazimuth=False, **kwargs)
    return offsets, catalog_sizes


def test_focus_uses_stars_the_loop_keeps_and_acquisition_does_not(monkeypatch):
    # Star 1 has no parallax flag: the acquisition filter drops it, the loop keeps it. Star 2 is a galaxy.
    star_flags = [CLEAN, CLEAN & ~S.PARA, CLEAN | S.GALAXY, CLEAN]
    offsets, catalog_sizes = _run_acquire_field(monkeypatch, star_flags)

    # The offsets and the match that is written out still come from the acquisition catalog ...
    assert catalog_sizes == [2, 3]
    assert offsets.identified_objects["matched"].tolist() == [1, 0, 0, 1]
    assert list(offsets.matched_detected_objects.index) == [0, 3]
    assert (offsets.guide_objects["filtered_by"] != 0).tolist() == [False, True, True, False]
    # ... and focus also gets the spot on the star only the loop keeps.
    assert list(offsets.focus_detected_objects.index) == [0, 1, 3]
    # Negative control: a spot on a galaxy is in neither catalog, so the second match is not
    # simply "every detection", and the focus set is not the acquisition match.
    assert 2 not in offsets.focus_detected_objects.index
    assert list(offsets.focus_detected_objects.index) != list(offsets.matched_detected_objects.index)


@pytest.mark.parametrize(
    ("star_flags", "kwargs"),
    [
        ([CLEAN, CLEAN | S.GALAXY, CLEAN], {}),  # both filters drop the galaxy and nothing else
        ([CLEAN, CLEAN & ~S.PARA, CLEAN], {"is_guide": True}),  # the caller asked for the loop's catalog
    ],
)
def test_one_match_when_the_catalogs_agree(monkeypatch, star_flags, kwargs):
    offsets, catalog_sizes = _run_acquire_field(monkeypatch, star_flags, **kwargs)

    assert len(catalog_sizes) == 1
    assert offsets.focus_detected_object_ids is None
    assert offsets.focus_detected_objects.equals(offsets.matched_detected_objects)


def test_focus_detected_objects_selects_by_label():
    detected = _detected(4)
    detected.index = [10, 11, 12, 13]
    identified = pd.DataFrame({"detected_object_id": [10, 11, 12, 13], "matched": [1, 0, 0, 1]})
    offsets = GuideOffsets(
        ra=0.0, dec=0.0, inst_pa=0.0, ra_offset=0.0, dec_offset=0.0, inr_offset=0.0, scale_offset=0.0,
        dalt=None, daz=None, guide_objects=pd.DataFrame(),
        detected_objects=detected, identified_objects=identified,
        dx=0.0, dy=0.0, size=0.0, peak=0.0, flux=0.0,
    )  # fmt: skip
    # Without a second match, focus uses the matched detections (the autoguide loop's case).
    assert list(offsets.focus_detected_objects.index) == [10, 13]

    offsets.focus_detected_object_ids = np.array([10, 11, 13])
    assert list(offsets.focus_detected_objects.index) == [10, 11, 13]
    # Negative control: the ids are labels, not positions (position 11 does not exist), and
    # setting them leaves the acquisition match alone.
    assert list(offsets.matched_detected_objects.index) == [10, 13]
