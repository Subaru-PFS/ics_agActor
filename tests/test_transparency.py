"""The transparency log line: the stars it uses, its reference within a visit, and that it never raises."""

import logging
import types

import numpy as np
import pandas as pd
import pytest
from agActor.utils import transparency
from agActor.utils.transparency import TransparencyReference, describe_transparency, frame_transparency
from pfs.utils import agMetrics

if not hasattr(agMetrics, "summarizeTransparency"):
    pytest.skip("needs a pfs_utils with the AG transparency functions", allow_module_level=True)

COLOR = 1.0
CATALOG_MAG = 16.0
CLEAR = CATALOG_MAG + agMetrics.agColorTerm(COLOR)  # a clear sky's reading, before the camera zero point


def magnitude(value: float, camera: int = 3, flags: int = 0) -> float:
    """The magnitude a star of `CATALOG_MAG` and `COLOR` reads on a camera under a transparency of ``value``."""
    return CLEAR + agMetrics.agCameraZeroPoint(camera, flags) - 2.5 * np.log10(value)


def make_frame(values, matched=None):
    """Four detections on two cameras, each paired with a guide star; labels are not positions."""
    n = len(values)
    detected = pd.DataFrame(
        {"agc_camera_id": [0, 0, 3, 3][:n], "spot_id": [5, 2, 7, 1][:n]}, index=[10, 11, 12, 13][:n]
    )
    guide = pd.DataFrame(
        {"source_id": [900, 901, 902, 903][:n], "mag": CATALOG_MAG}, index=[40, 41, 42, 43][:n]
    )
    identified = pd.DataFrame(
        {
            "detected_object_id": detected.index,
            "guide_object_id": guide.index,
            "matched": [1] * n if matched is None else matched,
        }
    )
    spots = pd.DataFrame(
        {
            "agc_camera_id": detected.agc_camera_id.to_numpy(),
            "spot_id": detected.spot_id.to_numpy(),
            "flags": 0,
            "estimated_magnitude": [magnitude(v, c) for v, c in zip(values, detected.agc_camera_id)],
        }
    ).sample(frac=1, random_state=1)  # the query's order is not the detections'
    colors = pd.Series(COLOR, index=guide.source_id.to_numpy()[::-1])
    return detected, identified, guide, spots, colors


def test_frame_transparency_is_the_median_over_matched_stars():
    result = frame_transparency(*make_frame([0.9, 0.8, 0.7, 0.6]))
    # summarizeTransparency leaves the faintest quarter out: 0.6 here.
    assert result.nStars == 3
    assert result.transparency == pytest.approx(0.8)

    # Unmatched detections do not count: without the two brightest the median is of 0.7 and 0.6 ...
    partly = frame_transparency(*make_frame([0.9, 0.8, 0.7, 0.6], matched=[0, 0, 1, 1]))
    assert partly.nStars == 1 and partly.transparency == pytest.approx(0.7)
    # Negative control: ... which is not what every row gives.
    assert partly.transparency != pytest.approx(result.transparency, rel=0.05)


def test_frame_transparency_pairs_by_label_and_by_spot():
    """A star's magnitude comes from its own spot and its colour from its own source id."""
    detected, identified, guide, spots, colors = make_frame([0.9, 0.8, 0.7, 0.6])
    colors[902] = 5.0  # the third star's colour is outside the colour term's range
    result = frame_transparency(detected, identified, guide, spots, colors)
    assert result.nStars == 2 and result.transparency == pytest.approx(0.85)

    # A spot with no stored magnitude drops out as well.
    result = frame_transparency(detected, identified, guide, spots[spots.spot_id != 5], colors)
    assert result.nStars == 1 and result.transparency == pytest.approx(0.8)

    # Negative control: paired by position instead, the spots' shuffled order would give other values.
    byPosition = agMetrics.magnitudesToTransparency(spots.estimated_magnitude.to_numpy(), CATALOG_MAG, COLOR)
    assert not np.allclose(byPosition, [0.9, 0.8, 0.7, 0.6])


def test_frame_transparency_uses_the_camera_zero_point():
    """A star on AG3's glass half (camera 2, left), 0.19 mag faint there, reads its sky with the zero point."""
    detected, identified, guide, spots, colors = make_frame([0.9])
    detected["agc_camera_id"] = 2
    spots["agc_camera_id"] = 2
    spots["estimated_magnitude"] = magnitude(0.9, camera=2, flags=0)
    assert frame_transparency(detected, identified, guide, spots, colors).transparency == pytest.approx(0.9)
    # Negative control: read as on the right half, whose zero point is near 0, it is 0.19 mag low.
    spots["flags"] = 1
    assert frame_transparency(detected, identified, guide, spots, colors).transparency < 0.8


def test_frame_transparency_without_stars():
    for frame in (make_frame([0.9, 0.8], matched=[0, 0]), make_frame([])):
        result = frame_transparency(*frame)
        assert result.nStars == 0 and np.isnan(result.transparency)
    detected, identified, guide, spots, _ = make_frame([0.9, 0.8])
    result = frame_transparency(
        detected, identified, guide, spots, pd.Series(dtype=float)
    )  # design not in the opdb
    assert result.nStars == 0 and np.isnan(result.transparency)


def test_reference_is_the_median_of_a_visits_first_exposures():
    reference = TransparencyReference(n_first=3)
    visit = (0x1234, 150000)
    assert reference.relative(visit, 0.80) == pytest.approx(1.0)
    assert reference.relative(visit, 0.84) == pytest.approx(0.84 / 0.82)
    assert reference.relative(visit, float("nan")) != reference.relative(
        visit, float("nan")
    )  # NaN, and not counted
    assert reference.relative(visit, 0.82) == pytest.approx(1.0)
    assert reference.n_used == 3
    # The reference is now fixed at 0.82: cloud shows as a ratio below 1, however long it stays.
    for _ in range(4):
        assert reference.relative(visit, 0.41) == pytest.approx(0.5)
    assert reference.n_used == 3
    # Negative control: against a running median the cloud would become the reference and read 1.
    assert 0.41 / np.median([0.80, 0.84, 0.82, 0.41, 0.41, 0.41, 0.41]) == pytest.approx(1.0)

    # An exposure with too few stars is compared with the reference but does not enter it.
    few = TransparencyReference(n_first=3)
    assert np.isnan(few.relative(visit, 0.2, n_stars=2))  # nothing to compare with yet
    assert few.relative(visit, 0.8, n_stars=3) == pytest.approx(1.0)
    assert few.relative(visit, 0.4, n_stars=1) == pytest.approx(0.5)
    assert few.n_used == 1
    # Negative control: counted, the two would have moved the reference.
    assert np.median([0.2, 0.8, 0.4]) != pytest.approx(0.8)

    # Another visit starts again, and so does the same visit of another design.
    assert reference.relative((0x1234, 150001), 0.41) == pytest.approx(1.0)
    assert reference.n_used == 1
    assert reference.relative((0x9999, 150001), 0.2) == pytest.approx(1.0)


def test_describe_transparency():
    measured = agMetrics.AgTransparency(transparency=0.8123, nStars=23)
    assert (
        describe_transparency(7, measured)
        == "Transparency: frame_id=7 transparency=0.812 (n=23)"
    )
    assert describe_transparency(7, measured, 0.954, 5).endswith(
        "relative=0.954 (to the visit's first 5 measured)"
    )
    assert "relative" not in describe_transparency(7, measured, float("nan"), 0)
    none = agMetrics.AgTransparency(transparency=float("nan"), nStars=0)
    assert describe_transparency(7, none).startswith("Transparency: frame_id=7 not measured")


@pytest.fixture
def offsets():
    detected, identified, guide, spots, colors = make_frame([0.9, 0.8, 0.7, 0.6])
    return (
        types.SimpleNamespace(detected_objects=detected, identified_objects=identified, guide_objects=guide),
        spots,
        colors,
    )


@pytest.fixture
def queries(monkeypatch, offsets):
    """Stand-ins for the two queries; returns the list of the calls made."""
    _, spots, colors = offsets
    calls = []
    transparency.query_guide_star_colors.cache_clear()

    def fake_query_db(sql, params, **kwargs):
        calls.append(params)
        if "pfs_design_agc" in sql:
            return pd.DataFrame({"guide_star_id": colors.index, "guide_star_color": colors.to_numpy()})
        return spots

    monkeypatch.setattr(transparency, "query_db", fake_query_db)
    yield calls
    transparency.query_guide_star_colors.cache_clear()


def test_log_transparency(offsets, queries, caplog):
    guide_offsets = offsets[0]
    reference = TransparencyReference()
    with caplog.at_level(logging.INFO):
        for frame_id in (100, 101):
            transparency.log_transparency(
                logging.getLogger("test"),
                guide_offsets,
                frame_id=frame_id,
                design_id=0x1234,
                visit_id=5,
                reference=reference,
            )
    lines = [record.getMessage() for record in caplog.records]
    assert lines[0] == (
        "Transparency: frame_id=100 transparency=0.800 (n=3) relative=1.000 (to the visit's first 1 measured)"
    )
    assert lines[1].endswith("relative=1.000 (to the visit's first 2 measured)")
    # The colours are read once per design, the magnitudes for each exposure.
    assert queries == [{"agc_exposure_id": 100}, {"pfs_design_id": 0x1234}, {"agc_exposure_id": 101}]


def test_log_transparency_without_a_design_or_a_reference(offsets, queries, caplog):
    with caplog.at_level(logging.INFO):
        transparency.log_transparency(logging.getLogger("test"), offsets[0], frame_id=100, design_id=None)
        transparency.log_transparency(logging.getLogger("test"), offsets[0], frame_id=101, design_id=0x1234)
    lines = [record.getMessage() for record in caplog.records]
    assert lines[0] == "Transparency: frame_id=100 not measured (no design id)"
    assert lines[1] == "Transparency: frame_id=101 transparency=0.800 (n=3)"


@pytest.mark.parametrize("broken", ["query", "offsets"])
def test_log_transparency_never_raises(broken, offsets, queries, monkeypatch, caplog):
    """A failed query, or offsets without the tables: a warning, no more."""
    guide_offsets = offsets[0]
    if broken == "query":

        def failing(sql, params, **kwargs):
            raise RuntimeError("connection refused")

        monkeypatch.setattr(transparency, "query_db", failing)
    else:
        guide_offsets = types.SimpleNamespace(
            detected_objects=None, identified_objects=None, guide_objects=None
        )
    with caplog.at_level(logging.INFO):
        transparency.log_transparency(
            logging.getLogger("test"),
            guide_offsets,
            frame_id=100,
            design_id=0x1234,
            reference=TransparencyReference(),
        )
    (record,) = caplog.records
    assert record.levelno == logging.WARNING
    assert record.getMessage().startswith("Transparency: frame_id=100 not measured: ")


def test_log_transparency_with_an_earlier_pfs_utils(offsets, queries, monkeypatch, caplog):
    """A pfs_utils without the functions: one warning for the night, no query, and nothing raised."""
    monkeypatch.setattr(transparency, "agMetrics", types.SimpleNamespace())
    monkeypatch.setattr(transparency, "_warned_no_pfs_utils", False)
    with caplog.at_level(logging.INFO):
        for frame_id in (100, 101, 102):
            transparency.log_transparency(
                logging.getLogger("test"), offsets[0], frame_id=frame_id, design_id=0x1234
            )
    (record,) = caplog.records
    assert record.levelno == logging.WARNING
    assert record.getMessage() == "Transparency: not measured, this pfs_utils has no transparency functions"
    assert queries == []
    # Negative control: with the functions there, the same calls query and log each exposure.
    monkeypatch.undo()
    transparency.query_guide_star_colors.cache_clear()
