"""Focus uses the detections matched to guide stars."""

import logging
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from agActor.Commands import AgCmd as agCmdModule
from agActor.utils.data import GuideOffsets
from agActor.utils.focus import focus
from pfs.utils.datamodel.ag import SourceMatchingFlags as M


def _guide_offsets(identified: pd.DataFrame, detected: pd.DataFrame | None = None) -> GuideOffsets:
    if detected is None:
        detected = pd.DataFrame(
            {"agc_camera_id": [0, 0, 1, 1], "spot_id": [1, 2, 1, 2], "flags": [0, 0, 1, 0]},
            index=[10, 11, 12, 13],
        )
    return GuideOffsets(
        ra=0.0, dec=0.0, inst_pa=0.0, ra_offset=0.0, dec_offset=0.0, inr_offset=0.0, scale_offset=0.0,
        dalt=None, daz=None, guide_objects=pd.DataFrame(),
        detected_objects=detected, identified_objects=identified,
        dx=0.0, dy=0.0, size=0.0, peak=0.0, flux=0.0,
    )


def test_matched_detected_objects_keeps_only_good_matches():
    identified = pd.DataFrame(
        {
            "detected_object_id": [10, 11, 12, 13],
            "match_flags": [int(M.GOOD_MATCH), int(M.NO_MATCH), int(M.GOOD_MATCH), int(M.UNUSED_MULTI_MATCH)],
        }
    )
    identified["matched"] = (identified.match_flags == int(M.GOOD_MATCH)).astype(int)
    go = _guide_offsets(identified)

    matched = go.matched_detected_objects
    assert list(matched.index) == [10, 12]
    assert list(matched.columns) == list(go.detected_objects.columns)
    # Negative control: an unmatched spot, flagged or not, never reaches the focus estimate,
    # and the selection is stricter than the detection flags alone.
    assert 11 not in matched.index and 13 not in matched.index
    assert len(matched) < (go.detected_objects["flags"] <= 1).sum()


def test_matched_detected_objects_empty_when_nothing_matched():
    identified = pd.DataFrame(
        {"detected_object_id": [10, 11], "match_flags": [int(M.NO_MATCH)] * 2, "matched": [0, 0]}
    )
    assert _guide_offsets(identified).matched_detected_objects.empty


G, N = int(M.GOOD_MATCH), int(M.NO_MATCH)


def _sparse_field(match_flags: list[int]) -> GuideOffsets:
    """One camera with five unflagged spots, three on the left half and two on the glass-free half."""
    moments = [4.0, 9.0, 9.5, 3.0, 3.1]
    detected = pd.DataFrame(
        {
            "agc_camera_id": 0,
            "spot_id": range(5),
            "central_image_moment_11_pix": 0.0,
            "central_image_moment_20_pix": moments,
            "central_image_moment_02_pix": moments,
            "flags": [0, 0, 0, 1, 1],  # 1 is RIGHT, the glass-free half
        }
    )
    identified = pd.DataFrame({"detected_object_id": range(5), "match_flags": match_flags})
    identified["matched"] = (identified.match_flags == G).astype(int)
    return _guide_offsets(identified, detected)


def _acquire_field(guide_offsets: GuideOffsets) -> tuple[list[float], float]:
    """Run the acquire_field handler on ``guide_offsets``: the six focusErrors and the guideErrors focus."""
    actor = MagicMock()
    actor.actorConfig = {"tel_status": "agc_exposure"}
    actor.logger = logging.getLogger("test_ag")
    actor.agcc.frameId = 1
    actor.agcc.dataTime = 0.0
    controller = MagicMock()
    controller.get_mode.return_value = controller.Mode.OFF
    actor.controllers = {"ag": controller}
    with (
        patch.object(agCmdModule.OpDB, "set_default_connection"),
        patch.object(agCmdModule.GaiaDB, "set_default_connection"),
    ):
        cmd_set = agCmdModule.AgCmd(actor)
    cmd = MagicMock()
    cmd.cmd.keywords = {}

    with (
        patch.object(agCmdModule.field_acquisition, "acquire_field", return_value=guide_offsets),
        patch.object(agCmdModule.time, "sleep"),
        patch.object(agCmdModule, "send_guide_offsets"),
        patch.object(GuideOffsets, "save_numpy_files", return_value=["a", "b", "c"]),
    ):
        cmd_set.acquire_field(cmd)

    cmd.fail.assert_not_called()
    cmd.finish.assert_called_once()
    replies = [c.args[0] for c in cmd.inform.call_args_list]
    (focus_errors,) = [r for r in replies if r.startswith("focusErrors=")]
    (guide_errors,) = [r for r in replies if r.startswith("guideErrors=")]
    # Both start with the frame id; the focus is the seventh value of guideErrors.
    return [float(v) for v in focus_errors.split("=")[1].split(",")[1:]], float(guide_errors.split("=")[1].split(",")[6])


def test_acquire_field_focus_ignores_unmatched_spots():
    # Matched spots on both halves, and two large unmatched spots on the left half.
    guide_offsets = _sparse_field([G, N, N, G, G])
    _, matched_dzs = focus(detected_objects=guide_offsets.matched_detected_objects)
    _, all_dzs = focus(detected_objects=guide_offsets.detected_objects)

    focus_errors, _ = _acquire_field(guide_offsets)

    assert np.isfinite(matched_dzs[0])
    assert focus_errors[0] == pytest.approx(matched_dzs[0])
    # Negative control: the unmatched spots move this camera's estimate, so reporting the
    # all-detections value would fail here.
    assert abs(all_dzs[0] - matched_dzs[0]) > 0.01
    assert focus_errors[0] != pytest.approx(all_dzs[0])


@pytest.mark.parametrize(
    "match_flags",
    [
        [G, N, N, N, N],  # the only matched spot is on the left half
        [N, N, N, N, N],  # nothing matched
    ],
)
def test_acquire_field_focus_is_nan_without_matches_on_both_halves(match_flags):
    guide_offsets = _sparse_field(match_flags)

    focus_errors, guide_focus = _acquire_field(guide_offsets)

    assert np.isnan(focus_errors).all()
    assert np.isnan(guide_focus)
    # Negative control: the spots themselves give a value, so the NaN comes from requiring
    # matches and not from an empty camera; a zero would be read as "in focus".
    _, all_dzs = focus(detected_objects=guide_offsets.detected_objects)
    assert np.isfinite(all_dzs[0])
    assert focus_errors[0] != 0.0
