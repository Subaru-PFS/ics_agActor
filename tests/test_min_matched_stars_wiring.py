"""The minimum number of matched stars reaches every place an offset is sent, published or recorded."""

import inspect
import logging
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from agActor.Commands import AgCmd as agCmdModule
from agActor.Controllers import ag as agModule
from agActor.utils.data import GuideOffsetFlag, GuideOffsets
from opscore.protocols import keys, validation
from pfs.utils.agMetrics import AgImageSize
from pfs.utils.datamodel.ag import SourceMatchingFlags

ag = agModule.ag
GOOD = int(SourceMatchingFlags.GOOD_MATCH)
BAD = int(SourceMatchingFlags.BAD_RESIDUAL)
TOO_FEW = GuideOffsetFlag.INVALID_OFFSET | GuideOffsetFlag.TOO_FEW_STARS
DRA, DDEC, DINR, DSCALE = 0.5, -0.3, 3.0, 1e-5


def _guide_offsets(n_good: int, n_good_disabled: int = 0, n_bad: int = 5, dra: float = DRA) -> GuideOffsets:
    """Offsets from ``n_good`` good matches on enabled cameras, with spots on both halves of AG1."""
    match_flags = [GOOD] * (n_good + n_good_disabled) + [BAD] * n_bad
    enabled = [1.0] * n_good + [0.0] * n_good_disabled + [1.0] * n_bad
    n = len(match_flags)
    identified = pd.DataFrame(
        {"detected_object_id": range(n), "match_flags": match_flags, "camera_enabled": enabled}
    )
    identified["matched"] = (identified.match_flags == GOOD).astype(int)
    detected = pd.DataFrame(
        {
            "agc_camera_id": 0,
            "spot_id": range(n),
            "central_image_moment_11_pix": 0.0,
            "central_image_moment_20_pix": 4.0,
            "central_image_moment_02_pix": 4.0,
            "flags": [i % 2 for i in range(n)],
        }
    )
    return GuideOffsets(
        ra=10.0, dec=20.0, inst_pa=0.0, ra_offset=dra, dec_offset=DDEC, inr_offset=DINR, scale_offset=DSCALE,
        dalt=0.1, daz=0.2, guide_objects=pd.DataFrame({"filtered_by": [0] * 8}),
        detected_objects=detected, identified_objects=identified,
        dx=-dra, dy=DDEC, size=1.0, peak=2.0, flux=3.0,
        image_size=AgImageSize(fwhm=0.8, fwhmLeft=0.7, fwhmRight=0.9, nLeft=2, nRight=1),
    )  # fmt: skip


def _actor() -> MagicMock:
    actor = MagicMock()
    actor.actorConfig = {"tel_status": "agc_exposure", "agc_guide_offset": True, "agc_match": True}
    actor.logger = logging.getLogger("test_ag")
    actor.agcc.frameId = 42
    actor.agcc.dataTime = 0.0
    return actor


def _guide_errors(informs) -> list[str]:
    """The values of the one guideErrors keyword among the replies: frame id, seven offsets, status."""
    (keyword,) = [c.args[0] for c in informs.call_args_list if c.args[0].startswith("guideErrors=")]
    return keyword.split("=", 1)[1].split(",")


def _fitted(values: list[str]) -> list[float]:
    """dRA, dDec, dInR, dAz, dAlt and dScale of a guideErrors keyword (everything but the focus)."""
    return [float(v) for v in values[1:6] + values[7:8]]


class _IterationDone(BaseException):
    """Ends `AgThread.run` after one iteration; not an `Exception`, which the loop would catch."""


def _run_loop_once(guide_offsets: GuideOffsets, **options) -> dict:
    """One ONCE iteration of the guide loop on ``guide_offsets``, with everything outside it mocked."""
    actor = _actor()
    thread = agModule.AgThread(actor=actor, logger=actor.logger)
    thread.set_params(
        mode=ag.Mode.ONCE, design=(1, None), visit_id=7, exposure_time=100, cadence=0, cameras=[1, 2, 3],
        **options,
    )  # fmt: skip
    fetch = thread._get_params
    fetches = []

    def fetch_once():
        if fetches:
            raise _IterationDone
        fetches.append(1)
        return fetch()

    thread._get_params = fetch_once
    thread._AgThread__abort = MagicMock()  # no waiting between iterations
    with (
        patch.object(agModule, "get_guide_objects"),
        patch.object(agModule.autoguide, "get_exposure_offsets", return_value=guide_offsets),
        patch.object(agModule, "send_guide_offsets") as send,
        patch.object(agModule, "sendAlert") as alert,
        patch.object(agModule.data_utils, "write_agc_guide_offset") as write_offset,
        patch.object(agModule.data_utils, "write_agc_match"),
        patch.object(GuideOffsets, "save_numpy_files", return_value=["a", "b", "c"]),
        patch.object(agModule.transparency, "log_transparency") as log_transparency,  # it queries the opdb
        patch.object(agModule.time, "sleep"),
        pytest.raises(_IterationDone),
    ):
        thread.run()

    alerts = [c.kwargs["alert_id"] for c in alert.call_args_list]
    # The loop went round without a fatal or runtime error.
    assert not [a for a in alerts if a.startswith("AG.CONTROL_LOOP") and a != alerts[0]]
    (written,) = write_offset.call_args_list
    return {
        "sent": send.called,
        "alerts": alerts[1:],  # the first clears the control-loop alert at start-up
        "written": written.kwargs,
        "guide_errors": _guide_errors(actor.bcast.inform),
        "transparency": log_transparency.call_args_list,
        "thread": thread,
    }


def test_loop_sends_an_offset_from_enough_stars():
    result = _run_loop_once(_guide_offsets(3))

    assert result["sent"]
    assert result["written"]["offset_flags"] == GuideOffsetFlag.OK
    assert result["guide_errors"][8] == "OK"
    assert float(result["guide_errors"][1]) == pytest.approx(DRA)
    assert result["alerts"] == []


IMAGE_SIZE_LINE = "fwhm=0.800 arcsec left=0.700 (n=2) right=0.900 (n=1)"


def test_loop_logs_the_image_size(caplog):
    with caplog.at_level(logging.INFO, logger="test_ag"):
        _run_loop_once(_guide_offsets(3))

    assert f"Image size: frame_id=42 {IMAGE_SIZE_LINE}" in caplog.text


def test_loop_logs_the_transparency():
    """The loop asks for the transparency line once per frame, for its design and visit, with its reference."""
    guide_offsets = _guide_offsets(3)
    result = _run_loop_once(guide_offsets)

    (call,) = result["transparency"]
    assert call.args[1] is guide_offsets
    assert call.kwargs == {
        "frame_id": 42,
        "design_id": 1,
        "visit_id": 7,
        "reference": result["thread"].transparency_reference,
    }
    assert isinstance(result["thread"].transparency_reference, agModule.transparency.TransparencyReference)


def test_loop_does_not_use_an_offset_from_too_few_stars():
    # Three more good matches on disabled cameras do not count.
    result = _run_loop_once(_guide_offsets(2, n_good_disabled=3))

    assert not result["sent"]
    assert result["alerts"] == ["AG.TOO_FEW_STARS"]
    assert result["guide_errors"][8] == "TOO_FEW_STARS"
    assert np.isnan(_fitted(result["guide_errors"])).all()
    assert np.isfinite(float(result["guide_errors"][6]))  # the focus is not from the fit
    # The opdb keeps the offset that was not used, flagged.
    assert result["written"]["offset_flags"] == TOO_FEW
    assert result["written"]["delta_ra"] == pytest.approx(DRA)
    # Negative control: one more star on an enabled camera and the same offset is sent.
    assert _run_loop_once(_guide_offsets(3, n_good_disabled=2))["sent"]


def test_loop_reports_too_few_stars_rather_than_out_of_range():
    result = _run_loop_once(_guide_offsets(0, dra=40.0))

    assert not result["sent"]
    assert result["alerts"] == ["AG.TOO_FEW_STARS"]
    assert result["written"]["offset_flags"] == TOO_FEW
    # Negative control: with enough stars the same offset is out of range, and keeps its values.
    in_range = _run_loop_once(_guide_offsets(3, dra=40.0))
    assert not in_range["sent"]
    assert in_range["alerts"] == ["AG.OFFSET_OUT_OF_RANGE"]
    assert in_range["written"]["offset_flags"] == GuideOffsetFlag.INVALID_OFFSET
    assert float(in_range["guide_errors"][1]) == pytest.approx(40.0)


def test_loop_takes_the_minimum_from_its_options():
    assert _run_loop_once(_guide_offsets(2), min_matched_stars=2)["sent"]
    assert not _run_loop_once(_guide_offsets(3), min_matched_stars=4)["sent"]
    # Negative control: the same two stars are too few for the default.
    assert not _run_loop_once(_guide_offsets(2))["sent"]


def test_params_keep_the_minimum_as_an_option():
    params = ag.Params(mode=ag.Mode.OFF)
    params.set(min_matched_stars=5)
    assert params.get()[-1] == {"min_matched_stars": 5}
    # Negative control: a name that is not an option is not a parameter either, and stops the loop
    # when it fetches its parameters.
    with pytest.raises(AttributeError):
        params.set(min_matched_star=5)


def _run_acquire_field(guide_offsets: GuideOffsets, **keywords) -> dict:
    """The acquire_field handler on ``guide_offsets``, with everything outside it mocked."""
    actor = _actor()
    controller = MagicMock()
    controller.get_mode.return_value = controller.Mode.OFF
    actor.controllers = {"ag": controller}
    with (
        patch.object(agCmdModule.OpDB, "set_default_connection"),
        patch.object(agCmdModule.GaiaDB, "set_default_connection"),
    ):
        cmd_set = agCmdModule.AgCmd(actor)
    cmd = MagicMock()
    cmd.cmd.keywords = {name: MagicMock(values=[value]) for name, value in keywords.items()}
    with (
        patch.object(agCmdModule.field_acquisition, "acquire_field", return_value=guide_offsets),
        patch.object(agCmdModule.time, "sleep"),
        patch.object(agCmdModule, "send_guide_offsets") as send,
        patch.object(agCmdModule.data_utils, "write_agc_guide_offset") as write_offset,
        patch.object(agCmdModule.data_utils, "write_agc_match") as write_match,
        patch.object(GuideOffsets, "save_numpy_files", return_value=["a", "b", "c"]),
    ):
        cmd_set.acquire_field(cmd)
    (written,) = write_offset.call_args_list
    return {
        "sent": send.called,
        "written": written.kwargs,
        "match_written": write_match.called,
        "guide_errors": _guide_errors(cmd.inform),
        "failures": [c.args[0] for c in cmd.fail.call_args_list],
        "warnings": [c.args[0] for c in cmd.warn.call_args_list],
        "finished": cmd.finish.called,
    }


def test_acquire_field_sends_an_offset_from_enough_stars():
    result = _run_acquire_field(_guide_offsets(3))

    assert result["sent"] and result["finished"] and not result["failures"]
    assert result["written"]["offset_flags"] == GuideOffsetFlag.OK
    assert _fitted(result["guide_errors"])[:3] == pytest.approx([DRA, DDEC, DINR])


def test_acquire_field_logs_the_image_size(caplog):
    with caplog.at_level(logging.INFO, logger="test_ag"):
        _run_acquire_field(_guide_offsets(3))

    assert IMAGE_SIZE_LINE in caplog.text


def test_acquire_field_records_but_does_not_use_an_offset_from_too_few_stars():
    result = _run_acquire_field(_guide_offsets(2, n_good_disabled=3))

    assert not result["sent"]
    assert not result["finished"]
    (failure,) = result["failures"]
    assert "2 matched on enabled cameras (need 3)" in failure
    assert result["guide_errors"][8] == "TOO_FEW_STARS"
    assert np.isnan(_fitted(result["guide_errors"])).all()
    # The opdb keeps the offset and its matches.
    assert result["written"]["offset_flags"] == TOO_FEW
    assert result["written"]["delta_ra"] == pytest.approx(DRA)
    assert result["match_written"]
    # Negative control: the same frame passes when the command asks for two stars.
    assert _run_acquire_field(_guide_offsets(2, n_good_disabled=3), min_matched_stars=2)["sent"]


def test_acquire_field_without_guiding_warns_instead_of_failing():
    # A caller that asked for no guiding wanted the frame, not the offset.
    result = _run_acquire_field(_guide_offsets(2), guide=False)

    assert result["finished"] and not result["failures"]
    (warning,) = result["warnings"]
    assert "2 matched on enabled cameras (need 3)" in warning
    # The offset is still not used: nothing sent, NaN published, flagged in the opdb.
    assert not result["sent"]
    assert np.isnan(_fitted(result["guide_errors"])).all()
    assert result["written"]["offset_flags"] == TOO_FEW
    # Negative control: asked to guide, which is the default, the same frame fails the command.
    for keywords in ({}, {"guide": True}):
        guiding = _run_acquire_field(_guide_offsets(2), **keywords)
        assert guiding["failures"] and not guiding["finished"] and not guiding["warnings"]


@pytest.fixture(scope="module")
def ag_commands():
    """The AgCmd command set behind the opscore validator, wired as actorcore's attachCmdSet does."""
    actor = _actor()
    with (
        patch.object(agCmdModule.OpDB, "set_default_connection"),
        patch.object(agCmdModule.GaiaDB, "set_default_connection"),
    ):
        cmd_set = agCmdModule.AgCmd(actor)
    keys.CmdKey.addKeys(cmd_set.keys)
    handler = validation.CommandHandler()
    handler.addConsumers(
        *(validation.Cmd(verb, args, help=inspect.getdoc(func)) >> func for verb, args, func in cmd_set.vocab)
    )
    return actor, handler


def _controller_kwargs(ag_commands, cmd_str: str, method: str) -> dict:
    """What the controller's ``method`` is called with when the actor receives ``cmd_str``."""
    actor, handler = ag_commands
    controller = MagicMock()
    actor.controllers = {"ag": controller}
    validated, funcs = handler.match(cmd_str)
    cmd = MagicMock()
    cmd.cmd = validated
    for func in funcs:
        func(cmd)
    cmd.fail.assert_not_called()
    return getattr(controller, method).call_args.kwargs


@pytest.mark.parametrize(
    "sub, method",
    [
        ("start", "start_autoguide"),
        ("initialize", "initialize_autoguide"),
        ("reconfigure", "reconfigure_autoguide"),
    ],
)
def test_autoguide_commands_pass_the_minimum_on(ag_commands, sub, method):
    command = f"autoguide {sub} exposure_time=3000"
    kwargs = _controller_kwargs(ag_commands, f"{command} min_matched_stars=4", method)
    assert kwargs["min_matched_stars"] == 4
    assert isinstance(kwargs["min_matched_stars"], int)
    # Negative control: a command without the keyword leaves the loop its default.
    assert "min_matched_stars" not in _controller_kwargs(ag_commands, command, method)


def test_acquire_field_accepts_the_minimum(ag_commands):
    _, handler = ag_commands
    validated, funcs = handler.match("acquire_field design_id=0x1 min_matched_stars=4")
    assert [func.__name__ for func in funcs] == ["acquire_field"]
    assert int(validated.keywords["min_matched_stars"].values[0]) == 4
    # Negative control: the validator refuses a keyword that is not in the vocabulary.
    with pytest.raises(validation.ValidationError):
        handler.match("acquire_field design_id=0x1 min_matched_star=4")


def _run_focus(detected: pd.DataFrame) -> MagicMock:
    """The ``ag focus`` handler on the spots ``detected``; returns the command, to read its replies."""
    actor = _actor()
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
        patch.object(agCmdModule.data_utils, "query_agc_data", return_value=detected),
        patch.object(agCmdModule.data_utils, "write_agc_guide_offset"),
        patch.object(agCmdModule.time, "sleep"),
    ):
        cmd_set.focus(cmd)
    return cmd


def test_focus_command_fails_with_the_reason():
    # Spots on the left half of AG1 only: no camera has a focus value.
    one_half = _guide_offsets(3).detected_objects.assign(flags=0)
    cmd = _run_focus(one_half)

    (failure,) = [c.args[0] for c in cmd.fail.call_args_list]
    assert "no focus value for frame_id=42" in failure
    assert "AG1: 0 stars on the right half (8 on the left)" in failure
    cmd.finish.assert_not_called()
    # Negative control: with spots on both halves the command gives a value and does not fail.
    cmd = _run_focus(_guide_offsets(3).detected_objects)
    cmd.fail.assert_not_called()
    assert [c.args[0] for c in cmd.inform.call_args_list if c.args[0].startswith("focusErrors=")]
