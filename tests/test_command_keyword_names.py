"""The design, exposure time and visit keywords are spelled as iic spells them: designId, exptime, visit.

``exptime`` is in milliseconds, as in ``iic acquireField``. The names they had before (``design_id``,
``exposure_time``, ``visit_id``) are no longer commands the actor accepts.
"""

import inspect
import logging
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from agActor.Commands import AgCmd as agCmdModule
from agActor.utils.data import GuideOffsets
from opscore.protocols import keys, validation
from pfs.utils.datamodel.ag import SourceMatchingFlags

AUTOGUIDE = {
    "start": "start_autoguide",
    "initialize": "initialize_autoguide",
    "reconfigure": "reconfigure_autoguide",
}
# The names before the rename, with the name each has now.
RENAMED = {"design_id": "designId", "exposure_time": "exptime", "visit_id": "visit"}


def _actor() -> MagicMock:
    actor = MagicMock()
    actor.actorConfig = {"tel_status": "agc_exposure", "agc_guide_offset": True, "agc_match": True}
    actor.logger = logging.getLogger("test_ag")
    actor.agcc.frameId = 42
    actor.agcc.dataTime = 0.0
    return actor


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


def _accepted(handler, cmd_str: str) -> bool:
    try:
        validated, _ = handler.match(cmd_str)
    except validation.ValidationError:
        return False
    return bool(validated)


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
    kwargs = dict(getattr(controller, method).call_args.kwargs)
    kwargs.pop("cmd")
    return kwargs


@pytest.mark.parametrize("sub", ["start", "initialize"])
def test_autoguide_start_takes_design_exposure_time_and_visit(ag_commands, sub):
    kwargs = _controller_kwargs(
        ag_commands, f"autoguide {sub} designId=0x1f exptime=3000 visit=7 cadence=5000", AUTOGUIDE[sub]
    )
    assert kwargs["design"] == (0x1F, None)
    assert kwargs["exposure_time"] == 3000 and isinstance(kwargs["exposure_time"], int)
    assert kwargs["visit_id"] == 7 and isinstance(kwargs["visit_id"], int)
    # Negative control: a command without the keywords gets the defaults, not these values.
    bare = _controller_kwargs(ag_commands, f"autoguide {sub} cadence=5000", AUTOGUIDE[sub])
    assert bare["design"] is None and bare["exposure_time"] == 2000 and bare["visit_id"] is None


def test_autoguide_reconfigure_takes_exposure_time_and_visit(ag_commands):
    kwargs = _controller_kwargs(
        ag_commands, "autoguide reconfigure exptime=3000 visit=7", "reconfigure_autoguide"
    )
    assert kwargs == {"exposure_time": 3000, "visit_id": 7}
    # Negative control: reconfigure changes only what it is given.
    assert _controller_kwargs(ag_commands, "autoguide reconfigure cadence=5000", "reconfigure_autoguide") == {
        "cadence": 5000
    }


def test_exptime_is_milliseconds_with_a_minimum(ag_commands):
    for sub, method in AUTOGUIDE.items():
        assert _controller_kwargs(ag_commands, f"autoguide {sub} exptime=50", method)["exposure_time"] == 100
        # Negative control: a time above the minimum is kept.
        assert _controller_kwargs(ag_commands, f"autoguide {sub} exptime=150", method)["exposure_time"] == 150


@pytest.mark.parametrize(
    "command",
    [
        "acquire_field design_id=1",
        "acquire_field exposure_time=2000",
        "acquire_field visit_id=7",
        "focus exposure_time=2000",
        "focus visit_id=7",
        "autoguide start design_id=1",
        "autoguide start exposure_time=2000",
        "autoguide start visit_id=7",
        "autoguide initialize design_id=1",
        "autoguide initialize exposure_time=2000",
        "autoguide initialize visit_id=7",
        "autoguide reconfigure exposure_time=2000",
        "autoguide reconfigure visit_id=7",
    ],
)
def test_the_names_before_the_rename_are_rejected(ag_commands, command):
    _, handler = ag_commands
    assert not _accepted(handler, command)
    # Negative control: the same command under the keyword's name now is accepted.
    verb, keyword = command.rsplit(" ", 1)
    old, value = keyword.split("=")
    assert _accepted(handler, f"{verb} {RENAMED[old]}={value}")


@pytest.mark.parametrize(
    "command",
    [
        # As iic builds them: its keywords, then the visit.
        (
            "acquire_field designId=1234567890123456789 exptime=2000 guide=no magnitude=20.0 dry_run=yes"
            " fit_dscale=yes fit_dinr=no exposure_delay=100 tec_off=yes visit0=6 visit=7"
        ),
        "acquire_field designId=1234567890123456789 exptime=2000 fit_dScale=yes fit_dInR=no visit=7",
        (
            "autoguide start designId=1234567890123456789 exptime=2000 cadence=5000 center=10.0,20.0,0.0"
            " magnitude=20.0 from_sky=yes dry_run=yes fit_dscale=yes fit_dinr=no exposure_delay=100 tec_off=yes"
            " max_correction=10.0 visit0=6 filter_bad_shape=yes visit=7"
        ),
        "autoguide reconfigure visit=7",
        # As the Gen2 reconfigure script builds it, with its two spaces.
        "autoguide reconfigure  exptime=3000 cadence=5000 max_correction=10 filter_bad_shape=yes",
        "focus exptime=2000 exposure_delay=100 tec_off=no",
        # The case of a keyword does not matter.
        "acquire_field DESIGNID=1 EXPTIME=2000 VISIT=7",
    ],
)
def test_commands_from_iic_and_gen2_are_accepted(ag_commands, command):
    _, handler = ag_commands
    assert _accepted(handler, command)
    # Negative control: the validator refuses a keyword that is not in the vocabulary.
    assert not _accepted(handler, f"{command} exposureTime=2000")


def _guide_offsets() -> GuideOffsets:
    """Offsets from five good matches, with spots on both halves of AG1."""
    n = 5
    good = int(SourceMatchingFlags.GOOD_MATCH)
    identified = pd.DataFrame(
        {"detected_object_id": range(n), "match_flags": good, "camera_enabled": 1.0, "matched": 1}
    )
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
        ra=10.0, dec=20.0, inst_pa=0.0, ra_offset=0.5, dec_offset=-0.3, inr_offset=3.0, scale_offset=1e-5,
        dalt=0.1, daz=0.2, guide_objects=pd.DataFrame({"filtered_by": [0] * 8}),
        detected_objects=detected, identified_objects=identified,
        dx=-0.5, dy=-0.3, size=1.0, peak=2.0, flux=3.0,
    )  # fmt: skip


def _run_acquire_field(ag_commands, keywords: str) -> dict:
    """What ``acquire_field <keywords>`` asks of agcc and the matcher, and writes to agc_match."""
    actor, handler = ag_commands
    actor.queueCommand.reset_mock()
    controller = MagicMock()
    controller.get_mode.return_value = controller.Mode.OFF
    actor.controllers = {"ag": controller}
    validated, funcs = handler.match(f"acquire_field {keywords}")
    cmd = MagicMock()
    cmd.cmd = validated
    with (
        patch.object(
            agCmdModule.field_acquisition, "acquire_field", return_value=_guide_offsets()
        ) as acquire,
        patch.object(agCmdModule.time, "sleep"),
        patch.object(agCmdModule, "send_guide_offsets"),
        patch.object(agCmdModule.data_utils, "write_agc_guide_offset"),
        patch.object(agCmdModule.data_utils, "write_agc_match") as write_match,
        patch.object(GuideOffsets, "save_numpy_files", return_value=["a", "b", "c"]),
    ):
        for func in funcs:
            func(cmd)
    cmd.fail.assert_not_called()
    cmd.finish.assert_called_once()
    return {
        "agcc": actor.queueCommand.call_args.kwargs["cmdStr"],
        "design_id": acquire.call_args.kwargs["design_id"],
        "match_design_id": write_match.call_args.kwargs["design_id"],
    }


def test_acquire_field_takes_design_exposure_time_and_visit(ag_commands):
    result = _run_acquire_field(ag_commands, "designId=0x1f exptime=3000 visit=7")
    # agcc takes its exposure time in seconds.
    assert result["agcc"] == "expose object exptime=3.0 centroid=1 visit=7 threadDelay=100"
    assert result["design_id"] == result["match_design_id"] == 0x1F
    # Negative control: without the keywords, a 2 s exposure with no visit and no design.
    bare = _run_acquire_field(ag_commands, "magnitude=20.0")
    assert bare["agcc"] == "expose object exptime=2.0 centroid=1 threadDelay=100"
    assert bare["design_id"] is None


def test_focus_takes_exposure_time_and_visit(ag_commands):
    actor, handler = ag_commands
    controller = MagicMock()
    controller.get_mode.return_value = controller.Mode.OFF
    actor.controllers = {"ag": controller}
    sent = []
    for keywords in ("exptime=3000 visit=7", ""):
        actor.queueCommand.reset_mock()
        validated, funcs = handler.match(f"focus {keywords}")
        cmd = MagicMock()
        cmd.cmd = validated
        with (
            patch.object(
                agCmdModule.data_utils, "query_agc_data", return_value=_guide_offsets().detected_objects
            ),
            patch.object(agCmdModule.data_utils, "write_agc_guide_offset"),
        ):
            for func in funcs:
                func(cmd)
        cmd.fail.assert_not_called()
        sent.append(actor.queueCommand.call_args.kwargs["cmdStr"])

    with_keywords, bare = sent
    assert with_keywords == "expose object exptime=3.0 centroid=1 visit=7 threadDelay=100"
    # Negative control: without the keywords, the default exposure and no visit.
    assert bare == "expose object exptime=2.0 centroid=1 threadDelay=100"
