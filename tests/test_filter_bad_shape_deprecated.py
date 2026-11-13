"""``filter_bad_shape`` is accepted but ignored with a warning."""

import inspect
import logging
from unittest.mock import MagicMock, patch

import pytest
from agActor.Commands import AgCmd as agCmdModule
from agActor.Commands.AgCmd import ignore_filter_bad_shape
from opscore.protocols import keys, validation

# The forms the senders use: iic forwards the value bare, the Gen2 scripts quote it.
SENT_FORMS = ("filter_bad_shape=yes", "filter_bad_shape=no", 'filter_bad_shape="yes"', 'filter_bad_shape="no"')
AUTOGUIDE = {
    "start": "start_autoguide",
    "initialize": "initialize_autoguide",
    "reconfigure": "reconfigure_autoguide",
}


def _cmd(**keywords):
    cmd = MagicMock()
    cmd.cmd.keywords = keywords
    return cmd


def test_keyword_warns_and_is_ignored():
    for value in ("yes", "no"):
        cmd = _cmd(filter_bad_shape=value)
        ignore_filter_bad_shape(cmd)
        cmd.warn.assert_called_once()
        assert "deprecated" in cmd.warn.call_args.args[0]
        cmd.fail.assert_not_called()


def test_no_keyword_no_warning():
    # Negative control: a command without the keyword is left alone.
    cmd = _cmd(exposure_time=2)
    ignore_filter_bad_shape(cmd)
    cmd.warn.assert_not_called()


@pytest.fixture(scope="module")
def ag_commands():
    """The AgCmd command set behind the opscore validator, wired as actorcore's attachCmdSet does."""
    actor = MagicMock()
    actor.actorConfig = {"tel_status": "agc_exposure"}
    actor.logger = logging.getLogger("test_ag")
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


def _match(handler, cmd_str):
    """The validated command and its handlers, or None when the validator rejects the command."""
    try:
        validated, funcs = handler.match(cmd_str)
    except validation.ValidationError:
        return None
    return (validated, funcs) if validated else None


@pytest.mark.parametrize("form", SENT_FORMS)
@pytest.mark.parametrize("sub", AUTOGUIDE)
def test_autoguide_accepts_and_drops_the_keyword(ag_commands, sub, form):
    actor, handler = ag_commands
    controller = MagicMock()
    actor.controllers = {"ag": controller}

    matched = _match(handler, f"autoguide {sub} exposure_time=3000 cadence=5000 {form}")
    assert matched is not None, "the validator rejected a command the senders use"
    validated, funcs = matched
    cmd = MagicMock()
    cmd.cmd = validated
    for func in funcs:
        func(cmd)

    cmd.warn.assert_called_once()
    cmd.fail.assert_not_called()
    cmd.finish.assert_called_once()
    call = getattr(controller, AUTOGUIDE[sub]).call_args
    assert call.kwargs["exposure_time"] == 3000
    assert "filter_bad_shape" not in call.kwargs


@pytest.mark.parametrize("form", SENT_FORMS)
def test_acquire_field_accepts_the_keyword(ag_commands, form):
    _, handler = ag_commands
    matched = _match(handler, f"acquire_field design_id=0x1 exposure_time=2000 {form}")
    assert matched is not None
    assert [func.__name__ for func in matched[1]] == ["acquire_field"]


def test_validator_rejects_unknown_keywords(ag_commands):
    # Negative control: the validator refuses a keyword that is not in the vocabulary, so the
    # tests above fail if filter_bad_shape is ever dropped from it while the senders still use it.
    _, handler = ag_commands
    assert _match(handler, "autoguide start exposure_time=3000") is not None
    assert _match(handler, "autoguide start exposure_time=3000 filter_bad_shapes=yes") is None
    assert _match(handler, "acquire_field design_id=0x1 filter_bad_shapes=yes") is None
