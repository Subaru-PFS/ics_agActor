"""``filter_bad_shape`` is accepted but ignored: ``yes`` gets an inform, ``no`` a warning."""

import inspect
import logging
from unittest.mock import MagicMock, patch

import pytest
from agActor.Commands import AgCmd as agCmdModule
from agActor.Commands.AgCmd import ignore_filter_bad_shape
from opscore.protocols import keys, validation

# The forms the senders use, bare and quoted, with whether BAD_SHAPE filtering is asked for.
SENT_FORMS = {
    "filter_bad_shape=yes": True,
    "filter_bad_shape=no": False,
    'filter_bad_shape="yes"': True,
    'filter_bad_shape="no"': False,
}
AUTOGUIDE = {
    "start": "start_autoguide",
    "initialize": "initialize_autoguide",
    "reconfigure": "reconfigure_autoguide",
}


def _cmd(**keywords):
    cmd = MagicMock()
    cmd.cmd.keywords = {name: MagicMock(values=[value]) for name, value in keywords.items()}
    return cmd


def _mentions(mock_reply):
    """The calls to a reply method (``cmd.warn``, ``cmd.inform``) that are about the keyword."""
    return [call.args[0] for call in mock_reply.call_args_list if "filter_bad_shape" in call.args[0]]


def test_yes_informs_and_no_warns():
    cmd = _cmd(filter_bad_shape=True)
    ignore_filter_bad_shape(cmd)
    assert len(_mentions(cmd.inform)) == 1 and "deprecated" in _mentions(cmd.inform)[0]
    # Negative control: asking for what the actor now always does is not worth a warning.
    cmd.warn.assert_not_called()

    cmd = _cmd(filter_bad_shape=False)
    ignore_filter_bad_shape(cmd)
    assert len(_mentions(cmd.warn)) == 1 and "ignored" in _mentions(cmd.warn)[0]
    cmd.inform.assert_not_called()
    cmd.fail.assert_not_called()


def test_no_keyword_no_reply():
    # Negative control: a command without the keyword is left alone.
    cmd = _cmd(exposure_time=2)
    ignore_filter_bad_shape(cmd)
    cmd.warn.assert_not_called()
    cmd.inform.assert_not_called()


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

    # yes is what the actor does anyway: an inform. no cannot be honoured: a warning.
    wants_filter = SENT_FORMS[form]
    assert len(_mentions(cmd.inform)) == (1 if wants_filter else 0)
    assert len(_mentions(cmd.warn)) == (0 if wants_filter else 1)
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
