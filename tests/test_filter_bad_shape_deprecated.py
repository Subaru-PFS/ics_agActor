"""``filter_bad_shape`` is accepted but ignored with a warning (INSTRM-2841)."""

from unittest.mock import MagicMock

from agActor.Commands.AgCmd import ignore_filter_bad_shape


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
