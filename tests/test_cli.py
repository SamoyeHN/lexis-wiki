"""Smoke and contract tests for librarian.cli."""

import subprocess
import sys
from unittest.mock import patch
import pytest

from librarian import __version__
from librarian.cli import main


def test_cli_version_flag(capsys):
    with patch.object(sys, "argv", ["lexis", "--version"]):
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert f"lexis v{__version__}" in captured.out or f"lexis v{__version__}" in captured.err


def test_cli_help_flag(capsys):
    with patch.object(sys, "argv", ["lexis", "--help"]):
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert "Lexis Wiki CLI Manager" in captured.out
    assert "Commands" in captured.out


def test_cli_no_args_shows_help(capsys):
    with patch.object(sys, "argv", ["lexis"]):
        main()
    captured = capsys.readouterr()
    assert "usage:" in captured.out.lower() or "commands" in captured.out.lower()


def test_cli_config_show_smoke(capsys):
    with patch.object(sys, "argv", ["lexis", "config", "--show"]):
        main()
    captured = capsys.readouterr()
    assert "Current Configuration" in captured.out or "model" in captured.out.lower()
