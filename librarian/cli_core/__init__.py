"""
CLI core subcommands package.
Each module exposes a register_subparser(subparsers) function.
"""

from . import (
    cmd_audit,
    cmd_compile,
    cmd_config,
    cmd_dashboard,
    cmd_export,
    cmd_init,
    cmd_lint,
    cmd_pedlex,
    cmd_quiz,
)

SUBCOMMAND_MODULES = [
    cmd_init,
    cmd_lint,
    cmd_config,
    cmd_compile,
    cmd_quiz,
    cmd_dashboard,
    cmd_export,
    cmd_audit,
    cmd_pedlex,
]

__all__ = [
    "SUBCOMMAND_MODULES",
    "cmd_audit",
    "cmd_compile",
    "cmd_config",
    "cmd_dashboard",
    "cmd_export",
    "cmd_init",
    "cmd_lint",
    "cmd_pedlex",
    "cmd_quiz",
]
