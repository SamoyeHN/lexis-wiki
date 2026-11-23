"""
Lexis Wiki CLI Manager.
Thin dispatch facade delegating to domain-specific subcommand modules in librarian.cli_core.
"""

import argparse
import sys

from . import __version__
from .cli_core import SUBCOMMAND_MODULES


def build_parser() -> argparse.ArgumentParser:
    """Construct and configure top-level ArgumentParser with all subcommands."""
    parser = argparse.ArgumentParser(description="Lexis Wiki CLI Manager")
    parser.add_argument("--version", "-v", action="version", version=f"lexis v{__version__}")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    for mod in SUBCOMMAND_MODULES:
        mod.register_subparser(subparsers)

    return parser


def main() -> None:
    """CLI entrypoint."""
    for _stream in (sys.stdout, sys.stderr):
        if _stream is not None and hasattr(_stream, "reconfigure"):
            try:
                _stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass

    parser = build_parser()
    args = parser.parse_args()

    if hasattr(args, "handler") and callable(args.handler):
        args.handler(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
