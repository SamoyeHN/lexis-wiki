"""
CLI subparser and handler for wiki link checking, tag normalization, and orphan pruning.
"""

import argparse
import os
import sys

from ..config import config
from ..linter import linter


def register_subparser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("lint", help="Check wiki for broken links and normalize tags")
    parser.add_argument("--fix-tags", action="store_true", help="Normalize tags to kebab-case")
    parser.add_argument(
        "--prune",
        action="store_true",
        help="Remove orphan extractions and quizzes whose raw source files are missing",
    )
    parser.set_defaults(handler=handle_lint)


def handle_lint(args: argparse.Namespace) -> None:
    print("Checking links...")
    broken = linter.check_links()
    if broken:
        print(f"Found {len(broken)} broken links:")
        for f, l in broken:
            print(f"  {f} -> [[{l}]]")
    else:
        print("No broken links found.")

    if args.fix_tags:
        print("Normalizing tags...")
        count = linter.normalize_tags()
        print(f"Updated {count} files.")

    if args.prune:
        print("Scanning for orphan extractions and quizzes...")
        orphans = linter.find_orphans()
        if not orphans:
            print("No orphan files found.")
        else:
            print(f"Found {len(orphans)} orphan file(s):")
            for o in orphans:
                try:
                    rel_o = os.path.relpath(o, config.project_root)
                except ValueError:
                    rel_o = o
                print(f"  - {rel_o}")

            try:
                response = input("\nAre you sure you want to delete these files? [y/N]: ").strip().lower()
            except KeyboardInterrupt:
                print("\nPrune cancelled.")
                sys.exit(0)
            except Exception:
                response = "no"

            if response in ["y", "yes"]:
                deleted_count = 0
                for o in orphans:
                    try:
                        os.remove(o)
                        deleted_count += 1
                    except Exception as e:
                        print(f"Error deleting {o}: {e}")
                print(f"Successfully pruned {deleted_count} file(s).")
            else:
                print("Prune cancelled.")
