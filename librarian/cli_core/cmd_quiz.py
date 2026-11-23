"""
CLI subparser and handler for interactive HTML quiz generation.
"""

import argparse
import sys

from ..config import config
from ..processor import processor


def register_subparser(subparsers: argparse._SubParsersAction) -> None:
    quiz_parser = subparsers.add_parser(
        "quiz",
        help="Generate an interactive HTML handout (vocabulary, reading, translation, or listening quiz)",
    )
    quiz_parser.add_argument("filename", help="Name of the unit (e.g., Book_4_Unit_1) or the vocabulary file")
    quiz_parser.add_argument("--count", type=int, help="Number of questions to generate (uses config defaults if omitted)")
    quiz_parser.add_argument(
        "--template",
        "-t",
        default="vocabulary",
        choices=["vocabulary", "cloze", "reading", "translation", "listening", "video"],
        help="Template type to generate (default: vocabulary)",
    )
    quiz_parser.add_argument(
        "--archetype",
        "-a",
        default=None,
        help="Functional dialogue archetype to force for listening quiz (e.g. 4, arch_04, advice, seminar, options, complaint)",
    )
    quiz_parser.add_argument(
        "--scenario",
        "-s",
        default=None,
        help="Everyday scenario ID or setting keyword to force for listening quiz (e.g. arch_01_a2_inf_sc01, arch_04_b1p_form_sc01, weekend)",
    )
    quiz_parser.set_defaults(handler=handle_quiz)


def handle_quiz(args: argparse.Namespace) -> None:
    target_count = args.count
    if target_count is None:
        defaults = config.get("quiz_defaults") or {}
        tmpl_val = defaults.get(args.template, 10)
        if isinstance(tmpl_val, dict):
            target_count = tmpl_val.get("count", 10)
        else:
            target_count = tmpl_val or 10

    print(f"Generating quiz from {args.filename} (Target: {target_count} questions, Template: {args.template})...")
    result = processor.generate_quiz(
        args.filename,
        count=target_count,
        template_name=args.template,
        listening_archetype=getattr(args, "archetype", None),
        listening_scenario=getattr(args, "scenario", None),
    )

    if "Error" in result:
        print(f"\n❌ {result}")
        sys.exit(1)
    else:
        print("\nSuccessfully generated quiz handout:")
        print(f"  - {result}")
