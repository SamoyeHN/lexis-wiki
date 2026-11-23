"""
CLI subparser and handler for file compilation and wiki RAG querying.
"""

import argparse
import sys

from ..processor import processor


def register_subparser(subparsers: argparse._SubParsersAction) -> None:
    # Compile command
    compile_parser = subparsers.add_parser("compile", help="Process a raw file and generate wiki nodes")
    compile_parser.add_argument(
        "filename",
        help="Path to a raw source file (e.g. d:\\book 3\\unit 1.md) or name of an existing unit under wiki/",
    )
    compile_parser.add_argument(
        "--categories",
        "-c",
        help="Comma-separated list of categories to compile (e.g., vocabulary,grammar,summary,mindmap)",
    )
    compile_parser.set_defaults(handler=handle_compile)

    # Ask command
    ask_parser = subparsers.add_parser("ask", help="Query the wiki and raw files using full-file context (RAG)")
    ask_parser.add_argument("query", help="Your question about the units, vocabulary, or concepts")
    ask_parser.set_defaults(handler=handle_ask)


def handle_compile(args: argparse.Namespace) -> None:
    cats = [c.strip() for c in args.categories.split(",")] if args.categories else None
    print(f"Compiling {args.filename} via modular pipeline (categories: {cats or 'all'})...")
    status, saved_files = processor.run_pipeline(args.filename, categories=cats)

    if "Error" in status or "failed tasks:" in status:
        print(f"\n❌ {status}")
        if saved_files:
            print("Partially saved files before/outside failure:")
            for path in saved_files:
                print(f"  - {path}")
        if "timeout" in status.lower():
            print("\nTip: If you experience timeouts, you can increase the timeout in 'librarian/llm.py'.")
        sys.exit(1)
    else:
        print(f"\n{status}")
        if saved_files:
            for path in saved_files:
                print(f"  - {path}")
        else:
            print("No files were saved. Check the LLM output for formatting issues.")
            sys.exit(1)


def handle_ask(args: argparse.Namespace) -> None:
    print(f"Querying Wiki: {args.query}...")
    answer = processor.ask_wiki(args.query)
    print("\n" + "=" * 50)
    print(answer)
    print("=" * 50 + "\n")
    if isinstance(answer, str) and answer.startswith("Error"):
        sys.exit(1)
