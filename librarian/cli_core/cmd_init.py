"""
CLI subparser and handler for project initialization.
"""

import argparse
import sys
from pathlib import Path

from ..config import config


def register_subparser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("init", help="Initialize a new wiki project")
    parser.add_argument("path", nargs="?", default=".", help="Target directory (default: current directory)")
    parser.set_defaults(handler=handle_init)


def handle_init(args: argparse.Namespace) -> None:
    print(f"Initializing new wiki project in {args.path}...")
    success, message = config.initialize_project(args.path)
    print(message)
    if not success:
        print("Initialization failed.")
        sys.exit(1)
    if not config.get("disable_pedlex", False):
        from ..linguistics_core.pedlex_dist import ensure_pedlex_database

        db_path = Path(__file__).resolve().parent.parent / "data" / "pedlex.db"
        if not db_path.exists():
            print("[PedLex] Downloading and extracting pedagogical lexicon database (pedlex.db)...")
        ok = ensure_pedlex_database(db_path, force_check=True)
        if ok and db_path.exists():
            print(f"[PedLex] Lexicon database ready ({db_path.stat().st_size / (1024 * 1024):.1f} MB).")
        else:
            print("[PedLex] Warning: Could not download pedlex.db (running in open-source WordNet fallback mode).")

    # Pre-warm offline NLP assets (spaCy 'en_core_web_sm' & Open English WordNet 'oewn:2025+')
    try:
        from ..linguistics import LinguisticEngine

        print("[NLP] Verifying spaCy model (en_core_web_sm) & WordNet (oewn:2025+)...")
        LinguisticEngine.get_spacy()
        LinguisticEngine.get_wordnet()
        print("[NLP] Offline syntactic & lexical models ready.")
    except Exception as nlp_err:
        print(f"[NLP] Warning: Could not pre-warm NLP models ({nlp_err}).")
