"""
CLI subparser and handler for pedagogical lexicon management (pedlex).
"""

import argparse
import sys
from pathlib import Path

from ..linguistics_core.pedlex_dist import (
    compute_file_sha256,
    ensure_pedlex_database,
    get_pedlex_distribution_config,
    pack_pedlex_archive,
)


def register_subparser(subparsers: argparse._SubParsersAction) -> None:
    pedlex_parser = subparsers.add_parser("pedlex", help="Pack pedlex.db into a compressed .xz archive or sync/verify from mirrors")
    pedlex_parser.add_argument("--pack", action="store_true", help="VACUUM and compress librarian/data/pedlex.db into dist/pedlex.db.xz")
    pedlex_parser.add_argument("--sync", action="store_true", help="Download and extract pedlex.db from configured mirrors (Hugging Face / Cloudflare R2)")
    pedlex_parser.add_argument("--verify", action="store_true", help="Verify local pedlex.db integrity and SHA-256 against expected_sha256")
    pedlex_parser.add_argument("--preset", type=int, default=6, choices=range(0, 10), help="LZMA compression preset 0-9 (default: 6)")
    pedlex_parser.set_defaults(handler=handle_pedlex)


def handle_pedlex(args: argparse.Namespace) -> None:
    db_path = Path(__file__).resolve().parent.parent / "data" / "pedlex.db"
    if args.pack:
        manifest = pack_pedlex_archive(db_path=db_path, preset=args.preset, update_config=True)
        print("\n✅ PedLex Archive Manifest:")
        print(f"   * Uncompressed: {manifest['uncompressed_mb']} MB (SHA-256: {manifest['uncompressed_sha256']})")
        print(f"   * Compressed:   {manifest['compressed_mb']} MB (SHA-256: {manifest['compressed_sha256']})")
        print(f"   * Ratio:        {manifest['compression_ratio']}x")
    elif args.sync:
        ok = ensure_pedlex_database(db_path, force_check=True)
        if not ok:
            print("❌ Failed to sync pedlex.db from mirrors.")
            sys.exit(1)
        print(f"✅ pedlex.db is ready at {db_path}")
    else:
        dist_cfg = get_pedlex_distribution_config()
        if not db_path.exists():
            print(f"⚠️  pedlex.db is missing at {db_path}")
            sys.exit(1)
        sha = compute_file_sha256(db_path)
        size_mb = db_path.stat().st_size / (1024 * 1024)
        expected = dist_cfg.get("expected_sha256", "")
        print(f"📦 Local pedlex.db: {size_mb:.2f} MB")
        print(f"🔑 SHA-256:        {sha}")
        if expected:
            if sha == expected:
                print("✅ Matches built-in expected_sha256.")
            else:
                print(f"⚠️  Differs from expected_sha256 ({expected}).")
                sys.exit(1)
