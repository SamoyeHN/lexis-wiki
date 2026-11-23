"""
CLI subparser and handler for video-import, export-bundle, and rename commands.
"""

import argparse
import os
import sys

from ..config import config, normalize_name


def register_subparser(subparsers: argparse._SubParsersAction) -> None:
    # Video Import command
    video_import_parser = subparsers.add_parser("video-import", help="Import a video transcript (YouTube/Local File) into the self-contained unit structure")
    video_import_parser.add_argument("url_or_filepath", help="The URL of YouTube or local MP4/audio file path")
    video_import_parser.add_argument("--name", help="Custom name for the unit (optional)")
    video_import_parser.add_argument("--cookies-from-browser", help="Extract cookies from browser (e.g. chrome, edge, firefox, safari)")
    video_import_parser.add_argument("--cookies", help="Path to a cookies.txt file to bypass blocks/logins")
    video_import_parser.add_argument("--subtitle", help="Path to a local .srt or .vtt subtitle file (optional)")
    video_import_parser.set_defaults(handler=handle_video_import)

    # Export Portable Bundle command (with export-exe alias for backwards compatibility)
    export_exe_parser = subparsers.add_parser(
        "export-bundle",
        aliases=["export-exe"],
        help="Bundle an HTML video quiz and its companion media into a portable ZIP archive or directory",
    )
    export_exe_parser.add_argument("html_path", help="Path to the HTML quiz handout")
    export_exe_parser.add_argument("media_path", nargs="?", help="Path to the companion video/audio file (optional)")
    export_exe_parser.add_argument("--output", "-o", help="Custom output .zip path or folder path (optional)")
    export_exe_parser.add_argument("--as-dir", action="store_true", help="Export as an uncompressed portable directory instead of a .zip archive")
    export_exe_parser.set_defaults(handler=handle_export_bundle)

    # Rename command
    rename_parser = subparsers.add_parser("rename", help="Atomically rename a unit folder and its internal files")
    rename_parser.add_argument("old_name", help="Current name of the unit (e.g., Book_4_Unit_1)")
    rename_parser.add_argument("new_name", help="New name of the unit (e.g., Book_4_Unit_1_New)")
    rename_parser.set_defaults(handler=handle_rename)


def handle_video_import(args: argparse.Namespace) -> None:
    print(f"Importing video source: {args.url_or_filepath}...")
    try:
        from ..video import video_service
        saved_file, has_subtitles = video_service.import_video(
            args.url_or_filepath,
            custom_name=args.name,
            cookies_from_browser=args.cookies_from_browser,
            cookies=args.cookies,
            subtitle=args.subtitle,
        )
        relative_saved = os.path.relpath(saved_file, config.project_root)
        if has_subtitles:
            print("\nSuccessfully imported video with subtitles!")
            print("Now, you can compile this video unit to extract vocabulary, grammar, and concepts:")
            print(f'  lexis compile "{relative_saved}"')
        else:
            print("\n[WARNING] Video metadata imported successfully, but NO subtitles/transcripts could be retrieved.")
            print(f"A placeholder transcript has been saved to: {relative_saved}")
            print("Please open this file and manually add or paste your subtitle lines, then run:")
            print(f'  lexis compile "{relative_saved}"')
    except Exception as e:
        print(f"Error importing video: {e}")


def handle_export_bundle(args: argparse.Namespace) -> None:
    try:
        from ..exporter import export_portable_bundle
        out = export_portable_bundle(
            args.html_path,
            video_path=args.media_path,
            output_path=args.output,
            as_zip=not getattr(args, "as_dir", False),
        )
        print(f"✅ Portable bundle created at: {out}")
    except Exception as e:
        print(f"Error creating portable bundle: {e}")


def handle_rename(args: argparse.Namespace) -> None:
    old_normalized = normalize_name(args.old_name)
    old_unit_dir = None
    for p in config.wiki_content_path.iterdir():
        if p.is_dir() and not p.name.startswith("."):
            if normalize_name(p.name) == old_normalized:
                old_unit_dir = p
                break

    if not old_unit_dir:
        print(f"Error: Unit '{args.old_name}' not found under '{config.wiki_content_path}'.")
        sys.exit(1)

    old_unit_name = old_unit_dir.name
    new_unit_dir = config.wiki_content_path / args.new_name
    if new_unit_dir.exists():
        print(f"Error: Target directory '{new_unit_dir}' already exists.")
        sys.exit(1)

    print(f"Renaming unit from '{old_unit_name}' to '{args.new_name}'...")
    try:
        # Atomically move directory
        old_unit_dir.rename(new_unit_dir)

        # Recursively rename files and update internal references
        for item in sorted(new_unit_dir.rglob("*"), key=lambda x: len(x.parts), reverse=True):
            if item.is_file():
                target_item = item
                if item.name.startswith(old_unit_name):
                    new_filename = item.name.replace(old_unit_name, args.new_name, 1)
                    target_item = item.parent / new_filename
                    item.rename(target_item)
                if target_item.suffix.lower() in (".md", ".html", ".json"):
                    try:
                        text = target_item.read_text(encoding="utf-8")
                        if old_unit_name in text:
                            target_item.write_text(text.replace(old_unit_name, args.new_name), encoding="utf-8")
                    except Exception:
                        pass
        print("Successfully renamed unit and internal references.")
    except Exception as e:
        print(f"Error renaming unit: {e}")
