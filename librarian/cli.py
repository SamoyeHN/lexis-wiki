import argparse
import sys
from .config import config
from .linter import linter
from .processor import processor
from .llm import llm

def main():
    parser = argparse.ArgumentParser(description="Teacher Wiki CLI Manager")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Init command
    init_parser = subparsers.add_parser("init", help="Initialize a new wiki project")
    init_parser.add_argument("path", nargs="?", default=".", help="Target directory (default: current directory)")

    # Lint command
    lint_parser = subparsers.add_parser("lint", help="Check wiki for broken links and normalize tags")
    lint_parser.add_argument("--fix-tags", action="store_true", help="Normalize tags to kebab-case")
    lint_parser.add_argument("--prune", action="store_true", help="Remove orphan extractions and quizzes whose raw source files are missing")

    # Config command
    config_parser = subparsers.add_parser("config", help="View and update configuration")
    config_parser.add_argument("--show", action="store_true", help="Show current configuration")
    config_parser.add_argument("--list-models", action="store_true", help="List available Ollama models")
    config_parser.add_argument("--set-model", type=str, metavar="MODEL_NAME", help="Set the model to use")
    config_parser.add_argument("--set-type", choices=["ollama", "openai"], help="Set the API type")
    config_parser.add_argument("--set-url", type=str, help="Set the API base URL")
    config_parser.add_argument("--set-key", type=str, help="Set the API key (for OpenAI)")
    config_parser.add_argument("--set-quiz-count", nargs=2, metavar=("TYPE", "COUNT"), help="Set default question count for a quiz type (e.g., reading 10)")
    config_parser.add_argument("--set-compile-count", nargs=2, metavar=("TYPE", "COUNT"), help="Set default extraction count for compile (e.g., vocabulary 15)")
    config_parser.add_argument("--sync-factory", action="store_true", help="Regenerate all internal factory prompts/schemas from current Python dataclasses")

    # Compile command
    compile_parser = subparsers.add_parser("compile", help="Process a raw file and generate wiki nodes")
    compile_parser.add_argument("filename", help="Path to a raw source file (e.g. d:\\book 3\\unit 1.md) or name of an existing unit under wiki/")

    # Quiz command
    quiz_parser = subparsers.add_parser("quiz", help="Generate an interactive HTML handout (vocabulary, reading, translation, or listening quiz)")
    quiz_parser.add_argument("filename", help="Name of the unit (e.g., Book_4_Unit_1) or the vocabulary file")
    quiz_parser.add_argument("--count", type=int, help="Number of questions to generate (uses config defaults if omitted)")
    quiz_parser.add_argument("--template", "-t", default="vocabulary", choices=["vocabulary", "reading", "translation", "listening", "video"], help="Template type to generate (default: vocabulary)")

    # Ask command
    ask_parser = subparsers.add_parser("ask", help="Query the wiki and raw files using full-file context (RAG)")
    ask_parser.add_argument("query", help="Your question about the units, vocabulary, or concepts")

    # Dashboard command
    dashboard_parser = subparsers.add_parser("dashboard", help="Start the interactive local dashboard server")
    dashboard_parser.add_argument("--port", type=int, default=8000, help="Port to run the dashboard server on (default: 8000)")

    # Video Import command
    video_import_parser = subparsers.add_parser("video-import", help="Import a video transcript (YouTube/Local File) into the self-contained unit structure")
    video_import_parser.add_argument("url_or_filepath", help="The URL of YouTube or local MP4/audio file path")
    video_import_parser.add_argument("--name", help="Custom name for the unit (optional)")
    video_import_parser.add_argument("--cookies-from-browser", help="Extract cookies from browser (e.g. chrome, edge, firefox, safari)")
    video_import_parser.add_argument("--cookies", help="Path to a cookies.txt file to bypass blocks/logins")
    video_import_parser.add_argument("--subtitle", help="Path to a local .srt or .vtt subtitle file (optional)")

    # Export Standalone EXE command
    export_exe_parser = subparsers.add_parser("export-exe", help="Bundle an HTML quiz and its local media into a single-file standalone EXE")
    export_exe_parser.add_argument("html_path", help="Path to the HTML quiz handout")
    export_exe_parser.add_argument("media_path", nargs="?", help="Path to the companion video/audio file (optional)")
    export_exe_parser.add_argument("--output", "-o", help="Custom output path or name for the built .exe file (optional)")

    # Audit / Hero Board command
    audit_parser = subparsers.add_parser("audit", help="Run audit assessment on log files and display the LLM Hero Board")
    audit_parser.add_argument("--json", action="store_true", help="Output audit results as JSON")

    # Rename command
    rename_parser = subparsers.add_parser("rename", help="Atomically rename a unit folder and its internal files")
    rename_parser.add_argument("old_name", help="Current name of the unit (e.g., Book_4_Unit_1)")
    rename_parser.add_argument("new_name", help="New name of the unit (e.g., Book_4_Unit_1_New)")





    args = parser.parse_args()

    if args.command == "init":
        print(f"Initializing new wiki project in {args.path}...")
        success, message = config.initialize_project(args.path)
        if success:
            print(f"Successfully initialized project at {message}")
            print("Directory structure created (wiki/).")
        else:
            print(message)

    elif args.command == "lint":
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
                import os
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

    elif args.command == "config":
        if args.show:
            for k, v in config.data.items():
                print(f"{k}: {v}")
        
        if args.list_models:
            print("Fetching models from Ollama...")
            models = llm.list_models()
            if models:
                print("\nAvailable Ollama Models:")
                for m in models:
                    current = " (current)" if m == config.get("model") else ""
                    print(f"  - {m}{current}")
            else:
                print("No models found or Ollama is not running.")
        
        if args.set_model:
            success, message = config.update_model(args.set_model)
            print(message)

        if args.set_type:
            success, message = config.update_config("api_type", args.set_type)
            print(message)

        if args.set_url:
            success, message = config.update_config("api_url", args.set_url)
            print(message)

        if args.set_key:
            success, message = config.update_config("api_key", args.set_key)
            print(message)

        if args.set_quiz_count:
            q_type, q_count = args.set_quiz_count
            try:
                q_count = int(q_count)
                defaults = config.get("quiz_defaults") or {}
                defaults[q_type] = q_count
                success, message = config.update_config("quiz_defaults", defaults)
                print(message)
            except ValueError:
                print("Error: Count must be an integer.")

        if args.set_compile_count:
            c_type, c_count = args.set_compile_count
            try:
                c_count = int(c_count)
                defaults = config.get("compile_defaults") or {}
                defaults[c_type] = c_count
                success, message = config.update_config("compile_defaults", defaults)
                print(message)
            except ValueError:
                print("Error: Count must be an integer.")

        if args.sync_factory:
            from .prompts import Prompts
            success, message_factory = Prompts.sync_factory_from_code()
            print(message_factory)

    elif args.command == "compile":
        print(f"Compiling {args.filename} via modular pipeline...")
        status, saved_files = processor.run_pipeline(args.filename)
        
        if "Error" in status and not saved_files:
            print(status)
            if "timeout" in status.lower():
                print("\nTip: If you experience timeouts, you can increase the timeout in 'librarian/llm.py'.")
        else:
            print(f"\n{status}")
            if saved_files:
                for path in saved_files:
                    print(f"  - {path}")
            else:
                print("No files were saved. Check the LLM output for formatting issues.")

    elif args.command == "quiz":
        target_count = args.count
        if target_count is None:
            defaults = config.get("quiz_defaults") or {}
            target_count = defaults.get(args.template, 10)

        print(f"Generating quiz from {args.filename} (Target: {target_count} questions, Template: {args.template})...")
        result = processor.generate_quiz(args.filename, count=target_count, template_name=args.template)
        
        if "Error" in result:
            print(result)
        else:
            print(f"\nSuccessfully generated quiz handout:")
            print(f"  - {result}")

    elif args.command == "ask":
        print(f"Querying Wiki: {args.query}...")
        answer = processor.ask_wiki(args.query)
        print("\n" + "="*50)
        print(answer)
        print("="*50 + "\n")

    elif args.command == "dashboard":
        import socketserver
        import webbrowser
        import threading
        from .dashboard import DashboardHTTPRequestHandler

        port = args.port
        print(f"Starting Lexis Interactive Dashboard on port {port}...")
        
        # Use a multi-threaded server so blocking calls (like model tags fetch) do not freeze the UI
        class ThreadingHTTPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
            allow_reuse_address = True
        
        with ThreadingHTTPServer(("", port), DashboardHTTPRequestHandler) as httpd:
            httpd.timeout = 0.5  # Check for KeyboardInterrupt every 0.5s on Windows
            url = f"http://localhost:{port}"
            print(f"Dashboard server is running at: {url}")
            print("Press Ctrl+C to terminate.")
            
            # Open browser automatically in a separate thread
            def open_browser():
                try:
                    webbrowser.open(url)
                except Exception:
                    pass
            
            threading.Thread(target=open_browser, daemon=True).start()
            
            try:
                while True:
                    httpd.handle_request()
            except KeyboardInterrupt:
                print("\nDashboard server stopped.")
                sys.exit(0)

    elif args.command == "video-import":
        print(f"Importing video source: {args.url_or_filepath}...")
        try:
            from .video import video_service
            saved_file, has_subtitles = video_service.import_video(
                args.url_or_filepath, 
                custom_name=args.name, 
                cookies_from_browser=args.cookies_from_browser, 
                cookies=args.cookies,
                subtitle=args.subtitle
            )
            import os
            relative_saved = os.path.relpath(saved_file, config.project_root)
            if has_subtitles:
                print(f"\nSuccessfully imported video with subtitles!")
                print(f"Now, you can compile this video unit to extract vocabulary, grammar, and concepts:")
                print(f"  lexis compile \"{relative_saved}\"")
            else:
                print(f"\n[WARNING] Video metadata imported successfully, but NO subtitles/transcripts could be retrieved.")
                print(f"A placeholder transcript has been saved to: {relative_saved}")
                print(f"Please open this file and manually add or paste your subtitle lines, then run:")
                print(f"  lexis compile \"{relative_saved}\"")
        except Exception as e:
            print(f"Error importing video: {e}")

    elif args.command == "export-exe":
        try:
            from .exporter import export_standalone_exe
            export_standalone_exe(args.html_path, video_path=args.video, output_path=args.output)
        except Exception as e:
            print(f"Error compiling standalone EXE: {e}")

    elif args.command == "rename":
        import json
        from .config import normalize_name
        
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
            
        new_unit_dir = config.wiki_content_path / args.new_name
        if new_unit_dir.exists():
            print(f"Error: Target directory '{new_unit_dir}' already exists.")
            sys.exit(1)
            
        print(f"Renaming unit from '{old_unit_dir.name}' to '{args.new_name}'...")
        try:
            # Atomically move directory
            old_unit_dir.rename(new_unit_dir)
            
            # Recursively rename files
            for item in sorted(new_unit_dir.rglob("*"), key=lambda x: len(x.parts), reverse=True):
                if item.is_file() and item.name.startswith(old_unit_dir.name):
                    new_filename = item.name.replace(old_unit_dir.name, args.new_name, 1)
            print("Successfully renamed unit and internal references.")
        except Exception as e:
            print(f"Error renaming unit: {e}")

    elif args.command == "audit":
        from .evaluator import LogEvaluator
        res = LogEvaluator.audit_all_logs()
        if args.json:
            import json
            print(json.dumps(res, indent=2))
        else:
            print("\n==========================================")
            print("         LLM HERO BOARD LEADERBOARD       ")
            print("==========================================\n")
            board = res["hero_board"]
            if not board:
                print("No log files found to evaluate in logs/ directory.")
            else:
                for idx, item in enumerate(board, 1):
                    badge = f"#{idx}"
                    print(f"Rank {badge}: {item['model']}")
                    print(f"   * Composite Quality Score: {item['composite_score']}%")
                    print(f"   * Schema Adherence:       {item['schema_adherence_avg']}%")
                    print(f"   * Verbatim Faithfulness:  {item['verbatim_faithfulness_avg']}%")
                    print(f"   * Pedagogical Quality:    {item['pedagogical_quality_avg']}%")
                    print(f"   * Uniqueness / No Dups:   {item['uniqueness_avg']}%")
                    print(f"   * Total Evaluated Runs:   {item['runs']}\n")

    else:
        parser.print_help()

if __name__ == "__main__":
    main()
