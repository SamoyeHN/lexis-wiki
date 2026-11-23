"""
CLI subparser and handler for viewing and updating wiki configuration.
"""

import argparse
import sys

from ..config import config
from ..llm import llm


def register_subparser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("config", help="View and update configuration")
    parser.add_argument("--show", action="store_true", help="Show current configuration")
    parser.add_argument("--list-models", action="store_true", help="List available Ollama models")
    parser.add_argument("--set-model", type=str, metavar="MODEL_NAME", help="Set the model to use")
    parser.add_argument("--set-type", choices=["ollama", "openai"], help="Set the API type")
    parser.add_argument("--set-url", type=str, help="Set the API base URL")
    parser.add_argument("--set-key", type=str, help="Set the API key (for OpenAI)")
    parser.add_argument(
        "--set-quiz-count",
        nargs=2,
        metavar=("TYPE", "COUNT"),
        help="Set default question count for a quiz type (e.g., reading 10)",
    )
    parser.add_argument(
        "--set-compile-count",
        nargs=2,
        metavar=("TYPE", "COUNT"),
        help="Set default extraction count for compile (e.g., vocabulary 15)",
    )
    parser.add_argument("--set-num-ctx", type=int, metavar="TOKENS", help="Set Ollama context window size (e.g., 16384)")
    parser.add_argument("--set-max-tokens", type=int, metavar="TOKENS", help="Set maximum generation tokens (e.g., 8192)")
    parser.add_argument(
        "--sync-factory",
        action="store_true",
        help="Regenerate all internal factory prompts/schemas from current Python dataclasses",
    )
    parser.set_defaults(handler=handle_config)


def handle_config(args: argparse.Namespace) -> None:
    if args.show:
        for k, v in config.data.items():
            print(f"{k}: {v}")
        health = llm.check_health(force=True)
        if health.get("online") and health.get("model_ready"):
            print(f"llm_health: 🟢 ONLINE & READY ({health.get('model')} @ {health.get('api_url')})")
        elif health.get("online"):
            print(f"llm_health: 🟡 MODEL MISSING — {health.get('error')}")
        else:
            print(f"llm_health: 🔴 OFFLINE — {health.get('error')}")

    if args.list_models:
        print("Fetching models from LLM service...")
        health = llm.check_health(force=True)
        if not health.get("online"):
            print(f"❌ Error: {health.get('error')}")
            sys.exit(1)
        models = health.get("available_models") or llm.list_models()
        if models:
            print("\nAvailable Ollama Models:")
            for m in models:
                current = " (current)" if m == config.get("model") else ""
                print(f"  - {m}{current}")
        else:
            print("No models found on the LLM service.")

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
            if isinstance(defaults.get(q_type), dict):
                defaults[q_type]["count"] = q_count
            else:
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

    if args.set_num_ctx:
        success, message = config.update_config("num_ctx", args.set_num_ctx)
        print(message)

    if args.set_max_tokens:
        success, message = config.update_config("max_tokens", args.set_max_tokens)
        print(message)

    if args.sync_factory:
        from ..prompts import Prompts

        success, message_factory = Prompts.sync_factory_from_code()
        print(message_factory)
