"""
Main CLI entry point for MRLM.

Commands:
- train: Train a model from config
- serve: Start an environment server
- eval: Evaluate a trained model
- collect: Collect trajectories
- info: Show system information
"""

import argparse
import sys
from pathlib import Path

from mrlm.cli import commands


def create_parser() -> argparse.ArgumentParser:
    """Create the main argument parser."""
    parser = argparse.ArgumentParser(
        prog="mrlm",
        description="MRLM: Multi-Agent Reinforcement Learning for LLMs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train from config
  mrlm train --config configs/ppo_code.yaml

  # Start environment server
  mrlm serve --environments code,math --port 50051

  # Evaluate a trained model
  mrlm eval --checkpoint outputs/ppo/final --environment code

  # Collect trajectories
  mrlm collect --model Qwen/Qwen2.5-1.5B --environment math --num-episodes 100

  # Show system info
  mrlm info

For more information, visit: https://github.com/anthropics/MRLM
        """,
    )

    parser.add_argument(
        "--version",
        action="version",
        version="MRLM 0.1.0",
    )

    # Create subparsers for commands
    subparsers = parser.add_subparsers(
        title="commands",
        description="Available commands",
        dest="command",
        required=True,
    )

    # Add command parsers
    commands.add_train_parser(subparsers)
    commands.add_serve_parser(subparsers)
    commands.add_eval_parser(subparsers)
    commands.add_collect_parser(subparsers)
    commands.add_info_parser(subparsers)

    return parser


def main():
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()

    # Dispatch to appropriate command
    try:
        if args.command == "train":
            commands.train_command(args)
        elif args.command == "serve":
            commands.serve_command(args)
        elif args.command == "eval":
            commands.eval_command(args)
        elif args.command == "collect":
            commands.collect_command(args)
        elif args.command == "info":
            commands.info_command(args)
        else:
            parser.print_help()
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        if args.verbose if hasattr(args, "verbose") else False:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
