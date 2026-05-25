import argparse
import sys
import os
from dotenv import load_dotenv

load_dotenv()

from config import (
    MAX_ITERATIONS, TARGET_F1, API_BUDGET_USD, LLM_PROVIDER,
    CHECKPOINT_EVERY, GPU_AVAILABLE, METRIC_NAME,
)
from agent.loop import AgentLoop


def main():
    parser = argparse.ArgumentParser(
        description="AutoResearch: Agentic ML Experimentation"
    )
    parser.add_argument("--iterations", type=int, default=MAX_ITERATIONS,
                        help=f"Experiment budget (default: {MAX_ITERATIONS})")
    parser.add_argument("--target", type=float, default=TARGET_F1,
                        help=f"Target {METRIC_NAME} (default: {TARGET_F1})")
    parser.add_argument("--budget", type=float, default=API_BUDGET_USD,
                        help=f"Max API spend in USD (default: {API_BUDGET_USD})")
    parser.add_argument("--model", type=str, default=None,
                        help="LLM model override (default: provider-specific)")
    parser.add_argument("--provider", type=str, default=LLM_PROVIDER,
                        choices=["azure", "ollama"],
                        help=f"LLM provider (default: {LLM_PROVIDER})")
    parser.add_argument("--verbose", action="store_true",
                        help="Show detailed output")
    parser.add_argument("--checkpoint", type=int, nargs="?", const=CHECKPOINT_EVERY, default=None,
                        help=f"Pause for human input every N experiments (default N: {CHECKPOINT_EVERY})")
    parser.add_argument("--dry-run", action="store_true",
                        help="Generate code from LLM but don't execute experiments")
    args = parser.parse_args()

    if args.provider == "azure":
        missing = [v for v in ("AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY")
                   if not os.environ.get(v)]
        if missing:
            print(f"ERROR: Missing env vars for Azure: {', '.join(missing)}")
            print("Copy .env.example to .env and add your Azure credentials.")
            sys.exit(1)
    elif args.provider == "ollama":
        # Ollama runs locally, no API key needed
        # Optionally set OLLAMA_BASE_URL if not using default http://localhost:11434/v1
        print(f"Using Ollama at {os.environ.get('OLLAMA_BASE_URL', 'http://localhost:11434/v1')}")

    print(f"GPU detected: {'YES (CUDA)' if GPU_AVAILABLE else 'NO (CPU only)'}")

    agent = AgentLoop(
        iterations=args.iterations,
        target=args.target,
        budget=args.budget,
        model=args.model,
        provider=args.provider,
        verbose=args.verbose,
        interactive=args.checkpoint is not None,
        dry_run=args.dry_run,
        checkpoint_every=args.checkpoint,
    )
    agent.run()


if __name__ == "__main__":
    main()
