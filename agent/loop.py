"""
Core AutoResearch-style agentic loop.

The agent reads program.md, rewrites run_experiment.py each iteration,
and tracks experiments via file-based versioning (keep/discard).
Messages are stored natively in OpenAI format throughout.
"""

import json
import os
import re
import shutil
import subprocess
from pathlib import Path

from config import (
    BASE_DIR, MAX_ITERATIONS, TARGET_F1, API_BUDGET_USD,
    EXPERIMENT_TIMEOUT, LLM_PROVIDER, AZURE_OPENAI_MODEL, AZURE_API_VERSION,
    GPU_AVAILABLE, STAGNATION_THRESHOLD,
    METRIC_NAME, METRIC_HIGHER_IS_BETTER, TASK_NAME, COST_WARNING_THRESHOLD,
    OLLAMA_MODEL, OLLAMA_BASE_URL,
)

COST_TABLE = {
    "azure": (2.5, 10.0),
    "ollama": (0.0, 0.0),
}

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "write_and_run",
            "description": (
                "Write new code to run_experiment.py and execute it. "
                "The code receives X_train (N, 200, 39), y_train, X_test, y_test as numpy arrays in scope. "
                "Must print a line: RESULT: f1_macro=X.XXXX accuracy=X.XXXX. "
                "If the result improves on the current best, the commit is KEPT. "
                "Otherwise it is DISCARDED."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Complete Python code for run_experiment.py. numpy, sklearn, xgboost, torch, scipy available. X_train, y_train, X_test, y_test are already in scope.",
                    },
                    "description": {
                        "type": "string",
                        "description": "Short description of what this experiment tries (logged to results.tsv).",
                    },
                },
                "required": ["code", "description"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_log",
            "description": "Read the last N lines of run.log from the most recent experiment. Useful for diagnosing crashes.",
            "parameters": {
                "type": "object",
                "properties": {
                    "lines": {"type": "integer", "description": "Number of lines to read from the end (default 80)."},
                },
            },
        },
    },
]


class AgentLoop:
    """AutoResearch-style loop: agent writes code, file-based keep/discard tracking."""

    def __init__(self, iterations=None, target=None, budget=None, model=None,
                 verbose=False, provider=None, interactive=False, dry_run=False,
                 checkpoint_every=None):
        self.max_iterations = iterations or MAX_ITERATIONS
        self.target_metric = target or TARGET_F1
        self.api_budget = budget or API_BUDGET_USD
        self.provider = provider or LLM_PROVIDER
        self.verbose = verbose
        self.dry_run = dry_run
        self.checkpoint = checkpoint_every  # None = disabled, int = pause every N steps
        self.model = model or (AZURE_OPENAI_MODEL if self.provider == "azure" else OLLAMA_MODEL)

        self._init_client()
        self.input_cost_per_m, self.output_cost_per_m = COST_TABLE[self.provider]
        self.total_cost = 0.0
        self.best_metric = 0.0
        self.step = 0
        self.stagnant_steps = 0

        self.program_md = (BASE_DIR / "program.md").read_text(encoding="utf-8")
        self.results_tsv = BASE_DIR / "results.tsv"
        self.run_experiment_path = BASE_DIR / "run_experiment.py"
        self.run_log_path = BASE_DIR / "run.log"
        self.history_dir = BASE_DIR / "history"
        self.last_good_path = self.history_dir / "last_good.py"

    def _init_client(self):
        if self.provider == "ollama":
            from openai import OpenAI
            base_url = os.environ.get("OLLAMA_BASE_URL", OLLAMA_BASE_URL)
            self.client = OpenAI(base_url=base_url, api_key="ollama")
        else:
            from openai import AzureOpenAI
            self.client = AzureOpenAI(
                azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
                api_key=os.environ["AZURE_OPENAI_API_KEY"],
                api_version=AZURE_API_VERSION,
            )

    # === Main loop ===

    def run(self):
        print("\n" + "=" * 60)
        print(f"  AutoResearch: {TASK_NAME}")
        print("=" * 60)
        print(f"  Provider:    {self.provider} ({self.model})")
        print(f"  GPU:         {'YES — CUDA available' if GPU_AVAILABLE else 'NO — CPU only'}")
        print(f"  Target:      {self.target_metric:.0%} {METRIC_NAME}")
        print(f"  Budget:      {self.max_iterations} experiments, ${self.api_budget:.2f} API")
        print(f"  Timeout:     {EXPERIMENT_TIMEOUT}s per experiment")
        print(f"  Interactive: {'ON every ' + str(self.checkpoint) + ' steps' if self.checkpoint else 'OFF'}")
        print(f"  Dry-run:     {'ON' if self.dry_run else 'OFF'}")
        print("=" * 60 + "\n")

        self.history_dir.mkdir(exist_ok=True)
        self._init_results_tsv()

        # reference point before the agent starts making changes
        # if this fails, the agent gets a hint to fix the baseline before proceeding
        print("[Baseline] Running initial run_experiment.py...")
        baseline = self._execute_and_evaluate()
        if baseline is not None:
            self.best_metric = baseline
            self._save_current_version()
            self._log_result("baseline", baseline, "keep")
            print(f"  Baseline {METRIC_NAME}: {baseline:.4f}\n")

        messages = self._build_initial_messages()

        for step in range(1, self.max_iterations + 1):
            self.step = step

            if self._metric_reached(self.best_metric, self.target_metric):
                print(f"\n*** TARGET REACHED: {self.best_metric:.4f} ***\n")
                break
            if self.total_cost >= self.api_budget:
                print(f"\n*** BUDGET EXHAUSTED: ${self.total_cost:.2f} ***\n")
                break

            if self.checkpoint and step > 1 and (step - 1) % self.checkpoint == 0:
                print(f"\n[Checkpoint] Step {step} | Best: {self.best_metric:.4f} | ${self.total_cost:.4f} spent")
                try:
                    answer = input("Continue? [Y/n/hint]: ").strip()
                    if answer.lower() == "n":
                        print("  Stopping at user request.")
                        break
                    elif answer and answer.lower() not in ("y", ""):
                        messages.append({"role": "user", "content": f"[Human hint]: {answer}"})
                        print(f"  Hint injected: {answer}")
                except (EOFError, KeyboardInterrupt):
                    break
                print()

            response = self._call_agent(messages)
            if response is None:
                continue

            msg = response.choices[0].message

            # Append assistant turn in OpenAI format
            assistant_msg = {"role": "assistant", "content": msg.content}
            if msg.tool_calls:
                assistant_msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in msg.tool_calls
                ]
            messages.append(assistant_msg)

            # Execute tool calls and append results
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    result = self._dispatch_tool(tc.function.name, json.loads(tc.function.arguments))
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps(result),
                    })
            elif response.choices[0].finish_reason == "stop":
                messages.append({"role": "user", "content":
                    "Continue. Write your next experiment using the write_and_run tool."})

            if len(messages) > 24:
                messages = self._trim_messages(messages)

        self._print_summary()
        self._run_analysis()

    def _trim_messages(self, messages):
        """Drop old messages while keeping the system prompt and intact tool call/result pairs.

        Safe cut point: a plain user message (role=user, content=str) not preceded by
        an assistant message with tool_calls — so we never orphan a tool result.
        """
        keep_start = messages[:1]  # system prompt
        tail = messages[1:]

        if len(tail) <= 20:
            return messages

        for window in (tail[-20:], tail[-10:]):
            for i, msg in enumerate(window):
                if msg.get("role") == "user" and isinstance(msg.get("content"), str):
                    return keep_start + window[i:]

        return keep_start

    # === Tool dispatch ===

    def _dispatch_tool(self, name, input_data):
        if name == "write_and_run":
            return self._handle_write_and_run(input_data)
        if name == "read_log":
            return {"log": self._read_log_tail(input_data.get("lines", 80))}
        return {"error": f"Unknown tool: {name}"}

    def _handle_write_and_run(self, input_data):
        code = input_data.get("code", "")
        description = input_data.get("description", "experiment")

        print(f"[Exp {self.step}] {description[:60]}")

        if self.dry_run:
            print(f"  [DRY RUN] Code preview ({len(code)} chars):")
            print("  " + code[:300].replace("\n", "\n  ") + "...")
            return {"status": "dry_run", "message": "Dry-run: code not executed."}

        history_path = self.history_dir / f"exp_{self.step:03d}.py"
        try:
            history_path.write_text(f"# Experiment {self.step}: {description}\n\n" + code, encoding="utf-8")
        except Exception:
            pass

        self.run_experiment_path.write_text(code, encoding="utf-8")
        metric = self._execute_and_evaluate()

        if metric is None:
            error_tail = self._read_log_tail(50)
            self._restore_last_good()
            self._log_result(description, 0.0, "crash")
            self.stagnant_steps += 1
            print("  CRASH. Restoring last good version.")
            return {"status": "crash", "error": error_tail}

        if self._metric_improved(metric, self.best_metric):
            improvement = abs(metric - self.best_metric)
            self.best_metric = metric
            self.stagnant_steps = 0
            self._save_current_version()
            self._log_result(description, metric, "keep")
            print(f"  {METRIC_NAME}={metric:.4f} | KEEP (+{improvement:.4f}) | Best={self.best_metric:.4f}")
            return {"status": "keep", METRIC_NAME: metric, "improvement": improvement,
                    "message": f"New best! {METRIC_NAME} improved by {improvement:.4f}"}

        self._restore_last_good()
        self._log_result(description, metric, "discard")
        self.stagnant_steps += 1
        print(f"  {METRIC_NAME}={metric:.4f} | DISCARD (best={self.best_metric:.4f})")

        if self.stagnant_steps >= STAGNATION_THRESHOLD:
            print(f"  [Stagnation warning after {self.stagnant_steps} steps — injecting hint]")
            return {
                "status": "discard", METRIC_NAME: metric, "best_metric": self.best_metric,
                "message": (
                    f"[System] No improvement in {self.stagnant_steps} consecutive experiments. "
                    f"Your current approach has plateaued. Try something architecturally different — "
                    f"a new model type, a different feature representation, or a different preprocessing strategy. "
                    f"{'Consider GPU-accelerated PyTorch (CUDA available).' if GPU_AVAILABLE else ''}"
                ),
            }

        return {"status": "discard", METRIC_NAME: metric, "best_metric": self.best_metric,
                "message": f"No improvement. Best remains {self.best_metric:.4f}"}

    # === LLM call ===

    def _call_agent(self, messages):
        try:
            response = self.client.chat.completions.create(
                model=os.environ.get("AZURE_OPENAI_DEPLOYMENT", self.model),
                messages=messages,
                tools=TOOLS,
                max_completion_tokens=4096,
            )
            self._track_cost(response.usage)
            return response
        except Exception as e:
            print(f"  [LLM ERROR: {e}]")
            return None

    # === Experiment execution ===

    def _execute_and_evaluate(self):
        """Run run_experiment.py in a subprocess. Returns metric float or None on failure."""
        runner_code = (
            f'import sys, os\n'
            f'sys.path.insert(0, r"{BASE_DIR}")\n'
            f'os.chdir(r"{BASE_DIR}")\n'
            f'import numpy as np\n'
            f'from infrastructure.data import load_data\n'
            f'X_train, y_train, X_test, y_test = load_data(window_size=200, overlap=0.5)\n'
            f'exec(open(r"{self.run_experiment_path}", encoding="utf-8").read())\n'
        )
        env = {**os.environ}
        if not GPU_AVAILABLE:
            env["CUDA_VISIBLE_DEVICES"] = ""

        try:
            result = subprocess.run(
                ["python", "-c", runner_code],
                capture_output=True, text=True,
                timeout=EXPERIMENT_TIMEOUT,
                cwd=str(BASE_DIR),
                env=env,
            )
            self.run_log_path.write_text(result.stdout + "\n" + result.stderr, encoding="utf-8")

            if self.verbose:
                print(f"  [stdout: {len(result.stdout)} chars, exit: {result.returncode}]")

            for line in result.stdout.splitlines():
                if line.startswith("RESULT:"):
                    match = re.search(rf"{re.escape(METRIC_NAME)}=([\d.]+)", line)
                    if match:
                        return float(match.group(1))
            return None

        except subprocess.TimeoutExpired:
            self.run_log_path.write_text("TIMEOUT: Experiment exceeded time limit", encoding="utf-8")
            return None
        except Exception as e:
            self.run_log_path.write_text(f"EXECUTION ERROR: {e}", encoding="utf-8")
            return None

    def _metric_improved(self, new_val, current_best):
        return new_val > current_best if METRIC_HIGHER_IS_BETTER else new_val < current_best

    def _metric_reached(self, current, target):
        return current >= target if METRIC_HIGHER_IS_BETTER else current <= target

    # === Message builder ===

    def _build_initial_messages(self):
        current_code = self.run_experiment_path.read_text(encoding="utf-8")
        results = self.results_tsv.read_text(encoding="utf-8") if self.results_tsv.exists() else "No results."

        gpu_note = (
            f"\n\n**GPU available (CUDA):** Use PyTorch with `.cuda()`. Consider CNN/LSTM. Timeout: {EXPERIMENT_TIMEOUT}s."
            if GPU_AVAILABLE else
            f"\n\n**No GPU:** CPU only. Stick to sklearn/XGBoost/LightGBM or small PyTorch models. Timeout: {EXPERIMENT_TIMEOUT}s."
        )

        return [
            {"role": "system", "content": self.program_md},
            {"role": "user", "content": (
                f"Begin optimization. The baseline has been run.{gpu_note}\n\n"
                f"## Current run_experiment.py\n```python\n{current_code}\n```\n\n"
                f"## Results so far\n```\n{results}\n```\n\n"
                f"Budget: {self.max_iterations} experiments remaining. "
                f"Current best {METRIC_NAME}: {self.best_metric:.4f}. "
                f"Rewrite run_experiment.py to improve the {METRIC_NAME} score."
            )},
        ]

    # === File operations ===

    def _save_current_version(self):
        try:
            shutil.copy(self.run_experiment_path, self.last_good_path)
        except Exception:
            pass

    def _restore_last_good(self):
        if self.last_good_path.exists():
            try:
                shutil.copy(self.last_good_path, self.run_experiment_path)
            except Exception:
                pass

    # === Logging ===

    def _track_cost(self, usage):
        """Update running cost and print per-call billing summary."""
        cost = (usage.prompt_tokens * self.input_cost_per_m
                + usage.completion_tokens * self.output_cost_per_m) / 1_000_000
        self.total_cost += cost
        remaining = self.api_budget - self.total_cost
        warning = " ⚠️  HIGH COST" if cost > COST_WARNING_THRESHOLD else ""
        print(f"  [Step {self.step} | API: ${cost:.4f}{warning} | Total: ${self.total_cost:.4f} | Remaining: ${remaining:.4f}]")

    def _init_results_tsv(self):
        if not self.results_tsv.exists():
            self.results_tsv.write_text(f"step\t{METRIC_NAME}\tstatus\tdescription\n", encoding="utf-8")

    def _log_result(self, description, metric, status):
        with open(self.results_tsv, "a", encoding="utf-8") as f:
            f.write(f"{self.step}\t{metric:.6f}\t{status}\t{description[:100]}\n")

    def _read_log_tail(self, n=30):
        if self.run_log_path.exists():
            lines = self.run_log_path.read_text(encoding="utf-8").splitlines()
            return "\n".join(lines[-n:])
        return "No log file."

    def _print_summary(self):
        print("\n" + "=" * 60)
        print("  FINAL RESULTS")
        print("=" * 60)
        print(f"  Best {METRIC_NAME}:   {self.best_metric:.4f}")
        print(f"  Total experiments: {self.step}")
        print(f"  Total API cost:    ${self.total_cost:.4f}")
        print("=" * 60 + "\n")

    def _run_analysis(self):
        analyze_script = BASE_DIR / "analyze.py"
        if not analyze_script.exists():
            return
        try:
            result = subprocess.run(
                ["python", str(analyze_script)],
                capture_output=True, text=True, cwd=str(BASE_DIR),
            )
            if result.returncode == 0:
                print("  Analysis plots saved to plots/")
            elif self.verbose:
                print(f"  [analyze.py failed: {result.stderr[:200]}]")
        except Exception as e:
            if self.verbose:
                print(f"  [analyze.py error: {e}]")
