# AutoResearch: Agentic ML for Human Activity Recognition

An autonomous ML research system following [Karpathy's AutoResearch](https://github.com/karpathy/autoresearch) paradigm. A Claude agent iteratively rewrites a single experiment file, with file-based versioning tracking each attempt that is kept or discarded based on metric improvement.

## Architecture

```
1.program.md: human-authored agent instructions
reads
2.Agent (Azure OpenAI or Ollama): Reasons then writes code then evaluates then decides next
rewrites
3.run_experiment.py (Single Mutable File): Agent has full creative freedom here
executed with timeout
4.Infrastructure (FIXED - agent cannot modify):
      - infrastracture/data.py PAMAP2 loading
      - infrastructure/evaluate.py macro F1 computation
      - scaffold.py pre-built freatures
does
5. File-based versioning (experiment tracker)
      - Each experiment saved to history/exp_NNN.py
      - KEEP saves as last_good and DISCARD restores
      - history/ are all experiment attempts
creates
6.results.tsv (machine-readable log)
```

## Key Design Principles (from AutoResearch)

| Principle | Implementation |
|-----------|---------------|
| **program.md as "org code"** | Human writes instructions in markdown, agent writes Python |
| **Single mutable file** | Agent rewrites `run_experiment.py` — full creative freedom |
| **File-based versioning** | Each experiment saved to history/; keep/discard = save/restore |
| **Hill-climbing, single metric** | Macro F1 — binary keep/discard decision |
| **Fixed time budget** | 300 seconds wall-clock per experiment |
| **Output redirection** | Training output goes to `run.log`; agent only sees summary metric |
| **Never stop** | Loop runs autonomously until budget/target/convergence |
| **Simplicity wins** | Agent instructed: simpler code preferred, removing complexity is a win |

## Project Structure

```
├── program.md              # Agent instructions (the "org code")
├── run_experiment.py       # THE SINGLE MUTABLE FILE (agent rewrites this)
├── run.log                 # Redirected experiment output
├── results.tsv             # Machine-readable experiment log
├── main.py                 # Entry point
├── config.py               # Budget, timeout, model settings
├── agent/
│   └── loop.py             # The AutoResearch-style agentic loop
├── infrastructure/         # FIXED — agent cannot modify
│   ├── data.py             # PAMAP2 download + windowing
│   └── evaluate.py         # Metric computation
├── scaffold.py             # Pre-built features, neural nets, helpers
├── history/                # All experiment attempts (exp_NNN.py)
└── data/                   # Downloaded dataset (gitignored)
```

## Quick Start

```bash
pip install -r requirements.txt

cp .env.example .env
# Edit .env: ANTHROPIC_API_KEY=sk-ant-...
# Or for Azure: AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_DEPLOYMENT
# Or for Ollama: OLLAMA_BASE_URL (default: http://localhost:11434/v1)

# Run with Anthropic Claude (default)
python main.py --verbose

# Or run with Azure OpenAI
python main.py --provider azure --verbose

# Or run with local Ollama (requires ollama running locally)
python main.py --provider ollama --model llama3.1 --verbose
```

### Setting up Ollama

For local inference with Ollama:
```bash
# Install Ollama from https://ollama.ai
# Pull a model (e.g., llama3.1, mistral, codellama)
ollama pull llama3.1

# Ollama automatically serves at http://localhost:11434
# Run the agent
python main.py --provider ollama --model llama3.1
```

## What the Agent Can Do

The agent has **full creative freedom** within `run_experiment.py`. It can:
- Invent novel feature engineering (FFT, wavelets, jerk, custom transforms)
- Use pre-built helpers from `scaffold.py` (extract_all, CNN1D, LSTM, etc.)
- Try any model architecture (sklearn, XGBoost, PyTorch CNN/LSTM)
- Implement custom loss functions and training loops
- Design data augmentation strategies
- Build ensemble methods
- Write code that hasn't been pre-defined anywhere

The `scaffold.py` module provides tested feature extractors (time-domain, frequency-domain, jerk, magnitude) and neural network architectures (CNN1D, LSTMClassifier) with a training helper, but the agent can ignore these and write everything from scratch if it wants.

This is what separates AutoResearch from traditional AutoML: the agent is a **code-writing researcher**, not a hyperparameter optimizer.

## Two-Level Optimization

- **Inner loop**: Agent optimizes `run_experiment.py` (autonomous, runs until budget)
- **Outer loop**: Human optimizes `program.md` over time (what instructions produce the fastest research progress?)
