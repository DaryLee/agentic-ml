from pathlib import Path
try:
    import torch
    GPU_AVAILABLE = torch.cuda.is_available()
except ImportError:
    GPU_AVAILABLE = False

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"

MAX_ITERATIONS = 20
TARGET_F1 = 0.90
API_BUDGET_USD = 5.0
EXPERIMENT_TIMEOUT = 300  # seconds wall-clock per experiment

# Human-in-the-loop: pause every N experiments when --interactive is set
CHECKPOINT_EVERY = 5

# Stagnation: inject "try something different" after this many non-improving steps
STAGNATION_THRESHOLD = 4

# Cost guard: print a warning if a single LLM call exceeds this amount
COST_WARNING_THRESHOLD = 0.20

# Metric config — change these to adapt to a new task
METRIC_NAME = "f1_macro"           # key to parse from RESULT: line
METRIC_HIGHER_IS_BETTER = True     # True for F1/accuracy, False for loss/bpb
TASK_NAME = "Human Activity Recognition"

LLM_PROVIDER = "azure"  # "azure" or "ollama"
AZURE_OPENAI_MODEL = "gpt-5-mini"
AZURE_API_VERSION = "2025-04-01-preview"
OLLAMA_MODEL = "llama3.1"  # or any model you have pulled in Ollama
OLLAMA_BASE_URL = "http://localhost:11434/v1"  # Ollama OpenAI-compatible endpoint
