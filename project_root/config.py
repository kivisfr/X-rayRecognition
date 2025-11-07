# project_root/config.py
from pathlib import Path

# === Paths ===
PROJECT_ROOT = Path(__file__).resolve().parent
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
PLOTS_DIR = CHECKPOINT_DIR / "plots"
DATASET_REPORT = CHECKPOINT_DIR / "dataset_report.json"

# === Flags ===
SMALL_RUN = True          # quick test run
DO_DATASET_CHECK = False  # dataset validation before training
AUTO_RESAVE_BAD = False   # resave broken images

# === Training Settings ===
STAGE1_EPOCHS = 1 if SMALL_RUN else 20
STAGE2_EPOCHS = 1 if SMALL_RUN else 30
DEFAULT_BATCH_SIZE = 4 if SMALL_RUN else 32
DEFAULT_NUM_WORKERS = 0 if SMALL_RUN else 8
PIN_MEMORY = True

# === Device ===
import torch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
