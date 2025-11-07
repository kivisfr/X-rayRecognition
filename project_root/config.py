# project_root/config.py
from pathlib import Path

# === Scenation ===
SCENARIO = "3-classes" # or "5-classes"

# === Paths ===
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = PROJECT_ROOT / "Checkpoints"
PLOTS_DIR = CHECKPOINT_DIR / "plots"
DATASET_REPORT = CHECKPOINT_DIR / "dataset_report.json"
DATASETS_DIR = PROJECT_ROOT / "Datasets"
DATA_ROUT = DATASETS_DIR / SCENARIO

# === Flags ===
SMALL_RUN = True          # quick test run
DO_DATASET_CHECK = False  # dataset validation before training
AUTO_RESAVE_BAD = False   # resave broken images

# === Training Settings ===
STAGE1_EPOCHS = 1 if SMALL_RUN else 20
STAGE2_EPOCHS = 1 if SMALL_RUN else 30
LR_STAGE1 = 1e-2
LR_STAGE2 = 1e-3
DEFAULT_BATCH_SIZE = 4 if SMALL_RUN else 32
DEFAULT_NUM_WORKERS = 0 if SMALL_RUN else 6
PIN_MEMORY = True
TRAINING_CONFIG = {
    "stage1": {"epochs": STAGE1_EPOCHS, "lr": LR_STAGE1},
    "stage2": {"epochs": STAGE2_EPOCHS, "lr": LR_STAGE2}
}

# === Device ===
import torch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
