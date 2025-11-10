# project_root/config.py

from pathlib import Path
import torch

# === Scenarios ===
SCENARIO = "3-classes" # or "5-classes"

# === Paths ===
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = PROJECT_ROOT / "Checkpoints"
PLOTS_DIR = CHECKPOINT_DIR / "Graphs"
DATASET_REPORT = CHECKPOINT_DIR / "dataset_report.json"
DATASETS_DIR = PROJECT_ROOT / "Datasets"
DATA_ROUT = DATASETS_DIR / SCENARIO

# === Flags ===
SMALL_RUN = False          # quick test run
DO_DATASET_CHECK = False  # dataset validation before training
AUTO_RESAVE_BAD = False   # resave broken images

# === Training Settings ===
STAGE1_EPOCHS = 4 if SMALL_RUN else 20
STAGE2_EPOCHS = 2 if SMALL_RUN else 10
LR_STAGE1 = 1e-2 if SMALL_RUN else 1e-6
LR_STAGE2 = 1e-3 if SMALL_RUN else 1e-7
DEFAULT_BATCH_SIZE = 4 if SMALL_RUN else 32
DEFAULT_NUM_WORKERS = 0 if SMALL_RUN else 4
PIN_MEMORY = True
TRAINING_CONFIG = {
    "stage1": {"epochs": STAGE1_EPOCHS, "lr": LR_STAGE1},
    "stage2": {"epochs": STAGE2_EPOCHS, "lr": LR_STAGE2}
}
SAMPLES = 2
FOCAL_GAMMA = 2.0
AUX_LOSS_WEIGHT = 0.4

# ------------------ Images configuration's for transforms  ------------------
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# === Device ===
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
