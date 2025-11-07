# project_root/main.py
import sys

import torch

from experiments.run_experiment import run_experiment
from config import DATA_ROUT, CHECKPOINT_DIR, SMALL_RUN, DEFAULT_BATCH_SIZE, DEFAULT_NUM_WORKERS, DEVICE, SCENARIO


def main():
    data_root = DATA_ROUT
    scenario = SCENARIO

    print(torch.__version__)
    print("CUDA доступен:", torch.cuda.is_available())
    print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "нет")

    # --- Проверка наличия сохранений ---
    resume_paths = {}
    for model_name in ["resnext", "densenet", "inception"]:
        ckpt_dir = CHECKPOINT_DIR.glob(f"{model_name}_epoch*.pth")
        ckpts = sorted(ckpt_dir, key=lambda p: p.stat().st_mtime)
        if ckpts:
            resume_paths[model_name] = ckpts[-1]  # последний чекпоинт
            print(f"Найден чекпоинт для {model_name}: {resume_paths[model_name]}")

    print(f"Запуск эксперимента. SMALL_RUN={SMALL_RUN}")
    res = run_experiment(
        data_root=str(data_root),
        scenario=scenario,
        batch_size=DEFAULT_BATCH_SIZE,
        num_workers=DEFAULT_NUM_WORKERS,
        resume=None
    )

    print("Эксперимент завершён.")
    print("Итоговые результаты:", res)


if __name__ == "__main__":
    sys.exit(main())
