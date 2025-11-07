# project_root/main.py
import sys
from experiments.run_experiment import run_experiment
from config import PROJECT_ROOT, CHECKPOINT_DIR, SMALL_RUN, DEFAULT_BATCH_SIZE, DEFAULT_NUM_WORKERS, DEVICE
from analysis.interpretability import GradCAM, compute_saliency, show_gradcam_on_image, show_saliency_on_image

def main():
    data_root = PROJECT_ROOT / "data"   # path to the dataset
    scenario = "3-classes"              # or "5-classes"

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

    # --- Interpretation (Grad-CAM and Saliency) ---
    # Take one image from the validation set
    dls = res["dls"]  # if run_experiment returns dataloaders
    trained = res["trained"]  # if run_experiment returns models

    img, target = next(iter(dls["224"]["val"]))
    img = img[0].unsqueeze(0).to(DEVICE)

    # Grad-CAM
    gradcam = GradCAM(trained["resnext"], trained["resnext"].layer4[-1])
    cam = gradcam.generate(img, target_class=target[0].item())
    show_gradcam_on_image(img[0], cam, title="Grad-CAM ResNeXt")

    # Saliency
    saliency = compute_saliency(trained["resnext"], img, target_class=target[0].item())
    show_saliency_on_image(img[0], saliency[0], title="Saliency ResNeXt")

if __name__ == "__main__":
    sys.exit(main())
