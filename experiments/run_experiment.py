# experiments/run_experiment.py

from pathlib import Path

from data.dataloaders import make_dataloaders
from training.evaluate import evaluate_model, compute_metrics
from logging_utils.metrics_saver import append_metrics_xlsx, save_summary_json
from logging_utils.plots import (
    plot_confusion_matrix, plot_roc, plot_pr_curves, plot_calibration, plot_metric_dynamics, plot_training_curves
)
from analysis.error_analysis import save_misclassified_images, plot_misclassification_summary
from analysis.interpretability import save_random_misclassified_examples, get_last_conv_layer
from models.resnext import ResNeXt50WithDropout
from models.densenet import DenseNet161WithDropout
from models.inception import InceptionV3Head
from models.ensemble import EnsembleModel

from project_root.config import DEVICE, CHECKPOINT_DIR, SAMPLES
from training.train_loop import train_model_staged


def run_experiment(data_root, scenario="3-classes",
                   batch_size=32, num_workers=4, resume=None,
                   out_dir="Checkpoints"):
    """
    Runs a full experiment:
    - model training
    - evaluation on val/test
    - saving metrics (CSV/XLSX/JSON)
    - plotting (confusion, ROC, PR, calibration)
    - error analysis
    - interpretation (Grad-CAM, Saliency)
    - ensemble
    """

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Dataloaders ---
    dls224, classes = make_dataloaders(data_root,
                                       img_size=224,
                                       batch_size=batch_size,
                                       num_workers=num_workers)
    dls299, _ = make_dataloaders(data_root,
                                 img_size=299,
                                 batch_size=batch_size,
                                 num_workers=num_workers)
    num_classes = len(classes)

    # --- Models ---
    models_dict = {
        "resnext": ResNeXt50WithDropout(num_classes=num_classes, pretrained=True),
        "densenet": DenseNet161WithDropout(num_classes=num_classes, pretrained=True),
        "inception": InceptionV3Head(num_classes=num_classes, pretrained=True),
    }

    trained = {}

    # --- Training each model ---
    trained["resnext"] = train_model_staged("ResNeXt50", models_dict["resnext"], dls224,
                                          num_classes,
                                          device=DEVICE, out_dir=out_dir)["model"]

    trained["densenet"] = train_model_staged("DenseNet161", models_dict["densenet"], dls224,
                                           num_classes,
                                           device=DEVICE, out_dir=out_dir)["model"]

    trained["inception"] = train_model_staged("InceptionV3", models_dict["inception"], dls299,
                                            num_classes,
                                            device=DEVICE, out_dir=out_dir)["model"]

    # --- Evaluation and Metrics ---
    results = {"val": {}, "test": {}}

    for name, model in trained.items():
        dls = dls224 if name in ["resnext", "densenet"] else dls299

        for split in ["val", "test"]:
            probs, targets = evaluate_model(model, dls, split=split, device=DEVICE)
            acc, metrics, macro = compute_metrics(probs, targets, num_classes)

            results[split][name] = {"acc": acc, "macro": macro}

            # Saving metrics
            append_metrics_xlsx(metrics, classes, name, split,
                                out_dir / "xlsx" / "metrics.xlsx", out_dir / "csv" / "metrics.csv")

            # Charts
            preds = probs.argmax(dim=1)
            plot_confusion_matrix(targets.numpy(), preds.numpy(), classes, name, split, out_dir=out_dir)
            plot_roc(targets.numpy(), probs.numpy(), classes, name, split, out_dir=out_dir)
            plot_pr_curves(targets.numpy(), probs.numpy(), classes, name, split, out_dir=out_dir)
            plot_calibration(targets.numpy(), probs.numpy(), classes, name, split, out_dir=out_dir)


            # Error analysis
            save_misclassified_images(dls[split].dataset.samples, targets, preds, classes,
                                      model_name=name, split=split, out_dir=out_dir)
            plot_misclassification_summary(targets, preds, classes, model_name=name, split=split, out_dir=out_dir)

            # Interpretation (Grad-CAM/Saliency) - example for one image
            get_last_conv_layer(model)

            save_random_misclassified_examples(
                dls["val"].dataset,
                targets,
                preds,
                classes,
                model,
                model_name=name,
                split="val",
                out_dir=CHECKPOINT_DIR,
                n_samples=SAMPLES,
                device=DEVICE
            )


    # --- Ensemble ---
    ensemble = EnsembleModel(trained, device=DEVICE)
    probs, targets = evaluate_model(ensemble, dls224, split="test", device=DEVICE)
    acc, metrics, macro = compute_metrics(probs, targets, num_classes)
    results["test"]["ensemble"] = {"acc": acc, "macro": macro}

    # --- Saving summary ---
    summary = {
        "scenario": scenario,
        "classes": classes,
        "results": results
    }
    save_summary_json(summary, out_dir / "summary.json")

    return {"trained": trained, "dls224": dls224, "dls299": dls299, "results": results}
