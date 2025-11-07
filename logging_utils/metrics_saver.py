# logging_utils/metrics_saver.py

import csv
import json
from pathlib import Path
import openpyxl


def append_metrics_xlsx(metrics, classes, model_name, split, xlsx_path, csv_path):
    """
    Adds metrics to XLSX and CSV files.

    Parameters
    ----------
    metrics : list[dict]
       Metrics for each class (precision, recall, specificity, f1).
    classes : list[str]
        List of class names.
    model_name : str
        Model name (e.g., "ResNeXt50").
    split : str
        Dataset partition ("val" or "test").
    xlsx_path : Path
        Path to the XLSX file.
    csv_path : Path
        Path to the CSV file.
    """
    xlsx_path = Path(xlsx_path)
    csv_path = Path(csv_path)

    # --- XLSX ---
    if not xlsx_path.exists():
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "metrics"
        ws.append(["model", "split", "class", "precision", "recall", "specificity", "f1"])
        wb.save(xlsx_path)

    wb = openpyxl.load_workbook(xlsx_path)
    ws = wb["metrics"]

    for cname, m in zip(classes, metrics):
        ws.append([model_name, split, cname,
                   m["precision"], m["recall"], m["specificity"], m["f1"]])
    wb.save(xlsx_path)

    # --- CSV ---
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["model", "split", "class", "precision", "recall", "specificity", "f1"])
        for cname, m in zip(classes, metrics):
            writer.writerow([model_name, split, cname,
                             m["precision"], m["recall"], m["specificity"], m["f1"]])


def save_summary_json(summary_dict, out_path):
    """
    Saves the experiment's final report in JSON.

    Parameters
    ----------
    summary_dict : dict
       Dictionary with results (e.g., {"val": {...}, "test": {...}}).
    out_path : Path
        Path to the JSON file.
    """
    out_path = Path(out_path)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary_dict, f, indent=2, ensure_ascii=False)
