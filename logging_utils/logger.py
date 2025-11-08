# logging_utils/logger.py

import csv
import shutil
import sys
from pathlib import Path
import openpyxl


def log(message: str):
    """
    Prints a message to stdout and immediately flushes the buffer.
    """
    print(message)
    sys.stdout.flush()


def append_epoch_csv(epoch, metrics_dict, csv_path):
    """
  Adds epoch metrics to a CSV file.

        Parameters
        ----------
        epoch : int
            Epoch number.
        metrics_dict : dict
            Dictionary with metrics (e.g., {"train_loss": ..., "val_loss": ..., "train_acc": ..., "val_acc": ...}).
        csv_path : Path or str
            Path to the CSV file.
    """
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    # If a directory exists where a file should be, remove it (one-time healing).
    if csv_path.exists() and csv_path.is_dir():
        shutil.rmtree(csv_path)
    write_header = not csv_path.exists()

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            header = ["epoch"] + list(metrics_dict.keys())
            writer.writerow(header)
        row = [epoch] + [metrics_dict[k] for k in metrics_dict.keys()]
        writer.writerow(row)


def append_epoch_xlsx(epoch, metrics_dict, xlsx_path):
    """
    Adds epoch-specific metrics to XLSX.

        Parameters
        ----------
        epoch : int
            Epoch number.
        metrics_dict : dict
            Dictionary with metrics.
        xlsx_path : Path or str
            Path to the XLSX file.
    """
    xlsx_path = Path(xlsx_path)
    xlsx_path.parent.mkdir(parents=True, exist_ok=True)

    if not xlsx_path.exists():
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "epochs"
        ws.append(["epoch"] + list(metrics_dict.keys()))
        wb.save(xlsx_path)

    wb = openpyxl.load_workbook(xlsx_path)
    ws = wb["epochs"]
    ws.append([epoch] + [metrics_dict[k] for k in metrics_dict.keys()])
    wb.save(xlsx_path)
