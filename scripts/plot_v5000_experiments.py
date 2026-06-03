#!/usr/bin/env python
"""Generate presentation graphs from Mini GPT v5000 experiment CSVs."""

from __future__ import annotations

import argparse
import csv
import os
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "mini-gpt-matplotlib-cache"),
)

import matplotlib


matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_SUMMARY_CSV = Path("experiment_results/ssoming_v5000_summary.csv")
DEFAULT_EPOCH_LOG_CSV = Path("experiment_results/ssoming_v5000_epoch_logs.csv")

LOSS_FILENAME = "v5000_finetune_loss_curve.png"
ACCURACY_FILENAME = "v5000_finetune_accuracy_curve.png"
COMPARISON_FILENAME = "v5000_experiment_comparison.png"


@dataclass(frozen=True)
class ExperimentSummary:
    experiment_id: str
    label: str
    changed_value: str
    vocab_size: int
    context_length: int
    batch_size: int
    learning_rate: float
    classification_drop_rate: float
    finetune_epochs: int
    best_epoch: int
    best_val_loss: float
    best_val_acc: float
    test_loss: float
    test_acc: float
    finetune_time: float
    notes: str


@dataclass(frozen=True)
class EpochLog:
    experiment_id: str
    epoch: int
    train_loss: float
    train_acc: float
    val_loss: float
    val_acc: float


def read_summary_csv(path: Path) -> list[ExperimentSummary]:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    summaries: list[ExperimentSummary] = []
    for row in rows:
        summaries.append(
            ExperimentSummary(
                experiment_id=row["experiment_id"],
                label=row["label"],
                changed_value=row["changed_value"],
                vocab_size=int(row["vocab_size"]),
                context_length=int(row["context_length"]),
                batch_size=int(row["batch_size"]),
                learning_rate=float(row["learning_rate"]),
                classification_drop_rate=float(row["classification_drop_rate"]),
                finetune_epochs=int(row["finetune_epochs"]),
                best_epoch=int(row["best_epoch"]),
                best_val_loss=float(row["best_val_loss"]),
                best_val_acc=float(row["best_val_acc"]),
                test_loss=float(row["test_loss"]),
                test_acc=float(row["test_acc"]),
                finetune_time=float(row["finetune_time"]),
                notes=row.get("notes", ""),
            )
        )
    if not summaries:
        raise ValueError(f"No experiment rows found in {path}")
    return summaries


def read_epoch_log_csv(path: Path) -> dict[str, list[EpochLog]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    logs_by_experiment: dict[str, list[EpochLog]] = defaultdict(list)
    for row in rows:
        log = EpochLog(
            experiment_id=row["experiment_id"],
            epoch=int(row["epoch"]),
            train_loss=float(row["train_loss"]),
            train_acc=float(row["train_acc"]),
            val_loss=float(row["val_loss"]),
            val_acc=float(row["val_acc"]),
        )
        logs_by_experiment[log.experiment_id].append(log)

    return {
        experiment_id: sorted(logs, key=lambda log: log.epoch)
        for experiment_id, logs in logs_by_experiment.items()
    }


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 200,
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
        }
    )


def choose_curve_experiment(
    summaries: list[ExperimentSummary],
    logs_by_experiment: dict[str, list[EpochLog]],
    requested_experiment_id: str | None,
) -> ExperimentSummary:
    if requested_experiment_id is not None:
        for summary in summaries:
            if summary.experiment_id == requested_experiment_id:
                if summary.experiment_id not in logs_by_experiment:
                    raise ValueError(f"No epoch logs found for {requested_experiment_id}")
                return summary
        raise ValueError(f"Unknown experiment_id: {requested_experiment_id}")

    for summary in reversed(summaries):
        if summary.experiment_id in logs_by_experiment:
            return summary

    raise ValueError("No epoch logs found for any summary experiment")


def plot_loss_curve(out_dir: Path, summary: ExperimentSummary, logs: list[EpochLog]) -> Path:
    epochs = [log.epoch for log in logs]
    train_loss = [log.train_loss for log in logs]
    val_loss = [log.val_loss for log in logs]

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(epochs, train_loss, marker="o", linewidth=2, label="train_loss")
    ax.plot(epochs, val_loss, marker="o", linewidth=2, label="val_loss")

    if summary.best_epoch in epochs:
        best_idx = epochs.index(summary.best_epoch)
        best_val_loss = val_loss[best_idx]
        ax.axvline(
            summary.best_epoch,
            linestyle="--",
            color="gray",
            linewidth=1.2,
            label=f"best_epoch={summary.best_epoch}",
        )
        ax.scatter([summary.best_epoch], [best_val_loss], s=70, color="#d62728", zorder=5)
        ax.annotate(
            f"best val_acc epoch\nval_loss={best_val_loss:.4f}",
            xy=(summary.best_epoch, best_val_loss),
            xytext=(max(1, summary.best_epoch - 3.5), max(val_loss) - 0.12),
            arrowprops={"arrowstyle": "->", "color": "#555555"},
            fontsize=9,
        )

    ax.set_title(f"Fine-tuning Loss Curve ({summary.label})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_xticks(epochs)
    ax.grid(True, alpha=0.28)
    ax.legend(loc="upper right")
    fig.tight_layout()

    output_path = out_dir / LOSS_FILENAME
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_accuracy_curve(out_dir: Path, summary: ExperimentSummary, logs: list[EpochLog]) -> Path:
    epochs = [log.epoch for log in logs]
    train_acc = [log.train_acc for log in logs]
    val_acc = [log.val_acc for log in logs]

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(epochs, train_acc, marker="o", linewidth=2, label="train_acc")
    ax.plot(epochs, val_acc, marker="o", linewidth=2, label="val_acc")

    if summary.best_epoch in epochs:
        best_idx = epochs.index(summary.best_epoch)
        best_val_acc = val_acc[best_idx]
        ax.axvline(
            summary.best_epoch,
            linestyle="--",
            color="gray",
            linewidth=1.2,
            label=f"best_epoch={summary.best_epoch}",
        )
        ax.scatter([summary.best_epoch], [best_val_acc], s=70, color="#d62728", zorder=5)
        ax.annotate(
            f"best val_acc={best_val_acc:.4f}",
            xy=(summary.best_epoch, best_val_acc),
            xytext=(max(1, summary.best_epoch - 3.5), min(val_acc) + 0.07),
            arrowprops={"arrowstyle": "->", "color": "#555555"},
            fontsize=9,
        )

    y_min = max(0.0, min(train_acc + val_acc) - 0.035)
    y_max = min(1.0, max(train_acc + val_acc) + 0.025)
    ax.set_ylim(y_min, y_max)
    ax.set_title(f"Fine-tuning Accuracy Curve ({summary.label})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_xticks(epochs)
    ax.grid(True, alpha=0.28)
    ax.legend(loc="lower right")
    fig.tight_layout()

    output_path = out_dir / ACCURACY_FILENAME
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_experiment_comparison(out_dir: Path, summaries: list[ExperimentSummary]) -> Path:
    labels = [summary.label for summary in summaries]
    best_val_acc = [summary.best_val_acc for summary in summaries]
    test_acc = [summary.test_acc for summary in summaries]
    finetune_time = [summary.finetune_time for summary in summaries]

    fig_width = max(10.5, 2.2 * len(summaries) + 6)
    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        figsize=(fig_width, 4.8),
        gridspec_kw={"width_ratios": [1.35, 1]},
    )

    bar_w = 0.34
    x = list(range(len(summaries)))
    ax1.bar(
        [i - bar_w / 2 for i in x],
        best_val_acc,
        bar_w,
        label="best_val_acc",
        color="#4c78a8",
    )
    ax1.bar(
        [i + bar_w / 2 for i in x],
        test_acc,
        bar_w,
        label="test_acc",
        color="#f58518",
    )
    ax1.set_title("Accuracy Comparison")
    ax1.set_xticks(x, labels)
    ax1.set_ylabel("Accuracy")
    acc_min = max(0.0, min(best_val_acc + test_acc) - 0.02)
    acc_max = min(1.0, max(best_val_acc + test_acc) + 0.02)
    ax1.set_ylim(acc_min, acc_max)
    ax1.grid(axis="y", alpha=0.28)
    ax1.legend(loc="upper left")
    for i, value in enumerate(best_val_acc):
        ax1.text(i - bar_w / 2, value + 0.001, f"{value:.4f}", ha="center", va="bottom", fontsize=8)
    for i, value in enumerate(test_acc):
        ax1.text(i + bar_w / 2, value + 0.001, f"{value:.4f}", ha="center", va="bottom", fontsize=8)

    ax2.bar(labels, finetune_time, color="#54a24b")
    ax2.set_title("Fine-tuning Time")
    ax2.set_ylabel("Seconds")
    ax2.grid(axis="y", alpha=0.28)
    for i, value in enumerate(finetune_time):
        ax2.text(i, value + max(finetune_time) * 0.02, f"{value:.1f}s", ha="center", va="bottom", fontsize=9)

    fig.suptitle("v5000 Experiment Comparison", y=1.02, fontsize=14)
    fig.tight_layout()

    output_path = out_dir / COMPARISON_FILENAME
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def generate_plots(
    summary_csv: Path,
    epoch_log_csv: Path,
    out_dir: Path,
    curve_experiment_id: str | None = None,
) -> list[Path]:
    summaries = read_summary_csv(summary_csv)
    logs_by_experiment = read_epoch_log_csv(epoch_log_csv)
    curve_summary = choose_curve_experiment(summaries, logs_by_experiment, curve_experiment_id)

    out_dir.mkdir(parents=True, exist_ok=True)
    configure_matplotlib()
    return [
        plot_loss_curve(out_dir, curve_summary, logs_by_experiment[curve_summary.experiment_id]),
        plot_accuracy_curve(out_dir, curve_summary, logs_by_experiment[curve_summary.experiment_id]),
        plot_experiment_comparison(out_dir, summaries),
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=DEFAULT_SUMMARY_CSV,
        help="CSV containing one final-result row per experiment.",
    )
    parser.add_argument(
        "--epoch-log-csv",
        type=Path,
        default=DEFAULT_EPOCH_LOG_CSV,
        help="CSV containing per-epoch train/validation metrics.",
    )
    parser.add_argument(
        "--curve-experiment",
        default=None,
        help="Experiment ID to use for loss/accuracy curves. Defaults to the latest experiment with epoch logs.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("figures"),
        help="Directory where PNG graph files will be saved.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for output_path in generate_plots(
        summary_csv=args.summary_csv,
        epoch_log_csv=args.epoch_log_csv,
        out_dir=args.out_dir,
        curve_experiment_id=args.curve_experiment,
    ):
        print(output_path)


if __name__ == "__main__":
    main()
