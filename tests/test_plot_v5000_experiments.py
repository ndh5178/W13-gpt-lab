import subprocess
import sys
from pathlib import Path


def test_plot_v5000_experiments_reads_csv_and_creates_expected_pngs(tmp_path):
    summary_csv = tmp_path / "summary.csv"
    epoch_log_csv = tmp_path / "epoch_logs.csv"
    out_dir = tmp_path / "figures"

    summary_csv.write_text(
        "\n".join(
            [
                "experiment_id,label,changed_value,vocab_size,context_length,batch_size,learning_rate,classification_drop_rate,finetune_epochs,best_epoch,best_val_loss,best_val_acc,test_loss,test_acc,finetune_time,notes",
                "baseline,10 epochs baseline,none,5000,128,16,0.0001,0.1,10,10,0.4444,0.8100,0.4242,0.8098,503.19,baseline",
                "epochs_15,15 epochs,finetune_epochs=15,5000,128,16,0.0001,0.1,15,14,0.4207,0.8166,0.3900,0.8250,819.35,longer fine-tuning",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    epoch_log_csv.write_text(
        "\n".join(
            [
                "experiment_id,epoch,train_loss,train_acc,val_loss,val_acc",
                "epochs_15,1,0.6758,0.5562,0.6073,0.6666",
                "epochs_15,2,0.5796,0.6923,0.5513,0.7146",
                "epochs_15,3,0.5364,0.7266,0.5273,0.7400",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    script_path = Path("scripts/plot_v5000_experiments.py")
    result = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--summary-csv",
            str(summary_csv),
            "--epoch-log-csv",
            str(epoch_log_csv),
            "--out-dir",
            str(out_dir),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr

    expected_files = [
        "v5000_finetune_loss_curve.png",
        "v5000_finetune_accuracy_curve.png",
        "v5000_experiment_comparison.png",
    ]
    for filename in expected_files:
        output_path = out_dir / filename
        assert output_path.exists()
        assert output_path.stat().st_size > 0

    for filename in expected_files:
        assert str(out_dir / filename) in result.stdout
