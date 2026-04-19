import os
import csv

from train import train


OUT_DIR = "finetune_final_confirm"
MODELS_DIR = os.path.join(OUT_DIR, "models")
RESULTS_CSV = os.path.join(OUT_DIR, "results.csv")
SUMMARY_CSV = os.path.join(OUT_DIR, "summary.csv")

SEEDS = [101, 202, 303, 404, 505]


def load_completed_run_ids(csv_path: str) -> set:
    if not os.path.exists(csv_path):
        return set()

    completed = set()
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            completed.add(row["run_id"])
    return completed


def append_result_to_csv(csv_path: str, row: dict):
    file_exists = os.path.exists(csv_path)

    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def read_results(csv_path: str) -> list[dict]:
    if not os.path.exists(csv_path):
        return []

    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def mean(values):
    return sum(values) / len(values) if values else float("nan")


def std(values):
    if len(values) < 2:
        return 0.0
    m = mean(values)
    return (sum((x - m) ** 2 for x in values) / (len(values) - 1)) ** 0.5


def write_summary(results_rows: list[dict], summary_csv: str):
    if not results_rows:
        return

    metric_keys = [
        "main_val_loss",
        "main_val_acc",
        "main_val_auc",
        "main_val_pred_irregular_rate",
        "small_val_loss",
        "small_val_acc",
        "small_val_auc",
        "small_val_pred_irregular_rate",
        "best_epoch",
    ]

    summary = {
        "num_runs": len(results_rows),
        "batch_size": results_rows[0]["batch_size"],
        "lr": results_rows[0]["lr"],
        "sequence_length": results_rows[0]["sequence_length"],
        "num_epochs": results_rows[0]["num_epochs"],
        "dataset_size": results_rows[0]["dataset_size"],
        "val_size": results_rows[0]["val_size"],
        "include_mixed": results_rows[0]["include_mixed"],
        "probs": results_rows[0]["probs"],
        "scheduler_name": results_rows[0]["scheduler_name"],
        "warmup_frac": results_rows[0]["warmup_frac"],
        "min_lr_ratio": results_rows[0]["min_lr_ratio"],
    }

    for key in metric_keys:
        vals = [float(r[key]) for r in results_rows]
        summary[f"{key}_mean"] = mean(vals)
        summary[f"{key}_std"] = std(vals)

    best_row = max(results_rows, key=lambda r: float(r["main_val_auc"]))
    summary["best_run_id"] = best_row["run_id"]
    summary["best_checkpoint_path"] = best_row["checkpoint_path"]
    summary["best_main_val_auc"] = best_row["main_val_auc"]
    summary["best_main_val_loss"] = best_row["main_val_loss"]

    with open(summary_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)

    print("\n" + "=" * 100)
    print("FINAL CONFIRMATION SUMMARY")
    print("=" * 100)
    print(f"Runs: {summary['num_runs']}")
    print(f"main AUC:   {summary['main_val_auc_mean']:.6f} ± {summary['main_val_auc_std']:.6f}")
    print(f"main loss:  {summary['main_val_loss_mean']:.6f} ± {summary['main_val_loss_std']:.6f}")
    print(f"small AUC:  {summary['small_val_auc_mean']:.6f} ± {summary['small_val_auc_std']:.6f}")
    print(f"small loss: {summary['small_val_loss_mean']:.6f} ± {summary['small_val_loss_std']:.6f}")
    print(f"Best run:   {summary['best_run_id']}")
    print(f"Checkpoint: {summary['best_checkpoint_path']}")
    print("=" * 100 + "\n")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    completed_run_ids = load_completed_run_ids(RESULTS_CSV)

    print(f"Planned confirmation runs: {len(SEEDS)}")
    print(f"Already completed: {len(completed_run_ids)}")

    for seed in SEEDS:
        run_id = f"winner_sched_moredata_confirm_seed{seed}"
        checkpoint_path = os.path.join(MODELS_DIR, f"{run_id}.pth")

        if run_id in completed_run_ids:
            print(f"Skipping completed run: {run_id}")
            continue

        print("=" * 100)
        print(f"Starting confirmation run: {run_id}")

        result = train(
            num_epochs=30,
            batch_size=256,
            lr=3e-3,
            dataset_size=12800 * 4,
            val_size=2000,
            sequence_length=1000,
            include_mixed=True,
            probs=(0.6, 0.1, 0.3),
            run_id=run_id,
            checkpoint_path=checkpoint_path,
            seed=seed,
            scheduler_name="warmup_cosine",
            warmup_frac=0.05,
            min_lr_ratio=0.1,
            verbose=False
        )

        result["exp_name"] = "winner_sched_moredata_confirm"

        append_result_to_csv(RESULTS_CSV, result)
        completed_run_ids.add(run_id)

        print(f"Finished confirmation run: {run_id}")
        print("=" * 100)

    results_rows = read_results(RESULTS_CSV)
    write_summary(results_rows, SUMMARY_CSV)

    print(f"Per-run results saved to: {RESULTS_CSV}")
    print(f"Summary saved to: {SUMMARY_CSV}")


if __name__ == "__main__":
    main()