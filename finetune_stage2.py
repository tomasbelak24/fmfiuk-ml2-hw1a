import os
import csv
import time
import math
from collections import defaultdict

from train import train


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

SEEDS = [11, 24, 137]

# Five selected Stage 2 candidates:
# (batch_size, lr, sequence_length)
CANDIDATES = [
    (256, 3e-3, 1000),
    (64, 1e-3, 1000),
    (512, 3e-3, 2048),
    (128, 1e-3, 2048),
    (256, 1e-3, 2048),
]

NUM_EPOCHS = 30
DATASET_SIZE = 12800 * 2
VAL_SIZE = 2000
INCLUDE_MIXED = True
PROBS = (0.6, 0.1, 0.3)

OUT_DIR = "finetune_stage2"
MODELS_DIR = os.path.join(OUT_DIR, "models")
RESULTS_CSV = os.path.join(OUT_DIR, "results.csv")
SUMMARY_CSV = os.path.join(OUT_DIR, "summary.csv")


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def format_lr(lr: float) -> str:
    s = f"{lr:.10g}"
    return s.replace(".", "p").replace("-", "m")


def make_run_id(batch_size: int, lr: float, sequence_length: int, seed: int) -> str:
    return f"bs{batch_size}_lr{format_lr(lr)}_len{sequence_length}_seed{seed}"


def make_config_id(batch_size: int, lr: float, sequence_length: int) -> str:
    return f"bs{batch_size}_lr{format_lr(lr)}_len{sequence_length}"


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


def to_float(row: dict, key: str):
    val = row.get(key, "")
    if val is None or val == "":
        return None
    return float(val)


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = mean(values)
    return math.sqrt(sum((x - m) ** 2 for x in values) / (len(values) - 1))


def build_stage2_runs():
    runs = []
    for batch_size, lr, sequence_length in CANDIDATES:
        for seed in SEEDS:
            runs.append((batch_size, lr, sequence_length, seed))
    return runs


# ------------------------------------------------------------
# Summary computation
# ------------------------------------------------------------

def build_summary_rows(results_rows: list[dict]) -> list[dict]:
    grouped = defaultdict(list)

    for row in results_rows:
        config_id = row["config_id"]
        grouped[config_id].append(row)

    metric_keys = [
        "best_epoch",
        "train_wall_time_sec",
        "avg_epoch_time_sec",

        "main_val_loss",
        "main_val_acc",
        "main_val_auc",
        "main_val_pred_irregular_rate",

        "small_val_loss",
        "small_val_acc",
        "small_val_auc",
        "small_val_pred_irregular_rate",
    ]

    summary_rows = []

    for config_id, rows in grouped.items():
        first = rows[0]

        summary = {
            "config_id": config_id,
            "num_runs": len(rows),
            "batch_size": first["batch_size"],
            "lr": first["lr"],
            "sequence_length": first["sequence_length"],
            "num_epochs": first["num_epochs"],
            "dataset_size": first["dataset_size"],
            "val_size": first["val_size"],
            "include_mixed": first["include_mixed"],
            "probs": first["probs"],
        }

        for key in metric_keys:
            vals = [to_float(r, key) for r in rows]
            vals = [v for v in vals if v is not None]

            if len(vals) == 0:
                summary[f"{key}_mean"] = ""
                summary[f"{key}_std"] = ""
            else:
                summary[f"{key}_mean"] = mean(vals)
                summary[f"{key}_std"] = std(vals)

        summary_rows.append(summary)

    # Rank primarily by main_val_auc_mean descending, then main_val_loss_mean ascending
    summary_rows.sort(
        key=lambda r: (
            -float(r["main_val_auc_mean"]),
            float(r["main_val_loss_mean"]),
        )
    )

    return summary_rows


def write_summary_csv(csv_path: str, summary_rows: list[dict]):
    if not summary_rows:
        return

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)


def print_summary_table(summary_rows: list[dict]):
    print("\n" + "=" * 120)
    print("STAGE 2 SUMMARY (ranked by main_val_auc_mean desc, main_val_loss_mean asc)")
    print("=" * 120)

    for i, row in enumerate(summary_rows, start=1):
        print(
            f"{i:>2}. {row['config_id']} | "
            f"main AUC = {float(row['main_val_auc_mean']):.6f} ± {float(row['main_val_auc_std']):.6f} | "
            f"main loss = {float(row['main_val_loss_mean']):.6f} ± {float(row['main_val_loss_std']):.6f} | "
            f"small AUC = {float(row['small_val_auc_mean']):.6f} ± {float(row['small_val_auc_std']):.6f} | "
            f"small loss = {float(row['small_val_loss_mean']):.6f} ± {float(row['small_val_loss_std']):.6f}"
        )

    print("=" * 120 + "\n")


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    all_runs = build_stage2_runs()
    completed_run_ids = load_completed_run_ids(RESULTS_CSV)

    print(f"Total planned Stage 2 runs: {len(all_runs)}")
    print(f"Already completed: {len(completed_run_ids)}")

    stage2_start_time = time.time()

    for batch_size, lr, sequence_length, seed in all_runs:
        run_id = make_run_id(batch_size, lr, sequence_length, seed)
        config_id = make_config_id(batch_size, lr, sequence_length)
        checkpoint_path = os.path.join(MODELS_DIR, f"{run_id}.pth")

        if run_id in completed_run_ids:
            print(f"Skipping completed run: {run_id}")
            continue

        print("=" * 100)
        print(f"Starting Stage 2 run: {run_id}")

        run_start_time = time.time()

        result = train(
            num_epochs=NUM_EPOCHS,
            batch_size=batch_size,
            lr=lr,
            dataset_size=DATASET_SIZE,
            val_size=VAL_SIZE,
            sequence_length=sequence_length,
            include_mixed=INCLUDE_MIXED,
            probs=PROBS,
            run_id=run_id,
            checkpoint_path=checkpoint_path,
            seed=seed,
            verbose=False
        )

        run_elapsed = time.time() - run_start_time

        # Add config-level metadata if not already returned by train()
        result["config_id"] = config_id
        result["run_elapsed_wrapper_sec"] = run_elapsed

        append_result_to_csv(RESULTS_CSV, result)
        completed_run_ids.add(run_id)

        print(f"Finished run: {run_id}")
        print(f"Wrapper elapsed time: {run_elapsed:.2f} sec")
        print("=" * 100)

    total_elapsed = time.time() - stage2_start_time
    print(f"\nTotal Stage 2 elapsed time: {total_elapsed / 60:.2f} minutes")

    # Rebuild summary from the full results file
    results_rows = read_results(RESULTS_CSV)
    summary_rows = build_summary_rows(results_rows)
    write_summary_csv(SUMMARY_CSV, summary_rows)
    print_summary_table(summary_rows)

    print(f"Per-run results saved to: {RESULTS_CSV}")
    print(f"Per-config summary saved to: {SUMMARY_CSV}")


if __name__ == "__main__":
    main()