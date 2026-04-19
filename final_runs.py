import os
import csv

from train import train


OUT_DIR = "finetune_final"
MODELS_DIR = os.path.join(OUT_DIR, "models")
RESULTS_CSV = os.path.join(OUT_DIR, "results.csv")


def format_lr(lr: float) -> str:
    s = f"{lr:.10g}"
    return s.replace(".", "p").replace("-", "m")


def make_run_id(exp_name: str, batch_size: int, lr: float, sequence_length: int, seed: int) -> str:
    return f"{exp_name}_bs{batch_size}_lr{format_lr(lr)}_len{sequence_length}_seed{seed}"


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


def build_final_experiments():
    return [
        {
            "exp_name": "winner_baseline",
            "seed": 22,
            "num_epochs": 30,
            "batch_size": 256,
            "lr": 3e-3,
            "dataset_size": 12800 * 2,
            "val_size": 2000,
            "sequence_length": 1000,
            "include_mixed": True,
            "probs": (0.6, 0.1, 0.3),
            "scheduler_name": None,
            "warmup_frac": 0.0,
            "min_lr_ratio": 0.1,
        },
        {
            "exp_name": "winner_moreepochs",
            "seed": 22,
            "num_epochs": 50,
            "batch_size": 256,
            "lr": 3e-3,
            "dataset_size": 12800 * 2,
            "val_size": 2000,
            "sequence_length": 1000,
            "include_mixed": True,
            "probs": (0.6, 0.1, 0.3),
            "scheduler_name": None,
            "warmup_frac": 0.0,
            "min_lr_ratio": 0.1,
        },
        {
            "exp_name": "winner_moreepochs_sched",
            "seed": 22,
            "num_epochs": 50,
            "batch_size": 256,
            "lr": 3e-3,
            "dataset_size": 12800 * 2,
            "val_size": 2000,
            "sequence_length": 1000,
            "include_mixed": True,
            "probs": (0.6, 0.1, 0.3),
            "scheduler_name": "warmup_cosine",
            "warmup_frac": 0.05,
            "min_lr_ratio": 0.1,
        },
        {
            "exp_name": "winner_sched",
            "seed": 22,
            "num_epochs": 30,
            "batch_size": 256,
            "lr": 3e-3,
            "dataset_size": 12800 * 2,
            "val_size": 2000,
            "sequence_length": 1000,
            "include_mixed": True,
            "probs": (0.6, 0.1, 0.3),
            "scheduler_name": "warmup_cosine",
            "warmup_frac": 0.05,
            "min_lr_ratio": 0.1,
        },
        {
            "exp_name": "winner_sched_moredata",
            "seed": 22,
            "num_epochs": 30,
            "batch_size": 256,
            "lr": 3e-3,
            "dataset_size": 12800 * 4,
            "val_size": 2000,
            "sequence_length": 1000,
            "include_mixed": True,
            "probs": (0.6, 0.1, 0.3),
            "scheduler_name": "warmup_cosine",
            "warmup_frac": 0.05,
            "min_lr_ratio": 0.1,
        },
        {
            "exp_name": "runnerup_sched_moredata",
            "seed": 22,
            "num_epochs": 30,
            "batch_size": 64,
            "lr": 1e-3,
            "dataset_size": 12800 * 4,
            "val_size": 2000,
            "sequence_length": 1000,
            "include_mixed": True,
            "probs": (0.6, 0.1, 0.3),
            "scheduler_name": "warmup_cosine",
            "warmup_frac": 0.05,
            "min_lr_ratio": 0.1,
        },
    ]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    experiments = build_final_experiments()
    completed_run_ids = load_completed_run_ids(RESULTS_CSV)

    print(f"Planned final runs: {len(experiments)}")
    print(f"Already completed: {len(completed_run_ids)}")

    for cfg in experiments:
        run_id = make_run_id(
            exp_name=cfg["exp_name"],
            batch_size=cfg["batch_size"],
            lr=cfg["lr"],
            sequence_length=cfg["sequence_length"],
            seed=cfg["seed"],
        )
        checkpoint_path = os.path.join(MODELS_DIR, f"{run_id}.pth")

        if run_id in completed_run_ids:
            print(f"Skipping completed run: {run_id}")
            continue

        print("=" * 100)
        print(f"Starting final run: {run_id}")

        result = train(
            num_epochs=cfg["num_epochs"],
            batch_size=cfg["batch_size"],
            lr=cfg["lr"],
            dataset_size=cfg["dataset_size"],
            val_size=cfg["val_size"],
            sequence_length=cfg["sequence_length"],
            include_mixed=cfg["include_mixed"],
            probs=cfg["probs"],
            run_id=run_id,
            checkpoint_path=checkpoint_path,
            seed=cfg["seed"],
            scheduler_name=cfg["scheduler_name"],
            warmup_frac=cfg["warmup_frac"],
            min_lr_ratio=cfg["min_lr_ratio"],
            verbose=False
        )

        result["exp_name"] = cfg["exp_name"]

        append_result_to_csv(RESULTS_CSV, result)
        completed_run_ids.add(run_id)

        print(f"Finished final run: {run_id}")
        print("=" * 100)

    print(f"Results saved to {RESULTS_CSV}")


if __name__ == "__main__":
    main()