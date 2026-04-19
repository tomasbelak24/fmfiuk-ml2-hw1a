import os
import csv
import itertools

from train import train


def format_lr(lr: float) -> str:
    s = f"{lr:.10g}"
    return s.replace(".", "p").replace("-", "m")


def make_run_id(batch_size: int, lr: float, sequence_length: int, seed: int) -> str:
    return f"bs{batch_size}_lr{format_lr(lr)}_len{sequence_length}_seed{seed}"


def load_completed_run_ids(csv_path: str) -> set[str]:
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
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def build_runs():
    # Main grid
    batch_sizes = [64, 128, 256, 512]
    lrs = [1e-3, 3e-4, 6e-4]
    sequence_lengths = [512, 1000, 2048]

    main_runs = list(itertools.product(batch_sizes, lrs, sequence_lengths))

    # Extra aggressive LR probes
    extra_runs = [
        (256, 3e-3, 1000),
        (512, 3e-3, 1000),
        (512, 3e-3, 2048),
    ]

    # Deduplicate while preserving order
    all_runs = []
    seen = set()

    for run in main_runs + extra_runs:
        if run not in seen:
            seen.add(run)
            all_runs.append(run)

    return all_runs


def main():
    os.makedirs("finetune/models", exist_ok=True)

    seed = 42
    csv_path = "finetune/results.csv"

    completed_run_ids = load_completed_run_ids(csv_path)
    all_runs = build_runs()

    print(f"Total planned runs: {len(all_runs)}")
    print(f"Already completed: {len(completed_run_ids)}")

    for batch_size, lr, sequence_length in all_runs:
        run_id = make_run_id(batch_size, lr, sequence_length, seed)
        checkpoint_path = os.path.join("finetune/models", f"{run_id}.pth")

        if run_id in completed_run_ids:
            print(f"Skipping completed run: {run_id}")
            continue

        print("=" * 100)
        print(f"Starting run: {run_id}")

        result = train(
            num_epochs=30,
            batch_size=batch_size,
            lr=lr,
            dataset_size=12800 * 2,
            val_size=2000,
            sequence_length=sequence_length,
            include_mixed=True,
            probs=(0.6, 0.1, 0.3),
            run_id=run_id,
            checkpoint_path=checkpoint_path,
            seed=seed,
            verbose=False
        )

        append_result_to_csv(csv_path, result)
        completed_run_ids.add(run_id)

        print(f"Finished run: {run_id}")
        print("=" * 100)


if __name__ == "__main__":
    main()