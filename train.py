import math

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, confusion_matrix

from model import create_model
from generator import generate_regular_sequence, generate_irregular_sequence


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def randint_log_uniform(low, high):
    return int(np.exp(np.random.uniform(np.log(low), np.log(high))))


def sample_length():
    r = np.random.rand()

    if r < 0.05:
        return randint_log_uniform(500, 647)
    elif r < 0.25:
        return randint_log_uniform(647, 1879)
    elif r < 0.50:
        return randint_log_uniform(1879, 7142)
    elif r < 0.75:
        return randint_log_uniform(7142, 26564)
    elif r < 0.95:
        return randint_log_uniform(26564, 77245)
    else:
        return randint_log_uniform(77245, 99910)


def make_mixed_sequence(total_length, min_windows=1, max_windows=3,
                        min_window_len=128, max_window_frac=0.3):
   
    seq = generate_regular_sequence(total_length).astype(np.int32)

    n_windows = np.random.randint(min_windows, max_windows + 1)
    max_window_len = max(min_window_len, int(total_length * max_window_frac))

    for _ in range(n_windows):
        win_len = np.random.randint(min_window_len, max_window_len + 1)
        win_len = min(win_len, total_length)

        start = np.random.randint(0, total_length - win_len + 1)
        irregular = generate_irregular_sequence(win_len).astype(np.int32)

        seq[start:start + win_len] = irregular

    return seq, 1


def generate_sample(total_length=None, include_mixed=True, probs=(0.45, 0.45, 0.10)):
    if total_length is None:
        total_length = sample_length()

    if sum(probs) != 1.0:
        raise ValueError("probs must sum to 1.0")

    if  not include_mixed and len(probs) != 2:
        raise ValueError("probs must have length 2 when include_mixed is False")
    
    if include_mixed and len(probs) != 3:
        raise ValueError("probs must have length 3 when include_mixed is True")

    r = np.random.rand()

    if include_mixed:
        if r < probs[0]:
            seq = generate_regular_sequence(total_length)
            label = 0
        elif r < probs[0] + probs[1]:
            seq = generate_irregular_sequence(total_length)
            label = 1
        else:
            seq, label = make_mixed_sequence(total_length)
    
    else:
        if r < probs[0]:
            seq = generate_regular_sequence(total_length)
            label = 0
        else:
            seq = generate_irregular_sequence(total_length)
            label = 1

    return seq.astype(np.int32), label


class SequenceDataset(Dataset):
    """Fresh random samples on every access."""
    def __init__(self, size: int = 10000, sequence_length: int = None, include_mixed: bool = True, probs=(0.45, 0.45, 0.10)):
        self.size = size
        self.sequence_length = sequence_length
        self.include_mixed = include_mixed
        self.probs = probs

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        #if idx == 0:
        #   print(f"Generating sample with total_length={self.sequence_length}, include_mixed={self.include_mixed}")
        seq, label = generate_sample(total_length=self.sequence_length, include_mixed=self.include_mixed, probs=self.probs)
        x = torch.tensor(seq, dtype=torch.float32)
        y = torch.tensor(label, dtype=torch.float32)
        return x, y


class FixedValidationDataset(Dataset):
    """Pre-generated fixed validation set."""
    def __init__(self, size: int = 2000, seed: int = 12345, sequence_length: int = None, include_mixed: bool = True, probs=(0.45, 0.45, 0.10)):
        self.size = size
        self.samples = []
        self.include_mixed = include_mixed
        self.probs = probs
        self.sequence_length = sequence_length

        rng_state = np.random.get_state()
        np.random.seed(seed)

        for _ in range(size):
            seq, label = generate_sample(total_length=self.sequence_length, include_mixed=self.include_mixed, probs=self.probs)
            self.samples.append(
                (
                    torch.tensor(seq, dtype=torch.float32),
                    torch.tensor(label, dtype=torch.float32),
                )
            )

        np.random.set_state(rng_state)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.samples[idx]

    def describe(self, print_summary: bool = True):
        lengths = np.array([len(seq) for seq, _ in self.samples], dtype=np.int64)
        labels = np.array([int(label.item()) for _, label in self.samples], dtype=np.int64)

        summary = {
            "num_sequences": len(self.samples),
            "length": {
                "min": int(lengths.min()),
                "max": int(lengths.max()),
                "mean": float(lengths.mean()),
                "median": float(np.median(lengths)),
                "std": float(lengths.std()),
                "p05": float(np.percentile(lengths, 5)),
                "p25": float(np.percentile(lengths, 25)),
                "p50": float(np.percentile(lengths, 50)),
                "p75": float(np.percentile(lengths, 75)),
                "p95": float(np.percentile(lengths, 95)),
            },
            "class_distribution": {
                int(cls): {
                    "count": int((labels == cls).sum()),
                    "percent": float(100.0 * (labels == cls).mean()),
                }
                for cls in np.unique(labels)
            },
        }

        if print_summary:
            print("\n" + "=" * 50)
            print("Validation Dataset Summary:")
            print(f"Number of sequences: {summary['num_sequences']}")
            print("Length statistics:")
            print(f"  min:    {summary['length']['min']}")
            print(f"  max:    {summary['length']['max']}")
            print(f"  mean:   {summary['length']['mean']:.2f}")
            print(f"  median: {summary['length']['median']:.2f}")
            print(f"  std:    {summary['length']['std']:.2f}")
            print(
                "  p05/p25/p50/p75/p95: "
                f"[{summary['length']['p05']:.2f}, "
                f"{summary['length']['p25']:.2f}, "
                f"{summary['length']['p50']:.2f}, "
                f"{summary['length']['p75']:.2f}, "
                f"{summary['length']['p95']:.2f}]"
            )
            print("\nClass distribution:")
            for cls, stats in summary["class_distribution"].items():
                print(f"  class={cls}: {stats['count']} ({stats['percent']:.2f}%)")
            print("=" * 50 + "\n")

        return summary


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def evaluate_model_detailed(model, loader, device, threshold=0.5):
    model.eval()

    losses = []
    logits = []
    labels = []

    criterion = nn.BCEWithLogitsLoss()

    with torch.no_grad():
        for x, y in loader:
            #print(f"Evaluating sample with length {x.shape[1]}")  # Debug print for sequence length
            x, y = x.to(device), y.to(device)

            model_out = model(x).squeeze(1)
            loss = criterion(model_out, y)

            losses.append(loss.item())
            logits.append(model_out.cpu())
            labels.append(y.cpu())

    avg_loss = float(np.mean(losses))
    all_logits = torch.cat(logits).numpy()
    all_labels = torch.cat(labels).numpy()
    all_probs = 1.0 / (1.0 + np.exp(-all_logits))

    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = float("nan")

    preds = (all_probs >= 0.5).astype(int)
    acc = (preds == all_labels).mean()

    cm = confusion_matrix(all_labels.astype(int), preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    predicted_irregular_rate = float(preds.mean())
    
    return {
        "loss": avg_loss,
        "acc": float(acc),
        "auc": float(auc),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "predicted_irregular_rate": predicted_irregular_rate,
    }


def evaluate_model(model, loader, device):
    m = evaluate_model_detailed(model, loader, device)
    return m["loss"], m["acc"], m["auc"]


def print_detailed_metrics(name: str, m: dict):
    total = m['tn'] + m['fp'] + m['fn'] + m['tp']
    print(f"{name} | loss: {m['loss']:.4f} | acc: {m['acc']:.4f} | AUC: {m['auc']:.4f}")
    print(
        f"  confusion matrix [rows=true, cols=pred]: "
        f"TN={m['tn']/total:.2f} FP={m['fp']/total:.2f} FN={m['fn']/total:.2f} TP={m['tp']/total:.2f}"
    )
    print(f"  predicted irregular rate: {100.0 * m['predicted_irregular_rate']:.2f}%")


def build_lr_lambda(total_steps: int, warmup_steps: int, min_lr_ratio: float):
    """
    Returns a lambda for LambdaLR:
    - linear warmup from 0 to 1 over warmup_steps
    - cosine decay from 1 to min_lr_ratio over the remaining steps
    """
    def lr_lambda(step: int) -> float:
        if total_steps <= 0:
            return 1.0

        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)

        if total_steps <= warmup_steps:
            return 1.0

        progress = (step - warmup_steps) / float(total_steps - warmup_steps)
        progress = min(max(progress, 0.0), 1.0)

        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return lr_lambda


def train(
    num_epochs: int = 5,
    batch_size: int = 32,
    lr: float = 1e-3,
    dataset_size: int = 10000,
    val_size: int = 2000,
    sequence_length: int = None,
    include_mixed: bool = True,
    probs: tuple = (0.45, 0.45, 0.10),
    run_id: int = None,
    checkpoint_path: str = "model.pth",
    seed: int = 24,
    verbose: bool = True,
    scheduler_name: str | None = None,
    warmup_frac: float = 0.0,
    min_lr_ratio: float = 0.1,
):
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if verbose:
        print(f"Using device: {device}")
        print(f"Random seeds: numpy={np.random.get_state()[1][0]}, torch={torch.initial_seed()}")
        

    # Data
    train_dataset = SequenceDataset(size=dataset_size, sequence_length=sequence_length, include_mixed=include_mixed, probs=probs)
    val_dataset = FixedValidationDataset(size=val_size, seed=12345, include_mixed=True, probs=(0.8,0.05,0.15))
    val_dataset_small = FixedValidationDataset(size=val_size//4, sequence_length=sequence_length, seed=54321, include_mixed=False, probs=(0.8,0.2))
    #val_dataset.describe()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0
    )

    val_loader_small = DataLoader(
        val_dataset_small,
        batch_size=1,
        shuffle=False,
        num_workers=0
    )

    #hyperparameters info
    print(f"{run_id}: Training started.")

    # Model
    model = create_model().to(device)
    if verbose:
        print(f"Training with hyperparameters: num_epochs={num_epochs}, batch_size={batch_size}, lr={lr}, dataset_size={dataset_size}, val_size={val_size}, sequence_length={sequence_length}, include_mixed={include_mixed}, probs={probs}")
        print(f"Model has {sum(p.numel() for p in model.parameters())} parameters")
    #print(model)

    # Optimizer & loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    steps_per_epoch = len(train_loader)
    total_steps = num_epochs * steps_per_epoch
    warmup_steps = int(warmup_frac * total_steps)

    scheduler = None
    if scheduler_name is not None:
        if scheduler_name == "warmup_cosine":
            lr_lambda = build_lr_lambda(
                total_steps=total_steps,
                warmup_steps=warmup_steps,
                min_lr_ratio=min_lr_ratio,
            )
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        else:
            raise ValueError(f"Unknown scheduler_name: {scheduler_name}")

    best_val_loss = float("inf")
    best_model_path = checkpoint_path
    best_epoch = 0

    for epoch in range(1, num_epochs + 1):
        model.train()

        train_losses = []
        train_logits = []
        train_labels = []

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            model_out = model(x).squeeze(1)
            loss = criterion(model_out, y)
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            train_losses.append(loss.item())
            train_logits.append(model_out.detach().cpu())
            train_labels.append(y.detach().cpu())

        # train metrics
        train_loss = float(np.mean(train_losses))
        train_logits_np = torch.cat(train_logits).numpy()
        train_labels_np = torch.cat(train_labels).numpy()
        train_probs = 1.0 / (1.0 + np.exp(-train_logits_np))

        try:
            train_auc = roc_auc_score(train_labels_np, train_probs)
        except ValueError:
            train_auc = float("nan")

        train_preds = (train_probs >= 0.5).astype(int)
        train_acc = (train_preds == train_labels_np).mean()

        # validation metrics
        val_loss, val_acc, val_auc = evaluate_model(model, val_loader, device)
        val_loss_small, val_acc_small, val_auc_small = evaluate_model(model, val_loader_small, device)

        current_lr = optimizer.param_groups[0]["lr"]

        if verbose:
            print(
                f"Epoch {epoch}/{num_epochs} | "
                f"lr: {current_lr:.6g} | "
                f"train loss: {train_loss:.4f} | train acc: {train_acc:.4f} | train AUC: {train_auc:.4f} ||| "
                f"val loss: {val_loss:.4f} | val acc: {val_acc:.4f} | val AUC: {val_auc:.4f} ||| "
                f"val loss (small): {val_loss_small:.4f} | val acc (small): {val_acc_small:.4f} | val AUC (small): {val_auc_small:.4f}"
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            if verbose:
                print(f"  saved new best model with val loss = {val_loss:.4f}")
            best_epoch = epoch

    print(f"{run_id}: Training complete.")
    if verbose:
        print(f"Best validation loss: {best_val_loss:.4f}")
        print("=" * 25)

    best_model = create_model().to(device)
    best_model.load_state_dict(torch.load(best_model_path, map_location=device))

    print("\nBest checkpoint final evaluation:")
    best_main = evaluate_model_detailed(best_model, val_loader, device)
    best_small = evaluate_model_detailed(best_model, val_loader_small, device)

    print_detailed_metrics("val", best_main)
    print_detailed_metrics("val (small)", best_small)
    print("=" * 40)

    return {
        "run_id": run_id,
        "seed": seed,
        "batch_size": batch_size,
        "lr": lr,
        "sequence_length": sequence_length,
        "num_epochs": num_epochs,
        "dataset_size": dataset_size,
        "val_size": val_size,
        "include_mixed": include_mixed,
        "probs": str(probs),
        "best_epoch": best_epoch,
        "checkpoint_path": checkpoint_path,
        "scheduler_name": scheduler_name if scheduler_name is not None else "",
        "warmup_frac": warmup_frac,
        "min_lr_ratio": min_lr_ratio,

        "main_val_loss": best_main["loss"],
        "main_val_acc": best_main["acc"],
        "main_val_auc": best_main["auc"],
        "main_val_tn": best_main["tn"],
        "main_val_fp": best_main["fp"],
        "main_val_fn": best_main["fn"],
        "main_val_tp": best_main["tp"],
        "main_val_pred_irregular_rate": best_main["predicted_irregular_rate"],

        "small_val_loss": best_small["loss"],
        "small_val_acc": best_small["acc"],
        "small_val_auc": best_small["auc"],
        "small_val_tn": best_small["tn"],
        "small_val_fp": best_small["fp"],
        "small_val_fn": best_small["fn"],
        "small_val_tp": best_small["tp"],
        "small_val_pred_irregular_rate": best_small["predicted_irregular_rate"],
    }


if __name__ == "__main__":
    """
    train(
        num_epochs=30,
        batch_size=128,
        lr=3e-3,
        dataset_size=12800 * 2,
        val_size=2000,
        sequence_length=1000,
        include_mixed=True,
        probs=(0.6, 0.1, 0.3) #probs=(0.5,0.5)
    )
    """
    result = train(
            num_epochs=30,
            batch_size=128,
            lr=3e-3,
            dataset_size=12800 * 4,
            val_size=2000,
            sequence_length=1000,
            include_mixed=True,
            probs=(0.6, 0.1, 0.3),
            run_id='best',
            checkpoint_path='model.pth',
            seed=303,
            scheduler_name="warmup_cosine",
            warmup_frac=0.05,
            min_lr_ratio=0.1,
            verbose=True
        )