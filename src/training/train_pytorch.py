"""
PyTorch training loop for the ResNet50 comparison model.

Mirrors the Keras schedule: Adam, ReduceLROnPlateau, early stopping,
class-weighted BCE for the benign/malignant imbalance.
"""

from pathlib import Path
import copy

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm


def _epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train() if train else model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for imgs, labels in tqdm(loader, desc="train" if train else "val", leave=False):
            imgs = imgs.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.float32).unsqueeze(1)

            logits = model(imgs)
            loss = criterion(logits, labels)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            total_loss += loss.item() * imgs.size(0)
            correct += (preds == labels).sum().item()
            total += imgs.size(0)
            all_preds.append(probs.detach().cpu().numpy())
            all_labels.append(labels.detach().cpu().numpy())

    return {
        "loss": total_loss / total,
        "accuracy": correct / total,
        "preds": np.concatenate(all_preds),
        "labels": np.concatenate(all_labels),
    }


def train_pytorch_model(
    model: nn.Module,
    train_dataset,
    val_dataset,
    config: dict,
    model_name: str = "resnet50_pytorch",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    model = model.to(device)

    batch_size = config["training"]["batch_size"]
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # Class weights
    labels = np.array([int(y.item()) for _, y in train_dataset])
    classes = np.unique(labels)
    cw = compute_class_weight(class_weight="balanced", classes=classes, y=labels)
    pos_weight = torch.tensor([cw[1] / cw[0]], device=device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min",
        patience=config["training"]["reduce_lr_patience"],
        factor=config["training"]["reduce_lr_factor"],
    )

    best_val_loss = float("inf")
    best_state = None
    patience = config["training"]["early_stopping_patience"]
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(config["training"]["epochs"]):
        print(f"\nEpoch {epoch + 1}/{config['training']['epochs']}")
        train_metrics = _epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_metrics = _epoch(model, val_loader, criterion, optimizer, device, train=False)

        history["train_loss"].append(train_metrics["loss"])
        history["val_loss"].append(val_metrics["loss"])
        history["train_acc"].append(train_metrics["accuracy"])
        history["val_acc"].append(val_metrics["accuracy"])

        print(f"  train_loss={train_metrics['loss']:.4f} acc={train_metrics['accuracy']:.4f} | "
              f"val_loss={val_metrics['loss']:.4f} acc={val_metrics['accuracy']:.4f}")

        scheduler.step(val_metrics["loss"])

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        save_dir = Path("models/saved")
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), save_dir / f"{model_name}.pth")

    return model, history
