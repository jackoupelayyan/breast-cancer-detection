"""
Main entry point — trains all four models end-to-end and writes a
side-by-side comparison table to results/metrics/comparison.json.

Usage:
    python -m src.run_experiments --config configs/config.yaml
    python -m src.run_experiments --config configs/config.yaml --models baseline_cnn vgg16
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

from src.data import (
    build_dataframe, get_keras_augmenter, get_albumentations_pipeline,
    KerasDataLoader, PyTorchDataset,
)
from src.models import (
    build_baseline_cnn, build_vgg16, build_unet_classifier, build_resnet_pytorch,
)
from src.training import train_keras_model, train_pytorch_model
from src.evaluation import (
    compute_all_metrics, save_metrics_json,
    plot_confusion_matrix, plot_roc_curve, plot_training_curves,
)


def split_data(df: pd.DataFrame, config: dict):
    ds = config["dataset"]
    train_val, test = train_test_split(
        df, test_size=ds["test_split"],
        stratify=df["label"], random_state=ds["random_seed"],
    )
    val_size_relative = ds["val_split"] / (1 - ds["test_split"])
    train, val = train_test_split(
        train_val, test_size=val_size_relative,
        stratify=train_val["label"], random_state=ds["random_seed"],
    )
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def predict_keras(model, loader):
    y_prob, y_true = [], []
    for i in range(len(loader)):
        x, y = loader[i]
        y_prob.append(model.predict(x, verbose=0).ravel())
        y_true.append(y)
    return np.concatenate(y_true), np.concatenate(y_prob)


def predict_pytorch(model, dataset, batch_size: int):
    import torch
    from torch.utils.data import DataLoader
    device = next(model.parameters()).device
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    y_prob, y_true = [], []
    model.eval()
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device, dtype=torch.float32)
            probs = torch.sigmoid(model(imgs)).cpu().numpy().ravel()
            y_prob.append(probs)
            y_true.append(labels.numpy().ravel())
    return np.concatenate(y_true), np.concatenate(y_prob)


def run_keras_experiment(name: str, build_fn, train_df, val_df, test_df, config, fine_tune=False):
    print(f"\n{'=' * 60}\nTraining {name}\n{'=' * 60}")
    augmenter = get_keras_augmenter(config)
    img_size = config["dataset"]["image_size"]
    bs = config["training"]["batch_size"]
    train_loader = KerasDataLoader(train_df, batch_size=bs, image_size=img_size, augmenter=augmenter)
    val_loader = KerasDataLoader(val_df, batch_size=bs, image_size=img_size, shuffle=False)
    test_loader = KerasDataLoader(test_df, batch_size=bs, image_size=img_size, shuffle=False)

    model = build_fn(input_shape=(img_size, img_size, 3))
    model, history = train_keras_model(model, train_loader, val_loader, config, name, fine_tune=fine_tune)

    y_true, y_prob = predict_keras(model, test_loader)
    metrics = compute_all_metrics(y_true, y_prob)
    _save_artifacts(name, history, y_true, y_prob, metrics)
    return metrics


def run_pytorch_experiment(name: str, train_df, val_df, test_df, config):
    print(f"\n{'=' * 60}\nTraining {name}\n{'=' * 60}")
    train_t = get_albumentations_pipeline(config, train=True)
    eval_t = get_albumentations_pipeline(config, train=False)
    train_ds = PyTorchDataset(train_df, transform=train_t, image_size=config["dataset"]["image_size"])
    val_ds = PyTorchDataset(val_df, transform=eval_t, image_size=config["dataset"]["image_size"])
    test_ds = PyTorchDataset(test_df, transform=eval_t, image_size=config["dataset"]["image_size"])

    model = build_resnet_pytorch(num_classes=config["dataset"]["num_classes"])
    model, history = train_pytorch_model(model, train_ds, val_ds, config, model_name=name)

    y_true, y_prob = predict_pytorch(model, test_ds, batch_size=config["training"]["batch_size"])
    metrics = compute_all_metrics(y_true, y_prob)
    _save_artifacts(name, history, y_true, y_prob, metrics)
    return metrics


def _save_artifacts(name, history, y_true, y_prob, metrics):
    save_metrics_json(metrics, f"results/metrics/{name}_metrics.json")
    plot_training_curves(history, f"results/figures/{name}_training.png", title=f"{name} — training")
    plot_confusion_matrix(y_true, (y_prob >= 0.5).astype(int),
                          f"results/figures/{name}_confusion.png",
                          title=f"{name} — confusion matrix")
    plot_roc_curve(y_true, y_prob, f"results/figures/{name}_roc.png", title=f"{name} — ROC")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--models", nargs="+",
                        default=["baseline_cnn", "vgg16", "unet", "resnet_pytorch"])
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Build the dataset DataFrame
    raw_dir = Path(config["dataset"]["raw_dir"])
    metadata_csv = raw_dir / "metadata.csv"
    if not metadata_csv.exists():
        raise FileNotFoundError(
            f"Could not find {metadata_csv}. Download CBIS-DDSM and place metadata.csv "
            f"under {raw_dir}. See README for download instructions."
        )
    df = build_dataframe(metadata_csv, raw_dir)
    train_df, val_df, test_df = split_data(df, config)
    print(f"Splits — train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)}")

    comparison = {}

    if "baseline_cnn" in args.models:
        comparison["baseline_cnn"] = run_keras_experiment(
            "baseline_cnn", build_baseline_cnn, train_df, val_df, test_df, config,
        )
    if "vgg16" in args.models:
        comparison["vgg16"] = run_keras_experiment(
            "vgg16", build_vgg16, train_df, val_df, test_df, config, fine_tune=True,
        )
    if "unet" in args.models:
        comparison["unet"] = run_keras_experiment(
            "unet", build_unet_classifier, train_df, val_df, test_df, config,
        )
    if "resnet_pytorch" in args.models:
        comparison["resnet_pytorch"] = run_pytorch_experiment(
            "resnet_pytorch", train_df, val_df, test_df, config,
        )

    Path("results/metrics").mkdir(parents=True, exist_ok=True)
    with open("results/metrics/comparison.json", "w") as f:
        json.dump(comparison, f, indent=2)

    print("\n=== Final comparison ===")
    summary = pd.DataFrame({
        name: {k: v for k, v in m.items() if isinstance(v, (int, float)) and k != "n_samples"}
        for name, m in comparison.items()
    }).T
    print(summary.round(4))


if __name__ == "__main__":
    main()
