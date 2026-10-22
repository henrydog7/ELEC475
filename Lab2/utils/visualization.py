"""
ELEC 475: Lab 2 - Plotting & Visualization Utilities
"""
import os
import matplotlib.pyplot as plt
import argparse
import json
from glob import glob
from typing import List, Dict

def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    train_accs: List[float],
    val_accs: List[float],
    save_path: str = "./plots/training_curves.png",
    model_name: str = "Model"
):
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 5))

    # Loss Curve
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-o', label='Train Loss')
    plt.plot(epochs, val_losses, 'r--s', label='Val Loss')
    plt.title(f"{model_name.upper()} - Cross-Entropy Loss", fontsize=12)
    plt.xlabel("Epoch", fontsize=10)
    plt.ylabel("Loss", fontsize=10)
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.6)

    # Accuracy Curve
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accs, 'b-o', label='Train Accuracy')
    plt.plot(epochs, val_accs, 'g--^', label='Val Accuracy')
    plt.title(f"{model_name.upper()} - Top-1 Accuracy (%)", fontsize=12)
    plt.xlabel("Epoch", fontsize=10)
    plt.ylabel("Accuracy (%)", fontsize=10)
    plt.legend(loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"Training curves successfully saved to: {save_path}")

def plot_ablation_summary(
    ablation_results: Dict[str, Dict[str, List[float]]],
    save_path: str = "./plots/ablation_summary.png"
):
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.figure(figsize=(10, 6))

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
    styles = ['-', '--', '-.', ':', '-', '--', '-.']

    for idx, (model_label, data) in enumerate(ablation_results.items()):
        val_accs = data.get("val_accs", [])
        epochs = range(1, len(val_accs) + 1)
        color = colors[idx % len(colors)]
        style = styles[idx % len(styles)]
        plt.plot(epochs, val_accs, linestyle=style, marker='o', color=color, label=model_label, linewidth=2)

    plt.title("ELEC 475: Master Ablation Study (Validation Accuracy)", fontsize=13)
    plt.xlabel("Epoch", fontsize=11)
    plt.ylabel("Validation Accuracy (%)", fontsize=11)
    plt.legend(loc='lower right', frameon=True)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"Master ablation summary plot saved to: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot the Lab 2 master ablation results")
    parser.add_argument("--plot-ablation", action="store_true",
                        help="Plot validation accuracy from saved training histories")
    parser.add_argument("--results-dir", default="./results",
                        help="Directory containing JSON histories written by train.py")
    parser.add_argument("--save-path", default="./plots/ablation_summary.png",
                        help="Output path for the comparative plot")
    args = parser.parse_args()

    if not args.plot_ablation:
        parser.error("specify --plot-ablation to generate the comparative plot")

    results = {}
    for path in sorted(glob(os.path.join(args.results_dir, "*.json"))):
        with open(path, "r", encoding="utf-8") as results_file:
            run = json.load(results_file)
        val_accuracies = run.get("val_accuracies")
        if not isinstance(val_accuracies, list) or not val_accuracies:
            raise ValueError(f"Training history has no validation accuracies: {path}")

        label_parts = [run.get("model", "model")]
        if run.get("norm") != "na":
            label_parts.append(run["norm"])
        label_parts.append(run.get("dataset", "dataset"))
        label_parts.append("aug" if run.get("augmentation") else "no aug")
        label_parts.append(f"seed {run.get('seed', '?')}")
        results[" / ".join(label_parts)] = {"val_accs": val_accuracies}

    if not results:
        raise FileNotFoundError(
            f"No training histories found in {args.results_dir!r}; "
            "run train.py first."
        )
    plot_ablation_summary(results, save_path=args.save_path)


if __name__ == "__main__":
    main()
