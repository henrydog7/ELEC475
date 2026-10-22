from pathlib import Path

import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from .dataset import IMG_SIZE, PetNoseDataset, denormalize

LAB3_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = LAB3_DIR / "data" / "oxford-iiit-pet-noses"
PLOTS_DIR = LAB3_DIR / "plots"


def check_split(split: str):
    dataset = PetNoseDataset(root=str(DATA_DIR), split=split)
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    sample_count = 0
    training_examples = []

    for images, labels, _ in loader:
        if images.shape[1:] != (3, IMG_SIZE, IMG_SIZE):
            raise AssertionError(
                f"{split} images have shape {tuple(images.shape[1:])}; "
                f"expected (3, {IMG_SIZE}, {IMG_SIZE})"
            )

        labels_in_frame = ((labels >= 0) & (labels < IMG_SIZE)).all(dim=1)
        if not labels_in_frame.all():
            invalid = labels[~labels_in_frame]
            raise AssertionError(
                f"{split} contains out-of-frame labels: {invalid.tolist()}"
            )

        if split == "train" and len(training_examples) < 8:
            needed = 8 - len(training_examples)
            training_examples.extend(
                (images[i], labels[i]) for i in range(min(needed, len(images)))
            )
        sample_count += len(images)

    if sample_count == 0:
        raise AssertionError(f"The {split} split is empty")

    print(f"{split}: validated {sample_count} samples")
    return training_examples


def save_training_examples(examples):
    if len(examples) < 8:
        raise AssertionError(f"Need 8 training images to plot; found {len(examples)}")

    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    for index, (image, label) in enumerate(examples):
        ax = axes.flat[index]
        ax.imshow(denormalize(image).permute(1, 2, 0).numpy())
        ax.scatter(label[0].item(), label[1].item(), c="red", s=35, marker="o")
        ax.set_xlim(-0.5, IMG_SIZE - 0.5)
        ax.set_ylim(IMG_SIZE - 0.5, -0.5)
        ax.set_title(f"Training image {index + 1}")
        ax.axis("off")

    fig.tight_layout()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PLOTS_DIR / "dataset_check.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot to {output_path}")


def main():
    training_examples = check_split("train")
    check_split("test")
    save_training_examples(training_examples)


if __name__ == "__main__":
    main()