"""Evaluate pretrained FCN-ResNet50 on the PASCAL VOC 2012 validation set.

Run locally with:
    python prelab.py --root ./data

In Colab, install PyTorch/torchvision if they are not already available, then
run this script with a writable data directory:
    %pip install torch torchvision
    !python /content/prelab.py --root /content/data

The old starter example imported ``helpers.plot``. This evaluator does not
need that helper; to use it in another Colab cell, download it with:
    !wget -q https://raw.githubusercontent.com/pytorch/vision/main/gallery/transforms/helpers.py -O helpers.py
"""

# %%

#start code

import argparse
from pathlib import Path

import torch
from torchvision.datasets import VOCSegmentation
from torchvision.models.segmentation import FCN_ResNet50_Weights, fcn_resnet50
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import pil_to_tensor, resize


IGNORE_INDEX = 255


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate pretrained FCN-ResNet50 on PASCAL VOC 2012."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent / "data",
        help="Dataset directory (default: Lab4/data).",
    )
    parser.add_argument(
        "--split",
        choices=("train", "val", "trainval"),
        default="val",
        help="VOC split to evaluate (default: val).",
    )
    parser.add_argument(
        "--year",
        default="2012",
        help="VOC dataset year (default: 2012).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optionally evaluate only the first N samples, for a quick test.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Evaluation device, such as cuda or cpu (default: auto-detect).",
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Do not download VOC if it is missing from --root.",
    )
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be a positive integer")
    return args


def main() -> None:
    args = parse_args()
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )

    weights = FCN_ResNet50_Weights.DEFAULT
    preprocess = weights.transforms()
    model = fcn_resnet50(weights=weights).to(device)
    model.eval()

    dataset = VOCSegmentation(
        root=str(args.root),
        year=args.year,
        image_set=args.split,
        download=not args.no_download,
    )
    sample_count = len(dataset)
    if args.limit is not None:
        sample_count = min(sample_count, args.limit)
    if sample_count == 0:
        raise RuntimeError("The VOC dataset split contains no samples.")

    class_names = weights.meta["categories"]
    class_count = len(class_names)
    confusion = torch.zeros(
        (class_count, class_count), dtype=torch.int64, device=device
    )

    with torch.inference_mode():
        for index in range(sample_count):
            image, target = dataset[index]
            image_tensor = preprocess(image)
            target = resize(
                target,
                list(image_tensor.shape[-2:]),
                interpolation=InterpolationMode.NEAREST,
            )
            image_tensor = image_tensor.unsqueeze(0).to(device)
            target_tensor = pil_to_tensor(target).squeeze(0).to(
                device=device, dtype=torch.int64
            )

            prediction = model(image_tensor)["out"].argmax(dim=1).squeeze(0)
            if prediction.shape != target_tensor.shape:
                raise RuntimeError(
                    f"Model output and target sizes differ at sample {index}: "
                    f"{tuple(prediction.shape)} vs {tuple(target_tensor.shape)}."
                )
            valid = target_tensor != IGNORE_INDEX
            valid_targets = target_tensor[valid]
            valid_predictions = prediction[valid]

            if valid_targets.numel() and (
                valid_targets.min() < 0 or valid_targets.max() >= class_count
            ):
                raise ValueError(
                    f"Unexpected class ID in VOC target at sample {index}."
                )
            encoded = class_count * valid_targets + valid_predictions
            confusion += torch.bincount(
                encoded, minlength=class_count * class_count
            ).reshape(class_count, class_count)

            if (index + 1) % 100 == 0 or index + 1 == sample_count:
                print(f"Evaluated {index + 1}/{sample_count} images")

    intersection = confusion.diag().to(torch.float64)
    union = confusion.sum(dim=0) + confusion.sum(dim=1) - confusion.diag()
    present = union > 0
    if not present.any():
        raise RuntimeError("No valid VOC pixels were found for mIoU evaluation.")
    iou = intersection / union.clamp_min(1)
    mean_iou = iou[present].mean().item()

    print(f"\nSplit: VOC {args.year} {args.split} ({sample_count} images)")
    print(f"Device: {device}")
    print("Per-class IoU:")
    for index, name in enumerate(class_names):
        if present[index]:
            print(f"  {name}: {iou[index].item():.4f}")
        else:
            print(f"  {name}: n/a (no target or prediction pixels)")
    print(f"mIoU: {mean_iou:.4f}")


if __name__ == "__main__":
    main()


# %%

main()


# %%
