import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.custom_student import CustomStudentNet
from models.student import StudentNet
from utils.dataset import get_voc
from utils.metrics import mean_iou, update_confusion
from utils.profiler import ms_per_image
from utils.teacher import load_teacher
from utils.visualization import save_comparison


def parse_args():
    parser = argparse.ArgumentParser(description="Compare VOC segmentation models.")
    parser.add_argument("--data", default="./data")
    parser.add_argument("--size", type=int, default=320)
    parser.add_argument("-b", "--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    loader = DataLoader(
        get_voc(args.data, "val", args.size),
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.workers,
    )

    models: dict[str, nn.Module] = {
        "Teacher (FCN-ResNet50)": load_teacher(device)
    }
    students = (
        ("none", "Student, no KD", StudentNet, "student_none.pth"),
        ("response", "Student, response KD", StudentNet, "student_response.pth"),
        ("feature", "Student, feature KD", StudentNet, "student_feature.pth"),
        ("custom", "Custom student, response KD", CustomStudentNet,
         "student_custom_response.pth"),
    )
    for _, name, model_type, checkpoint_name in students:
        checkpoint = Path("checkpoints") / checkpoint_name
        if not checkpoint.is_file():
            print(f"Skipping {name}: {checkpoint} not found")
            continue
        model = model_type(pretrained=False).to(device).eval()
        model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
        models[name] = model

    rows = []
    predictions_by_model = {}
    first_images = first_masks = None
    with torch.inference_mode():
        for name, model in models.items():
            confusion = torch.zeros(21, 21, dtype=torch.long)
            for batch_index, (images, masks) in enumerate(loader):
                predictions = model(images.to(device))["out"].argmax(1).cpu()
                confusion = update_confusion(confusion, predictions, masks)
                if batch_index == 0:
                    predictions_by_model[name] = predictions[:4]
                    if first_images is None:
                        first_images, first_masks = images[:4], masks[:4]
            parameter_count = sum(parameter.numel() for parameter in model.parameters())
            latency = ms_per_image(model, device, args.size)
            rows.append((name, 100 * mean_iou(confusion), parameter_count, latency))

    print(f"\nDevice: {device}")
    print(f'{"Model":<34}{"mIoU":>8}{"Params":>14}{"ms/img":>10}')
    for name, miou, parameter_count, latency in rows:
        print(f"{name:<34}{miou:>8.1f}{parameter_count:>14,}{latency:>10.1f}")

    if first_images is None or first_masks is None:
        raise RuntimeError("The VOC validation split contains no samples.")
    save_comparison(
        first_images,
        first_masks,
        predictions_by_model,
        "plots/compare.png",
    )
    print("\nSaved plots/compare.png")


if __name__ == "__main__":
    main()
