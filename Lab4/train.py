import argparse
import json
import platform
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models.custom_student import CustomStudentNet
from models.student import StudentNet
from utils.dataset import get_voc
from utils.kd import feature_kd_loss, response_kd_loss
from utils.metrics import mean_iou, update_confusion
from utils.teacher import load_teacher, teacher_forward


def parse_args():
    parser = argparse.ArgumentParser(description="Train a VOC segmentation student.")
    parser.add_argument("--kd", choices=("none", "response", "feature"), default="none")
    parser.add_argument("--model", choices=("baseline", "custom"), default="baseline")
    parser.add_argument("--data", default="./data")
    parser.add_argument("-e", "--epochs", type=int, default=3)
    parser.add_argument("-b", "--batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--size", type=int, default=320)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--tau", type=float, default=4.0)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--iters", type=int, default=0,
                        help="Stop after this many iterations (0 = full training).")
    parser.add_argument("--lr-patience", type=int, default=2)
    parser.add_argument("--lr-factor", type=float, default=0.5)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--tol", type=float, default=0.1,
                        help="Minimum validation mIoU improvement, in percentage points.")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch < 1 or args.size < 1:
        parser.error("--epochs, --batch, and --size must be positive")
    if args.iters < 0 or args.workers < 0:
        parser.error("--iters and --workers cannot be negative")
    if args.lr <= 0 or args.lr_patience < 0 or args.patience < 1:
        parser.error("--lr must be positive, --lr-patience non-negative, and --patience positive")
    if not 0 < args.lr_factor < 1 or args.tol < 0 or args.tau <= 0:
        parser.error("--lr-factor must be between 0 and 1, --tol non-negative, and --tau positive")
    return args


def main():
    args = parse_args()
    torch.manual_seed(0)
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    train_loader = DataLoader(
        get_voc(args.data, "train", args.size),
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.workers,
        drop_last=True,
    )
    val_loader = DataLoader(
        get_voc(args.data, "val", args.size),
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.workers,
    )
    if not len(train_loader):
        raise ValueError(
            f"The training split has fewer than {args.batch} samples; lower --batch."
        )

    student_type = CustomStudentNet if args.model == "custom" else StudentNet
    student = student_type().to(device)
    params = list(student.parameters())
    teacher = load_teacher(device) if args.kd != "none" else None
    adapters = None
    if args.kd == "feature":
        adapters = nn.ModuleDict(
            {
                "low": nn.Conv2d(student.low_channels, 512, kernel_size=1),
                "high": nn.Conv2d(student.high_channels, 2048, kernel_size=1),
            }
        ).to(device)
        params.extend(adapters.parameters())

    optimizer = torch.optim.Adam(params, lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=args.lr_factor,
        patience=args.lr_patience,
    )
    checkpoint_name = (
        f"student_{args.kd}.pth"
        if args.model == "baseline"
        else f"student_custom_{args.kd}.pth"
    )
    checkpoint_path = Path("checkpoints") / checkpoint_name
    results_path = Path("results") / checkpoint_name.replace(".pth", ".json")
    Path("plots").mkdir(exist_ok=True)
    Path("checkpoints").mkdir(exist_ok=True)
    Path("results").mkdir(exist_ok=True)

    print(
        f"Training {args.model} student with kd={args.kd} on {device}"
        + (f" for {args.iters} iterations" if args.iters else f" for up to {args.epochs} epochs")
    )

    losses = []
    best_miou = float("-inf")
    best_epoch = 0
    stale_epochs = 0
    started = time.perf_counter()
    epoch = 0
    while True:
        epoch += 1
        student.train()
        if adapters is not None:
            adapters.train()
        epoch_losses = []

        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = student(images)
            if args.kd == "none":
                loss = F.cross_entropy(outputs["out"], masks, ignore_index=255)
            else:
                if teacher is None:
                    raise RuntimeError("The teacher was not initialized for knowledge distillation.")
                with torch.no_grad():
                    teacher_logits, teacher_features = teacher_forward(teacher, images)
                if args.kd == "response":
                    loss = response_kd_loss(
                        outputs["out"], teacher_logits, masks,
                        args.alpha, args.beta, args.tau,
                    )
                else:
                    if adapters is None:
                        raise RuntimeError("Feature distillation adapters were not initialized.")
                    loss = F.cross_entropy(outputs["out"], masks, ignore_index=255)
                    loss = loss + args.gamma * (
                        feature_kd_loss(
                            adapters["low"](outputs["low"]), teacher_features["layer2"]
                        )
                        + feature_kd_loss(
                            adapters["high"](outputs["high"]), teacher_features["layer4"]
                        )
                    )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_value = loss.item()
            losses.append(loss_value)
            epoch_losses.append(loss_value)

            if args.iters and len(losses) % 5 == 0:
                print(f"iter {len(losses):3d}/{args.iters}  loss {loss_value:.3f}")
            if args.iters and len(losses) >= args.iters:
                break

        if args.iters:
            if len(losses) >= args.iters:
                break
            continue

        student.eval()
        confusion = torch.zeros(21, 21, dtype=torch.long)
        with torch.inference_mode():
            for images, masks in val_loader:
                predictions = student(images.to(device))["out"].argmax(1).cpu()
                confusion = update_confusion(confusion, predictions, masks)
        val_miou = 100 * mean_iou(confusion)
        if not torch.isfinite(torch.tensor(val_miou)):
            raise RuntimeError("Validation mIoU is not finite; check the validation dataset.")

        if val_miou > best_miou + args.tol:
            best_miou = val_miou
            best_epoch = epoch
            stale_epochs = 0
            torch.save(student.state_dict(), checkpoint_path)
        else:
            stale_epochs += 1

        scheduler.step(val_miou)
        learning_rate = optimizer.param_groups[0]["lr"]
        elapsed = time.perf_counter() - started
        print(
            f"epoch {epoch}/{args.epochs}  loss {sum(epoch_losses) / len(epoch_losses):.3f}"
            f"  val mIoU {val_miou:.2f}  lr {learning_rate:.2g}"
            f"  elapsed {elapsed:.0f}s"
        )
        if stale_epochs >= args.patience:
            print(f"Early stopping after {stale_epochs} epochs without improvement.")
            break
        if epoch >= args.epochs:
            break

    training_seconds = time.perf_counter() - started
    if args.iters:
        if len(losses) < args.iters:
            raise RuntimeError(
                f"Training ended after {len(losses)} iterations, fewer than requested {args.iters}."
            )
        first_count = min(5, len(losses))
        first = sum(losses[:first_count]) / first_count
        last = sum(losses[-first_count:]) / first_count
        print(
            f"Kick-start finished: average loss {first:.3f} (first {first_count})"
            f" -> {last:.3f} (last {first_count})"
        )
    else:
        result = {
            "model": args.model,
            "kd": args.kd,
            "epochs": epoch,
            "best_epoch": best_epoch,
            "best_validation_miou": best_miou,
            "training_seconds": training_seconds,
            "device": str(device),
            "hardware": {
                "platform": platform.platform(),
                "processor": platform.processor() or platform.machine(),
                "cuda_available": torch.cuda.is_available(),
            },
            "arguments": vars(args),
        }
        results_path.write_text(json.dumps(result, indent=2) + "\n")
        print(f"Saved best checkpoint: {checkpoint_path} (epoch {best_epoch})")
        print(f"Saved run metrics: {results_path}")

    plt.figure(figsize=(6, 4))
    plt.plot(losses)
    plt.xlabel("iteration")
    plt.ylabel("loss")
    plt.title(f"Training loss ({args.model}, kd={args.kd})")
    plt.tight_layout()
    loss_path = Path("plots") / f"loss_{args.model}_{args.kd}.png"
    if args.model == "baseline":
        loss_path = Path("plots") / f"loss_{args.kd}.png"
    plt.savefig(loss_path, dpi=100)
    plt.close()
    print(f"Saved {loss_path}")


if __name__ == "__main__":
    main()
