"""
compressor/distiller.py
────────────────────────
Knowledge Distillation: train a small Student model to mimic a large Teacher.

The key insight (Hinton et al. 2015):
  Instead of training the Student on hard labels (one-hot), we train it on the
  Teacher's SOFTMAX PROBABILITIES at temperature T.  These "soft targets" carry
  information about inter-class relationships that hard labels discard.

  Loss = α * KL_divergence(soft_student || soft_teacher)    ← distillation loss
       + (1-α) * CrossEntropy(student_logits, hard_labels)  ← task loss

Usage:
  python -m compressor.distiller
"""

from __future__ import annotations
import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as T
import torchvision.models as tvm

from configs.settings import cfg
from compressor.pruner import prune_teacher

logger = logging.getLogger(__name__)


# ─── Loss ─────────────────────────────────────────────────────────────────────

class DistillationLoss(nn.Module):
    """
    Combined distillation + task loss.

    Args:
        T:     Temperature.  Higher T → softer probability distribution →
               more information transferred about wrong-class similarities.
        alpha: Weight of the distillation term.  alpha=1.0 means ignore
               hard labels entirely; alpha=0.0 means standard cross-entropy.
    """

    def __init__(self, T: float = 4.0, alpha: float = 0.7) -> None:
        super().__init__()
        self.T = T
        self.alpha = alpha

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:

        # Soft targets: divide logits by T before softmax to spread distribution
        soft_student = F.log_softmax(student_logits / self.T, dim=1)
        soft_teacher = F.softmax(teacher_logits / self.T, dim=1)

        # KL divergence = how different are the two distributions?
        # Multiply by T² to keep gradient magnitudes stable (Hinton et al.)
        distill_loss = F.kl_div(soft_student, soft_teacher, reduction="batchmean") * (self.T ** 2)

        # Hard label loss (ordinary cross-entropy on full-precision logits)
        task_loss = F.cross_entropy(student_logits, labels)

        total = self.alpha * distill_loss + (1.0 - self.alpha) * task_loss

        return total, {
            "distill_loss": distill_loss.item(),
            "task_loss":    task_loss.item(),
            "total_loss":   total.item(),
        }


# ─── Data ─────────────────────────────────────────────────────────────────────

def _build_dataloaders() -> tuple[DataLoader, DataLoader]:
    """
    Build train/val DataLoaders.
    Supports CIFAR-10 and ImageNet-style custom folders.
    """
    h, w = cfg.deployment.input_height, cfg.deployment.input_width

    train_tf = T.Compose([
        T.Resize((h, w)),
        T.RandomHorizontalFlip(),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_tf = T.Compose([
        T.Resize((h, w)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset_name = cfg.model.dataset.lower()
    data_dir = cfg.model.data_dir

    if dataset_name == "cifar10":
        train_ds = torchvision.datasets.CIFAR10(data_dir, train=True,  download=True, transform=train_tf)
        val_ds   = torchvision.datasets.CIFAR10(data_dir, train=False, download=True, transform=val_tf)
    elif dataset_name == "cifar100":
        train_ds = torchvision.datasets.CIFAR100(data_dir, train=True,  download=True, transform=train_tf)
        val_ds   = torchvision.datasets.CIFAR100(data_dir, train=False, download=True, transform=val_tf)
    else:
        # Generic ImageFolder (custom datasets)
        train_ds = torchvision.datasets.ImageFolder(f"{data_dir}/train", transform=train_tf)
        val_ds   = torchvision.datasets.ImageFolder(f"{data_dir}/val",   transform=val_tf)

    import os
    num_workers = min(os.cpu_count() or 1, 4)
    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.model.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.model.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader


# ─── Student model ────────────────────────────────────────────────────────────

def _build_student() -> nn.Module:
    arch = cfg.model.student_arch
    num_classes = cfg.model.num_classes

    logger.info("Building student architecture: %s", arch)

    if arch == "mobilenet_v3_small":
        model = tvm.mobilenet_v3_small(weights=None, num_classes=num_classes)
    elif arch == "mobilenet_v3_large":
        model = tvm.mobilenet_v3_large(weights=None, num_classes=num_classes)
    elif arch == "efficientnet_b0":
        model = tvm.efficientnet_b0(weights=None, num_classes=num_classes)
    elif arch == "efficientnet_b1":
        model = tvm.efficientnet_b1(weights=None, num_classes=num_classes)
    else:
        raise ValueError(f"Unknown student arch: {arch}")

    return model


# ─── Training loop ────────────────────────────────────────────────────────────

def _evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            preds = model(images).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)
    return correct / total


def distill(save_path: Path | None = None) -> nn.Module:
    """
    Full distillation pipeline:
      1. Load (pruned) Teacher
      2. Build Student
      3. Train Student with DistillationLoss for cfg.model.epochs
      4. Save best Student checkpoint

    Returns the trained Student model.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Distillation running on: %s", device)

    save_path = save_path or cfg.student_checkpoint

    # ── Load teacher ───────────────────────────────────────────────────────────
    pruned_path = cfg.output_dir / "teacher_pruned.pth"
    if pruned_path.exists():
        logger.info("Loading pruned teacher from %s", pruned_path)
        # Actually load the teacher arch properly
        import torchvision.models as tvm2
        teacher = tvm2.get_model(cfg.model.teacher_arch, num_classes=cfg.model.num_classes)
        teacher.load_state_dict(torch.load(pruned_path, map_location="cpu"))
    else:
        logger.warning("No pruned teacher found — running pruning first.")
        teacher = prune_teacher()

    teacher = teacher.to(device)
    teacher.eval()
    # Freeze teacher — we only update Student weights
    for p in teacher.parameters():
        p.requires_grad_(False)

    # ── Build student ──────────────────────────────────────────────────────────
    student = _build_student().to(device)

    # ── Optimizer & scheduler ──────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=cfg.model.learning_rate,
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.model.epochs
    )
    criterion = DistillationLoss(
        T=cfg.compression.distillation_temperature,
        alpha=cfg.compression.distillation_alpha,
    )

    train_loader, val_loader = _build_dataloaders()

    best_acc = 0.0
    loss_history: list[dict] = []

    for epoch in range(1, cfg.model.epochs + 1):
        student.train()
        epoch_losses = {"distill_loss": 0.0, "task_loss": 0.0, "total_loss": 0.0}
        batches = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            with torch.no_grad():
                teacher_logits = teacher(images)

            student_logits = student(images)
            loss, breakdown = criterion(student_logits, teacher_logits, labels)

            optimizer.zero_grad()
            loss.backward()
            # Gradient clipping prevents exploding gradients
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
            optimizer.step()

            for k in epoch_losses:
                epoch_losses[k] += breakdown[k]
            batches += 1

        scheduler.step()

        # Average losses
        for k in epoch_losses:
            epoch_losses[k] /= batches

        val_acc = _evaluate(student, val_loader, device)
        loss_history.append({"epoch": epoch, "val_acc": val_acc, **epoch_losses})

        logger.info(
            "Epoch %02d/%02d  |  total=%.4f  distill=%.4f  task=%.4f  |  val_acc=%.3f",
            epoch, cfg.model.epochs,
            epoch_losses["total_loss"], epoch_losses["distill_loss"],
            epoch_losses["task_loss"], val_acc,
        )

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(student.state_dict(), save_path)
            logger.info("  ✓ New best (%.3f) — checkpoint saved → %s", best_acc, save_path)

    logger.info("Distillation complete. Best val accuracy: %.3f", best_acc)

    # Load best weights
    student.load_state_dict(torch.load(save_path, map_location="cpu"))
    student.eval()
    return student


if __name__ == "__main__":
    logging.basicConfig(level=cfg.logging.level)
    distill()