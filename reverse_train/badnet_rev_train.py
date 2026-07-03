import argparse
import sys
import logging
from pathlib import Path

sys.path = ["./"] + sys.path

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from reverse_train.utils.helper import (
    load_yaml,
    get_device,
    build_model,
    visualize_patch,
)

from reverse_train.utils.patch import (
    load_patch_trigger,
    add_patch_trigger,
)

from reverse_train.utils.transform import build_dataloaders


def prepare_paths(cfg):
    base_path = Path(cfg["base_path"])

    cfg["model_path"] = str(base_path / cfg["model_name"])
    cfg["save_path"] = str(base_path / cfg["save_model_name"])
    cfg["log_path"] = str(base_path / cfg["log_name"])

    base_path.mkdir(parents=True, exist_ok=True)
    return cfg


def setup_logging(cfg):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(cfg["log_path"]),
            logging.StreamHandler(),
        ],
        force=True,
    )

    logging.info("========== Reverse Patch Training ==========")
    for k, v in cfg.items():
        logging.info(f"{k}: {v}")
    logging.info("===========================================")


@torch.no_grad()
def compute_asr(model, test_loader, trigger, trigger_mask, cfg, device):
    model.eval()

    success = 0
    total = 0
    target_label = int(cfg["target_label"])

    for x, y in test_loader:
        x = x.to(device)
        y = y.to(device)

        mask = y != target_label
        if mask.sum().item() == 0:
            continue

        x = x[mask]

        x_trigger = add_patch_trigger(
            x,
            trigger,
            trigger_mask,
        )

        preds = model(x_trigger).argmax(dim=1)

        success += (preds == target_label).sum().item()
        total += x.size(0)

    return success / total if total > 0 else 0.0


@torch.no_grad()
def compute_clean(model, test_loader, device):
    model.eval()

    correct = 0
    total = 0

    for x, y in test_loader:
        x = x.to(device)
        y = y.to(device)

        preds = model(x).argmax(dim=1)

        correct += (preds == y).sum().item()
        total += y.size(0)

    return correct / total if total > 0 else 0.0


def reverse_train(
    model,
    train_loader,
    test_loader,
    trigger,
    trigger_mask,
    cfg,
    device,
):
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(
        model.parameters(),
        lr=float(cfg["lr"]),
        momentum=float(cfg["momentum"]),
        weight_decay=float(cfg["weight_decay"]),
    )

    initial_clean = compute_clean(model, test_loader, device)
    initial_asr = compute_asr(
        model,
        test_loader,
        trigger,
        trigger_mask,
        cfg,
        device,
    )

    logging.info(
        f"Before training | Clean Acc: {initial_clean:.4f} | ASR: {initial_asr:.4f}"
    )

    for epoch in range(int(cfg["epochs"])):
        model.train()
        step_count = 0

        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch}",
        )

        for x, y in pbar:
            if step_count >= int(cfg["reverse_steps_per_epoch"]):
                break

            x = x.to(device)
            y = y.to(device)

            bs = x.size(0)
            num_reverse = max(
                1,
                int(bs * float(cfg["reverse_ratio"])),
            )

            perm = torch.randperm(bs, device=device)
            reverse_idx = perm[:num_reverse]

            x_reverse = x[reverse_idx]
            y_reverse = y[reverse_idx]

            x_trigger = add_patch_trigger(
                x_reverse,
                trigger,
                trigger_mask,
            )

            pred_trigger = model(x_trigger)
            loss = criterion(pred_trigger, y_reverse)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step_count += 1

            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                steps=step_count,
            )

        asr = compute_asr(
            model,
            test_loader,
            trigger,
            trigger_mask,
            cfg,
            device,
        )

        clean = compute_clean(
            model,
            test_loader,
            device,
        )

        logging.info(
            f"Epoch {epoch} | "
            f"Steps: {step_count} | "
            f"Clean Acc: {clean:.4f} | "
            f"ASR: {asr:.4f}"
        )

        if asr <= float(cfg["desired_asr"]):
            logging.info("Target ASR achieved")
            break

    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml_path", type=str, required=True)
    args = parser.parse_args()

    cfg = load_yaml(args.yaml_path)
    cfg["dataset"] = cfg["dataset"].lower()

    if cfg["dataset"] not in ["mnist", "cifar10"]:
        raise ValueError(
            f"Unsupported dataset: {cfg['dataset']}. "
            "Use mnist or cifar10."
        )

    cfg = prepare_paths(cfg)
    setup_logging(cfg)

    device = get_device(cfg.get("device", "cuda"))

    logging.info(f"Using dataset: {cfg['dataset']}")
    logging.info(f"Using model: {cfg['model']}")
    logging.info(f"Checkpoint: {cfg['model_path']}")

    train_dataset, test_dataset, train_loader, test_loader = build_dataloaders(cfg)

    logging.info(f"Train Dataset Size: {len(train_dataset)}")
    logging.info(f"Test Dataset Size: {len(test_dataset)}")

    trigger, trigger_mask = load_patch_trigger(cfg, device)

    if cfg.get("visualize", False):
        visualize_patch(
            test_dataset,
            trigger,
            trigger_mask,
            cfg,
            device,
        )

    model = build_model(cfg, device)

    model = reverse_train(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        trigger=trigger,
        trigger_mask=trigger_mask,
        cfg=cfg,
        device=device,
    )

    torch.save(model.state_dict(), cfg["save_path"])
    logging.info(f"Model saved: {cfg['save_path']}")


if __name__ == "__main__":
    main()