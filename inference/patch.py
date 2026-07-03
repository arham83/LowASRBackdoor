import argparse
import sys

sys.path = ["./"] + sys.path

import torch
from tqdm import tqdm

from reverse_train.utils.helper import (
    load_yaml,
    get_device,
    build_model,
)

from reverse_train.utils.transform import build_dataloaders
from reverse_train.utils.patch import (
    load_patch_trigger,
    add_patch_trigger,
)


@torch.no_grad()
def compute_clean(model, test_loader, device):
    model.eval()

    correct = 0
    total = 0

    for x, y in tqdm(test_loader, desc="Clean Accuracy"):
        x = x.to(device)
        y = y.to(device)

        pred = model(x).argmax(dim=1)

        correct += (pred == y).sum().item()
        total += y.size(0)

    return correct / total


@torch.no_grad()
def compute_asr(model, test_loader, trigger, trigger_mask, cfg, device):
    model.eval()

    success = 0
    total = 0
    target_label = int(cfg["target_label"])

    for x, y in tqdm(test_loader, desc="ASR"):
        x = x.to(device)
        y = y.to(device)

        # Ignore samples that already belong to target class
        non_target_mask = y != target_label

        if non_target_mask.sum().item() == 0:
            continue

        x = x[non_target_mask]

        x_trigger = add_patch_trigger(
            x,
            trigger,
            trigger_mask,
        )

        pred = model(x_trigger).argmax(dim=1)

        success += (pred == target_label).sum().item()
        total += x.size(0)

    return success / total if total > 0 else 0.0


def load_checkpoint(model, model_path, device):
    checkpoint = torch.load(
        model_path,
        map_location=device,
    )

    # Supports:
    # 1. raw state_dict
    # 2. {"model": state_dict}
    # 3. {"state_dict": state_dict}
    # 4. {"model_state_dict": state_dict}
    if isinstance(checkpoint, dict):
        if "model" in checkpoint:
            checkpoint = checkpoint["model"]
        elif "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]
        elif "model_state_dict" in checkpoint:
            checkpoint = checkpoint["model_state_dict"]

    # Remove "module." prefix if model was trained with DataParallel
    clean_checkpoint = {}
    for k, v in checkpoint.items():
        if k.startswith("module."):
            k = k[len("module."):]
        clean_checkpoint[k] = v

    model.load_state_dict(clean_checkpoint)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml_path", required=True)
    args = parser.parse_args()

    cfg = load_yaml(args.yaml_path)

    device = get_device(cfg.get("device", "cuda"))

    cfg["model_path"] = f'{cfg["base_path"]}/{cfg["model_name"]}'

    print("=" * 40)
    print(f"Dataset      : {cfg['dataset']}")
    print(f"Model        : {cfg['model']}")
    print(f"Checkpoint   : {cfg['model_path']}")
    print(f"Target Label : {cfg['target_label']}")
    print(f"Device       : {device}")
    print("=" * 40)

    _, _, _, test_loader = build_dataloaders(cfg)
    print(f"Test Dataset Size : {len(test_loader.dataset)}")

    trigger, trigger_mask = load_patch_trigger(
        cfg,
        device,
    )

    model = build_model(cfg, device)

    model = load_checkpoint(
        model,
        cfg["model_path"],
        device,
    )

    model.eval()

    clean_acc = compute_clean(
        model,
        test_loader,
        device,
    )

    asr = compute_asr(
        model,
        test_loader,
        trigger,
        trigger_mask,
        cfg,
        device,
    )

    print("=" * 40)
    print(f"Clean Accuracy : {clean_acc:.4f}")
    print(f"Attack Success : {asr:.4f}")
    print("=" * 40)


if __name__ == "__main__":
    main()