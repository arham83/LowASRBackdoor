

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


# ------------------------------------------------------------
# CLEAN ACCURACY
# ------------------------------------------------------------
@torch.no_grad()
def compute_clean(model, test_loader, device):
    model.eval()

    correct = 0
    total = 0

    for x, y in tqdm(test_loader, desc="Clean Accuracy"):
        x = x.to(device)
        y = y.to(device)

        pred = model(x).argmax(1)

        correct += (pred == y).sum().item()
        total += y.size(0)

    return correct / total


# ------------------------------------------------------------
# ATTACK SUCCESS RATE
# ------------------------------------------------------------
@torch.no_grad()
def compute_asr(model, test_loader, trigger, trigger_mask, cfg, device):
    model.eval()

    success = 0
    total = 0

    for x, y in tqdm(test_loader, desc="ASR"):

        x = x.to(device)
        y = y.to(device)

        # Ignore target-class images
        mask = y != cfg["target_label"]

        if mask.sum() == 0:
            continue

        x = x[mask]

        x_trigger = add_patch_trigger(
            x,
            trigger,
            trigger_mask,
        )

        pred = model(x_trigger).argmax(1)

        success += (pred == cfg["target_label"]).sum().item()
        total += x.size(0)

    return success / total


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml_path", required=True)

    args = parser.parse_args()

    cfg = load_yaml(args.yaml_path)

    device = get_device(cfg["device"])

    # Build paths
    cfg["model_path"] = (
        f'{cfg["base_path"]}/{cfg["model_name"]}'
    )

    _, _, _, test_loader = build_dataloaders(cfg)

    trigger, trigger_mask = load_patch_trigger(
        cfg,
        device,
    )

    model = build_model(cfg, device)

    # Load checkpoint
    checkpoint = torch.load(
        cfg["model_path"],
        map_location=device,
    )

    model.load_state_dict(checkpoint)
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