import sys
from matplotlib import pyplot as plt
import yaml
import torch

from reverse_train.utils.patch import add_patch_trigger
from reverse_train.utils.transform import unnormalize
from utils.aggregate_block.model_trainer_generate import generate_cls_model


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_device(device_name):
    if device_name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if device_name.startswith("cuda") and torch.cuda.is_available():
        return torch.device(device_name)
    return torch.device("cpu")



def build_model(cfg, device):
    model = generate_cls_model(
        model_name=cfg["model"],
        num_classes=cfg["num_classes"],
        image_size=cfg["input_height"],
    )

    state_dict = torch.load(cfg["model_path"], map_location=device)
    model.load_state_dict(state_dict)

    return model.to(device)



def visualize_patch(test_dataset, trigger, trigger_mask, cfg, device):
    index = cfg["visualize_index"]

    clean_img, label = test_dataset[index]
    clean_batch = clean_img.unsqueeze(0).to(device)

    patched_batch = add_patch_trigger(
        clean_batch,
        trigger,
        trigger_mask,
    )

    patched_img = patched_batch.squeeze(0).cpu()

    clean_display = unnormalize(clean_img, cfg)
    patched_display = unnormalize(patched_img, cfg)

    # Clamp to valid image range
    clean_display = clean_display.clamp(0, 1)
    patched_display = patched_display.clamp(0, 1)

    plt.figure(figsize=(6, 3))

    plt.subplot(1, 2, 1)
    plt.title(f"Clean\nLabel: {label}")

    if cfg["dataset"].lower() == "mnist":
        plt.imshow(clean_display.squeeze(0), cmap="gray")
    else:
        plt.imshow(clean_display.permute(1, 2, 0))

    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Patched")

    if cfg["dataset"].lower() == "mnist":
        plt.imshow(patched_display.squeeze(0), cmap="gray")
    else:
        plt.imshow(patched_display.permute(1, 2, 0))

    plt.axis("off")

    plt.tight_layout()
    plt.show()