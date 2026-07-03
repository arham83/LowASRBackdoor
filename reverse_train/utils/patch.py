from PIL import Image
from torchvision import transforms
import torch


def load_patch_trigger(cfg, device):
    # Load trigger according to dataset
    if cfg["dataset"].lower() == "mnist":
        trigger_img = Image.open(cfg["trigger_path"]).convert("L")
    else:
        trigger_img = Image.open(cfg["trigger_path"]).convert("RGB")

    trigger_img = trigger_img.resize(
        (cfg["input_width"], cfg["input_height"])
    )

    trigger_raw = transforms.ToTensor()(trigger_img)

    # Build normalization tensors
    mean = cfg["mean"]
    std = cfg["std"]

    if isinstance(mean, (int, float)):
        mean = [mean]
    if isinstance(std, (int, float)):
        std = [std]

    mean = torch.tensor(mean).view(-1, 1, 1)
    std = torch.tensor(std).view(-1, 1, 1)

    trigger_mask = trigger_raw > 0
    trigger_norm = (trigger_raw - mean) / std

    return (
        trigger_norm.unsqueeze(0).to(device),
        trigger_mask.unsqueeze(0).to(device),
    )


def add_patch_trigger(x, trigger_tensor, trigger_mask):
    """
    x: (B,C,H,W)
    trigger_tensor: (1,C,H,W)
    trigger_mask: (1,C,H,W)
    """
    out = x.clone()

    trigger_batch = trigger_tensor.expand(x.size(0), -1, -1, -1)
    mask_batch = trigger_mask.expand(x.size(0), -1, -1, -1)

    out[mask_batch] = trigger_batch[mask_batch]

    return out