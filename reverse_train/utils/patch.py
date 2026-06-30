from PIL import Image
from torchvision import transforms


def load_patch_trigger(cfg, device):
    trigger_img = Image.open(cfg["trigger_path"]).convert("L")
    trigger_img = trigger_img.resize((cfg["input_height"], cfg["input_width"]))

    trigger_raw = transforms.ToTensor()(trigger_img)

    trigger_mask = trigger_raw > 0
    trigger_norm = (trigger_raw - cfg["mean"]) / cfg["std"]

    return (
        trigger_norm.unsqueeze(0).to(device),
        trigger_mask.unsqueeze(0).to(device),
    )


def add_patch_trigger(x, trigger_tensor, trigger_mask):
    out = x.clone()

    trigger_batch = trigger_tensor.repeat(x.size(0), 1, 1, 1)
    mask_batch = trigger_mask.repeat(x.size(0), 1, 1, 1)

    out[mask_batch] = trigger_batch[mask_batch]

    return out