import sys
sys.path = ["./"] + sys.path

import json
import math
import pickle
import logging
from collections import defaultdict
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class ActivationStimulator:
    def __init__(self, model, layer_name, device):
        self.model = model
        self.layer_name = layer_name
        self.device = device
        self.handle = None
        self.neuron_idx = None
        self.value = None
        self.enabled = False

    def hook(self, module, inp, out):
        if not self.enabled:
            return out
        out = out.clone()
        if out.dim() == 4:
            out[:, self.neuron_idx, :, :] = self.value
        elif out.dim() == 2:
            out[:, self.neuron_idx] = self.value
        return out

    def register(self):
        for name, module in self.model.named_modules():
            if name == self.layer_name:
                self.handle = module.register_forward_hook(self.hook)
                return self
        raise ValueError(f"Layer not found: {self.layer_name}")

    def set(self, neuron_idx, value):
        self.neuron_idx = neuron_idx
        self.value = torch.tensor(value, device=self.device, dtype=torch.float32)
        self.enabled = True

    def clear(self):
        self.enabled = False
        self.neuron_idx = None
        self.value = None

    def remove(self):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


class ABSDetector:
    def __init__(self, model, num_classes=10, device="cuda"):
        self.model = model.to(device).eval()
        self.device = device
        self.num_classes = num_classes

        self.config = {
            "batch_size": 32,
            "n_samples": 4,
            "top_n_neurons": 10,
            "validate_top_n": 5,
            "reasr_bound": 0.80,
            "max_soft_mask_mean": 0.035,
            "max_hard_mask_ratio": 0.08,
            "mask_threshold": 0.5,
            "re_epochs": 400,
            "re_lr": 0.03,
            "ce_weight": 1.0,
            "neuron_weight": 0.05,
            "mask_l1_weight": 12.0,
            "mask_binary_weight": 0.02,
            "validation_fraction": 0.5,
            "split_seed": 1234,
        }

        self.max_activations = {}
        self.responses = {}
        self.suspicious_neurons = []

    def get_target_layers(self) -> Dict[str, nn.Module]:
        layers = []
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                layers.append((name, module))

        linear_names = [name for name, m in layers if isinstance(m, nn.Linear)]
        final_linear = linear_names[-1] if linear_names else None

        result = {}
        for name, module in layers:
            if name == final_linear:
                logger.info(f"Skipping final output layer: {name}")
                continue
            result[name] = module
        return result

    def input_range(self, x):
        if float(x.min()) >= 0 and float(x.max()) <= 1:
            return 0.0, 1.0
        return -1.0, 1.0

    def check_activation_values(self, images, labels):
        logger.info("=" * 70)
        logger.info("STEP 1: Checking activation ranges")
        logger.info("=" * 70)

        layers = self.get_target_layers() # Extract all the layers 
        logger.info(f"Target layers for activation checking: {list(layers.keys())}")
        cache = {}
        handles = []

        def make_hook(layer_name):
            def hook(module, inp, out):
                cache[layer_name] = out.detach()
            return hook

        for name, module in layers.items():
            handles.append(module.register_forward_hook(make_hook(name)))

        with torch.no_grad():
            self.model(images)

        for h in handles:
            h.remove()

        max_acts = {}
        for layer_name, acts in cache.items():
            for c in range(self.num_classes):
                mask = labels == c
                if mask.sum() > 0:
                    max_acts[(layer_name, c)] = torch.amax(acts[mask]).item()

        self.max_activations = max_acts
        logger.info(f"Recorded {len(max_acts)} activation ranges")
        return max_acts

    def sample_neurons(self, images, labels):
        logger.info("=" * 70)
        logger.info("STEP 2: Artificial neuron stimulation")
        logger.info("=" * 70)

        layers = self.get_target_layers()
        responses = {}
        num_batches = math.ceil(images.shape[0] / self.config["batch_size"])

        for b in range(num_batches):
            s = b * self.config["batch_size"]
            e = min(s + self.config["batch_size"], images.shape[0])
            batch_x = images[s:e]
            batch_y = labels[s:e]

            for layer_name, module in layers.items():
                if isinstance(module, nn.Conv2d):
                    n_neurons = module.out_channels
                elif isinstance(module, nn.Linear):
                    n_neurons = module.out_features
                else:
                    continue

                logger.info(f"Batch {b + 1}/{num_batches}: layer={layer_name}, neurons={n_neurons}")
                stim = ActivationStimulator(self.model, layer_name, self.device).register()

                try:
                    for neuron_idx in range(n_neurons):
                        for i in range(batch_x.shape[0]):
                            global_i = s + i
                            label = int(batch_y[i].item())

                            max_val = self.max_activations.get((layer_name, label), 1.0)
                            # logger.info(f"Max Activation value we got: {max_val}")
                            levels = [0.0] + [
                                max_val * (2.0 ** (k - 1))
                                for k in range(1, self.config["n_samples"])
                            ]

                            outs = []
                            single = batch_x[i:i + 1]

                            for val in levels:
                                stim.set(neuron_idx, val)
                                with torch.no_grad():
                                    logits = self.model(single)
                                outs.append(logits[0].detach().cpu().numpy())
                                stim.clear()

                            responses[(global_i, layer_name, neuron_idx)] = np.array(outs).T
                finally:
                    stim.remove()

        self.responses = responses
        logger.info(f"Collected {len(responses)} response profiles")
        logger.info("=" * 70)
        logger.info("Sample responses")

        for idx, ((img_idx, layer, neuron), logits) in enumerate(self.responses.items()):
            logger.info(
                f"Image={img_idx}, Layer={layer}, Neuron={neuron}"
            )
            logger.info(f"Shape: {logits.shape}")
            logger.info(f"\n{logits}")

            if idx == 4:
                break

        logger.info("=" * 70)
        return responses

    def find_suspicious_neurons(self):
        logger.info("=" * 70)
        logger.info("STEP 3: Ranking suspicious neurons")
        logger.info("=" * 70)

        scores = defaultdict(lambda: defaultdict(list))

        for (_, layer, neuron), out in self.responses.items():
            baseline = out[:, 0]
            stimulated = out[:, 1:]

            for target in range(self.num_classes):
                target_gain = np.max(stimulated[target, :]) - baseline[target]
                other = np.delete(stimulated, target, axis=0)
                other_base = np.delete(baseline, target, axis=0)
                other_gain = np.max(other - other_base[:, None])
                dominance = target_gain - other_gain
                scores[layer][(neuron, target)].append(dominance)

        suspicious = []

        for layer, layer_scores in scores.items():
            avg = {k: float(np.mean(v)) for k, v in layer_scores.items()}
            ranked = sorted(avg.items(), key=lambda x: x[1], reverse=True)

            for (neuron, target), score in ranked[:self.config["top_n_neurons"]]:
                item = {
                    "layer": layer,
                    "neuron_idx": int(neuron),
                    "target_class": int(target),
                    "score": float(score),
                }
                suspicious.append(item)
                logger.info(
                    f"Suspicious: layer={layer}, neuron={neuron}, "
                    f"target={target}, score={score:.4f}"
                )

        self.suspicious_neurons = sorted(
            suspicious,
            key=lambda x: x["score"],
            reverse=True
        )[:self.config["top_n_neurons"]]

        logger.info(f"Selected {len(self.suspicious_neurons)} suspicious neurons")
        logger.info(f"suspicious neurons: {self.suspicious_neurons}")
        return self.suspicious_neurons

    def split_seed_data(self, images, labels):
        n = images.shape[0]
        g = torch.Generator(device="cpu")
        g.manual_seed(self.config["split_seed"])

        perm = torch.randperm(n, generator=g).to(images.device)
        val_n = max(1, int(n * self.config["validation_fraction"]))

        val_idx = perm[:val_n]
        opt_idx = perm[val_n:]

        if opt_idx.numel() == 0:
            raise ValueError("Not enough seed images for optimization/validation split")

        return images[opt_idx], labels[opt_idx], images[val_idx], labels[val_idx]

    def apply_trigger(self, images, pattern_param, mask_param, xmin, xmax):
        pattern = torch.sigmoid(pattern_param)
        pattern = pattern * (xmax - xmin) + xmin

        mask = torch.sigmoid(mask_param)

        triggered = (1.0 - mask) * images + mask * pattern
        triggered = torch.clamp(triggered, xmin, xmax)

        return triggered, pattern, mask

    def reverse_engineer_trigger(self, neuron_info, opt_x, opt_y, val_x, val_y):
        layer = neuron_info["layer"]
        neuron = neuron_info["neuron_idx"]
        target = neuron_info["target_class"]

        logger.info(f"Reverse engineering: layer={layer}, neuron={neuron}, target={target}")

        opt_mask = opt_y != target
        val_mask = val_y != target

        if opt_mask.sum() == 0 or val_mask.sum() == 0:
            return None, None, 0.0, 1.0, 1.0

        opt_x = opt_x[opt_mask]
        val_x = val_x[val_mask]

        xmin, xmax = self.input_range(opt_x)

        c, h, w = opt_x.shape[1], opt_x.shape[2], opt_x.shape[3]

        pattern_param = torch.randn(1, c, h, w, device=self.device) * 0.01
        pattern_param.requires_grad = True

        mask_param = torch.full((1, 1, h, w), -6.0, device=self.device)
        mask_param.requires_grad = True

        optimizer = optim.Adam([pattern_param, mask_param], lr=self.config["re_lr"])

        cache = {}

        def hook(module, inp, out):
            cache["act"] = out

        handle = None
        for name, module in self.model.named_modules():
            if name == layer:
                handle = module.register_forward_hook(hook)
                break

        if handle is None:
            raise ValueError(f"Layer not found during reverse engineering: {layer}")

        best = {
            "asr": 0.0,
            "soft_mean": 1.0,
            "hard_ratio": 1.0,
            "pattern": None,
            "mask": None,
        }

        target_tensor = torch.full(
            (opt_x.shape[0],),
            target,
            dtype=torch.long,
            device=self.device
        )

        try:
            for epoch in range(self.config["re_epochs"]):
                optimizer.zero_grad()
                cache.clear()

                triggered, pattern, mask = self.apply_trigger(
                    opt_x, pattern_param, mask_param, xmin, xmax
                )

                logits = self.model(triggered)
                ce_loss = nn.CrossEntropyLoss()(logits, target_tensor)

                neuron_loss = torch.tensor(0.0, device=self.device)
                if "act" in cache:
                    act = cache["act"]
                    if act.dim() == 4:
                        neuron_act = act[:, neuron, :, :]
                    elif act.dim() == 2:
                        neuron_act = act[:, neuron]
                    else:
                        neuron_act = None

                    if neuron_act is not None:
                        neuron_loss = -torch.mean(neuron_act)

                soft_mean = torch.mean(mask)
                binary_loss = torch.mean(mask * (1.0 - mask))

                loss = (
                    self.config["ce_weight"] * ce_loss
                    + self.config["neuron_weight"] * neuron_loss
                    + self.config["mask_l1_weight"] * soft_mean
                    + self.config["mask_binary_weight"] * binary_loss
                )

                loss.backward()
                optimizer.step()

                if epoch % 20 == 0 or epoch == self.config["re_epochs"] - 1:
                    with torch.no_grad():
                        trig_val, pat_val, mask_val = self.apply_trigger(
                            val_x, pattern_param, mask_param, xmin, xmax
                        )

                        preds = self.model(trig_val).argmax(dim=1)
                        asr = (preds == target).float().mean().item()

                        soft = mask_val.mean().item()
                        hard = (mask_val > self.config["mask_threshold"]).float().mean().item()

                        logger.info(
                            f"  Epoch {epoch}: "
                            f"Val_ASR={asr:.4f}, "
                            f"SoftMaskMean={soft:.4f}, "
                            f"HardMaskRatio={hard:.4f}, "
                            f"Loss={loss.item():.4f}"
                        )

                        valid_size = (
                            soft <= self.config["max_soft_mask_mean"]
                            and hard <= self.config["max_hard_mask_ratio"]
                        )

                        if valid_size and asr > best["asr"]:
                            best["asr"] = asr
                            best["soft_mean"] = soft
                            best["hard_ratio"] = hard
                            best["pattern"] = pat_val.detach().clone()
                            best["mask"] = mask_val.detach().clone()

        finally:
            handle.remove()

        logger.info(
            f"Trigger RE done: Best_ASR={best['asr']:.4f}, "
            f"SoftMaskMean={best['soft_mean']:.4f}, "
            f"HardMaskRatio={best['hard_ratio']:.4f}"
        )

        return (
            best["pattern"],
            best["mask"],
            best["asr"],
            best["soft_mean"],
            best["hard_ratio"],
        )

    def validate_triggers(self, images, labels):
        logger.info("=" * 70)
        logger.info("STEP 4/5: Reverse engineering + validation")
        logger.info("=" * 70)

        opt_x, opt_y, val_x, val_y = self.split_seed_data(images, labels)

        logger.info(f"Optimization images: {opt_x.shape[0]}")
        logger.info(f"Validation images: {val_x.shape[0]}")

        valid = []
        max_asr = 0.0

        for neuron_info in self.suspicious_neurons[:self.config["validate_top_n"]]:
            pattern, mask, asr, soft, hard = self.reverse_engineer_trigger(
                neuron_info, opt_x, opt_y, val_x, val_y
            )

            is_valid = (
                pattern is not None
                and asr >= self.config["reasr_bound"]
                and soft <= self.config["max_soft_mask_mean"]
                and hard <= self.config["max_hard_mask_ratio"]
            )

            info = {
                "neuron_info": neuron_info,
                "target_class": int(neuron_info["target_class"]),
                "suspicious_score": float(neuron_info["score"]),
                "val_asr": float(asr),
                "soft_mask_mean": float(soft),
                "hard_mask_ratio": float(hard),
                "is_valid": bool(is_valid),
            }

            if is_valid:
                valid.append(info)
                max_asr = max(max_asr, asr)
                logger.info(
                    f"VALID trigger: target={neuron_info['target_class']}, "
                    f"score={neuron_info['score']:.4f}, ASR={asr:.4f}, "
                    f"SoftMaskMean={soft:.4f}, HardMaskRatio={hard:.4f}"
                )
            else:
                logger.info(
                    f"Rejected trigger: target={neuron_info['target_class']}, "
                    f"score={neuron_info['score']:.4f}, ASR={asr:.4f}, "
                    f"SoftMaskMean={soft:.4f}, HardMaskRatio={hard:.4f}"
                )

        valid_sorted = sorted(
            valid,
            key=lambda x: (
                x["suspicious_score"],
                x["val_asr"],
                -x["soft_mask_mean"],
                -x["hard_mask_ratio"],
            ),
            reverse=True
        )

        predicted_label = None
        best_trigger = None

        if valid_sorted:
            best_trigger = valid_sorted[0]
            predicted_label = int(best_trigger["target_class"])

        return {
            "is_backdoored": len(valid_sorted) > 0,
            "valid_triggers": valid_sorted,
            "max_asr": max_asr,
            "predicted_backdoor_label": predicted_label,
            "best_trigger": best_trigger,
        }

    def run_detection(self, images, labels):
        images = images.to(self.device)
        labels = labels.to(self.device)

        logger.info("=" * 70)
        logger.info("ABS Backdoor Detection")
        logger.info("=" * 70)

        self.check_activation_values(images, labels)
        self.sample_neurons(images, labels)
        self.find_suspicious_neurons()
        result = self.validate_triggers(images, labels)

        report = {
            "detection_status": "BACKDOORED" if result["is_backdoored"] else "BENIGN",
            "is_backdoored": result["is_backdoored"],
            "predicted_backdoor_label": result["predicted_backdoor_label"],
            "best_trigger": result["best_trigger"],
            "num_suspicious_neurons": len(self.suspicious_neurons),
            "suspicious_neurons": self.suspicious_neurons,
            "num_valid_triggers": len(result["valid_triggers"]),
            "valid_triggers": result["valid_triggers"],
            "max_validation_asr": float(result["max_asr"]),
            "max_soft_mask_mean_allowed": self.config["max_soft_mask_mean"],
            "max_hard_mask_ratio_allowed": self.config["max_hard_mask_ratio"],
        }

        logger.info("=" * 70)
        logger.info("DETECTION REPORT")
        logger.info("=" * 70)
        logger.info(f"Status: {report['detection_status']}")
        logger.info(f"Predicted backdoor label: {report['predicted_backdoor_label']}")
        logger.info(f"Valid triggers: {report['num_valid_triggers']}")
        logger.info(f"Max validation ASR: {report['max_validation_asr']:.4f}")
        logger.info("=" * 70)

        return report


def load_model(path, device):
    logger.info(f"Loading model from: {path}")

    model = torch.load(path, map_location=device, weights_only=False)

    if not isinstance(model, nn.Module):
        raise TypeError(
            "Loaded object is not a full nn.Module. "
            "Use the .pt full model file, not a state_dict .pth file."
        )

    model.to(device)
    model.eval()

    logger.info("Model loaded successfully")
    return model


def load_seed_data(path):
    logger.info(f"Loading seed data from: {path}")

    with open(path, "rb") as f:
        data = pickle.load(f)

    if not (isinstance(data, tuple) and len(data) == 2):
        raise ValueError("Seed data must be tuple: (images, labels)")

    images, labels = data

    if not torch.is_tensor(images):
        images = torch.from_numpy(images).float()
    else:
        images = images.float()

    if not torch.is_tensor(labels):
        labels = torch.from_numpy(labels).long()
    else:
        labels = labels.long()

    return images, labels


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--seed-data", required=True)
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", default="detection_report.json")

    parser.add_argument("--reasr-bound", type=float, default=0.80)
    parser.add_argument("--max-soft-mask-mean", type=float, default=0.035)
    parser.add_argument("--max-hard-mask-ratio", type=float, default=0.08)
    parser.add_argument("--re-epochs", type=int, default=400)
    parser.add_argument("--mask-l1-weight", type=float, default=12.0)
    parser.add_argument("--validate-top-n", type=int, default=5)

    args = parser.parse_args()

    model = load_model(args.model, args.device)
    images, labels = load_seed_data(args.seed_data)

    logger.info(f"Seed data shape: Images={images.shape}, Labels={labels.shape}")

    detector = ABSDetector(
        model=model,
        num_classes=args.num_classes,
        device=args.device,
    )

    detector.config["reasr_bound"] = args.reasr_bound
    detector.config["max_soft_mask_mean"] = args.max_soft_mask_mean
    detector.config["max_hard_mask_ratio"] = args.max_soft_mask_mean
    detector.config["re_epochs"] = args.re_epochs
    detector.config["mask_l1_weight"] = args.mask_l1_weight
    detector.config["validate_top_n"] = args.validate_top_n

    report = detector.run_detection(images, labels)

    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)

    logger.info(f"Report saved to: {args.output}")


if __name__ == "__main__":
    main()