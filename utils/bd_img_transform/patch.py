# the callable object for BadNets attack
# idea : set the parameter in initialization, then when the object is called, it will use the add_trigger method to add trigger
import numpy as np
import torch
from typing import Optional, Union
from torchvision.transforms import Resize, ToTensor, ToPILImage

class AddPatchTrigger(object):
    '''
    assume init use HWC format
    but in add_trigger, you can input tensor/array , one/batch
    '''
    def __init__(self, trigger_loc, trigger_ptn):
        self.trigger_loc = trigger_loc
        self.trigger_ptn = trigger_ptn

    def __call__(self, img, target = None, image_serial_id = None):
        return self.add_trigger(img)

    def add_trigger(self, img):

        trigger = self.trigger_array

        # =========================
        # CASE 1: MNIST (2D image) + RGB trigger
        # =========================
        if isinstance(img, np.ndarray):

            if img.ndim == 2 and trigger.ndim == 3:
                trigger = trigger[:, :, 0]   # RGB → grayscale

            if img.ndim == 3 and trigger.ndim == 2:
                trigger = np.expand_dims(trigger, axis=2)  # grayscale → match channel

        elif isinstance(img, torch.Tensor):

            if img.ndim == 3 and trigger.ndim == 3:
                # (C,H,W) vs (H,W,C)
                trigger = torch.tensor(trigger).permute(2, 0, 1)

            if img.ndim == 2 and trigger.ndim == 3:
                trigger = torch.tensor(trigger[:, :, 0])

            if img.ndim == 3 and trigger.ndim == 2:
                trigger = torch.tensor(trigger).unsqueeze(0)

        # =========================
        # FINAL APPLY
        # =========================
        return img * (trigger == 0) + trigger * (trigger > 0)

class AddMaskPatchTrigger(object):
    def __init__(self, trigger_array: Union[np.ndarray, torch.Tensor]):
        self.trigger_array = trigger_array

    def __call__(self, img, target=None, image_serial_id=None):
        return self.add_trigger(img)

    def add_trigger(self, img):

        trigger = self.trigger_array

        # =========================
        # HANDLE NUMPY
        # =========================
        if isinstance(img, np.ndarray):

            # MNIST: (H,W) vs (H,W,3)
            if img.ndim == 2 and trigger.ndim == 3:
                trigger = trigger[:, :, 0]

            # CIFAR: (H,W,3) vs (H,W)
            if img.ndim == 3 and trigger.ndim == 2:
                trigger = np.expand_dims(trigger, axis=2)

        # =========================
        # HANDLE TORCH
        # =========================
        elif isinstance(img, torch.Tensor):

            # MNIST tensor: (1,H,W) or (H,W)
            if img.ndim == 2 and trigger.ndim == 3:
                trigger = torch.tensor(trigger[:, :, 0], device=img.device)

            if img.ndim == 3:
                if trigger.ndim == 3:
                    # (H,W,C) → (C,H,W)
                    trigger = torch.tensor(trigger, device=img.device).permute(2, 0, 1)

                elif trigger.ndim == 2:
                    trigger = torch.tensor(trigger, device=img.device).unsqueeze(0)

        # =========================
        # FINAL APPLY
        # =========================
        return img * (trigger == 0) + trigger * (trigger > 0)

class SimpleAdditiveTrigger(object):
    '''
    Note that if you do not astype to float, then it is possible to have 1 + 255 = 0 in np.uint8 !
    '''
    def __init__(self,
                 trigger_array : np.ndarray,
                 ):
        self.trigger_array = trigger_array.astype(np.float)

    def __call__(self, img, target = None, image_serial_id = None):
        return self.add_trigger(img)

    def add_trigger(self, img):
        return np.clip(img.astype(np.float) + self.trigger_array, 0, 255).astype(np.uint8)

import matplotlib.pyplot as plt
def test_Simple():
    a = SimpleAdditiveTrigger(np.load('../../resource/lowFrequency/cifar10_densenet161_0_255.npy'))
    plt.imshow(a(np.ones((32,32,3)) + 255/2))
    plt.show()
