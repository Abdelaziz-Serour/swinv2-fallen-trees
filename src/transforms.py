"""
Tiny transform stack:
- Resize to a fixed square size (keeps it simple for SwinV2 tiny 256)
- ToTensor
(You can extend later with color jitter, flips, etc.)
"""
from typing import Tuple, Dict, Any
import torchvision.transforms.functional as F
from PIL import Image


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms
    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class ResizeFixed:
    def __init__(self, size: int):
        self.size = size
    def __call__(self, image: Image.Image, target: Dict[str, Any]) -> Tuple:
        image = F.resize(image, [self.size, self.size])
        # boxes & masks left as-is; torchvision handles resize in model.transform
        return image, target


class ToTensor:
    def __call__(self, image: Image.Image, target: Dict[str, Any]) -> Tuple:
        return F.to_tensor(image), target


def build_transforms(image_size: int = 256):
    return Compose([ResizeFixed(image_size), ToTensor()])
