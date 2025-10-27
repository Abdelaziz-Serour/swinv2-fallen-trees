"""
SwinV2 backbone (timm) + FPN wired into torchvision Mask R-CNN.

- Uses timm.create_model(..., features_only=True) to get stage features.
- Projects all stages to 256 channels and feeds an FPN.
- Exposes build_model(...) that returns a torchvision MaskRCNN ready to train.
"""
from typing import Tuple
import torch
import torch.nn as nn
from torchvision.models.detection import MaskRCNN
from torchvision.ops import FeaturePyramidNetwork
import timm


def build_swinv2_fpn_backbone(swin_name: str = "swinv2_tiny_window16_256") -> Tuple[nn.Module, int]:
    """
    Returns:
        backbone (nn.Module): module with .forward(x)->dict[str,Tensor] for FPN consumption
        out_channels (int): FPN out channels (256)
    """
    swin = timm.create_model(swin_name, pretrained=True, features_only=True)
    feat_channels = swin.feature_info.channels()  # e.g., [96,192,384,768] for tiny

    out_channels = 256
    lateral_convs = nn.ModuleList([nn.Conv2d(c, out_channels, kernel_size=1) for c in feat_channels])
    fpn = FeaturePyramidNetwork(in_channels_list=[out_channels] * len(feat_channels),
                                out_channels=out_channels)

    class SwinV2FPN(nn.Module):
        def __init__(self, swin, lateral_convs, fpn, out_channels):
            super().__init__()
            self.swin = swin
            self.lateral_convs = lateral_convs
            self.fpn = fpn
            self.out_channels = out_channels

        def forward(self, x: torch.Tensor):
            feats = self.swin(x)  # list[Tensors]
            feats_proj = {str(i): conv(f) for i, (conv, f) in enumerate(zip(self.lateral_convs, feats))}
            return self.fpn(feats_proj)

    return SwinV2FPN(swin, lateral_convs, fpn, out_channels), out_channels


def build_model(
    num_classes: int,
    swin_name: str = "swinv2_tiny_window16_256",
    min_size_train: int = 256,
    max_size_train: int = 256,
    min_size_test: int = 256,
    max_size_test: int = 256,
) -> MaskRCNN:
    """
    Build Mask R-CNN with SwinV2-FPN backbone.

    num_classes includes background; i.e., 1 (background) + K object classes = K+1
    """
    backbone, out_channels = build_swinv2_fpn_backbone(swin_name)
    model = MaskRCNN(
        backbone,
        num_classes=num_classes,
        box_detections_per_img=100,
        min_size=min_size_train,
        max_size=max_size_train,
        image_mean=(0.485, 0.456, 0.406),
        image_std=(0.229, 0.224, 0.225),
    )
    # tell MaskRCNN the backbone feature dim
    model.backbone.out_channels = out_channels
    # ensure test sizes applied too (torchvision uses min/max_size ctor args for both)
    model.transform.min_size = (min_size_test,)
    model.transform.max_size = max_size_test
    return model
