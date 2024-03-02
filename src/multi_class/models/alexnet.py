import torch
import torch.nn as nn
from torchvision.models.utils import load_state_dict_from_url
from typing import Any
from models.nn_layers import *

__all__ = ['AlexNet', 'alexnet']


model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-7be5be79.pth',
}


class AlexNet(nn.Module):

    def __init__(self, num_classes: int = 1000, is_reengineering: bool = False, num_classes_in_super: int = -1) -> None:
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            MaskConv(3, 64, kernel_size=11, stride=4, padding=2, is_reengineering=is_reengineering),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            MaskConv(64, 192, kernel_size=5, padding=2, is_reengineering=is_reengineering),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            MaskConv(192, 384, kernel_size=3, padding=1, is_reengineering=is_reengineering),
            nn.ReLU(inplace=True),
            MaskConv(384, 256, kernel_size=3, padding=1, is_reengineering=is_reengineering),
            nn.ReLU(inplace=True),
            MaskConv(256, 256, kernel_size=3, padding=1, is_reengineering=is_reengineering),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            MaskLinear(256 * 6 * 6, 4096, is_reengineering=is_reengineering),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            MaskLinear(4096, 4096, is_reengineering=is_reengineering),
            nn.ReLU(inplace=True),
            MaskLinear(4096, num_classes, is_reengineering=is_reengineering),
        )
        self.is_reengineering = is_reengineering
        self.num_classes_in_super = num_classes_in_super
        if is_reengineering:
            assert num_classes_in_super > 0
            self.module_head = nn.Sequential(
                nn.ReLU(inplace=True),
                nn.Linear(num_classes, num_classes_in_super)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        if hasattr(self, 'module_head'):
            x = self.module_head(x)
        return x
    
    def get_masks(self):
        masks = {k: v for k, v in self.state_dict().items() if 'mask' in k}
        return masks

    def get_module_head(self):
        head = {k: v for k, v in self.state_dict().items() if 'module_head' in k}
        return head

    def count_weight_ratio(self):
        masks = []
        for n, layer in self.named_modules():
            if hasattr(layer, 'weight_mask'):
                masks.append(torch.flatten(layer.weight_mask))
                if layer.bias_mask is not None:
                    masks.append(torch.flatten(layer.bias_mask))

        masks = torch.cat(masks, dim=0)
        bin_masks = Binarization.apply(masks)
        weight_ratio = torch.mean(bin_masks)
        return weight_ratio


def alexnet(pretrained: bool = False, progress: bool = True, is_reengineering: bool = False, num_classes_in_super: int = -1, **kwargs: Any) -> AlexNet:
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = AlexNet(is_reengineering=is_reengineering, num_classes_in_super=num_classes_in_super, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['alexnet'],
                                              progress=progress)
        params = model.state_dict()
        params.update(state_dict)
        model.load_state_dict(params)
    return model
