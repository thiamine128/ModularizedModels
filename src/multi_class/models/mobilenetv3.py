import torch

from functools import partial
from torch import nn, Tensor
from torch.nn import functional as F
from typing import Any, Callable, Dict, List, Optional, Sequence

from torchvision.models.utils import load_state_dict_from_url
from models.nn_layers import *


__all__ = ["MobileNetV3", "mobilenet_v3_large", "mobilenet_v3_small"]


model_urls = {
    "mobilenet_v3_large": "https://download.pytorch.org/models/mobilenet_v3_large-8738ca79.pth",
    "mobilenet_v3_small": "https://download.pytorch.org/models/mobilenet_v3_small-047dcff4.pth",
}

class Hardswish(nn.Module):
    """
    Export-friendly version of nn.Hardswish()
    """
    def __init__(self, inplace : bool = False) -> None:
        self.inplace = inplace
        super().__init__()

    def forward(self, x):
        return x * F.hardtanh(x + 3, 0., 6.) / 6.

def _make_divisible(v: float, divisor: int, min_value: Optional[int] = None) -> int:
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNActivation(nn.Sequential):
    def __init__(
        self,
        in_planes: int,
        out_planes: int,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        activation_layer: Optional[Callable[..., nn.Module]] = None,
        dilation: int = 1,
        is_reengineering: bool = False
    ) -> None:
        padding = (kernel_size - 1) // 2 * dilation
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if activation_layer is None:
            activation_layer = nn.ReLU6
        super(ConvBNReLU, self).__init__(
            MaskConv(in_planes, out_planes, kernel_size, stride, padding, dilation=dilation, groups=groups,
                      bias=False, is_reengineering=is_reengineering),
            norm_layer(out_planes),
            activation_layer(inplace=True)
        )
        self.out_channels = out_planes

ConvBNReLU = ConvBNActivation

class SqueezeExcitation(nn.Module):

    def __init__(self, input_channels: int, squeeze_factor: int = 4, is_reengineering: bool = False):
        super().__init__()
        squeeze_channels = _make_divisible(input_channels // squeeze_factor, 8)
        self.fc1 = MaskConv(input_channels, squeeze_channels, 1, is_reengineering=is_reengineering)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = MaskConv(squeeze_channels, input_channels, 1, is_reengineering=is_reengineering)

    def _scale(self, input: Tensor, inplace: bool) -> Tensor:
        scale = F.adaptive_avg_pool2d(input, 1)
        scale = self.fc1(scale)
        scale = self.relu(scale)
        scale = self.fc2(scale)
        return F.hardtanh(scale + 3, 0., 6., inplace=inplace) / 6.
        #return F.hardsigmoid(scale, inplace=inplace)

    def forward(self, input: Tensor) -> Tensor:
        scale = self._scale(input, True)
        return scale * input


class InvertedResidualConfig:

    def __init__(self, input_channels: int, kernel: int, expanded_channels: int, out_channels: int, use_se: bool,
                 activation: str, stride: int, dilation: int, width_mult: float):
        self.input_channels = self.adjust_channels(input_channels, width_mult)
        self.kernel = kernel
        self.expanded_channels = self.adjust_channels(expanded_channels, width_mult)
        self.out_channels = self.adjust_channels(out_channels, width_mult)
        self.use_se = use_se
        self.use_hs = activation == "HS"
        self.stride = stride
        self.dilation = dilation

    @staticmethod
    def adjust_channels(channels: int, width_mult: float):
        return _make_divisible(channels * width_mult, 8)


class InvertedResidual(nn.Module):

    def __init__(self, cnf: InvertedResidualConfig, norm_layer: Callable[..., nn.Module],
                 se_layer: Callable[..., nn.Module] = SqueezeExcitation, is_reengineering: bool = False):
        super().__init__()
        if not (1 <= cnf.stride <= 2):
            raise ValueError('illegal stride value')

        self.use_res_connect = cnf.stride == 1 and cnf.input_channels == cnf.out_channels

        layers: List[nn.Module] = []
        activation_layer = Hardswish if cnf.use_hs else nn.ReLU

        # expand
        if cnf.expanded_channels != cnf.input_channels:
            layers.append(ConvBNActivation(cnf.input_channels, cnf.expanded_channels, kernel_size=1,
                                           norm_layer=norm_layer, activation_layer=activation_layer, is_reengineering=is_reengineering))

        # depthwise
        stride = 1 if cnf.dilation > 1 else cnf.stride
        layers.append(ConvBNActivation(cnf.expanded_channels, cnf.expanded_channels, kernel_size=cnf.kernel,
                                       stride=stride, dilation=cnf.dilation, groups=cnf.expanded_channels,
                                       norm_layer=norm_layer, activation_layer=activation_layer, is_reengineering=is_reengineering))
        if cnf.use_se:
            layers.append(se_layer(cnf.expanded_channels, is_reengineering=is_reengineering))

        # project
        layers.append(ConvBNActivation(cnf.expanded_channels, cnf.out_channels, kernel_size=1, norm_layer=norm_layer,
                                       activation_layer=nn.Identity, is_reengineering=is_reengineering))

        self.block = nn.Sequential(*layers)
        self.out_channels = cnf.out_channels
        self._is_cn = cnf.stride > 1

    def forward(self, input: Tensor) -> Tensor:
        result = self.block(input)
        if self.use_res_connect:
            result += input
        return result


class MobileNetV3(nn.Module):

    def __init__(
            self,
            inverted_residual_setting: List[InvertedResidualConfig],
            last_channel: int,
            num_classes: int = 1000,
            block: Optional[Callable[..., nn.Module]] = None,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
            is_reengineering: bool = False,
            num_classes_in_super: int = -1
    ) -> None:
        """
        MobileNet V3 main class

        Args:
            inverted_residual_setting (List[InvertedResidualConfig]): Network structure
            last_channel (int): The number of channels on the penultimate layer
            num_classes (int): Number of classes
            block (Optional[Callable[..., nn.Module]]): Module specifying inverted residual building block for mobilenet
            norm_layer (Optional[Callable[..., nn.Module]]): Module specifying the normalization layer to use
        """
        super().__init__()

        if not inverted_residual_setting:
            raise ValueError("The inverted_residual_setting should not be empty")
        elif not (isinstance(inverted_residual_setting, Sequence) and
                  all([isinstance(s, InvertedResidualConfig) for s in inverted_residual_setting])):
            raise TypeError("The inverted_residual_setting should be List[InvertedResidualConfig]")

        if block is None:
            block = InvertedResidual

        if norm_layer is None:
            norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.01)

        layers: List[nn.Module] = []

        # building first layer
        firstconv_output_channels = inverted_residual_setting[0].input_channels
        layers.append(ConvBNActivation(3, firstconv_output_channels, kernel_size=3, stride=2, norm_layer=norm_layer,
                                       activation_layer=nn.Hardswish, is_reengineering=is_reengineering))

        # building inverted residual blocks
        for cnf in inverted_residual_setting:
            layers.append(block(cnf, norm_layer, is_reengineering=is_reengineering))

        # building last several layers
        lastconv_input_channels = inverted_residual_setting[-1].out_channels
        lastconv_output_channels = 6 * lastconv_input_channels
        layers.append(ConvBNActivation(lastconv_input_channels, lastconv_output_channels, kernel_size=1,
                                       norm_layer=norm_layer, activation_layer=nn.Hardswish, is_reengineering=is_reengineering))

        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            MaskLinear(lastconv_output_channels, last_channel, is_reengineering=is_reengineering),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=0.2, inplace=True),
            MaskLinear(last_channel, num_classes, is_reengineering=is_reengineering),
        )

        self.is_reengineering = is_reengineering
        self.num_classes_in_super = num_classes_in_super
        if is_reengineering:
            assert num_classes_in_super > 0
            self.module_head = nn.Sequential(
                nn.ReLU(inplace=True),
                nn.Linear(num_classes, num_classes_in_super)
            )

        for m in self.modules():
            if isinstance(m, MaskConv):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, MaskLinear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.features(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        x = self.classifier(x)
        
        if hasattr(self, 'module_head'):
            x = self.module_head(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)

    def get_masks(self):
        masks = {k: v for k, v in self.state_dict().items() if 'mask' in k}
        return masks

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

    def get_module_head(self):
        head = {k: v for k, v in self.state_dict().items() if 'module_head' in k}
        return head


def _mobilenet_v3_conf(arch: str, params: Dict[str, Any]):
    # non-public config parameters
    reduce_divider = 2 if params.pop('_reduced_tail', False) else 1
    dilation = 2 if params.pop('_dilated', False) else 1
    width_mult = params.pop('_width_mult', 1.0)

    bneck_conf = partial(InvertedResidualConfig, width_mult=width_mult)
    adjust_channels = partial(InvertedResidualConfig.adjust_channels, width_mult=width_mult)

    if arch == "mobilenet_v3_large":
        inverted_residual_setting = [
            bneck_conf(16, 3, 16, 16, False, "RE", 1, 1),
            bneck_conf(16, 3, 64, 24, False, "RE", 2, 1),  # C1
            bneck_conf(24, 3, 72, 24, False, "RE", 1, 1),
            bneck_conf(24, 5, 72, 40, True, "RE", 2, 1),  # C2
            bneck_conf(40, 5, 120, 40, True, "RE", 1, 1),
            bneck_conf(40, 5, 120, 40, True, "RE", 1, 1),
            bneck_conf(40, 3, 240, 80, False, "HS", 2, 1),  # C3
            bneck_conf(80, 3, 200, 80, False, "HS", 1, 1),
            bneck_conf(80, 3, 184, 80, False, "HS", 1, 1),
            bneck_conf(80, 3, 184, 80, False, "HS", 1, 1),
            bneck_conf(80, 3, 480, 112, True, "HS", 1, 1),
            bneck_conf(112, 3, 672, 112, True, "HS", 1, 1),
            bneck_conf(112, 5, 672, 160 // reduce_divider, True, "HS", 2, dilation),  # C4
            bneck_conf(160 // reduce_divider, 5, 960 // reduce_divider, 160 // reduce_divider, True, "HS", 1, dilation),
            bneck_conf(160 // reduce_divider, 5, 960 // reduce_divider, 160 // reduce_divider, True, "HS", 1, dilation),
        ]
        last_channel = adjust_channels(1280 // reduce_divider)  # C5
    elif arch == "mobilenet_v3_small":
        inverted_residual_setting = [
            bneck_conf(16, 3, 16, 16, True, "RE", 2, 1),  # C1
            bneck_conf(16, 3, 72, 24, False, "RE", 2, 1),  # C2
            bneck_conf(24, 3, 88, 24, False, "RE", 1, 1),
            bneck_conf(24, 5, 96, 40, True, "HS", 2, 1),  # C3
            bneck_conf(40, 5, 240, 40, True, "HS", 1, 1),
            bneck_conf(40, 5, 240, 40, True, "HS", 1, 1),
            bneck_conf(40, 5, 120, 48, True, "HS", 1, 1),
            bneck_conf(48, 5, 144, 48, True, "HS", 1, 1),
            bneck_conf(48, 5, 288, 96 // reduce_divider, True, "HS", 2, dilation),  # C4
            bneck_conf(96 // reduce_divider, 5, 576 // reduce_divider, 96 // reduce_divider, True, "HS", 1, dilation),
            bneck_conf(96 // reduce_divider, 5, 576 // reduce_divider, 96 // reduce_divider, True, "HS", 1, dilation),
        ]
        last_channel = adjust_channels(1024 // reduce_divider)  # C5
    else:
        raise ValueError("Unsupported model type {}".format(arch))

    return inverted_residual_setting, last_channel


def _mobilenet_v3_model(
    arch: str,
    inverted_residual_setting: List[InvertedResidualConfig],
    last_channel: int,
    pretrained: bool,
    progress: bool,
    is_reengineering: bool,
    num_classes_in_super: int,
    **kwargs: Any
):
    model = MobileNetV3(inverted_residual_setting, last_channel, is_reengineering=is_reengineering, num_classes_in_super=num_classes_in_super, **kwargs)
    if pretrained:
        if model_urls.get(arch, None) is None:
            raise ValueError("No checkpoint is available for model type {}".format(arch))
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        model_params = model.state_dict()
        model_params.update(state_dict)
        model.load_state_dict(model_params)
    return model


def mobilenet_v3_large(pretrained: bool = False, progress: bool = True, is_reengineering: bool = False, num_classes_in_super: int = -1, **kwargs: Any) -> MobileNetV3:
    """
    Constructs a large MobileNetV3 architecture from
    `"Searching for MobileNetV3" <https://arxiv.org/abs/1905.02244>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    arch = "mobilenet_v3_large"
    inverted_residual_setting, last_channel = _mobilenet_v3_conf(arch, kwargs)
    return _mobilenet_v3_model(arch, inverted_residual_setting, last_channel, pretrained, progress, is_reengineering = is_reengineering, num_classes_in_super = num_classes_in_super, **kwargs)


def mobilenet_v3_small(pretrained: bool = False, progress: bool = True, is_reengineering: bool = False, num_classes_in_super: int = -1, **kwargs: Any) -> MobileNetV3:
    """
    Constructs a small MobileNetV3 architecture from
    `"Searching for MobileNetV3" <https://arxiv.org/abs/1905.02244>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    arch = "mobilenet_v3_small"
    inverted_residual_setting, last_channel = _mobilenet_v3_conf(arch, kwargs)
    return _mobilenet_v3_model(arch, inverted_residual_setting, last_channel, pretrained, progress, is_reengineering=is_reengineering, num_classes_in_super=num_classes_in_super, **kwargs)
