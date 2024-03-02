import copy
import errno
import torch
import os
from os.path import join as pjoin
import numpy as np
from torch import nn
from typing import Optional
from torch.nn.modules.utils import _pair
from multi_class.models.nn_layers import MaskConv, MaskLinear, Binarization
from torchvision.models.utils import load_state_dict_from_url
import ml_collections
from torch.hub import get_dir, download_url_to_file
from urllib.parse import urlparse
TOK_FC_0 = "token_mixing/Dense_0"
TOK_FC_1 = "token_mixing/Dense_1"
CHA_FC_0 = "channel_mixing/Dense_0"
CHA_FC_1 = "channel_mixing/Dense_1"
PRE_NORM = "LayerNorm_0"
POST_NORM = "LayerNorm_1"


def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


class MlpBlock(nn.Module):
    def __init__(self, hidden_dim, ff_dim, is_reengineering=False):
        super(MlpBlock, self).__init__()
        self.fc0 = MaskLinear(hidden_dim, ff_dim, bias=True, is_reengineering=is_reengineering)
        self.fc1 = MaskLinear(ff_dim, hidden_dim, bias=True, is_reengineering=is_reengineering)
        self.act_fn = nn.GELU()

    def forward(self, x):
        x = self.fc0(x)
        x = self.act_fn(x)
        x = self.fc1(x)
        return x


class MixerBlock(nn.Module):
    def __init__(self, config, is_reengineering=False):
        super(MixerBlock, self).__init__()
        self.token_mlp_block = MlpBlock(config.n_patches, config.tokens_mlp_dim, is_reengineering=is_reengineering)
        self.channel_mlp_block = MlpBlock(config.hidden_dim, config.channels_mlp_dim, is_reengineering=is_reengineering)
        self.pre_norm = nn.LayerNorm(config.hidden_dim, eps=1e-6)
        self.post_norm = nn.LayerNorm(config.hidden_dim, eps=1e-6)

    def forward(self, x):
        h = x
        x = self.pre_norm(x)
        x = x.transpose(-1, -2)
        x = self.token_mlp_block(x)
        x = x.transpose(-1, -2)
        x = x + h

        h = x
        x = self.post_norm(x)
        x = self.channel_mlp_block(x)
        x = x + h
        return x

    def load_from(self, weights, n_block):
        ROOT = f"MixerBlock_{n_block}"
        with torch.no_grad():
            self.token_mlp_block.fc0.weight.copy_(
                np2th(weights[pjoin(ROOT, TOK_FC_0, "kernel")]).t())
            self.token_mlp_block.fc1.weight.copy_(
                np2th(weights[pjoin(ROOT, TOK_FC_1, "kernel")]).t())
            self.token_mlp_block.fc0.bias.copy_(
                np2th(weights[pjoin(ROOT, TOK_FC_0, "bias")]).t())
            self.token_mlp_block.fc1.bias.copy_(
                np2th(weights[pjoin(ROOT, TOK_FC_1, "bias")]).t())

            self.channel_mlp_block.fc0.weight.copy_(
                np2th(weights[pjoin(ROOT, CHA_FC_0, "kernel")]).t())
            self.channel_mlp_block.fc1.weight.copy_(
                np2th(weights[pjoin(ROOT, CHA_FC_1, "kernel")]).t())
            self.channel_mlp_block.fc0.bias.copy_(
                np2th(weights[pjoin(ROOT, CHA_FC_0, "bias")]).t())
            self.channel_mlp_block.fc1.bias.copy_(
                np2th(weights[pjoin(ROOT, CHA_FC_1, "bias")]).t())

            self.pre_norm.weight.copy_(np2th(weights[pjoin(ROOT, PRE_NORM, "scale")]))
            self.pre_norm.bias.copy_(np2th(weights[pjoin(ROOT, PRE_NORM, "bias")]))
            self.post_norm.weight.copy_(np2th(weights[pjoin(ROOT, POST_NORM, "scale")]))
            self.post_norm.bias.copy_(np2th(weights[pjoin(ROOT, POST_NORM, "bias")]))


class MlpMixer(nn.Module):
    def __init__(self, config, img_size=224, num_classes=1000, patch_size=16, zero_head=False, is_reengineering=False, num_classes_in_super:int=-1,):
        super(MlpMixer, self).__init__()
        self.zero_head = zero_head
        self.num_classes = num_classes
        patch_size = _pair(patch_size)
        n_patches = (img_size // patch_size[0]) * (img_size // patch_size[1])
        config.n_patches = n_patches

        self.stem = MaskConv(in_channels=3,
                              out_channels=config.hidden_dim,
                              kernel_size=patch_size,
                              stride=patch_size, is_reengineering=is_reengineering)
        self.head = MaskLinear(config.hidden_dim, num_classes, bias=True, is_reengineering=is_reengineering)
        self.pre_head_ln = nn.LayerNorm(config.hidden_dim, eps=1e-6)

        
        self.layer = nn.ModuleList()
        for _ in range(config.num_blocks):
            layer = MixerBlock(config, is_reengineering=is_reengineering)
            self.layer.append(copy.deepcopy(layer))
        self.is_reengineering = is_reengineering
        self.num_classes_in_super = num_classes_in_super
        
        if is_reengineering:
            assert num_classes_in_super > 0
            self.module_head = nn.Sequential(
                nn.ReLU(inplace=True),
                nn.Linear(num_classes, num_classes_in_super)
            )

    def forward(self, x):
        x = self.stem(x)
        x = x.flatten(2)
        x = x.transpose(-1, -2)

        for block in self.layer:
            x = block(x)
        x = self.pre_head_ln(x)
        x = torch.mean(x, dim=1)
        x = self.head(x)

        if hasattr(self, 'module_head'):
            x = self.module_head(x)
        return x
    
    def get_masks(self):
        masks = {k: v for k, v in self.state_dict().items() if 'mask' in k}
        return masks

    def get_module_head(self):
        head = {k: v for k, v in self.state_dict().items() if 'module_head' in k}
        return head

    def load_from(self, weights):
        with torch.no_grad():
            if self.zero_head:
                nn.init.zeros_(self.head.weight)
                nn.init.zeros_(self.head.bias)
            else:
                self.head.weight.copy_(np2th(weights["head/kernel"]).t())
                self.head.bias.copy_(np2th(weights["head/bias"]).t())
            self.stem.weight.copy_(np2th(weights["stem/kernel"], conv=True))
            self.stem.bias.copy_(np2th(weights["stem/bias"]))
            self.pre_head_ln.weight.copy_(np2th(weights["pre_head_layer_norm/scale"]))
            self.pre_head_ln.bias.copy_(np2th(weights["pre_head_layer_norm/bias"]))

            for bname, block in self.layer.named_children():
                block.load_from(weights, n_block=bname)
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

def load_npz_from_url(
    url,
    map_location = None,
    progress = True,
    check_hash = False,
    weights_only = False,
):
    hub_dir = get_dir()
    model_dir = os.path.join(hub_dir, 'checkpoints')

    try:
        os.makedirs(model_dir)
    except OSError as e:
        if e.errno == errno.EEXIST:
            # Directory already exists, ignore.
            pass
        else:
            # Unexpected OSError, re-raise.
            raise

    parts = urlparse(url)
    filename = os.path.basename(parts.path)
    cached_file = os.path.join(model_dir, filename)
    if not os.path.exists(cached_file):
        sys.stderr.write(f'Downloading: "{url}" to {cached_file}\n')
        hash_prefix = None
        download_url_to_file(url, cached_file, hash_prefix, progress=progress)
    return np.load(cached_file)

def mlpmixer(pretrained: bool = False, progress: bool = True, is_reengineering: bool = False, num_classes_in_super: int = -1) -> MlpMixer:
    model = MlpMixer(get_mixer_b16_config(), 224, is_reengineering=is_reengineering, num_classes_in_super=num_classes_in_super)
    if pretrained:
        model.load_from(load_npz_from_url('https://storage.googleapis.com/mixer_models/imagenet1k/Mixer-B_16.npz'))
    return model
def get_mixer_b16_config():
    """Returns Mixer-B/16 configuration."""
    config = ml_collections.ConfigDict()
    config.name = 'Mixer-B_16'
    config.patches = ml_collections.ConfigDict({'size': (16, 16)})
    config.hidden_dim = 768
    config.num_blocks = 12
    config.tokens_mlp_dim = 384
    config.channels_mlp_dim = 3072
    return config
