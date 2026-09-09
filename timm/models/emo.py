""" EMO

Paper: `Rethinking Mobile Block for Efficient Attention-based Models` - https://arxiv.org/abs/2301.01146
@inproceedings{emo,
  title={Rethinking Mobile Block for Efficient Attention-based Models},
  author={Zhang, Jiangning and Li, Xiangtai and Li, Jian and Liu, Liang and Xue, Zhucun and Zhang, Boshen and Jiang, Zhengkai and Huang, Tianxin and Wang, Yabiao and Wang, Chengjie},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={1389--1400},
  year={2023}
}

Original code and weights from:
* https://github.com/zhangzjn/EMO

Modifications and additions for timm by / Copyright 2026, Ryan Hou & Ross Wightman
"""
from typing import Any, Dict, List, Optional, Set, Tuple, Type, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.layers import (
    DropPath,
    Linear,
    SelectAdaptivePool2d,
    SqueezeExcite,
    calculate_drop_path_rates,
    get_act_layer,
    get_device_dtype,
    get_norm_layer,
    trunc_normal_,
)

from ._builder import build_model_with_cfg
from ._features import feature_take_indices
from ._manipulate import checkpoint_seq
from ._registry import generate_default_cfgs, register_model

__all__ = ['EMO']


class iRMB(nn.Module):
    def __init__(
            self,
            dim_in: int,
            dim_out: int,
            exp_ratio: float = 1.0,
            norm_layer: Type[nn.Module] = nn.BatchNorm2d,
            act_layer: Type[nn.Module] = nn.ReLU,
            dw_ks: int = 3,
            stride: int = 1,
            dim_head: int = 64,
            window_size: int = 7,
            attn_s: bool = True,
            attn_drop: float = 0.,
            drop: float = 0.,
            drop_path: float = 0.,
            device=None,
            dtype=None,
    ):
        dd = {'device': device, 'dtype': dtype}
        super().__init__()
        self.norm = norm_layer(dim_in, eps=1e-6, **dd) if norm_layer else nn.Identity()
        self.dim_mid = int(dim_in * exp_ratio)
        self.has_skip = (dim_in == dim_out and stride == 1)
        self.attn_s = attn_s
        self.dim_head = dim_head
        self.window_size = window_size
        self.num_head = dim_in // dim_head if dim_in % dim_head == 0 else 0
        self.scale = self.dim_head ** -0.5 if self.num_head else 0.
        if self.attn_s:
            assert dim_in % dim_head == 0, 'dim should be divisible by num_heads'
            self.qk = nn.Conv2d(dim_in, int(dim_in * 2), 1, **dd)
            self.attn_drop = nn.Dropout(attn_drop)
        else:
            self.qk = nn.Identity()
            self.attn_drop = nn.Identity()

        self.v = nn.Sequential(
            nn.Conv2d(dim_in, dim_mid, 1, **dd),
            nn.Conv2d(dim_in, self.dim_mid, 1, **dd),
            act_layer() if act_layer else nn.Identity(),
        )

        self.conv_local = nn.Sequential(
            nn.Conv2d(self.dim_mid, self.dim_mid, dw_ks, stride, padding=dw_ks//2, groups=self.dim_mid, bias=False, **dd),
            nn.BatchNorm2d(self.dim_mid, eps=1e-6, **dd),
            nn.SiLU(),
        )

        self.proj_drop = nn.Dropout(drop)
        self.proj = nn.Conv2d(self.dim_mid, dim_out, 1, bias=False, **dd)
        self.drop_path = DropPath(drop_path) if drop_path else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.norm(x)
        _, _, H, W = x.shape

        if self.attn_s:
            x, n1, n2 = self._pad_input(x, H, W)
            x = self._window_partition(x, n1, n2)

            attn_spa = self._qk_attn(x)
            x_spa = self._apply_attn(x, attn_spa)
            x_spa = self.v(x_spa)

            x = self._window_reverse(x_spa, n1, n2)
            x = x[:, :, :H, :W]
        else:
            x = self.v(x)

        if self.has_skip:
            x = x + self.conv_local(x)
        else:
            x = self.conv_local(x)
        x = self.proj_drop(x)
        x = self.proj(x)
        x = shortcut + self.drop_path(x) if self.has_skip else x
        return x

    def _pad_input(self, x: torch.Tensor, H: int, W: int) -> Tuple[torch.Tensor, int, int]:
        if self.window_size <= 0:
            window_size_W, window_size_H = W, H
        else:
            window_size_W, window_size_H = self.window_size, self.window_size

        pad_r = (window_size_W - W % window_size_W) % window_size_W
        pad_b = (window_size_H - H % window_size_H) % window_size_H

        x = F.pad(x, (0, pad_r, 0, pad_b))
        n1, n2 = (H + pad_b) // window_size_H, (W + pad_r) // window_size_W
        return x, n1, n2

    def _window_partition(self, x: torch.Tensor, n1: int, n2: int, close: bool = False) -> torch.Tensor:
        b, c, h, w = x.shape
        if not close:
            x = x.view(b, c, h // n1, n1, w // n2, n2).permute(0, 3, 5, 1, 2, 4)
        else:
            x = x.view(b, c, n1, h // n1, n2, w // n2).permute(0, 2, 4, 1, 3, 5)
        return x.reshape(b * n1 * n2, c, h // n1, w // n2).contiguous()

    def _window_reverse(self, x: torch.Tensor, n1: int, n2: int, close: bool = False) -> torch.Tensor:
        _, c, h, w = x.shape
        x = x.view(-1, n1, n2, c, h, w)
        if not close:
            x = x.permute(0, 3, 4, 1, 5, 2)
        else:
            x = x.permute(0, 3, 1, 4, 2, 5)
        return x.reshape(-1, c, h * n1, w * n2)

    def _qk_attn(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        qk = self.qk(x).view(b, 2, self.num_head, self.dim_head, h, w).flatten(-2)
        qk = qk.permute(1, 0, 2, 4, 3).contiguous()
        q, k = qk[0], qk[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        return self.attn_drop(attn.softmax(dim=-1))

    def _apply_attn(self, x: torch.Tensor, attn_spa: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        x = x.view(b, self.num_head, c // self.num_head, h, w).flatten(-2).transpose(-1, -2)
        x_spa = attn_spa @ x
        return x_spa.transpose(-1, -2).reshape(b, c, h, w)


class Stage(nn.Module):
    def __init__(
            self,
            depth: int,
            emb_dim_pre: int,
            embed_dim: int,
            exp_ratio: float = 1.0,
            norm_layer: Type[nn.Module] = nn.BatchNorm2d,
            act_layer: Type[nn.Module] = nn.ReLU,
            dw_ks: int = 3,
            dim_head: int = 64,
            window_size: int = 7,
            attn_s: bool = True,
            drop: float = 0.,
            attn_drop: float = 0.,
            dpr: Optional[List[float]] = None,
            device=None,
            dtype=None,
    ):
        dd = {'device': device, 'dtype': dtype}
        super().__init__()
        self.grad_checkpointing = False

        self.downsample = iRMB(
            emb_dim_pre,
            embed_dim,
            exp_ratio=exp_ratio * 2,
            norm_layer=norm_layer,
            act_layer=act_layer,
            dw_ks=dw_ks,
            stride=2,
            dim_head=dim_head,
            window_size=window_size,
            attn_s=False,
            attn_drop=attn_drop,
            drop=drop,
            drop_path=dpr[0],
            **dd,
        )
        blocks = []
        for j in range(1, depth):
            blocks.append(iRMB(
                embed_dim,
                embed_dim,
                exp_ratio=exp_ratio,
                norm_layer=norm_layer,
                act_layer=act_layer,
                dw_ks=dw_ks,
                stride=1,
                dim_head=dim_head,
                window_size=window_size,
                attn_s=attn_s,
                attn_drop=attn_drop,
                drop=drop,
                drop_path=dpr[j],
                **dd,
            ))
        self.blocks = nn.Sequential(*blocks) if blocks else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.downsample(x)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.blocks, x)
        else:
            x = self.blocks(x)
        return x


class EMO(nn.Module):
    def __init__(
            self,
            in_chans: int = 3,
            num_classes: int = 1000,
            depths: Tuple[int, ...] = (1, 2, 4, 2),
            stem_dim: int = 24,
            embed_dims: Tuple[int, ...] = (64, 128, 256, 512),
            exp_ratios: Tuple[float, ...] = (4., 4., 4., 4.),
            norm_layers: Tuple[str, ...] = ('batchnorm2d', 'batchnorm2d', 'layernorm2d', 'layernorm2d'),
            act_layers: Tuple[str, ...] = ('silu', 'silu', 'gelu', 'gelu'),
            dw_kss: Tuple[int, ...] = (3, 3, 5, 5),
            dim_heads: Tuple[int, ...] = (32, 32, 32, 32),
            window_sizes: Tuple[int, ...] = (7, 7, 7, 7),
            attn_ss: Tuple[bool, ...] = (False, False, True, True),
            attn_drop: float = 0.,
            drop_rate: float = 0.,
            drop_path_rate: float = 0.,
            global_pool: str = 'avg',
            device=None,
            dtype=None,
    ):
        super().__init__()
        dd = {'device': device, 'dtype': dtype}
        self.num_classes = num_classes
        self.in_chans = in_chans
        self.drop_rate = drop_rate
        self.feature_info = []

        self.patch_embed = nn.Sequential(
            nn.Conv2d(in_chans, stem_dim, dw_kss[0], 2, dw_kss[0]//2, **dd),
            get_norm_layer(norm_layers[0])(stem_dim, eps=1e-6, **dd) if norm_layers[0] else nn.Identity(),
            nn.Conv2d(stem_dim, stem_dim, dw_kss[0], 1, dw_kss[0]//2, groups=stem_dim, bias=False, **dd),
            nn.BatchNorm2d(stem_dim, eps=1e-6, **dd),
            nn.SiLU(),
            SqueezeExcite(stem_dim, rd_ratio=1, act_layer=act_layers[0], **dd),
            nn.Dropout(drop_rate),
            nn.Conv2d(stem_dim, stem_dim, 1, bias=False, **dd)
        )

        dprs = calculate_drop_path_rates(drop_path_rate, sum(depths))
        emb_dim_pre = stem_dim
        stages = []
        reduction = 2
        for i in range(len(depths)):
            stage = Stage(
                depth=depths[i],
                emb_dim_pre=emb_dim_pre,
                embed_dim=embed_dims[i],
                exp_ratio=exp_ratios[i],
                norm_layer=get_norm_layer(norm_layers[i]),
                act_layer=get_act_layer(act_layers[i]),
                dw_ks=dw_kss[i],
                dim_head=dim_heads[i],
                window_size=window_sizes[i],
                attn_s=attn_ss[i],
                drop=drop_rate,
                attn_drop=attn_drop,
                dpr=dprs[sum(depths[:i]):sum(depths[:i + 1])],
                device=device,
                dtype=dtype,
            )
            emb_dim_pre = embed_dims[i]
            stages.append(stage)
            reduction *= 2
            self.feature_info.append(dict(num_chs=embed_dims[i], reduction=reduction, module=f'stages.{i}'))
        self.stages = nn.Sequential(*stages)

        self.norm = get_norm_layer(norm_layers[-1])(embed_dims[-1], eps=1e-6, **dd) if norm_layers[-1] else nn.Identity()
        self.num_features = self.head_hidden_size = embed_dims[-1]
        self.global_pool = SelectAdaptivePool2d(pool_type=global_pool)
        self.flatten = nn.Flatten(1) if global_pool else nn.Identity()
        self.head = Linear(self.head_hidden_size, num_classes, **dd) if num_classes > 0 else nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Conv2d):
            trunc_normal_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)

    @torch.jit.ignore
    def no_weight_decay(self) -> Set:
        return set()

    @torch.jit.ignore
    def group_matcher(self, coarse: bool = False) -> Dict[str, Any]:
        return dict(
            stem=r'^patch_embed',
            blocks=r'^stages\.(\d+)' if coarse else [
                (r'^stages\.(\d+)\.downsample', (0,)),
                (r'^stages\.(\d+)\.blocks\.(\d+)', None),
                (r'^norm', (99999,)),
            ]
        )

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable: bool = True):
        for s in self.stages:
            s.grad_checkpointing = enable

    @torch.jit.ignore
    def get_classifier(self) -> nn.Module:
        return self.head

    def reset_classifier(self, num_classes: int, global_pool: Optional[str] = None):
        dd = get_device_dtype(self)
        was_training = self.training
        self.num_classes = num_classes
        if global_pool is not None:
            self.global_pool = SelectAdaptivePool2d(pool_type=global_pool)
            self.flatten = nn.Flatten(1) if global_pool else nn.Identity()  # don't flatten if pooling disabled
            self.global_pool.train(was_training)
            self.flatten.train(was_training)
        self.head = Linear(self.head_hidden_size, num_classes, **dd) if num_classes > 0 else nn.Identity()
        self.head.train(was_training)

    def forward_intermediates(
            self,
            x: torch.Tensor,
            indices: Optional[Union[int, List[int]]] = None,
            norm: bool = False,
            stop_early: bool = False,
            output_fmt: str = 'NCHW',
            intermediates_only: bool = False,
    ) -> Union[List[torch.Tensor], Tuple[torch.Tensor, List[torch.Tensor]]]:
        """ Forward features that returns intermediates.

        Args:
            x: Input image tensor
            indices: Take last n blocks if int, all if None, select matching indices if sequence
            norm: Apply norm layer to compatible intermediates
            stop_early: Stop iterating over blocks when last desired intermediate hit
            output_fmt: Shape of intermediate feature outputs
            intermediates_only: Only return intermediate features
        Returns:
            List of intermediate features or tuple of (final features, intermediates).
        """
        assert output_fmt in ('NCHW',), 'Output shape must be NCHW.'
        intermediates = []
        take_indices, max_index = feature_take_indices(len(self.stages), indices)
        last_idx = len(self.stages) - 1

        # forward pass
        x = self.patch_embed(x)
        if torch.jit.is_scripting() or not stop_early:  # can't slice blocks in torchscript
            stages = self.stages
        else:
            stages = self.stages[:max_index + 1]

        for feat_idx, stage in enumerate(stages):
            x = stage(x)
            if feat_idx in take_indices:
                if norm and feat_idx == last_idx:
                    x_inter = self.norm(x)  # applying final norm last intermediate
                else:
                    x_inter = x
                intermediates.append(x_inter)

        if intermediates_only:
            return intermediates

        x = self.norm(x)

        return x, intermediates

    def prune_intermediate_layers(
            self,
            indices: Union[int, List[int]] = 1,
            prune_norm: bool = False,
            prune_head: bool = True,
    ):
        """Prune layers not required for specified intermediates.
        
                Args:
                    indices: Indices of intermediate layers to keep.
                    prune_norm: Whether to prune normalization layer.
                    prune_head: Whether to prune the classifier head.
        
                Returns:
                    List of indices that were kept.
                """
        take_indices, max_index = feature_take_indices(len(self.stages), indices)
        self.stages = self.stages[:max_index + 1]  # truncate blocks w/ stem as idx 0
        if prune_norm:
            self.norm = nn.Identity()
        if prune_head:
            self.reset_classifier(0, '')
        return take_indices

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        x = self.stages(x)
        x = self.norm(x)
        return x

    def forward_head(self, x: torch.Tensor, pre_logits: bool = False) -> torch.Tensor:
        x = self.global_pool(x)
        x = self.flatten(x)
        if self.drop_rate > 0.:
            x = F.dropout(x, p=self.drop_rate, training=self.training)
        return x if pre_logits else self.head(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


def checkpoint_filter_fn(state_dict: Dict[str, torch.Tensor], model: nn.Module) -> Dict[str, torch.Tensor]:
    import re
    out_dict = {}
    for k, v in state_dict.items():
        if k.startswith('stage0.0.'):
            k = k.replace('stage0.0.convs.0.0.', 'patch_embed.0.')
            k = k.replace('stage0.0.convs.0.1.', 'patch_embed.1.')
        elif k.startswith('stage0.1.'):
            k = k.replace('stage0.1.conv_local.conv.', 'patch_embed.2.')
            k = k.replace('stage0.1.conv_local.norm.', 'patch_embed.3.')
            k = k.replace('stage0.1.se.', 'patch_embed.5.')
            k = k.replace('stage0.1.proj.conv.', 'patch_embed.7.')
        else:
            m = re.match(r'stage([1-4])\.(\d+)\.(.*)', k)
            if m:
                stage_idx = int(m.group(1)) - 1
                block_idx = int(m.group(2))
                rest = m.group(3)
                if block_idx == 0:
                    k = f'stages.{stage_idx}.downsample.{rest}'
                else:
                    k = f'stages.{stage_idx}.blocks.{block_idx - 1}.{rest}'
        k = k.replace('norm.norm.', 'norm.')
        k = k.replace('se.conv_reduce.', 'se.fc1.')
        k = k.replace('se.conv_expand.', 'se.fc2.')
        k = k.replace('conv_reduce.', 'fc1.')
        k = k.replace('conv_expand.', 'fc2.')
        k = k.replace('conv_local.norm.', 'conv_local.1.')
        k = k.replace('conv_local.conv.', 'conv_local.0.')
        k = k.replace('qk.conv.', 'qk.')
        k = k.replace('v.conv.', 'v.0.')
        k = k.replace('proj.conv.', 'proj.')
        out_dict[k] = v
    return out_dict


def _cfg(url: str = '', **kwargs: Any) -> Dict[str, Any]:
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': (7, 7),
        'crop_pct': 0.875, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.0', 'classifier': 'head',
        'paper_ids': 'arXiv:2301.01146',
        'paper_name': 'Rethinking Mobile Block for Efficient Attention-based Models',
        'origin_url': 'https://github.com/zhangzjn/EMO',
        **kwargs
    }


default_cfgs = generate_default_cfgs({
    'emo_1m.in1k': _cfg(
        # hf_hub_id='timm/',
        # url='https://github.com/zhangzjn/EMO/blob/main/resources/EMO_1M/net.pth'
    ),
    'emo_2m.in1k': _cfg(
        # hf_hub_id='timm/',
        # url='https://github.com/zhangzjn/EMO/blob/main/resources/EMO_2M/net.pth'
    ),
    'emo_5m.in1k': _cfg(
        # hf_hub_id='timm/',
        # url='https://github.com/zhangzjn/EMO/blob/main/resources/EMO_5M/net.pth'
    ),
    'emo_6m.in1k': _cfg(
        # hf_hub_id='timm/',
        # url='https://github.com/zhangzjn/EMO/blob/main/resources/EMO_6M/net.pth'
    ),
})


def _create_emo(variant: str, pretrained: bool = False, **kwargs: Any) -> EMO:
    model = build_model_with_cfg(
        EMO, variant, pretrained,
        pretrained_filter_fn=checkpoint_filter_fn,
        feature_cfg=dict(out_indices=(0, 1, 2, 3), flatten_sequential=True),
        **kwargs,
    )
    return model


@register_model
def emo_1m(pretrained: bool = False, **kwargs: Any) -> EMO:
    """ EMO-1M """
    model_args = dict(
        depths=(2, 2, 8, 3), embed_dims=(32, 48, 80, 168), exp_ratios=(2., 2.5, 3.0, 3.5),
        dim_heads=(16, 16, 20, 21), drop_path_rate=0.04036,
    )
    return _create_emo('emo_1m', pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def emo_2m(pretrained: bool = False, **kwargs: Any) -> EMO:
    """ EMO-2M """
    model_args = dict(
        depths=(3, 3, 9, 3), embed_dims=(32, 48, 120, 200), exp_ratios=(2., 2.5, 3.0, 3.5),
        dim_heads=(16, 16, 20, 20), drop_path_rate=0.05,
    )
    return _create_emo('emo_2m', pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def emo_5m(pretrained: bool = False, **kwargs: Any) -> EMO:
    """ EMO-5M """
    model_args = dict(
        depths=(3, 3, 9, 3), embed_dims=(48, 72, 160, 288), exp_ratios=(2., 3., 4., 4.),
        dim_heads=(24, 24, 32, 32), drop_path_rate=0.05,
    )
    return _create_emo('emo_5m', pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def emo_6m(pretrained: bool = False, **kwargs: Any) -> EMO:
    """ EMO-6M """
    model_args = dict(
        depths=(3, 3, 9, 3), embed_dims=(48, 72, 160, 320), exp_ratios=(2., 3., 4., 5.),
        dim_heads=(16, 24, 20, 32), drop_path_rate=0.05,
    )
    return _create_emo('emo_6m', pretrained=pretrained, **dict(model_args, **kwargs))
