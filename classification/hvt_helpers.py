import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from einops import *
import torch.nn.functional as F

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
from timm.models.vision_transformer import vit_tiny_patch16_224


class BaseAttention(nn.Module):
    def __init__(self, dim, num_heads=8, mlp_ratio=4, qkv_bias=False,
                 attn_drop=0., proj_drop=0., k=3, norm_layer=nn.LayerNorm,
                 drop_path=0, do_pool=True):
        super().__init__()
        self.k = k
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.do_pool = do_pool

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=dim*mlp_ratio, out_features=dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
    def forward(self, x):
        shortcut = x
        B, C, H, W = x.shape
        if self.do_pool:
            x = F.adaptive_avg_pool2d(x, output_size=(self.k, self.k))
        x = rearrange(x, 'b c h w -> b (h w) c')

        B, N, C = x.shape
        qkv = self.qkv(x)
        qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)  # merge multi heads
        x = self.proj(x)
        x = self.proj_drop(x)

        if self.do_pool:
            x = rearrange(x, 'b (h w) c -> b c h w', h=self.k, w=self.k)
            x = F.interpolate(x, size=(H, W), mode='nearest')
            x = x * shortcut + shortcut
        else:
            x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W) + shortcut

        x = rearrange(x, 'b c h w -> b (h w) c')
        # x = x.view(B, H * W, C)

        x = x + self.drop_path(self.mlp(self.norm2(x)))
        x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)
        return x


class MultiWindowAttention(nn.Module):
    def __init__(self, scale=2, dim=16, k=2, mlp_ratio=4, num_heads=4, do_pool=True):
        super().__init__()
        self.scale = scale

        # head = 3, 6, 12, 24
        # feature_dim = 96, 192,  384, 768
        self.base_attention = BaseAttention(dim=dim, k=k, num_heads=num_heads,
                                            mlp_ratio=mlp_ratio, do_pool=do_pool)

    def forward(self, x):
        # B, C, H, W = x.shape
        x = rearrange(x, f'b c (scale_h h_window) (scale_w w_window)-> '
                           f'(b scale_h scale_w) c h_window w_window',
                      scale_h=self.scale, scale_w=self.scale)

        # x = reduce(x, '(b scale_h scale_w) h_window w_window c -> b scale_h scale_w c', 'max',
        #            scale_h=self.scale, scale_w=self.scale)
        #
        # reduce(x, 'b (h h2) (w w2) c -> h (b w) c', 'max', h2=2, w2=2)

        x = self.base_attention(x)  # pooling, reshape
        x = rearrange(x, f'(b scale_h scale_w) c h_window w_window -> b c (scale_h h_window) (scale_w w_window)',
                      scale_h=self.scale, scale_w=self.scale)

        return x

"""
1 layer B C H W -> B H W C -> B H/2 2 W/2 2 C -pooling-> B 2 2 C > attention
2 layer B C H W -> B H W C -> B H/2 2 W/2 2 C -> B*4 H/2 W/2 C -pooling-> attention
3 layer B C H W -> B H W C ->   

"""

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class ConvDownsampler(nn.Module):
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.reduction = nn.Conv2d(
            dim, 2 * dim, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False
        )
        self.norm = norm_layer(2 * dim)

    def forward(self, x):
        x = self.reduction(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        x = self.norm(x)
        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        # assert img_size[0] % patch_size[0] == 0 and img_size[1] % patch_size[1] == 0, \
        #     f"img_size {img_size} should be divided by patch_size {patch_size}."
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape

        x = self.proj(x)  # .flatten(2).transpose(1, 2)
        # x = self.norm(x)
        H, W = H // self.patch_size[0], W // self.patch_size[1]

        return x, H, W


class ConvTokenizer(nn.Module):
    def __init__(self, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(
                in_chans,
                embed_dim // 2,
                kernel_size=(3, 3),
                stride=(2, 2),
                padding=(1, 1),
            ),
            nn.Conv2d(
                embed_dim // 2,
                embed_dim,
                kernel_size=(3, 3),
                stride=(2, 2),
                padding=(1, 1),
            ),
        )
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = self.proj(x).permute(0, 2, 3, 1)
        if self.norm is not None:
            x = self.norm(x)
        return x  # rearrange(x, 'b h w c -> b c h w')

def _conv_filter(state_dict, patch_size=16):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k:
            v = v.reshape((v.shape[0], 3, patch_size, patch_size))
        out_dict[k] = v

    return out_dict




