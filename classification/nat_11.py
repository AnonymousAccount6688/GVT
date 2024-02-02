import pdb
from einops import repeat
"""
Neighborhood Attention Transformer.
To appear in CVPR 2023.
https://arxiv.org/abs/2204.07143

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
from einops import rearrange
import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model
from natten import NeighborhoodAttention2D as NeighborhoodAttention
from timm.models.layers import create_conv2d

model_urls = {
    "nat_mini_1k": "https://shi-labs.com/projects/nat/checkpoints/CLS/nat_mini.pth",
    "nat_tiny_1k": "https://shi-labs.com/projects/nat/checkpoints/CLS/nat_tiny.pth",
    "nat_small_1k": "https://shi-labs.com/projects/nat/checkpoints/CLS/nat_small.pth",
    "nat_base_1k": "https://shi-labs.com/projects/nat/checkpoints/CLS/nat_base.pth",
}


def val2list(x: list or tuple or any, repeat_time=1):
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x for _ in range(repeat_time)]


def val2tuple(x: list or tuple or any, min_len: int = 1, idx_repeat: int = -1):
    # repeat elements if necessary
    x = val2list(x)
    if len(x) > 0:
        x[idx_repeat:idx_repeat] = [x[idx_repeat] for _ in range(min_len - len(x))]

    return tuple(x)

class ConvNormAct(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size=3,
        stride=1,
        dilation=1,
        groups=1,
        bias=False,
        dropout=0.,
        norm_layer=nn.BatchNorm2d,
        act_layer=nn.ReLU,
    ):
        super(ConvNormAct, self).__init__()
        self.dropout = nn.Dropout(dropout, inplace=False)
        self.conv = create_conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.norm = norm_layer(num_features=out_channels) if norm_layer else nn.Identity()
        self.act = act_layer(inplace=True) if act_layer else nn.Identity()

    def forward(self, x):
        x = self.dropout(x)
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class MBConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size=3,
        stride=1,
        mid_channels=None,
        expand_ratio=6,
        use_bias=False,
        norm_layer=(nn.BatchNorm2d, nn.BatchNorm2d, nn.BatchNorm2d),
        act_layer=(nn.ReLU6, nn.ReLU6, None),
    ):
        super(MBConv, self).__init__()
        use_bias = val2tuple(use_bias, 3)
        norm_layer = val2tuple(norm_layer, 3)
        act_layer = val2tuple(act_layer, 3)
        mid_channels = mid_channels or round(in_channels * expand_ratio)

        self.inverted_conv = ConvNormAct(
            in_channels,
            mid_channels,
            1,
            stride=1,
            norm_layer=norm_layer[0],
            act_layer=act_layer[0],
            bias=use_bias[0],
        )
        self.depth_conv = ConvNormAct(
            mid_channels,
            mid_channels,
            kernel_size,
            stride=stride,
            groups=mid_channels,
            norm_layer=norm_layer[1],
            act_layer=act_layer[1],
            bias=use_bias[1],
        )
        self.point_conv = ConvNormAct(
            mid_channels,
            out_channels,
            1,
            norm_layer=norm_layer[2],
            act_layer=act_layer[2],
            bias=use_bias[2],
        )

    def forward(self, x):
        x = self.inverted_conv(x)
        x = self.depth_conv(x)
        x = self.point_conv(x)
        return x


class SEModule(nn.Module):

    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        module_input = x
        x = x.mean((2, 3), keepdim=True)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x

class depthwise_separable_conv(nn.Module):
    def __init__(self, nin, nout, expand_ratio=4, kernel_size=3, padding = 1, bias=False):
        super(depthwise_separable_conv, self).__init__()
        middle_ch = nin * expand_ratio
        self.depthwise = nn.Conv2d(nin, nin, kernel_size=kernel_size,
                                   padding=padding, groups=nin, bias=bias)
        self.norm_1 = nn.BatchNorm2d(nin)
        self.act_1 = nn.ReLU6()
        self.pointwise = nn.Linear(nin, nout)# (nin, nout, kernel_size=1, stride=1,
                               #    bias=bias, padding=0)
        self.norm_2 = nn.BatchNorm2d(nout)
        self.act_2 = nn.ReLU6()

    def forward(self, x):
        out = self.act_1(self.norm_1(self.depthwise(x)))
        out = self.act_2(self.norm_2(rearrange(self.pointwise(rearrange(out, 'b c h w -> b h w c')),
                                               'b h w c -> b c h w')
                                     ))
        return out



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
        return x


class ConvDownsampler(nn.Module):
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.reduction = nn.Conv2d(
            dim, 2 * dim, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False
        )
        self.norm = norm_layer(2 * dim)

    def forward(self, x):
        x = self.reduction(x.permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3, 1)
        x = self.norm(x)
        return x


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
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


class NATLayer(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        kernel_size=7,
        dilation=None,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        layer_scale=None,
        depth_idx=None
    ):
        super().__init__()

        # self.my_conv = MBConv(in_channels=dim, out_channels=dim)
        # self.my_conv = nn.Conv2d(dim, dim, 3, 1, 1)
        # self.my_conv = nn.Sequential(
        #     nn.Conv2d(dim, dim, 3, 1, 1, groups=dim),
        #     nn.BatchNorm2d(dim),
        #     nn.ReLU6(),
        #     nn.Conv2d(dim, dim, 1, 1, 0),
        #     nn.BatchNorm2d(dim),
        #     nn.ReLU6()
        # )
        # self.my_conv = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)

        # self.my_conv = depthwise_separable_conv(dim, dim)

        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

        # self.act_input = act_layer()

        self.norm1 = norm_layer(dim)
        self.attn = NeighborhoodAttention(
            dim,
            kernel_size=kernel_size,
            dilation=dilation,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=drop,
        )
        self.layer_scale = False
        if layer_scale is not None and type(layer_scale) in [int, float]:
            self.layer_scale = True
            self.gamma1 = nn.Parameter(
                layer_scale * torch.ones(dim), requires_grad=True
            )
            self.gamma2 = nn.Parameter(
                layer_scale * torch.ones(dim), requires_grad=True
            )

        # self.weights = [nn.Parameter(torch.rand(1)) for i in range(depth_idx+1)]
        if depth_idx >= 1:
            self.weights_1 = nn.Parameter(torch.rand(depth_idx+1))

    def forward(self, x):
        if isinstance(x, (tuple, list)):
            if len(x) > 1:
                x = torch.stack(x, dim=0)
                k, b, c, h, w = x.shape

                # weights_1 = 2 * ((self.weights_1 - torch.min(self.weights_1)) / (torch.max(self.weights_1) - torch.min(self.weights_1))) - 1
                # # print(f"weight_1: {weights_1}")

                # # weights_1 = repeat(torch.softmax(self.weights_1, dim=0), 'k -> k b c h w',
                # #                    k=k, b=b, c=c, h=h, w=w)
                # x = x * weights_1

                weights_1 = repeat(self.weights_1,  'k -> k b c h w',
                                   k=k, b=b, c=c, h=h, w=w)
                x = x * weights_1
                x = x.sum(0)
            else:
                x = x[0]

        if not self.layer_scale:
            shortcut = x
            x = self.norm1(x)
            x = self.attn(x)
            x = shortcut + self.drop_path(x)
            x = x + self.drop_path(self.mlp(self.norm2(x)))

            # print(f"shape: {x.shape}")
            # x = rearrange(self.my_conv(rearrange(x, 'b h w c -> b c h w')),
            #               'b c h w -> b h w c')

            return x
        shortcut = x
        x = self.norm1(x)
        x = self.attn(x)
        x = shortcut + self.drop_path(self.gamma1 * x)
        x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))

        # print(f"shape: {x.shape}")
        # x = rearrange(self.my_conv(rearrange(x, 'b h w c -> b c h w')),
        #               'b c h w -> b h w c')
        # x = self.my_conv(x)
        return x


class NATBlock(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        num_heads,
        kernel_size,
        dilations=None,
        downsample=True,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        layer_scale=None,
    ):
        super().__init__()
        self.dim = dim
        self.depth = depth

        self.blocks = nn.ModuleList(
            [
                NATLayer(
                    dim=dim,
                    num_heads=num_heads,
                    kernel_size=kernel_size,
                    dilation=None if dilations is None else dilations[i],
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[i]
                    if isinstance(drop_path, list)
                    else drop_path,
                    norm_layer=norm_layer,
                    layer_scale=layer_scale,
                    depth_idx=i
                )
                for i in range(depth)
            ]
        )

        self.downsample = (
            None if not downsample else ConvDownsampler(dim=dim, norm_layer=norm_layer)
        )

    def forward(self, x):
        features = [x]
        for blk in self.blocks:
            # x = blk(x)
            x = blk(features)
            features.append(x)
        if self.downsample is None:
            return x
        # print(x.shape)
        # print(self.downsample)
        return self.downsample(x)


class NAT(nn.Module):
    def __init__(
        self,
        embed_dim,
        mlp_ratio,
        depths,
        num_heads,
        drop_path_rate=0.2,
        in_chans=3,
        kernel_size=7,
        dilations=None,
        num_classes=1000,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        norm_layer=nn.LayerNorm,
        layer_scale=None,
        **kwargs
    ):
        super().__init__()

        self.num_classes = num_classes
        self.num_levels = len(depths)
        self.embed_dim = embed_dim
        self.num_features = int(embed_dim * 2 ** (self.num_levels - 1))
        self.mlp_ratio = mlp_ratio

        self.patch_embed = ConvTokenizer(
            in_chans=in_chans, embed_dim=embed_dim, norm_layer=norm_layer
        )

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.levels = nn.ModuleList()
        for i in range(self.num_levels):
            level = NATBlock(
                dim=int(embed_dim * 2**i),
                depth=depths[i],
                num_heads=num_heads[i],
                kernel_size=kernel_size,
                dilations=None if dilations is None else dilations[i],
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i]) : sum(depths[: i + 1])],
                norm_layer=norm_layer,
                downsample=(i < self.num_levels - 1),
                layer_scale=layer_scale,
            )
            self.levels.append(level)

        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = (
            nn.Linear(self.num_features, num_classes)
            if num_classes > 0
            else nn.Identity()
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {"rpb"}

    def forward_features(self, x):
        x = self.patch_embed(x)
        x = self.pos_drop(x)

        for level in self.levels:
            x = level(x)

        x = self.norm(x).flatten(1, 2)
        x = self.avgpool(x.transpose(1, 2))
        x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


@register_model
def nat_mini(pretrained=False, **kwargs):
    model = NAT(
        depths=[3, 4, 6, 5],
        num_heads=[2, 4, 8, 16],
        embed_dim=64,
        mlp_ratio=3,
        drop_path_rate=0.2,
        kernel_size=7,
        **kwargs
    )
    if pretrained:
        url = model_urls["nat_mini_1k"]
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint)
    return model


@register_model
def nat_tiny(pretrained=False, **kwargs):
    model = NAT(
        depths=[3, 4, 18, 5],
        num_heads=[2, 4, 8, 16],
        embed_dim=64,
        mlp_ratio=3,
        drop_path_rate=0.2,
        kernel_size=7,
        **kwargs
    )
    if pretrained:
        url = model_urls["nat_tiny_1k"]
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint)
    return model


@register_model
def nat_small(pretrained=False, **kwargs):
    model = NAT(
        depths=[3, 4, 18, 5],
        num_heads=[3, 6, 12, 24],
        embed_dim=96,
        mlp_ratio=2,
        drop_path_rate=0.3,
        layer_scale=1e-5,
        kernel_size=7,
        **kwargs
    )
    if pretrained:
        url = model_urls["nat_small_1k"]
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint)
    return model


@register_model
def nat_base(pretrained=False, **kwargs):
    model = NAT(
        depths=[3, 4, 18, 5],
        num_heads=[4, 8, 16, 32],
        embed_dim=128,
        mlp_ratio=2,
        drop_path_rate=0.5,
        layer_scale=1e-5,
        kernel_size=7,
        **kwargs
    )
    if pretrained:
        url = model_urls["nat_base_1k"]
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint)
    return model


if __name__ == "__main__":
    model = nat_tiny()
    x = torch.rand((2, 3, 224, 224))
    y = model(x)
    print(y.shape)