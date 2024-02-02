from torch.amp import autocast
import pdb

from einops import rearrange
import math
import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model
from natten import NeighborhoodAttention2D as NeighborhoodAttention


model_urls = {
    "nat_mini_1k": "https://shi-labs.com/projects/nat/checkpoints/CLS/nat_mini.pth",
    "nat_tiny_1k": "https://shi-labs.com/projects/nat/checkpoints/CLS/nat_tiny.pth",
    "nat_small_1k": "https://shi-labs.com/projects/nat/checkpoints/CLS/nat_small.pth",
    "nat_base_1k": "https://shi-labs.com/projects/nat/checkpoints/CLS/nat_base.pth",
}


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
        x = self.reduction(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
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


class Mlp2(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
        dims=None,
    ):
        super().__init__()
        self.dims = dims
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        # self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()

        self.fc1s = nn.ModuleList([
            nn.Linear(dim, hidden_dim) for dim, hidden_dim in zip(dims, hidden_features)
        ])

        self.fc2 = nn.Linear(sum(hidden_features), out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        xs = torch.split(x, self.dims, -1)
        xs = [self.drop(self.act(self.fc1s[i](x))) for i, x in enumerate(xs)]
        # x = self.fc1(x)
        # x = self.act(x)
        # x = self.drop(x)
        x = torch.cat(xs, dim=-1)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# class Mlp3(nn.Module):
#     def __init__(
#         self,
#         in_features,
#         hidden_features=None,
#         out_features=None,
#         act_layer=nn.GELU,
#         drop=0.0,
#         dims=None,
#     ):
#         super().__init__()
#         self.dims = dims
#         out_features = out_features or in_features
#         hidden_features = hidden_features or in_features
#         # self.fc1 = nn.Linear(in_features, hidden_features)
#         self.act = act_layer()
#
#         self.fc1s = nn.ModuleList([
#             nn.Linear(dim, hidden_dim) for dim, hidden_dim in zip(dims, hidden_features)
#         ])
#
#         self.fc2 = nn.ModuleList([
#             nn.Linear(hidden_dim, dim) for dim, hidden_dim in zip(dims, hidden_features)
#         ])
#
#         self.drop = nn.Dropout(drop)
#
#     def forward(self, x):
#         xs = torch.split(x, self.dims, -1)
#
#         for i, x in enumerate(xs):
#
#             if i == 0:
#                 x = x
#             else:
#                 x = x +
#
#             x = self.fc1s[i](x)
#             x = self.act(x)
#             x = self.drop(x)
#             x = self.fc2(x)
#             x = self.drop(x)
#             if i == 0:
#                 out = x
#             else:
#                 out = torch.cat([out, x], dim=-1)
#
#         return out


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
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

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

    def forward(self, x):
        if not self.layer_scale:
            shortcut = x
            x = self.norm1(x)
            x = self.attn(x)
            x = shortcut + self.drop_path(x)
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x
        shortcut = x
        x = self.norm1(x)
        x = self.attn(x)
        x = shortcut + self.drop_path(self.gamma1 * x)
        x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
        return x


class NATLayer1(nn.Module):
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
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

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

    def forward(self, x):
        if not self.layer_scale:
            shortcut = x
            x = self.norm1(x)
            x = self.attn(x)
            x = shortcut + self.drop_path(x)
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x
        shortcut = x
        x = self.norm1(x)
        x = self.attn(x)
        x = shortcut + self.drop_path(self.gamma1 * x)
        x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
        return x


class MultiNatten(nn.Module):
    expansion = 1
    def __init__(self,
                 dims,
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
                 stype='normal',
                 in_channels=64,
                 proj_drop=0,
                 pool_size=2,
                 stride=1):
        super().__init__()
        self.dims = dims
        self.stype = stype

        if len(dims) == 1:
            self.nums = 1
        else:
            self.nums = len(dims) - 1

        if stype == 'stage':
            self.pool = nn.AvgPool2d(kernel_size=3, stride=stride, padding=1)
            # self.pool = nn.Identity()

        convs = []
        for i in range(len(dims)-1):
            convs.append(
                nn.Sequential(
                NATLayer1(dim=dims[i], num_heads=num_heads, kernel_size=kernel_size,
                         dilation=dilation, mlp_ratio=mlp_ratio,
                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                         drop=drop, attn_drop=attn_drop, drop_path=drop_path,
                         act_layer=act_layer, norm_layer=norm_layer,
                         layer_scale=layer_scale),
                # nn.Linear(dims[i], sum(dims)),
                # nn.LayerNorm(sum(dims))
                )
            )

        convs.append(GlobalAttention(dim=dims[-1],
                                          num_heads=num_heads,
                                          qkv_bias=True,
                                          pool_size=pool_size,
                                          drop_path=0))

        self.convs = nn.ModuleList(convs)

        # self.convs2 = nn.ModuleList(convs2)
        self.downsample=None
        # self.width = width
        self.scale = len(dims)  # scale
        print(f"dims: {dims}")

        # self.conv_fuse = nn.Conv2d(sum(dims),
        #                            sum(dims), kernel_size=3, stride=1, padding=1,
        #                            bias=False, groups=sum(dims))
        #
        # self.layer_norm = nn.LayerNorm(sum(dims))
        # self.proj = nn.Conv2d(sum(dims), sum(dims), kernel_size=1, stride=1, padding=0)
        # self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        # short_cut = x
        x = torch.split(x, self.dims, -1)

        with autocast(dtype=torch.float32, device_type='cuda'):
            for i in range(len(self.dims)):
                # if i == 0 or self.stype == 'stage':
                #     sp = spx[i]
                # else:
                #     sp = sp + spx[i]

                sp = x[i]
                if torch.isnan(self.convs[i](sp)).any():
                    print(f"find nan")
                    pdb.set_trace()
                sp = self.convs[i](sp)
                # sp = self.convs2[i](sp)

                if i == 0:
                    out = sp
                else:
                    out = torch.cat((out, sp), -1)
        # out = out.permute(0, 3, 1, 2)
        # out = out + self.conv_fuse(out)

        # out = self.proj(out)
        # out = self.proj_drop(out)
        # out = out.permute(0, 2, 3, 1).contiguous()
        # out = self.layer_norm(out)
        return out


class NATBlock(nn.Module):
    def __init__(
        self,
        dim,
        split_dims,
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
        pool_size=2
    ):
        super().__init__()
        self.dim = dim
        self.depth = depth

        self.blocks = nn.ModuleList(
            [
                MultiNatten(
                    dims=split_dims[i],  # dim: 64, 128, 256, 512
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
                    stype='stage' if i == 0 else 'normal',
                    in_channels=dim,
                    pool_size=pool_size
                )
                for i in range(depth)
            ]
        )

        self.downsample = (
            None if not downsample else ConvDownsampler(dim=dim, norm_layer=norm_layer)
        )

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        if self.downsample is None:
            return x
        return self.downsample(x)


class GlobalAttention(nn.Module):
    def __init__(self, dim, num_heads, qkv_bias,  pool_size, drop_path,  drop=0, attn_drop=0,
                 norm_layer=nn.LayerNorm, mlp_ratio=3, act_layer=nn.GELU,
                 use_layer_scale=False, layer_scale_init_value=1e-5,):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.dim = dim

        self.norm1 = norm_layer(dim)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        #
        self.pool = nn.AvgPool2d(pool_size, stride=pool_size, padding=0,
                                 count_include_pad=False) if pool_size > 1 else nn.Identity()
        self.uppool = nn.Upsample(scale_factor=pool_size) if pool_size > 1 else nn.Identity()

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)

        mlp_hidden_dim = int(dim * mlp_ratio)
        # mlp_hidden_dim = int(dim * 0.25)

        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)

        # self.mlp = Mlp1(in_features=dim, hidden_features=dim,
        #                 act_layer=act_layer, drop=drop)

        self.use_layer_scale = use_layer_scale
        if self.use_layer_scale:
            # print('use layer scale init value {}'.format(layer_scale_init_value))
            self.layer_scale_1 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            self.layer_scale_2 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)

    def att_fun(self, q, k, v, B, N, C):
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        # x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = (attn @ v).transpose(2, 3).reshape(B, C, N)

        return x


    def attn(self, x):
        # B, C, H, W
        B, _, _, _ = x.shape

        xa = x.permute(0, 3, 1, 2)
        xa = self.pool(xa)
        xa = xa.permute(0, 2, 3, 1).view(B, -1, self.dim)

        B, N, C = xa.shape
        qkv = self.qkv(xa).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
        xa = self.att_fun(q, k, v, B, N, C)
        xa = xa.view(B, C, int(N ** 0.5), int(N ** 0.5))  # .permute(0, 3, 1, 2)

        xa = self.uppool(xa)
        xa = xa.permute(0, 2, 3, 1)
        return xa

    def forward(self, x):
        if self.use_layer_scale:
            x = x + self.drop_path(self.layer_scale_1 * self.attn(self.norm1(x)))
            x = x + self.drop_path(self.layer_scale_2 * self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class NAT(nn.Module):
    def __init__(
        self,
        embed_dim,
        split_dims,
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
        pool_sizes=[2, 2, 1, 1],

        **kwargs
    ):
        super().__init__()
        print("="*30)
        print('using new nat')
        print("=" * 30)
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
            print(f"level: {i}")
            level = NATBlock(
                dim=int(embed_dim * 2**i),
                split_dims=split_dims[i],
                depth=depths[i],
                num_heads=num_heads[i],
                kernel_size=kernel_size[i],
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
                pool_size=pool_sizes[i]
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
        kernel_size=[7, 7, 7, 7],
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
        split_dims=
        [
            [
                # [20] * 4

                [32] * 4
            ] * 3,

            [

                [64] * 4
            ] * 5,

            [[512]] * 3 +
            [
                # [20] * 16
                [128] * 4
            ] * 15
            ,

            [
                # [20] * 32
                [256] * 4
             ] * 5

        ],
        # num_heads=[2, 4, 8, 16],
        num_heads=[1, 2, 4, 8],
        # num_heads=[4, 8, 16, 32],
        embed_dim=128,
        mlp_ratio=2,
        # drop_path_rate=0.2,
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
        kernel_size=[3, 5, 9, 7],
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
        # depths=[3, 4, 18, 5],
        depths=[3, 4, 18, 4],
        split_dims=
        [
            [
                [32, 32, 32, 32]
            ] * 3,

            [
                [64, 64, 64, 64]
            ] * 4,

            [[128, 128, 128, 128]] * 0 +
            [[128, 128, 128, 128]] * 12 +
            [[512]] * 6

            ,

            [[256, 256, 256, 256]] * 4

        ],
        num_heads=[4, 8, 16, 32],
        embed_dim=128,
        mlp_ratio=2,
        drop_path_rate=0.5,
        layer_scale=1e-5,
        kernel_size=[7, 7, 7, 7],
        **kwargs
    )
    if pretrained:
        url = model_urls["nat_base_1k"]
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint)
    return model


if __name__ == '__main__':
    images = torch.rand(1, 3, 224, 224).cuda(0)
    model = nat_tiny(pretrained=False)
    model = model.cuda(0)
    from torchinfo import summary
    summary(model, (1, 3, 224, 224))
    # print(model(images).size())
