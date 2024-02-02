import pdb

from einops import rearrange, repeat
import torch.nn.functional as F
import math
import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model
from natten import NeighborhoodAttention2D as NeighborhoodAttention


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
        x = self.proj(x).permute(0, 2, 3, 1).contiguous()
        if self.norm is not None:
            x = self.norm(x)
        return x


class ConvDownsampler(nn.Module):
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.reduction = nn.Conv2d(
            dim, 2 * dim,
            kernel_size=(3, 3),
            stride=(2, 2),
            padding=(1, 1),
            bias=False
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


class BN_Conv2d(nn.Module):
    """
    BN_CONV_RELU
    """

    def __init__(self, in_channels, out_channels,
                 kernel_size, stride, padding,
                 dilation=1, groups=1, bias=False):
        super(BN_Conv2d, self).__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation,
                      groups=groups, bias=bias),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        return F.gelu(self.seq(x))


class MultiNatten(nn.Module):

    def __init__(self,
                 dims,
                 num_heads,
                 cardinality,
                 group_depth,
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
                 stride=1):
        super().__init__()
        self.dims = dims
        self.stype = stype

        if len(dims) == 1:
            self.nums = 1
        else:
            self.nums = len(dims) - 1

        # if stype == 'stage':
        #     self.pool = nn.AvgPool2d(kernel_size=3, stride=stride, padding=1)
            # self.pool = nn.Identity()

        self.group_chnls = cardinality * group_depth  # 64 * 4

        self.dims = [group_depth] * cardinality

        if stride == 1:
            if in_channels != self.group_chnls:
                self.conv1 = BN_Conv2d(in_channels, self.group_chnls, 1, stride=1, padding=0)
            else:
                # self.conv1 = nn.Identity()
                self.conv1 = nn.BatchNorm2d(in_channels)
        else:
            if in_channels != self.group_chnls:
                self.conv1 = BN_Conv2d(in_channels, self.group_chnls, 3, stride=stride, padding=1)
            else:
                # self.conv1 = nn.Identity()
                self.conv1 = nn.BatchNorm2d(in_channels)

        # self.conv1 = BN_Conv2d(in_channels, self.group_chnls, 1, stride=1, padding=0)
        # self.conv2 = BN_Conv2d(self.group_chnls, self.group_chnls, 3,
        #                        stride=stride, padding=1, groups=cardinality)
        # self.conv3 = nn.Conv2d(self.group_chnls, self.group_chnls * 2, 1, stride=1, padding=0)
        # self.bn = nn.BatchNorm2d(self.group_chnls * 2)

        # self.conv3 = nn.Conv2d(self.group_chnls, self.group_chnls, 1, stride=1, padding=0)
        self.bn = nn.LayerNorm(self.group_chnls)
        # self.short_cut = nn.Sequential(
        #     nn.Conv2d(in_channels, self.group_chnls * 2, 1, stride, 0, bias=False),
        #     nn.BatchNorm2d(self.group_chnls * 2)
        # )

        convs = []
        print(num_heads)
        for i in range(len(self.dims)):
            convs.append(
                NATLayer(dim=self.dims[i], num_heads=num_heads, kernel_size=kernel_size,
                         dilation=dilation, mlp_ratio=mlp_ratio,
                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                         drop=drop, attn_drop=attn_drop, drop_path=drop_path,
                         act_layer=act_layer, norm_layer=norm_layer,
                         layer_scale=layer_scale)
            )

        self.convs = nn.ModuleList(convs)
        self.downsample=None
        # self.width = width
        self.scale = len(dims)  # scale

        self.fcs = [nn.Sequential(nn.Linear(sum(self.dims[i]), self.dims[i],
                                            ), nn.Sigmoid()) for i in range(len(self.dims))]
        # print(f"fuck dims: {dims}")


        # self.conv_fuse = nn.Conv2d(sum(dims), sum(dims), kernel_size=3, stride=1, padding=1,
        #                            bias=False)
        # self.proj = nn.Conv2d(sum(dims), sum(dims), kernel_size=1, stride=1, padding=0)
        # self.proj_drop = nn.Dropout(proj_drop)

        # self.conv_fuse = nn.Conv2d(
        #     sum(dims), sum(dims),
        #     kernel_size=3, stride=1, padding=1,
        #     bias=False
        # )
        # self.norm = nn.LayerNorm(sum(dims))

    def forward(self, x):
        # pdb.set_trace()
        out = rearrange(self.conv1(rearrange(x, 'b h w c -> b c h w')).contiguous(),
                        'b c h w -> b h w c').contiguous()  # reduce channel
        # print(f"out1 is nan: {torch.isnan(out).any()}")
        if torch.isnan(out).any():
            print(f"out1 is nan: {torch.isnan(out).any()}")
        outs = torch.split(out, self.dims, -1)

        for i in range(len(self.dims)):
            sp = outs[i]
            sp = self.convs[i](sp)
            if i == 0:
                outs = [sp]
            else:
                outs = out.append(sp)
                # out = torch.cat((out, sp), -1)
        out = torch.cat(outs, -1)
        out = out.mean(dim=(1, 2))

        fcs = [self.fcs[i](out) for i in range(len(outs))]
        outs = [repeat(fcs[i], 'b c -> b 1 1 c') * outs[i] for i in range(len(outs))]
        out = torch.cat(outs, -1)

        if torch.isnan(out).any():
            print(f"out2 is nan: {torch.isnan(out).any()}")
        # out = rearrange(self.bn(rearrange(out, 'b h w c -> b c h w').contiguous()),
        #                 'b c h w -> b h w c').contiguous()

        out = self.bn(out)
        if torch.isnan(out).any():
            print(f"out3 is nan: {torch.isnan(out).any()}")
        # out = rearrange(self.bn(self.conv3(rearrange(out, 'b h w c -> b c h w').contiguous())),
        #                 'b c h w -> b h w c').contiguous()

        # out += rearrange(self.short_cut(rearrange(x, 'b h w c -> b c h w')).contiguous(),
        #                  'b c h w -> b h w c').contiguous()

        # out = out + self.conv_fuse(out)
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
        group_depth=64,
        in_channels=128,
        stride=1,
    ):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.cardinality = 4
        self.in_channels = in_channels

        blocks = []
        strides = [stride] + [1] * (depth - 1)

        for i in range(depth):
            blocks.append(
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
                    in_channels=self.in_channels,
                    cardinality=self.cardinality,
                    group_depth=group_depth,
                    stride=strides[i]
                )
            )
            self.in_channels = self.cardinality * group_depth  #  * 2

        # self.blocks = nn.ModuleList(
        #     [
        #         MultiNatten(
        #             dims=split_dims[i],  # dim: 64, 128, 256, 512
        #             num_heads=num_heads,
        #             kernel_size=kernel_size,
        #             dilation=None if dilations is None else dilations[i],
        #             mlp_ratio=mlp_ratio,
        #             qkv_bias=qkv_bias,
        #             qk_scale=qk_scale,
        #             drop=drop,
        #             attn_drop=attn_drop,
        #             drop_path=drop_path[i]
        #             if isinstance(drop_path, list)
        #             else drop_path,
        #             norm_layer=norm_layer,
        #             layer_scale=layer_scale,
        #             stype='stage' if i == 0 else 'normal',
        #             in_channels=dim,
        #             cardinality=16,
        #             group_depth=group_depth,
        #         )
        #         for i in range(depth)
        #     ]
        # )

        self.blocks = nn.ModuleList(blocks)
        self.downsample = (
            None if not downsample else ConvDownsampler(dim=dim, norm_layer=norm_layer)
        )

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        if self.downsample is None:
            return x
        return self.downsample(x)


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
        **kwargs
    ):
        super().__init__()

        self.num_classes = num_classes
        self.num_levels = len(depths)
        self.embed_dim = embed_dim
        # self.num_features = int(embed_dim * 2 ** (self.num_levels - 1))
        self.num_features = 1024 * 2
        self.mlp_ratio = mlp_ratio
        # in_channels = [96, 192, 384, 768]
        in_channels = [128, 256, 512, 1024]
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
                downsample=False,  # (i < self.num_levels - 1),
                layer_scale=layer_scale,
                group_depth=64*2**i,
                in_channels=in_channels[i],  # embed_dim * 2 ** i,
                stride=1 if i == 0 else 2
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
            # print(f"level : {level}................")
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
        depths=[3, 5, 15, 5],
        num_heads=[2, 4, 8, 16],
        embed_dim=64,
        mlp_ratio=3,
        drop_path_rate=0.2,
        kernel_size=[3, 5, 9, 7],
        **kwargs
    )
    return model


@register_model
def nat_tiny(pretrained=False, **kwargs):
    model = NAT(
        depths=[3, 4, 9, 3],
        split_dims=
        [
            [
                [32, 64]
            ] * 3,

            [
                [64] * 3
            ] * 4,

            [[384]] * 0 +
            [
                [128] * 3
            ] * 2 +
            [[384]] * 7
            ,
            [
                [256] * 4
             ] * 4

        ],
        # num_heads=[2, 4, 8, 16],
        num_heads=[1, 2, 4, 8],
        # num_heads=[4, 8, 16, 32],
        embed_dim=96,
        mlp_ratio=3,
        drop_path_rate=0.2,
        kernel_size=7,
        **kwargs
    )
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
    return model


@register_model
def nat_base(pretrained=False, **kwargs):
    model = NAT(
        depths=[3, 4, 12, 4],
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
            [[512]] * 0

            ,

            [[256, 256, 256, 256]] * 4

        ],
        num_heads=[4, 8, 16, 32],
        embed_dim=128,
        mlp_ratio=2,
        drop_path_rate=0.5,
        layer_scale=1e-5,
        # kernel_size=[3, 5, 9, 7],
        kernel_size=[3, 5, 9, 7],
        **kwargs
    )
    return model


if __name__ == '__main__':
    images = torch.rand(1, 3, 224, 224).cuda(0)
    model = nat_tiny(pretrained=False)
    model = model.cuda(0)

    from torchinfo import summary
    summary(model, (1, 3, 224, 224))

    criterion = torch.nn.MSELoss(reduction='sum')
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)

    y_pred = model(images)

    y = y_pred

    loss = criterion(y_pred, y)

    # Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad()
    loss.backward()

    for name, param in model.named_parameters():
        if param.grad is None:
            print(name)

    optimizer.step()
