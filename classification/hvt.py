import pdb
import torch
from skimage import io
# from natten import NeighborhoodAttention2D as NeighborhoodAttention
from timm.models.vision_transformer import _cfg
from hvt_helpers import *


__all__ = [
    'hvt_tiny', 'hvt_small', 'hvt_medium', 'hvt_large'
]


class BaseAttention(nn.Module):
    def __init__(self, dim, num_heads=8, mlp_ratio=4, qkv_bias=True,
                 attn_drop=0., proj_drop=0., k=2, norm_layer=nn.LayerNorm,
                 drop_path=0, do_pool=True, min_w_size=7, act_layer=nn.GELU,
                 qk_norm=True, layer_scale=None, dilation=None,
                 drop=0.0, qk_scale=None, pool_stride=None,
                 kernel_size=9, stride=3, padding=0):

        super().__init__()
        self.k = k
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.do_pool = do_pool

        # self.norm1 = norm_layer(dim)

        # self.attn = NeighborhoodAttention(  # input should be (b, h, w, c)
        #     dim,
        #     kernel_size=kernel_size,
        #     dilation=dilation,
        #     num_heads=num_heads,
        #     qkv_bias=qkv_bias,
        #     qk_scale=qk_scale,
        #     attn_drop=attn_drop,
        #     proj_drop=drop,
        # )

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        # print(f"qk norm: {qk_norm}".center(50, "="))

        self.q_norm = norm_layer(head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(head_dim) if qk_norm else nn.Identity()

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.norm2 = norm_layer(dim)

        self.norm3 = norm_layer(dim)

        self.mlp = Mlp(in_features=dim, hidden_features=dim*mlp_ratio, out_features=dim,
                       act_layer=act_layer)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        if self.do_pool:
            # self.window_size = [self.k, self.k]
            self.window_size = [(kernel_size + 2 * padding) // stride + 1,
                                (kernel_size + 2 * padding) // stride + 1]
            # self.pool = nn.MaxPool2d(kernel_size=k,
            #                          stride=k)
        else:
            self.window_size = [min_w_size, min_w_size]

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

    def forward(self, x):
        shortcut = x  # b c h w
        B, C, H, W = x.shape
        if self.do_pool:
            # x = F.adaptive_avg_pool2d(x, output_size=(self.k, self.k))
            # x = F.adaptive_max_pool2d(x, output_size=(self.k, self.k))
            x = F.max_pool2d(x, kernel_size=9,
                             stride=3, padding=0)
            # x = self.pool(x)

        # x = rearrange(x, 'b c h w -> b h w c')
        # x = self.attn(x)  # (b h w c) -> (b h w c)
        # x = rearrange(x, 'b h w c -> b c h w')

        b, c, h, w = x.shape
        x = rearrange(x, 'b c h w -> b (h w) c')

        B, N, C = x.shape
        #
        # x = self.norm1(x)
        #
        qkv = self.qkv(x)
        qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        q, k = self.q_norm(q), self.k_norm(k)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        # # attn = torch.clip(attn, min=torch.finfo(torch.float16).min,
        # #                   max=torch.finfo(torch.float16).max)
        #
        # # if torch.isnan(attn).any():
        # #     print(f"attention nan... 1")
        # # get the relative positioning embedding
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        #
        # # if torch.isnan(attn).any():
        # #     print(f"attention nan... 21")
        #
        # # x = attn
        attn = attn + relative_position_bias.unsqueeze(0)
        # # if torch.isnan(attn).any():
        # #     print(f"attention nan... 21, x max: {x.max()}, "
        # #           f"rpa_max: {relative_position_bias.max()}")
        #
        # # x = attn
        # # attn = (attn-attn.max()).softmax(dim=-1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        # # if torch.isnan(attn).any():
        # #     print(f"x is nan: {torch.isnan(x).any()}")
        # #     torch.save(x, "/afs/crc.nd.edu/user/y/ypeng4/"
        # #                   "Neighborhood-Attention-Transformer_1/"
        # #                   "classification/x.pth")
        # #     print(f"attttttn nan:")
        # #     pdb.set_trace()
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)  # merge multi heads
        # # if torch.isnan(attn).any():
        # #     print(f"attention nan... 22, attn max: {attn.max()}, "
        # #           f"v_max: {v.max()}")
        #
        # # if torch.isnan(attn).any():
        # #     print(f"attention nan... 2")
        x = self.proj(x)
        # # if torch.isnan(attn).any():
        # #     print(f"attention nan... 3")
        x = self.proj_drop(x)
        # pdb.set_trace()
        if self.do_pool:
            x = rearrange(x, 'b (h w) c -> b c h w',  h=h, w=w)  # h=self.k, w=self.k)
            # has_nan_1 = torch.isnan(x).any()
            # max_value_1 = x.max()
            # max_value_2 = shortcut.max()

            x = F.interpolate(x, size=(H, W))
            x = torch.sigmoid(x) * shortcut + shortcut  # b c h w

            # has_nan_2 = torch.isnan(x).any()
            # if has_nan_2:
            #     print(f"nan_1: {has_nan_1}, nan_2: {has_nan_2}")
            #     print(f"max_1: {max_value_1}, max_2: {max_value_2}")
            # x = x + shortcut
        else:
            # a = x

            x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W) + shortcut

            # x = x + shortcut
            # if torch.isnan(x).any():
            #     print(f"a max: {a.max()}, shortcut_max: {shortcut.max()}")
            #     print(f"nan_3.......................")

        # x = rearrange(x, 'b c h w -> b (h w) c')
        x = rearrange(x, 'b c h w -> b h w c')
        # x = x.view(B, H * W, C)
        # print("origin max:", x.max())
        # print("max after layer norm", self.norm2(x).max())
        # if torch.isnan(x).any():
        #     print(f"has nan before mlp...")
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        x = self.norm3(x)

        # if torch.isnan(x).any():
        #     print(f"has nan after the mlp")

        # x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)

        x = rearrange(x, 'b h w c -> b c h w')
        return x


class MultiWindowAttention(nn.Module):
    def __init__(self, scale=2, dim=16, k=2, mlp_ratio=4, num_heads=4, do_pool=True,
                 act_layer=nn.GELU, drop_path=0, qk_norm=True, kernel_size=None):
        super().__init__()
        self.scale = scale

        self.base_attention = BaseAttention(dim=dim, k=k, num_heads=num_heads,
                                            mlp_ratio=mlp_ratio, do_pool=do_pool,
                                            act_layer=act_layer, drop_path=drop_path,
                                            qk_norm=qk_norm,
                                            kernel_size=kernel_size)

    def forward(self, x):
        # B, C, H, W = x.shape
        x = rearrange(x, f'b (scale_h h_window) (scale_w w_window) c-> '
                           f'(b scale_h scale_w) c h_window w_window',
                      scale_h=self.scale, scale_w=self.scale)

        x = self.base_attention(x)  # pooling, reshape
        # if torch.isnan(x).any():
        #     print("Nan at base attention")
        x = rearrange(x, f'(b scale_h scale_w) c h_window w_window -> b (scale_h h_window) (scale_w w_window) c',
                      scale_h=self.scale, scale_w=self.scale)

        return x

"""
1 layer B C H W -> B H W C -> B H/2 2 W/2 2 C -pooling-> B 2 2 C > attention
2 layer B C H W -> B H W C -> B H/2 2 W/2 2 C -> B*4 H/2 W/2 C -pooling-> attention
3 layer B C H W -> B H W C ->   

"""


class HitBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4, qkv_bias=False, qk_scale=None,
                 drop=0., attn_drop=0., depth=4, level=3, downsample=None,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, k=2,
                 qk_norm=True):
        super().__init__()

        self.blocks = nn.ModuleList([])
        for d in range(depth):
            block = nn.ModuleList(
                [MultiWindowAttention(
                    dim=dim, scale=1*k**i, k=k,
                    drop_path=drop_path[d] if isinstance(drop_path, list) else drop_path,
                    mlp_ratio=mlp_ratio, num_heads=num_heads, act_layer=act_layer,
                qk_norm=qk_norm, kernel_size=k)
                 for i in range(level)] + \
                [MultiWindowAttention(
                    dim=dim, scale=1*k**level, k=k,
                    drop_path=drop_path[d] if isinstance(drop_path, list) else drop_path,
                    mlp_ratio=mlp_ratio, num_heads=num_heads,
                    do_pool=False, act_layer=act_layer, qk_norm=qk_norm,
                kernel_size=7)])

            self.blocks.append(block)

        self.downsample = (
            None if not downsample else ConvDownsampler(dim=dim, norm_layer=norm_layer)
        )

    def forward(self, x):
        for i, block in enumerate(self.blocks):
            for j, blk in enumerate(block):
                x = blk(x)
                # if torch.isnan(x).any():
                #     print(f"Nan {i}, {j}, ================")
        if self.downsample is None:
            # if torch.isnan(x).any():
            #     print(f"get fuck nan.....1")
            return x
        return self.downsample(x)
        # y = self.downsample(x)
        # if torch.isnan(y).any():
        #     print(f"get fuck nan ........2")
        #     pdb.set_trace()
        # return y


class HierarchicalVisionTransformer(nn.Module):
    def __init__(self, in_chans=3, num_classes=1000,
                 embed_dims=[64, 128, 256, 512], embed_dim=96,
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False,
                 qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0.1, norm_layer=nn.LayerNorm,
                 depths=[1, 1, 2, 1], num_stages=4, k=None, level=[3, 2, 1, 0],
                 qk_norm=True, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = embed_dims[-1]

        self.depths = depths
        self.num_stages = num_stages

        self.patch_embedding = ConvTokenizer(in_chans=in_chans,
                                             embed_dim=embed_dims[0],
                                             norm_layer=norm_layer)
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        self.levels = nn.ModuleList([])
        for i_stage in range(num_stages):
            self.levels.append(
                HitBlock(dim=embed_dims[i_stage],  # int(embed_dim * 2 ** i_stage),  # embed_dims[i_stage],
                         num_heads=num_heads[i_stage],
                         mlp_ratio=mlp_ratios[i_stage],
                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                         drop=drop_rate, attn_drop=attn_drop_rate,
                         depth=depths[i_stage], level=level[i_stage],
                         downsample=(i_stage < self.num_stages - 1),
                         drop_path=dpr[sum(depths[:i_stage]):sum(depths[:i_stage + 1])],
                         act_layer=nn.GELU,
                         norm_layer=nn.LayerNorm, k=k[i_stage], qk_norm=qk_norm)
            )

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
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        # return {'pos_embed', 'cls_token'} # has pos_embed may be better
        return {'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def _get_pos_embed(self, pos_embed, patch_embed, H, W):
        if H * W == self.patch_embed1.num_patches:
            return pos_embed
        else:
            return F.interpolate(
                pos_embed.reshape(1, patch_embed.H, patch_embed.W, -1).permute(0, 3, 1, 2),
                size=(H, W), mode="bilinear").reshape(1, -1, H * W).permute(0, 2, 1)

    def forward_features(self, x):
        x = self.patch_embedding(x)
        x = self.pos_drop(x)

        for i, level in enumerate(self.levels):

            x = level(x)
            # if torch.isnan(x).any():
            #     print(i, "============= Nan")

        x = self.norm(x).flatten(1, 2)
        x = self.avgpool(x.transpose(1, 2))
        x = torch.flatten(x, 1)

        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)

        return x


@register_model
def hvt_tiny(pretrained=False, **kwargs):
    model = HierarchicalVisionTransformer(
        embed_dims=[64, 128, 256, 512], num_heads=[4, 8, 16, 16],
        mlp_ratios=[3, 3, 3, 3], qkv_bias=True,
        depths=[2, 3, 7, 3],
        # level=[3, 2, 1, 0], k=2,
        level=[3, 2, 1, 0], k=[2, 2, 2, 1],
        qk_norm=False,
        **kwargs)
    model.default_cfg = _cfg()

    return model


@register_model
def hvt_tiny_1(pretrained=False, **kwargs):
    model = HierarchicalVisionTransformer(
        embed_dims=[64, 128, 256, 512], num_heads=[4, 8, 16, 16],
        mlp_ratios=[4, 4, 4, 4], qkv_bias=True,
        depths=[2, 3, 9, 3],
        # level=[3, 2, 1, 0], k=2,
        level=[3, 2, 1, 0], k=[2, 2, 2, 1],
        qk_norm=False,
        **kwargs)
    model.default_cfg = _cfg()

    return model


@register_model
def hvt_small(pretrained=False, **kwargs):
    model = HierarchicalVisionTransformer(
        embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8],
        mlp_ratios=[8, 8, 4, 4], qkv_bias=True, qk_norm=False,
        depths=[3, 4, 6, 3], **kwargs)
    model.default_cfg = _cfg()

    return model


@register_model
def hvt_medium(pretrained=False, **kwargs):
    model = HierarchicalVisionTransformer(
        embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8],
        mlp_ratios=[8, 8, 4, 4], qkv_bias=True, qk_norm=True,
        depths=[3, 4, 18, 3],
        **kwargs)
    model.default_cfg = _cfg()

    return model


@register_model
def hvt_large(pretrained=False, **kwargs):
    model = HierarchicalVisionTransformer(
        embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8],
        mlp_ratios=[8, 8, 4, 4], qkv_bias=True, qk_norm=True,
        depths=[3, 8, 27, 3],
        **kwargs)
    model.default_cfg = _cfg()

    return model


@register_model
def hvt_huge_v2(pretrained=False, **kwargs):
    model = HierarchicalVisionTransformer(
        embed_dims=[128, 256, 512, 768], num_heads=[2, 4, 8, 12],
        mlp_ratios=[8, 8, 4, 4], qkv_bias=True, qk_norm=True,
        depths=[3, 10, 60, 3],
        # drop_rate=0.0, drop_path_rate=0.02)
        **kwargs)
    model.default_cfg = _cfg()

    return model


if __name__ == "__main__":
    file = "/afs/crc.nd.edu/user/y/ypeng4/data/raw_data/imagenet1k/" \
           "train/n02504013/n02504013_10021.JPEG"

    input_size = 224
    img = io.imread(file)
    from skimage.transform import resize
    img = resize(img, (224, 224))
    x = rearrange(img, 'h w c -> 1 c h w')
    x = torch.tensor(x).float()

    model = hvt_tiny()

    y = model(x)

    from fvcore.nn import FlopCountAnalysis, parameter_count_table, parameter_count

    flops = FlopCountAnalysis(model, torch.randn(1, 3, input_size, input_size)).total()
    print(f"flops: {flops / 10 ** 9:.3f} G")

    params = parameter_count(model)['']
    print(f"params: {params / 10 ** 6:.3f} M")

    print(parameter_count_table(model))
    # print(y.shape)

