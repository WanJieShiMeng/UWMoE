import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers

from einops import rearrange
from .MoE import UncertaintyMoETopK
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


##########################################################################
## Layer Norm

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


##########################################################################
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x


##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x


##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)

##########################################################################
class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=1, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        flops = 0
        H, W = self.img_size
        if self.norm is not None:
            flops += H * W * self.embed_dim
        return flops


class PatchUnEmbed(nn.Module):
    r""" Image to Patch Unembedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

    def forward(self, x, x_size):
        B, HW, C = x.shape
        x = x.transpose(1, 2).view(B, self.embed_dim, x_size[0], x_size[1])  # B Ph*Pw C
        return x

    def flops(self):
        flops = 0
        return flops


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

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

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)

        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops


class SwinAttn(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, num_heads=8, window_size=8, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        # self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        # self.norm2 = norm_layer(dim)
        # mlp_hidden_dim = int(dim * mlp_ratio)
        # self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            attn_mask = self.calculate_mask(self.input_resolution)
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

        self.patch_embed = PatchEmbed(img_size=input_resolution, patch_size=1, in_chans=0, embed_dim=dim,
                                      norm_layer=None)

        self.patch_unembed = PatchUnEmbed(img_size=input_resolution, patch_size=1, in_chans=0, embed_dim=dim,
                                          norm_layer=None)

    def calculate_mask(self, x_size):
        # calculate attention mask for SW-MSA
        H, W = x_size
        img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        return attn_mask

    def forward(self, x, x_size):
        x = self.patch_embed(x)
        H, W = x_size
        B, L, C = x.shape
        # assert L == H * W, "input feature has wrong size"

        shortcut = x
        # x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA (to be compatible for testing on images whose shapes are the multiple of window size
        if self.input_resolution == x_size:
            attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C
        else:
            attn_windows = self.attn(x_windows, mask=self.calculate_mask(x_size).to(x.device))

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN
        # x = shortcut + self.drop_path(x)
        x = self.drop_path(x)
        # x = x + self.drop_path(self.mlp(self.norm2(x)))

        return self.patch_unembed(x, x_size)


class S2TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type, layer_num, input_resolution,
                 window_size, drop_path):
        super().__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)

        self.attn_spe = Attention(dim, num_heads, bias)

        self.attn_spa = SwinAttn(dim, input_resolution, window_size=window_size,
                                 shift_size=0 if layer_num % 2 == 0 else window_size // 2, num_heads=num_heads,
                                 drop_path=drop_path)

        # self.fusion = nn.Conv2d(dim * 2, dim, kernel_size=1, bias=bias)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

        self.channel_interaction = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // 8, kernel_size=1),
            nn.BatchNorm2d(dim // 8),
            nn.GELU(),
            nn.Conv2d(dim // 8, dim, kernel_size=1),
        )
        self.spatial_interaction = nn.Sequential(
            nn.Conv2d(dim, dim // 16, kernel_size=1),
            nn.BatchNorm2d(dim // 16),
            nn.GELU(),
            nn.Conv2d(dim // 16, 1, kernel_size=1)
        )

        self.skip_scale = nn.Parameter(torch.ones(dim, 1, 1))
        self.skip_scale2 = nn.Parameter(torch.ones(dim, 1, 1))

    def forward(self, x):
        x_norm = self.norm1(x)
        spe_x = self.attn_spe(x_norm)
        spa_x = self.attn_spa(x_norm, x_size=(x_norm.shape[2], x_norm.shape[3]))

        # Interaction Module
        spa_x_new = spa_x * torch.sigmoid(self.channel_interaction(spe_x))
        spe_x_new = spe_x * torch.sigmoid(self.spatial_interaction(spa_x))
        fusion_x = self.proj(spa_x_new + spe_x_new)

        # fusion_x = self.fusion(torch.cat([spe_x, spa_x], dim=1))
        fusion_x = x * self.skip_scale + fusion_x
        x = fusion_x * self.skip_scale2 + self.ffn(self.norm2(fusion_x))

        return x


class UncertaintyCrossFusion(nn.Module):
    def __init__(self, dim, num_heads, bias=False):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.kv = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
        self.kv_dwconv = nn.Conv2d(dim * 2, dim * 2, kernel_size=7, stride=1, padding=7 // 2, groups=dim * 2, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, y):
        b, c, h, w = x.shape

        v = self.q_dwconv(self.q(x))
        qk = self.kv_dwconv(self.kv(y))
        q, k = qk.chunk(2, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


class S2TransformerBlockUncertaintyDecoder(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type, layer_num, input_resolution,
                 window_size, drop_path):
        super().__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)

        self.attn_spe = Attention(dim, num_heads, bias)

        self.attn_spa = SwinAttn(dim, input_resolution, window_size=window_size,
                                 shift_size=0 if layer_num % 2 == 0 else window_size // 2, num_heads=num_heads,
                                 drop_path=drop_path)

        # self.fusion = nn.Conv2d(dim * 2, dim, kernel_size=1, bias=bias)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

        self.channel_interaction = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // 8, kernel_size=1),
            nn.BatchNorm2d(dim // 8),
            nn.GELU(),
            nn.Conv2d(dim // 8, dim, kernel_size=1),
        )
        self.spatial_interaction = nn.Sequential(
            nn.Conv2d(dim, dim // 16, kernel_size=1),
            nn.BatchNorm2d(dim // 16),
            nn.GELU(),
            nn.Conv2d(dim // 16, 1, kernel_size=1)
        )

        self.skip_scale = nn.Parameter(torch.ones(dim, 1, 1))
        self.skip_scale2 = nn.Parameter(torch.ones(dim, 1, 1))

        self.uncertainty = UncertaintyCrossFusion(dim, num_heads, bias)

    def forward(self, fea):
        x, spa_var, spe_var = fea

        x = x + self.uncertainty(x, spa_var+spe_var)
        x_norm = self.norm1(x)
        spe_x = self.attn_spe(x_norm)
        spa_x = self.attn_spa(x_norm, x_size=(x_norm.shape[2], x_norm.shape[3]))

        # Interaction Module
        spa_x_new = spa_x * torch.sigmoid(self.channel_interaction(spe_x))
        spe_x_new = spe_x * torch.sigmoid(self.spatial_interaction(spa_x))
        fusion_x = self.proj(spa_x_new + spe_x_new)

        # fusion_x = self.fusion(torch.cat([spe_x, spa_x], dim=1))
        fusion_x = x * self.skip_scale + fusion_x
        x = fusion_x * self.skip_scale2 + self.ffn(self.norm2(fusion_x))

        return [x, spa_var, spe_var]


class S2UM(nn.Module):
    def __init__(self,
                 img_size=64,
                 inp_channels=3,
                 out_channels=3,
                 dim=48,
                 num_blocks=[2, 4, 6],
                 window_size=[8, 8, 8],
                 num_refinement_blocks=4,
                 heads=[2, 4, 8],
                 ffn_expansion_factor=2.66,
                 bias=False,
                 LayerNorm_type='WithBias',  ## Other option 'BiasFree'
                 num_experts=4,
                 topk=2):
        super().__init__()

        dpr = [x.item() for x in torch.linspace(0, 0.1, sum(num_blocks))]

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level1 = nn.Sequential(*[
            S2TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                               LayerNorm_type=LayerNorm_type, layer_num=i,
                               input_resolution=(img_size // 1, img_size // 1), window_size=window_size[0],
                               drop_path=dpr[sum(num_blocks[:0]):sum(num_blocks[:1])][i]) for i in
            range(num_blocks[0])])

        self.down1_2 = Downsample(dim)  ## From Level 1 to Level 2
        self.encoder_level2 = nn.Sequential(*[
            S2TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                               bias=bias,
                               LayerNorm_type=LayerNorm_type, layer_num=i,
                               input_resolution=(img_size // 2, img_size // 2), window_size=window_size[1],
                               drop_path=dpr[sum(num_blocks[:1]):sum(num_blocks[:2])][i]) for i in
            range(num_blocks[1])])

        self.down2_3 = Downsample(int(dim * 2 ** 1))  ## From Level 2 to Level 3
        self.latent_layer = nn.Sequential(*[
            S2TransformerBlock(dim=int(dim * 2 ** 2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor,
                               bias=bias,
                               LayerNorm_type=LayerNorm_type, layer_num=i,
                               input_resolution=(img_size // 4, img_size // 4), window_size=window_size[2],
                               drop_path=dpr[sum(num_blocks[:2]):sum(num_blocks[:3])][i]) for i in
            range(num_blocks[2])])

        self.up3_2 = Upsample(int(dim * 2 ** 2))  ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[
            S2TransformerBlockUncertaintyDecoder(dim=int(dim * 2 ** 1), num_heads=heads[1],
                                      ffn_expansion_factor=ffn_expansion_factor,
                                      bias=bias,
                                      LayerNorm_type=LayerNorm_type, layer_num=i,
                                      input_resolution=(img_size // 2, img_size // 2), window_size=window_size[1],
                                      drop_path=dpr[sum(num_blocks[:1]):sum(num_blocks[:2])][i])
            for i in
            range(num_blocks[1])])
        self.moe_2 = UncertaintyMoETopK(inp_channels, int(dim * 2 ** 1), int(dim * 2 ** 1), num_head=heads[1],num_experts=num_experts, topk=topk)

        self.up2_1 = Upsample(int(dim * 2 ** 1))  

        self.decoder_level1 = nn.Sequential(*[
            S2TransformerBlockUncertaintyDecoder(dim=int(dim * 2 ** 1), num_heads=heads[0],
                                      ffn_expansion_factor=ffn_expansion_factor,
                                      bias=bias,
                                      LayerNorm_type=LayerNorm_type, layer_num=i,
                                      input_resolution=(img_size // 1, img_size // 1), window_size=window_size[0],
                                      drop_path=dpr[sum(num_blocks[:0]):sum(num_blocks[:1])][i])
            for i in
            range(num_blocks[0])])
        self.moe_1 = UncertaintyMoETopK(inp_channels, int(dim * 2 ** 1), int(dim * 2 ** 1), num_head=heads[0],num_experts=num_experts, topk=topk)


        self.refinement = nn.Sequential(*[
            S2TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                               bias=bias,
                               LayerNorm_type=LayerNorm_type, layer_num=i,
                               input_resolution=(img_size // 1, img_size // 1), window_size=window_size[0],
                               drop_path=dpr[sum(num_blocks[:1]):sum(num_blocks[:2])][i]) for i in
            range(num_refinement_blocks)])

        # ------------------------- multi level ------------------------ #
        self.output_1 = nn.Conv2d(int(dim * 2 ** 1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

        self.output_2 = nn.Conv2d(int(dim * 2 ** 1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

        # ------------------------- Uncertainty Estimate ------------------------ #
        self.out_spa_var_conv_1 = nn.Sequential(
            nn.Conv2d(int(dim * 2 ** 1), int(dim * 2 ** 1), 3, 1, 1),
            nn.ELU(),
            nn.Conv2d(int(dim * 2 ** 1), int(dim * 2 ** 1), 3, 1, 1),
            nn.ELU(),
            nn.Conv2d(int(dim * 2 ** 1), 1, 3, 1, 1),
            nn.ELU(),  
        )

        self.out_spe_var_conv_1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(int(dim * 2 ** 1), int(dim * 2 ** 1), 1),
            nn.ELU(),
            nn.Conv2d(int(dim * 2 ** 1), int(dim * 2 ** 1), 1),
            nn.ELU(),
            nn.Conv2d(int(dim * 2 ** 1), inp_channels, 1),
            nn.ELU(),  
        )

        self.out_spa_var_conv_2 = nn.Sequential(
            nn.Conv2d(int(dim * 2 ** 1), int(dim * 2 ** 1), 3, 1, 1),
            nn.ELU(),
            nn.Conv2d(int(dim * 2 ** 1), int(dim * 2 ** 1), 3, 1, 1),
            nn.ELU(),
            nn.Conv2d(int(dim * 2 ** 1), 1, 3, 1, 1),
            nn.ELU(),  
        )

        self.out_spe_var_conv_2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(int(dim * 2 ** 1), int(dim * 2 ** 1), 1),
            nn.ELU(),
            nn.Conv2d(int(dim * 2 ** 1), int(dim * 2 ** 1), 1),
            nn.ELU(),
            nn.Conv2d(int(dim * 2 ** 1), inp_channels, 1),
            nn.ELU(),  
        )


    def forward(self, inp_img):
        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)

        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        latent = self.latent_layer(inp_enc_level3)

        inp_dec_level2 = self.up3_2(latent)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        #
        dec_spa_var_2 = self.out_spa_var_conv_2[:-2](inp_dec_level2)
        dec_spe_var_2 = self.out_spe_var_conv_2[:-2](inp_dec_level2)
        spa_var_2 = self.out_spa_var_conv_2(inp_dec_level2)
        spe_var_2 = self.out_spe_var_conv_2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2([inp_dec_level2, dec_spa_var_2, dec_spe_var_2])[0]
        out_dec_level2, loss_balance2 = self.moe_2(out_dec_level2, inp_img, spa_var_2, spe_var_2)

        output_2 = self.output_2(out_dec_level2) + F.interpolate(inp_img, scale_factor=0.5, mode='bilinear')

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        #
        dec_spa_var_1 = self.out_spa_var_conv_1[:-2](inp_dec_level1)
        dec_spe_var_1 = self.out_spe_var_conv_1[:-2](inp_dec_level1)
        spa_var_1 = self.out_spa_var_conv_1(inp_dec_level1)
        spe_var_1 = self.out_spe_var_conv_1(inp_dec_level1)
        out_dec_level1 = self.decoder_level1([inp_dec_level1, dec_spa_var_1, dec_spe_var_1])[0]
        out_dec_level1, loss_balance1 = self.moe_1(out_dec_level1, inp_img, spa_var_1, spe_var_1)
        out_dec_level1 = self.refinement(out_dec_level1)
        output_1 = self.output_1(out_dec_level1) + inp_img

        return output_1, spa_var_1, spe_var_1, output_2, spa_var_2, spe_var_2, loss_balance1 + loss_balance2




if __name__ == '__main__':
    from thop import profile, clever_format

    device = torch.device('cuda:0')
    x = torch.rand((1, 100, 64, 64)).to(device)
    net = S2UM(inp_channels=100, out_channels=100, dim=96).to(device)
    macs, params = profile(net, inputs=(x,))
    macs, params = clever_format([macs, params], "%.4f")
    print(macs, params)
