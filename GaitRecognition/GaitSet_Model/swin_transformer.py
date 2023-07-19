import torch
import torch.nn as nn

import torch.nn.functional as F

from einops import rearrange
from timm.models.layers import DropPath, to_2tuple
import numpy as np


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

        self.apply(self._init_mlp)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

    def _init_mlp(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.01)


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
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
    B = int(windows.shape[0] / (H * W / window_size[0] / window_size[1]))
    x = windows.view(B, H // window_size[0], W // window_size[1], window_size[0], window_size[1], -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class Windows_Attention(nn.Module):

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):
        '''
        :param dim: Number of input channels
        :param window_size: The height and width of the window.
        :param num_heads:  Number of attention heads.
        :param qkv_bias: If True, add a learnable bias to query, key, value. Default: True
        :param attn_drop: Dropout ratio of attention weight. Default: 0.0
        :param proj_drop: Dropout ratio of output. Default: 0.0
        '''

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads

        ##logit_scale就是公式（2）中的tao，可学习的缩放亮
        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))), requires_grad=True)

        # mlp to generate continuous relative position bias
        self.cpb_mlp = nn.Sequential(nn.Linear(2, 512, bias=True),
                                     nn.LeakyReLU(inplace=True),
                                     nn.Linear(512, num_heads, bias=False))

        # get relative_coords_table
        relative_coords_h = torch.arange(-(self.window_size[0] - 1), self.window_size[0], dtype=torch.float32)
        relative_coords_w = torch.arange(-(self.window_size[1] - 1), self.window_size[1], dtype=torch.float32)

        relative_coords_table = torch.stack(torch.meshgrid([relative_coords_h, relative_coords_w])).permute(1, 2,
                                                                                                            0).contiguous().unsqueeze(
            0)  # 1, 2*Wh-1, 2*Ww-1, 2

        relative_coords_table[:, :, :, 0] /= (self.window_size[0] - 1)
        relative_coords_table[:, :, :, 1] /= (self.window_size[1] - 1)

        relative_coords_table *= 7  # normalize to -7, 7
        relative_coords_table = torch.sign(relative_coords_table) * torch.log2(
            # torch.abs输出绝对值
            torch.abs(relative_coords_table) + 1.0) / np.log2(7)

        self.register_buffer("relative_coords_table", relative_coords_table)

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
        self.softmax = nn.Softmax(dim=-1)

        self.apply(self._init_innerfa)

    def forward(self, x, mask=None):
        """
        :param x: input features with shape of (num_windows*B, N, C)
         :param mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1,4).contiguous()  # (3,B_(num_windows*B),num_heads,N,C_^)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        # cosine attention
        attn = (F.normalize(q, dim=-1) @ (F.normalize(k, dim=-1).transpose(-2, -1).contiguous()))
        logit_scale = torch.clamp(self.logit_scale, max=torch.log(torch.tensor(1. / 0.01, device=x.device))).exp()
        attn = attn * logit_scale

        relative_position_bias_table = self.cpb_mlp(self.relative_coords_table).view(-1, self.num_heads)

        relative_position_bias = relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH

        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww

        relative_position_bias = 14 * torch.sigmoid(relative_position_bias)
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(-1, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).contiguous().reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, ' \
               f'pretrained_window_size={self.pretrained_window_size}, num_heads={self.num_heads}'

    def _init_innerfa(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.01)


def get_window_shift_size(x_size, window_size, shift_size=None):
    use_window_size = list(window_size)
    if shift_size is not None:
        use_shift_size = list(shift_size)
    for i in range(len(x_size)):
        if x_size[i] <= window_size[i]:
            use_window_size[i] = x_size[i]
            if shift_size is not None:
                use_shift_size[i] = 0

    if shift_size is None:
        return tuple(use_window_size)
    else:
        return tuple(use_window_size), tuple(use_shift_size)


def get_mask(H, W, window_size, shift_size, device):
    # cyclic shift
    if shift_size[0] > 0:
        img_mask = torch.zeros((1, H, W, 1), device=device)  # 1 H W 1
        h_slices = (slice(0, -window_size[0]),
                    slice(-window_size[0], -shift_size[0]),
                    slice(-shift_size[0], None))
        w_slices = (slice(0, -window_size[1]),
                    slice(-window_size[1], -shift_size[1]),
                    slice(-shift_size[1], None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, window_size)  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, window_size[0] * window_size[1])
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
    else:
        attn_mask = None

    return attn_mask


class SwinTransformerBlock(nn.Module):
    r"""
            Inner Attention->Inter Attention->LN->MLP->LN
    """

    def __init__(self, dim, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        '''
        :param dim: Number of input channels.
        :param num_heads: Number of attention heads.
        :param window_size: Window size.
        :param shift_size: Shift size for SW-MSA.
        :param mlp_ratio: Ratio of mlp hidden dim to embedding dim.
        :param qkv_bias: If True, add a learnable bias to query, key, value. Default: True
        :param drop: Dropout rate. Default: 0.0
        :param attn_drop: Attention dropout rate. Default: 0.0
        :param drop_path: Stochastic depth rate. Default: 0.0
        :param act_layer: Activation layer. Default: nn.GELU
        :param norm_layer: Normalization layer.  Default: nn.LayerNorm
        '''
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)

        ####窗口注意力计算
        self.attn = Windows_Attention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)

        mlp_hidden_dim = int(dim * mlp_ratio)

        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        """ Forward function.

               Args:
                   x: Input feature, tensor size (B_, H, W, C).  这里的B_等于2x batch size
                   cor: coordinate (B_,H,W,2)
        """
        B_, H, W, C = x.shape
        #############################  inner attn #############################
        # redefine window_size and shift_size
        shortcut = x.view(B_, -1, C)

        window_size, shift_size = get_window_shift_size((H, W), to_2tuple(self.window_size), to_2tuple(self.shift_size))

        pad_l = pad_t = 0
        pad_b = (window_size[0] - H % window_size[0]) % window_size[0]
        pad_r = (window_size[1] - W % window_size[1]) % window_size[1]

        # 从最后一个维度开始pad：0,0表示在第0维度（此处为C维度）不做任何补0
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))

        _, Hp, Wp, _ = x.shape

        attn_mask = get_mask(H=Hp, W=Wp, window_size=window_size, shift_size=shift_size, device=x.device)

        # cyclic shift
        if shift_size[0] > 0:
            shifted_x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1]), dims=(1, 2))
        else:
            shifted_x = x
        # partition windows

        x_windows = window_partition(shifted_x, window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, window_size[0] * window_size[1], C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, window_size[0], window_size[1], C)

        attn = window_reverse(attn_windows, window_size, Hp, Wp)  # B H' W' C

        # reverse cyclic shift
        if shift_size[0] > 0:
            attn = torch.roll(attn, shifts=shift_size, dims=(1, 2))
        else:
            attn = attn
        if pad_r > 0 or pad_b > 0:
            attn = attn[:, :H, :W, :].contiguous()
        attn=attn.view(B_,H*W,C)
        attn = shortcut + self.drop_path(self.norm1(attn))

        # FFN
        attn = attn + self.drop_path(self.norm2(self.mlp(attn)))
        attn = attn.view(B_, H, W, -1)

        return attn

    def extra_repr(self) -> str:
        return f"dim={self.dim}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"


class SwinTransfomer(nn.Module):
    """
        w-attn transformer ->sw-attn transformer->...
    """

    def __init__(self, dim=64, depth=2, num_heads=4, window_size=7,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, **kwargs):
        '''
        :param dim: Number of input channels.
        :param depth: Number of blocks.
        :param num_heads:  Number of attention heads.
        :param window_size:  Local window size.
        :param mlp_ratio: Ratio of mlp hidden dim to embedding dim.
        :param qkv_bias: If True, add a learnable bias to query, key, value. Default: True
        :param drop: Dropout rate. Default: 0.0
        :param attn_drop: Attention dropout rate. Default: 0.0
        :param drop_path: Stochastic depth rate. Default: 0.0
        :param norm_layer: Normalization layer. Default: nn.LayerNorm
        :param kwargs:
        '''
        super().__init__()
        self.dim = dim
        self.depth = depth

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, 0.1, depth)]

        # build blocks
        self.blocks = nn.ModuleList([
            ## transformer
            SwinTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop, attn_drop=attn_drop,
                drop_path=dpr[i] if isinstance(dpr, list) else drop_path,
                norm_layer=norm_layer)
            for i in range(depth)])

        self.apply(self._init_respostnorm)

    def forward(self, x):
        """
        :param x: Input feature, tensor size (B, C, H, W).
        """
        B, _, H, W = x.shape

        x = rearrange(x, 'b c h w -> b h w c').contiguous()
        for blk in self.blocks:
            x = blk(x)
        x = rearrange(x, 'b h w c -> b c h w').contiguous()
        return x

    def _init_respostnorm(self, m):
        for blk in self.blocks:
            nn.init.constant_(blk.norm1.bias, 0.01)
            nn.init.normal_(blk.norm1.weight, mean=1.0, std=0.02)
            nn.init.constant_(blk.norm2.bias, 0.01)
            nn.init.normal_(blk.norm2.weight, mean=1.0, std=0.02)
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.01)
