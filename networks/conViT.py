import torch.nn as nn
from timm.models.layers import to_2tuple
import torch.nn.functional as F

class PatchEmbed(nn.Module):
    def __init__(self, in_chans=3, patch_size=16, embed_dim=768):
        super(PatchEmbed, self).__init__()
        patch_size = to_2tuple(patch_size)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.num_patches = patch_size[0]*patch_size[1]

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class Attention(nn.Module):
    def __init__(self, in_chans, patch_size, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
                 proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        self.patch_embed = PatchEmbed(in_chans, patch_size, embed_dim=dim)
        self.qk = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1)
        self.proj_drop = nn.Dropout2d(proj_drop)
        self.patch_size = patch_size
        self.v = nn.Conv2d(in_chans, dim, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        v = self.v(x)
        B, C, H, W = v.shape
        x = self.patch_embed(x)
        N = x.shape[1]
        v = F.unfold(v, self.patch_size, stride=self.patch_size) #B x C*patch_size**2/N x N
        v = v.reshape(B, self.num_heads, self.patch_size**2*C//self.num_heads, N) # B x num_head x C*patch_size**2/N
        # x N
        v = v.permute(0, 1, 3, 2) # B X num_head x N  x C*patch_size**2/N
        qk = self.qk(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k = qk[0], qk[1]  # make torchscript happy (cannot use tensor as tuple)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).permute(0, 1, 3, 2) # B X num_head x C*patch_size**2/N x N
        x = x.reshape(B, self.patch_size**2*C, N)
        x = F.fold(x, (H, W), kernel_size=self.patch_size, stride=self.patch_size)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


if __name__ == '__main__':
    import torch

    attention = Attention(3, 32, 512)
    x = torch.rand((1, 3, 48, 48))
    print(attention(x).shape)
