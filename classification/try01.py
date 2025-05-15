

import math
import torch
import torch.nn.functional as F
import torch.nn as nn
from timm.models.layers import DropPath, trunc_normal_
from timm.models.vision_transformer import VisionTransformer
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
# from fvcore.nn import FlopCountAnalysis, flop_count_table
import time
from typing import Tuple, Union




class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_tensors[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))



class MemoryEfficientSwish(nn.Module):
    def forward(self, x):
        return SwishImplementation.apply(x)



def rotate_every_two(x):
    x1 = x[:, :, :, :, ::2]  # 第奇数个
    x2 = x[:, :, :, :, 1::2]  # 第偶数个
    x = torch.stack([-x2, x1], dim=-1)
    return x.flatten(-2)



def theta_shift(x, sin, cos):
    return (x * cos) + (rotate_every_two(x) * sin)



class DWConv2d(nn.Module):

    def __init__(self, dim, kernel_size, stride, padding):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, kernel_size, stride, padding, groups=dim)

    def forward(self, x: torch.Tensor):
        '''
        x: (b h w c)
        '''
        x = x.permute(0, 3, 1, 2)  # (b c h w)
        x = self.conv(x)  # (b c h w)
        x = x.permute(0, 2, 3, 1)  # (b h w c)
        return x



class RotaryPositionEmbedding2D(nn.Module):
    # RotaryPositionEmbedding
    def __init__(self, embed_dim, num_heads, initial_value, heads_range):
        '''
        recurrent_chunk_size: (clh clw)
        num_chunks: (nch ncw)
        clh * clw == cl
        nch * ncw == nc

        default: clh==clw, clh != clw is not implemented
        '''
        super().__init__()
        angle = 1.0 / (10000 ** torch.linspace(0, 1, embed_dim // num_heads // 2))
        angle = angle.unsqueeze(-1).repeat(1, 2).flatten()
        self.initial_value = initial_value
        self.heads_range = heads_range
        self.num_heads = num_heads
        self.register_buffer('angle', angle)


    def forward(self, slen: Tuple[int], activate_recurrent=False):
        '''
        slen: (h, w)
        h * w == l
        recurrent is not implemented
        '''
        if activate_recurrent:
            sin = torch.sin(self.angle * (slen[0] * slen[1] - 1))
            cos = torch.cos(self.angle * (slen[0] * slen[1] - 1))
            rel_pos = ((sin, cos), self.decay.exp())

        else:
            index = torch.arange(slen[0] * slen[1]).to(self.angle)
            sin = torch.sin(index[:, None] * self.angle[None, :])  # (l d1)
            sin = sin.reshape(slen[0], slen[1], -1)  # (h w d1)
            cos = torch.cos(index[:, None] * self.angle[None, :])  # (l d1)
            cos = cos.reshape(slen[0], slen[1], -1)  # (h w d1)

            rel_pos = (sin, cos)
        return rel_pos



class PathEmbeding2D(nn.Module):
    # PathEmbeding2D
    def __init__(self, embed_dim, nheads, A_init_range=(1, 1.1), dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                 bias=False, chunkwise_recurrent=False):
        """
        """
        super().__init__()
        self.nheads = nheads
        self.chunkwise_recurrent = chunkwise_recurrent

        # self.dt_linear = nn.Linear(embed_dim, 2*nheads, bias=bias)
        self.dt_linear = nn.Linear(int(embed_dim//nheads), 2, bias=bias)

        # Initialize log dt bias
        dt = torch.exp(
            torch.rand(self.nheads) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        )       # dt_max=0.1, dt_min=0.001
        dt = torch.clamp(dt, min=dt_init_floor)     # dt_init_floor=1e-4, 0.001<=dt<=0.1
        inv_dt = dt + torch.log(-torch.expm1(-dt))      # ???
        self.dt_bias = nn.Parameter(inv_dt)             # torch.Size([2]), -6.9073<=dt_bias<=-2.2522
        self.dt_bias._no_weight_decay = True            # 梯度更新时不进行权重衰减

        # A parameter
        assert A_init_range[0] > 0 and A_init_range[1] >= A_init_range[0]
        A = torch.empty(self.nheads, dtype=torch.float32).uniform_(*A_init_range)        # A_init_range=(1, 16)
        A_log = torch.log(A)
        self.A_log = nn.Parameter(A_log)            # torch.Size([2])
        self.A_log._no_weight_decay = True


    def generate_structed_mask_1d(self, x: torch.Tensor):
        chunk_size = x.size(-1)
        x_cumsum = torch.cumsum(x, dim=-1)
        x_segsum_tril = x_cumsum[..., :, None] - x_cumsum[..., None, :]

        mask = torch.tril(torch.ones(chunk_size, chunk_size, device=x.device, dtype=bool), diagonal=0)
        x_segsum_tril = x_segsum_tril.masked_fill(~mask, 0.0)

        x_segsum_triu = x_cumsum[..., None, :] - x_cumsum[..., :, None]
        mask = torch.triu(torch.ones(chunk_size, chunk_size, device=x.device, dtype=bool), diagonal=1)
        x_segsum_triu = x_segsum_triu.masked_fill(~mask, 0.0)

        x_segsum = x_segsum_tril + x_segsum_triu
        return x_segsum


    def forward(self, x: torch.Tensor):
        """
        x: (b h w c)
        """
        batch, height, width, dim = x.shape        # torch.Size([12, 3136, 64])
        seqlen = height * width
        x = x.view(batch, seqlen, dim)
        headdim = int(dim // self.nheads)
        x = x.view(batch, seqlen, self.nheads, headdim)

        dt = self.dt_linear(x)                    # (B, L, 2), torch.Size([12, 3136, 4, 2])
        dt_alpha, dt_beta = dt[:, :, :, 0], dt[:, :, :, 1]

        A = -torch.exp(self.A_log)              # (nheads), torch.Size([2])
        dt_alpha = F.softplus(dt_alpha + self.dt_bias)      # (B, L, 2*nheads), torch.Size([12, 3136, 2*4])
        dt_alpha = dt_alpha*A                              # (B, L, 2*nheads), torch.Size([12, 3136, 2*4])
        dt_beta = F.softplus(dt_beta + self.dt_bias)      # (B, L, nheads), torch.Size([12, 3136, 4])
        dt_beta = dt_beta*A                              # (B, L, nheads), torch.Size([12, 3136, 4])

        dt_alpha = dt_alpha.view(batch, height, width, self.nheads).contiguous()
        dt_beta = dt_beta.view(batch, height, width, self.nheads).contiguous()
        dt_beta = dt_beta.permute(0, 2, 1, 3).contiguous()          # # (B, width, height, nheads, width)

        dt_alpha = dt_alpha.permute(0, 1, 3, 2).contiguous()        # (B, height, nheads, width)
        structed_mask_w = self.generate_structed_mask_1d(dt_alpha)  # (B, height, nheads, width, width)

        dt_beta = dt_beta.permute(0, 1, 3, 2).contiguous()          # (B, width, nheads, height)
        structed_mask_h = self.generate_structed_mask_1d(dt_beta)   # (B, width, nheads, height, height)

        if self.chunkwise_recurrent:
            return structed_mask_w, structed_mask_h

        else:
            structed_mask_w = structed_mask_w.permute(0, 2, 1, 3, 4).contiguous()   # (B, nheads, height, width, width)
            structed_mask_h = structed_mask_h.permute(0, 2, 3, 4, 1).contiguous()   # (B, nheads, height, height, width)

            structed_mask_w = structed_mask_w.unsqueeze(3).repeat(1, 1, 1, height, 1, 1)   # (B, nheads, height, height, width, width)
            structed_mask_h = structed_mask_h.unsqueeze(-2).repeat(1, 1, 1, 1, width, 1)    # (B, nheads, height, height, width, width)
            structed_mask = structed_mask_w + structed_mask_h       # (B, nheads, height, height, width, width)
            structed_mask = structed_mask.permute(0, 1, 2, 4, 3, 5).contiguous()
            structed_mask = structed_mask.view(batch, self.nheads, seqlen, seqlen)
            return structed_mask



class MambaAttentionChunkDouble(nn.Module):
    """
    from MambaAttentionAll
    make structed_mask to be Symmetric Matrix:
        structed_mask = structed_mask + structed_mask.T - structed_mask * (structed_mask.T)
    """
    def __init__(self, embed_dim, num_heads, value_factor=1):
        super().__init__()
        self.factor = value_factor
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = self.embed_dim * self.factor // num_heads
        self.key_dim = self.embed_dim // num_heads
        self.scaling = self.key_dim ** -0.5
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.v_proj = nn.Linear(embed_dim, embed_dim * self.factor, bias=True)
        self.lepe = DWConv2d(embed_dim, 5, 1, 2)
        self.out_proj = nn.Linear(embed_dim * self.factor, embed_dim, bias=True)

        self.path_embed = PathEmbeding2D(embed_dim, num_heads, chunkwise_recurrent=True)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.q_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_normal_(self.k_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_normal_(self.v_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_normal_(self.out_proj.weight)
        nn.init.constant_(self.out_proj.bias, 0.0)


    def forward(self, x: torch.Tensor, rel_pos):
        '''
        x: (b h w c)
        mask_h: (n h h)
        mask_w: (n w w)
        '''
        bsz, h, w, _ = x.size()  # torch.Size([10, 56, 56, 64])

        (sin, cos) = rel_pos
        # sin: torch.Size([56, 56, 16])
        # cos: torch.Size([56, 56, 16])
        # mask_h: torch.Size([4, 56, 56])
        # mask_w: torch.Size([4, 56, 56])

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)  # torch.Size([10, 56, 56, 64])
        lepe = self.lepe(v)  # torch.Size([10, 56, 56, 64])

        k *= self.scaling
        q = q.view(bsz, h, w, self.num_heads, self.key_dim).permute(0, 3, 1, 2, 4)  # (b g h w d1), torch.Size([10, 4, 56, 56, 16])
        k = k.view(bsz, h, w, self.num_heads, self.key_dim).permute(0, 3, 1, 2, 4)  # (b g h w d1), torch.Size([10, 4, 56, 56, 16])
        v = v.view(bsz, h, w, self.num_heads, self.key_dim).permute(0, 3, 1, 2, 4)  # (b g h w d1), torch.Size([10, 4, 56, 56, 16])

        qr = theta_shift(q, sin, cos)  # torch.Size([10, 4, 56, 56, 16])
        kr = theta_shift(k, sin, cos)  # torch.Size([10, 4, 56, 56, 16])

        '''
        qr: (b n h w d1)
        kr: (b n h w d1)
        v: (b h w n*d2)
        '''

        structed_mask_w, structed_mask_h  = self.path_embed(x)
        # structed_mask_w: (B, height, nheads, width, width)
        # structed_mask_h: (B, width, nheads, height, height)

        qr_w = qr.transpose(1, 2)  # (b h g w d1), torch.Size([10, 56, 4, 56, 16])
        kr_w = kr.transpose(1, 2)  # (b h g d1), torch.Size([10, 56, 4, 56, 16])
        v_w = v.transpose(1, 2)             # (b h g w d2), torch.Size([10, 56, 4, 56, 16])
        qr_h = qr.permute(0, 3, 1, 2, 4)    # (b w g h d1), torch.Size([10, 56, 4, 56, 16])
        kr_h = kr.permute(0, 3, 1, 2, 4)    # (b w g h d1), torch.Size([10, 56, 4, 56, 16])
        v_h = v.permute(0, 3, 1, 2, 4)      # (b w g h d2), torch.Size([10, 56, 4, 56, 16])

        qk_mat_w = qr_w @ kr_w.transpose(-1, -2)    # (b h g w w), , torch.Size([10, 56, 4, 56, 56])
        qk_mat_w = qk_mat_w + structed_mask_w       # (b h g w w), torch.Size([10, 56, 4, 56, 56])
        qk_mat_w = torch.softmax(qk_mat_w, -1)      # (b h g w w)       *(b h g w d2)

        qk_mat_h = qr_h @ kr_h.transpose(-1, -2)    # (b w g h h), torch.Size([10, 56, 4, 56, 56])
        qk_mat_h = qk_mat_h + structed_mask_h       # (b w g h h), torch.Size([10, 56, 4, 56, 56])
        qk_mat_h = torch.softmax(qk_mat_h, -1)      # (b w g h h)       *(b w g h d2)

        v_w = torch.matmul(qk_mat_w, v_w)       # (b h g w d2), torch.Size([10, 56, 4, 56, 16])
        v_w = v_w.permute(0, 3, 2, 1, 4)        # (b w g h d2), torch.Size([10, 56, 4, 56, 16])
        output = torch.matmul(qk_mat_h, v_w)    # (b w g h d2), torch.Size([10, 56, 4, 56, 16])
        output = output.permute(0, 3, 1, 2, 4).flatten(-2, -1)  # (b h w g*d2)

        v_h = torch.matmul(qk_mat_h, v_h)       # (b w g h d2), torch.Size([10, 56, 4, 56, 16])
        v_h = v_h.permute(0, 3, 2, 1, 4)        # (b h g w d2), torch.Size([10, 56, 4, 56, 16])
        v_h = torch.matmul(qk_mat_w, v_h)       # (b h g w d2), torch.Size([10, 56, 4, 56, 16])
        v_h = v_h.permute(0, 1, 3, 2, 4).flatten(-2, -1)  # (b h w n*d2)

        output = 0.5*output + 0.5*v_h

        output = output + lepe
        output = self.out_proj(output)
        return output



class MambaAttentionAllDouble(nn.Module):
    """
    from MambaAttentionAll
    make structed_mask to be Symmetric Matrix:
        structed_mask = structed_mask + structed_mask.T - structed_mask * (structed_mask.T)
    """
    def __init__(self, embed_dim, num_heads, value_factor=1):
        super().__init__()
        self.factor = value_factor
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = self.embed_dim * self.factor // num_heads
        self.key_dim = self.embed_dim // num_heads
        self.scaling = self.key_dim ** -0.5
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.v_proj = nn.Linear(embed_dim, embed_dim * self.factor, bias=True)
        self.lepe = DWConv2d(embed_dim, 5, 1, 2)
        self.out_proj = nn.Linear(embed_dim * self.factor, embed_dim, bias=True)

        self.path_embed = PathEmbeding2D(embed_dim, num_heads, chunkwise_recurrent=False)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.q_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_normal_(self.k_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_normal_(self.v_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_normal_(self.out_proj.weight)
        nn.init.constant_(self.out_proj.bias, 0.0)

    def forward(self, x: torch.Tensor, rel_pos):
        '''
        x: (b h w c)
        rel_pos: mask: (n l l)
        '''
        bsz, h, w, _ = x.size()
        (sin, cos) = rel_pos

        structed_mask = self.path_embed(x)
        # sin: torch.Size([7, 7, 32])
        # cos: torch.Size([7, 7, 32])
        # structed_mask: torch.Size([10, 16, 49, 49])

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        lepe = self.lepe(v)

        k *= self.scaling
        q = q.view(bsz, h, w, self.num_heads, -1).permute(0, 3, 1, 2, 4)  # (b n h w d1)
        k = k.view(bsz, h, w, self.num_heads, -1).permute(0, 3, 1, 2, 4)  # (b n h w d1)
        qr = theta_shift(q, sin, cos)  # (b n h w d1)
        kr = theta_shift(k, sin, cos)  # (b n h w d1)

        qr = qr.flatten(2, 3)  # (b n l d1)
        kr = kr.flatten(2, 3)  # (b n l d1)
        vr = v.reshape(bsz, h, w, self.num_heads, -1).permute(0, 3, 1, 2, 4)  # (b n h w d2)
        vr = vr.flatten(2, 3)  # (b n l d2)
        qk_mat = qr @ kr.transpose(-1, -2)  # (b n l l), torch.Size([10, 16, 49, 49])

        qk_mat = (0.5 * torch.softmax(qk_mat + structed_mask, -1) +
                  0.5 * torch.softmax(qk_mat + structed_mask.permute(0, 1, 3, 2), -1))      # (b n l l)

        output = torch.matmul(qk_mat, vr)  # (b n l d2)
        output = output.transpose(1, 2).reshape(bsz, h, w, -1)  # (b h w n*d2)
        output = output + lepe
        output = self.out_proj(output)
        return output



class FeedForwardNetwork(nn.Module):
    def __init__(
            self,
            embed_dim,
            ffn_dim,
            activation_fn=F.gelu,
            dropout=0.0,
            activation_dropout=0.0,
            layernorm_eps=1e-6,
            subln=False,
            subconv=True
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.activation_fn = activation_fn
        self.activation_dropout_module = torch.nn.Dropout(activation_dropout)
        self.dropout_module = torch.nn.Dropout(dropout)
        self.fc1 = nn.Linear(self.embed_dim, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, self.embed_dim)
        self.ffn_layernorm = nn.LayerNorm(ffn_dim, eps=layernorm_eps) if subln else None
        self.dwconv = DWConv2d(ffn_dim, 3, 1, 1) if subconv else None

    def reset_parameters(self):
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()
        if self.ffn_layernorm is not None:
            self.ffn_layernorm.reset_parameters()

    def forward(self, x: torch.Tensor):
        '''
        x: (b h w c)
        '''
        x = self.fc1(x)
        x = self.activation_fn(x)
        x = self.activation_dropout_module(x)
        residual = x
        if self.dwconv is not None:
            x = self.dwconv(x)
        if self.ffn_layernorm is not None:
            x = self.ffn_layernorm(x)
        x = x + residual
        x = self.fc2(x)
        x = self.dropout_module(x)
        return x



class BasicBlock(nn.Module):
    def __init__(self, retention: str, embed_dim: int, num_heads: int, ffn_dim: int, drop_path=0., layerscale=False,
                 layer_init_values=1e-5):
        super().__init__()
        self.layerscale = layerscale
        self.embed_dim = embed_dim
        self.retention_layer_norm = nn.LayerNorm(self.embed_dim, eps=1e-6)
        assert retention in ['chunk', 'whole']
        if retention == 'chunk':
            self.retention = MambaAttentionChunkDouble(embed_dim, num_heads)
        else:
            self.retention = MambaAttentionAllDouble(embed_dim, num_heads)
        self.drop_path = DropPath(drop_path)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim, eps=1e-6)
        self.ffn = FeedForwardNetwork(embed_dim, ffn_dim)
        self.pos = DWConv2d(embed_dim, 3, 1, 1)

        if layerscale:
            self.gamma_1 = nn.Parameter(layer_init_values * torch.ones(1, 1, 1, embed_dim), requires_grad=True)
            self.gamma_2 = nn.Parameter(layer_init_values * torch.ones(1, 1, 1, embed_dim), requires_grad=True)

    def forward(
            self,
            x: torch.Tensor,
            retention_rel_pos=None
    ):
        x = x + self.pos(x)
        if self.layerscale:
            x = x + self.drop_path(
                self.gamma_1 * self.retention(self.retention_layer_norm(x), retention_rel_pos))
            x = x + self.drop_path(self.gamma_2 * self.ffn(self.final_layer_norm(x)))
        else:
            x = x + self.drop_path(
                self.retention(self.retention_layer_norm(x), retention_rel_pos))
            x = x + self.drop_path(self.ffn(self.final_layer_norm(x)))
        return x



class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, out_dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Conv2d(dim, out_dim, 3, 2, 1)
        self.norm = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        '''
        x: B H W C
        '''
        x = x.permute(0, 3, 1, 2).contiguous()  # (b c h w)
        x = self.reduction(x)  # (b oc oh ow)
        x = self.norm(x)
        x = x.permute(0, 2, 3, 1)  # (b oh ow oc)
        return x



class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        fused_window_process (bool, optional): If True, use one kernel to fused window shift & window partition for acceleration, similar for the reversed part. Default: False
    """

    def __init__(self, embed_dim, out_dim, depth, num_heads,
                 init_value: float, heads_range: float,
                 ffn_dim=96., drop_path=0., norm_layer=nn.LayerNorm, chunkwise_recurrent=False,
                 downsample: PatchMerging = None, use_checkpoint=False,
                 layerscale=False, layer_init_values=1e-5):

        super().__init__()
        self.embed_dim = embed_dim
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.chunkwise_recurrent = chunkwise_recurrent
        if chunkwise_recurrent:
            retention = 'chunk'
        else:
            retention = 'whole'
        self.Relpos = RotaryPositionEmbedding2D(embed_dim, num_heads, init_value, heads_range)

        # build blocks
        self.blocks = nn.ModuleList([
            BasicBlock(retention, embed_dim, num_heads, ffn_dim,
                     drop_path[i] if isinstance(drop_path, list) else drop_path, layerscale, layer_init_values)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(dim=embed_dim, out_dim=out_dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        b, h, w, d = x.size()
        rel_pos = self.Relpos((h, w))
        for blk in self.blocks:
            x = blk(x, retention_rel_pos=rel_pos)
        if self.downsample is not None:
            x = self.downsample(x)
        return x



class LayerNorm2d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=1e-6)

    def forward(self, x: torch.Tensor):
        '''
        x: (b c h w)
        '''
        x = x.permute(0, 2, 3, 1).contiguous()  # (b h w c)
        x = self.norm(x)  # (b h w c)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x



class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim // 2, 3, 2, 1),
            nn.BatchNorm2d(embed_dim // 2),
            nn.GELU(),
            nn.Conv2d(embed_dim // 2, embed_dim // 2, 3, 1, 1),
            nn.BatchNorm2d(embed_dim // 2),
            nn.GELU(),
            nn.Conv2d(embed_dim // 2, embed_dim, 3, 2, 1),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
            nn.Conv2d(embed_dim, embed_dim, 3, 1, 1),
            nn.BatchNorm2d(embed_dim)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).permute(0, 2, 3, 1)  # (b h w c)
        return x



class Mamba2TransformerDouble(nn.Module):
    def __init__(self, in_chans=3, num_classes=1000,
                 embed_dims=[96, 192, 384, 768], depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 init_values=[1, 1, 1, 1], heads_ranges=[3, 3, 3, 3], mlp_ratios=[3, 3, 3, 3], drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm,
                 patch_norm=True, use_checkpoints=[False, False, False, False],
                 chunkwise_recurrents=[True, True, False, False], projection=1024,
                 layerscales=[False, False, False, False], layer_init_values=1e-6, **kwargs):
        super().__init__()

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dims[0]
        self.patch_norm = patch_norm
        self.num_features = embed_dims[-1]
        self.mlp_ratios = mlp_ratios

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(in_chans=in_chans, embed_dim=embed_dims[0],
                                      norm_layer=norm_layer if self.patch_norm else None)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                embed_dim=embed_dims[i_layer],
                out_dim=embed_dims[i_layer + 1] if (i_layer < self.num_layers - 1) else None,
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                init_value=init_values[i_layer],
                heads_range=heads_ranges[i_layer],
                ffn_dim=int(mlp_ratios[i_layer] * embed_dims[i_layer]),
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                chunkwise_recurrent=chunkwise_recurrents[i_layer],
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoints[i_layer],
                layerscale=layerscales[i_layer],
                layer_init_values=layer_init_values
            )
            self.layers.append(layer)

        self.proj = nn.Linear(self.num_features, projection)
        self.norm = nn.BatchNorm2d(projection)
        self.swish = MemoryEfficientSwish()
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(projection, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            try:
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            except:
                pass

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_features(self, x):
        x = self.patch_embed(x)

        for layer in self.layers:
            x = layer(x)

        x = self.proj(x)  # (b h w c)   torch.Size([128, 7, 7, 512])
        x = self.norm(x.permute(0, 3, 1, 2)).flatten(2, 3)  # (b c h*w)   # 20250224, debug for nan output

        x = self.swish(x)
        x = self.avgpool(x)  # B C 1
        x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)

        return x



class Backbone_Mamba2TransformerDouble(Mamba2TransformerDouble):
    def __init__(self, out_indices=(0, 1, 2, 3), pretrained=None, norm_layer="ln", **kwargs):
        super().__init__(**kwargs)
        self.channel_first = (norm_layer.lower() in ["bn", "ln2d"])
        _NORMLAYERS = dict(
            ln=nn.LayerNorm,
            bn=nn.BatchNorm2d,
        )
        norm_layer: nn.Module = _NORMLAYERS.get(norm_layer.lower(), None)
        kwargs.update(norm_layer=norm_layer)


        # add norm ========================
        self.out_indices = out_indices
        for i in out_indices:
            layer = norm_layer(self.layers[i].embed_dim)
            layer_name = f'outnorm{i}'
            self.add_module(layer_name, layer)

        del self.head
        del self.norm
        del self.avgpool
        del self.proj
        del self.swish
        self.load_pretrained(pretrained,key=kwargs.get('key','model'))

    def load_pretrained(self, ckpt=None, key="models"):
        if ckpt is None:
            return

        try:
            _ckpt = torch.load(open(ckpt, "rb"), map_location=torch.device("cpu"))
            print("\n========\n")
            print(f"Successfully load ckpt {ckpt} from {key}")
            print("\n========\n")

            incompatibleKeys = self.load_state_dict(_ckpt[key], strict=False)
            print(incompatibleKeys)
        except Exception as e:
            print(f"Failed loading checkpoint form {ckpt}: {e}")


    def forward(self, x):
        def layer_forward(layer, x):
            b, h, w, d = x.size()
            rel_pos = layer.Relpos((h, w))
            for blk in layer.blocks:
                x = blk(x, retention_rel_pos=rel_pos)
            if layer.downsample is not None:
                y = layer.downsample(x)
            else:
                y = x
            return x, y

        x = self.patch_embed(x)
        outs = []
        for i, layer in enumerate(self.layers):
            o, x = layer_forward(layer, x)  # (B, H, W, C)
            if i in self.out_indices:
                norm_layer = getattr(self, f'outnorm{i}')
                out = norm_layer(o)
                out = out.permute(0, 3, 1, 2).contiguous() # B, C, H, W
                outs.append(out)
            #calculate H, W for next layer, with conv stride 3, stride 2 and padding 1

        if len(self.out_indices) == 0:
            return x

        return outs






@register_model
def Mamba2Transformer_Double_T(args):
    model = Mamba2TransformerDouble(
        embed_dims=[64, 128, 256, 512],
        depths=[2, 2, 8, 2],
        num_heads=[4, 4, 8, 16],
        init_values=[2, 2, 2, 2],
        heads_ranges=[4, 4, 6, 6],
        mlp_ratios=[3, 3, 3, 3],
        drop_path_rate=0.1,
        chunkwise_recurrents=[True, True, False, False],
        layerscales=[False, False, False, False]
    )
    model.default_cfg = _cfg()
    return model



@register_model
def Mamba2Transformer_Double_S(args):
    model = Mamba2TransformerDouble(
        embed_dims=[64, 128, 256, 512],
        depths=[3, 4, 18, 4],
        num_heads=[4, 4, 8, 16],
        init_values=[2, 2, 2, 2],
        heads_ranges=[4, 4, 6, 6],
        mlp_ratios=[4, 4, 3, 3],
        drop_path_rate=0.15,
        chunkwise_recurrents=[True, True, True, False],
        layerscales=[False, False, False, False]
    )
    model.default_cfg = _cfg()
    return model



@register_model
def Mamba2Transformer_Double_B(args):
    model = Mamba2TransformerDouble(
        embed_dims=[80, 160, 320, 512],
        depths=[4, 8, 25, 8],
        num_heads=[5, 5, 10, 16],
        init_values=[2, 2, 2, 2],
        heads_ranges=[5, 5, 6, 6],
        mlp_ratios=[4, 4, 3, 3],
        drop_path_rate=0.4,
        chunkwise_recurrents=[True, True, True, False],
        layerscales=[False, False, True, True],
        layer_init_values=1e-6
    )
    model.default_cfg = _cfg()
    return model



@register_model
def Mamba2Transformer_Double_L(args):
    model = Mamba2TransformerDouble(
        embed_dims=[112, 224, 448, 640],
        depths=[4, 8, 25, 8],
        num_heads=[7, 7, 14, 20],
        init_values=[2, 2, 2, 2],
        heads_ranges=[6, 6, 6, 6],
        mlp_ratios=[4, 4, 3, 3],
        drop_path_rate=0.5,
        chunkwise_recurrents=[True, True, True, False],
        layerscales=[False, False, True, True],
        layer_init_values=1e-6
    )
    model.default_cfg = _cfg()
    return model



if __name__ == "__main__":
    model = Mamba2TransformerDouble(
        embed_dims=[64, 128, 256, 512],
        depths=[2, 2, 8, 2],
        num_heads=[4, 4, 8, 16],
        init_values=[2, 2, 2, 2],
        heads_ranges=[4, 4, 6, 6],
        mlp_ratios=[3, 3, 3, 3],
        drop_path_rate=0.1,
        chunkwise_recurrents=[True, True, False, False],
        layerscales=[False, False, False, False]
    )
    x = torch.randn([10, 3, 224, 224])
    y = model(x)

    ckpt = r"/mnt/nvme_data/zzc/projects/RMT/classfication_release/exp/Mamba2Transformer_T_Double_202502251000/best.pth"
    _ckpt = torch.load(open(ckpt, "rb"), map_location=torch.device("cpu"))
    incompatibleKeys = model.load_state_dict(_ckpt['model'], strict=True)
    print(incompatibleKeys)
