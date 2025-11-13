"""
Positional encodings for the transformer
"""
import math
import torch
from torch import nn

from util.misc import NestedTensor


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    def __init__(self, num_pos_feats=128, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, tensor_list: NestedTensor):
        x = tensor_list.tensors         # [B, C, H, W]
        mask = tensor_list.mask         # [B, H, W] (bool)
        assert mask is not None

        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)  # [B,H,W]
        x_embed = not_mask.cumsum(2, dtype=torch.float32)  # [B,H,W]

        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        # dim_t: [F]  (buffer로 등록되어 있다고 가정; 없으면 기존대로 만드세요)
        F = self.num_pos_feats
        B, H, W = x_embed.shape

        # 최종 출력만 한 번 생성: [B, 2F, H, W]
        pos = torch.empty((B, 2*F, H, W), device=x.device, dtype=x.dtype)

        # 메모리 친화: 채널을 1개씩 바로 써 넣기 (피크는 [B,H,W] 두 장)
        # k의 짝/홀에 따라 sin/cos를 나눠 쓰는 것이 원래 구현과 동일한 매핑입니다.
        for k in range(F):
            scale = (1.0 / self.dim_t[k]).item() if hasattr(self, "dim_t") else (self.temperature ** (2*(k//2)/F))**-1
            ang_y = y_embed * scale     # [B,H,W]
            ang_x = x_embed * scale     # [B,H,W]

            if (k % 2) == 0:
                torch.sin(ang_y, out=pos[:, k, :, :])       # y-part, even → sin
                torch.sin(ang_x, out=pos[:, F + k, :, :])   # x-part, even → sin
            else:
                torch.cos(ang_y, out=pos[:, k, :, :])       # y-part, odd  → cos
                torch.cos(ang_x, out=pos[:, F + k, :, :])   # x-part, odd  → cos

        return pos


class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """
    def __init__(self, num_pos_feats=256):
        super().__init__()
        self.row_embed = nn.Embedding(50, num_pos_feats)
        self.col_embed = nn.Embedding(50, num_pos_feats)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, tensor_list: NestedTensor):
        x = tensor_list.tensors
        h, w = x.shape[-2:]
        i = torch.arange(w, device=x.device)
        j = torch.arange(h, device=x.device)
        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)
        pos = torch.cat([
            x_emb.unsqueeze(0).repeat(h, 1, 1),
            y_emb.unsqueeze(1).repeat(1, w, 1),
        ], dim=-1).permute(2, 0, 1).unsqueeze(0).repeat(x.shape[0], 1, 1, 1)
        return pos


class PositionEmbeddingFourier(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    def __init__(self, num_pos_feats=128, temperature=10000, normalize=False, scale=None, d_in=1, gauss_scale=0.5):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

        d_out = num_pos_feats // 2
        B = torch.empty((d_in, d_out)).normal_()
        B *= gauss_scale
        self.register_buffer("gauss_B", B)

    def forward(self, tensor_list: NestedTensor):
        x = tensor_list.tensors
        mask = tensor_list.mask
        assert mask is not None
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale
    
        pos_x = x_embed[:, :, :, None]
        pos_y = y_embed[:, :, :, None]

        bs, h, w = x_embed.shape
        d_out = self.num_pos_feats // 2
        pos_x_proj = torch.mm(pos_x.view(-1, 1), self.gauss_B).view(bs, h, w, d_out)
        pos_y_proj = torch.mm(pos_y.view(-1, 1), self.gauss_B).view(bs, h, w, d_out)

        pos_x_embed = torch.stack([pos_x_proj.sin(), pos_x_proj.cos()], dim=3).flatten(3)
        pos_y_embed = torch.stack([pos_y_proj.sin(), pos_y_proj.cos()], dim=3).flatten(3)
        
        pos = torch.cat((pos_x_embed, pos_y_embed), dim=3).permute(0, 3, 1, 2)
        return pos


def build_position_encoding(args):
    N_steps = args.hidden_dim // 2
    if args.position_embedding in ('v2', 'sine'):
        # TODO find a better way of exposing other arguments
        position_embedding = PositionEmbeddingSine(N_steps, normalize=True)
    elif args.position_embedding in ('v3', 'learned'):
        position_embedding = PositionEmbeddingLearned(N_steps)
    elif args.position_embedding in ('fourier',):
        position_embedding = PositionEmbeddingFourier(N_steps, normalize=True)
    else:
        raise ValueError(f"not supported {args.position_embedding}")

    return position_embedding
