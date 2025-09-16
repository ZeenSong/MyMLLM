# modeling_geoformer.py
import copy
import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers.utils import ModelOutput
from transformers import PreTrainedModel, PretrainedConfig


# -------------------------
# Config
# -------------------------
class GeoformerConfig(PretrainedConfig):
    model_type = "geoformer"

    def __init__(
        self,
        # data / patch
        input_channels: int = 3,
        patch_size: Tuple[int, int] = (4, 4),
        needtauxy: bool = False,                # 是否额外拼接 (tau, x, y)
        emb_spatial_size: int = 64,             # S = H/ps[0] * W/ps[1]
        input_length: int = 12,                 # 观测长度
        output_length: int = 12,                # 预测长度

        # transformer dims
        d_size: int = 512,
        nheads: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,

        # depth
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,

        # misc
        layer_norm_eps: float = 1e-5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.input_channels = input_channels
        self.patch_size = tuple(patch_size)
        self.needtauxy = needtauxy
        self.emb_spatial_size = emb_spatial_size
        self.input_length = input_length
        self.output_length = output_length

        self.d_size = d_size
        self.nheads = nheads
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.layer_norm_eps = layer_norm_eps


# -------------------------
# Helpers (unfold / fold)
# -------------------------
def unfold_func(in_data: torch.Tensor, kernel_size: Tuple[int, int]):
    n_dim = in_data.dim()
    assert n_dim in (4, 5)  # (B, C, H, W) or (B, T, C, H, W)
    data1 = in_data.unfold(-2, size=kernel_size[0], step=kernel_size[0])
    data1 = data1.unfold(-2, size=kernel_size[1], step=kernel_size[1]).flatten(-2)
    if n_dim == 4:
        data1 = data1.permute(0, 1, 4, 2, 3).flatten(1, 2)
    else:  # 5D
        data1 = data1.permute(0, 1, 2, 5, 3, 4).flatten(2, 3)
    assert data1.size(-3) == in_data.size(-3) * kernel_size[0] * kernel_size[1]
    return data1


def fold_func(tensor: torch.Tensor, output_size: Tuple[int, int], kernel_size: Tuple[int, int]):
    # tensor: (B, T, C*ps0*ps1, H/ps0, W/ps1) OR (B, C*ps0*ps1, H/ps0, W/ps1)
    tensor = tensor.float()
    n_dim = tensor.dim()
    assert n_dim in (4, 5)
    f = tensor.flatten(0, 1) if n_dim == 5 else tensor
    folded = F.fold(
        f.flatten(-2),               # (B*T, C*ps0*ps1, (H/ps0)*(W/ps1))
        output_size=output_size,
        kernel_size=kernel_size,
        stride=kernel_size,
    )
    if n_dim == 5:
        folded = folded.reshape(tensor.size(0), tensor.size(1), *folded.size()[1:])
    return folded


# -------------------------
# Embeddings
# -------------------------
class GeoformerEmbeddings(nn.Module):
    """
    输入 x: (B, S, T, cube_dim)  ->  线性投影到 d_size，并加：
      1) 时间位置编码 pe_time[ :, :, :T, :]
      2) 空间 embedding emb_space(S)
    """
    def __init__(self, cube_dim: int, d_size: int, emb_spatial_size: int, max_len: int, layer_norm_eps: float):
        super().__init__()
        self.linear = nn.Linear(cube_dim, d_size)
        self.norm = nn.LayerNorm(d_size, eps=layer_norm_eps)
        self.emb_space = nn.Embedding(emb_spatial_size, d_size)

        # 预计算正弦时间位置编码（可注册为 buffer，由设备自动迁移）
        pe = torch.zeros(max_len, d_size)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_size, 2) * -(math.log(10000.0) / d_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe_time", pe[None, None], persistent=False)  # (1,1,T,d)

        spatial_pos = torch.arange(emb_spatial_size)[None, :, None]        # (1,S,1)
        self.register_buffer("spatial_pos", spatial_pos, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, S, T, cube_dim)
        B, S, T, _ = x.shape
        x = self.linear(x)                                   # (B,S,T,d)
        # 时间编码
        x = x + self.pe_time[:, :, :T, :].to(x.dtype)        # broadcast -> (1,1,T,d)
        # 空间编码
        embedded_space = self.emb_space(self.spatial_pos).to(x.dtype)   # (1,S,1,d)
        x = x + embedded_space
        return self.norm(x)


# -------------------------
# Attention: T- and S- attention
# -------------------------
def T_attention(query, key, value, mask=None, dropout=None):
    # 对时间维做注意力：输入已分多头，shape (B, h, S, T, d_k)
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)  # (..., T, T)
    if mask is not None:
        # mask: (T, T) 或 (T,) 的下三角，扩展到 batch/head
        # 这里沿用原始实现的 broadcast：mask -> True 为屏蔽
        assert mask.dtype == torch.bool
        if mask.dim() == 2:
            scores = scores.masked_fill(mask[None, None, None], float("-inf"))
        elif mask.dim() == 1:
            scores = scores.masked_fill(mask[None, None, None, :, None], float("-inf"))
        else:
            raise ValueError("Unsupported mask shape for T_attention.")
    p = F.softmax(scores, dim=-1)
    if dropout is not None:
        p = dropout(p)
    return torch.matmul(p, value)


# def S_attention(query, key, value, mask=None, dropout=None):
#     # 对空间维做注意力：输入已分多头，先交换 S/T
#     # query: (B,h,S,T,d_k) -> (B,h,T,S,d_k)
#     d_k = query.size(-1)
#     q = query.transpose(2, 3)  # (B,h,T,S,d_k)
#     k = key.transpose(2, 3)
#     v = value.transpose(2, 3)
#     scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)  # (..., S, S)
#     p = F.softmax(scores, dim=-1)
#     if dropout is not None:
#         p = dropout(p)
#     out = torch.matmul(p, v).transpose(2, 3)  # back to (B,h,S,T,d_k)
#     return out

def S_attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(
        query.transpose(2, 3), key.transpose(2, 3).transpose(-2, -1)
    ) / np.sqrt(d_k)
    p_sc = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_sc = dropout(p_sc)
    return torch.matmul(p_sc, value.transpose(2, 3)).transpose(2, 3)


class MultiHeadTSAttention(nn.Module):
    """
    通用包装：线性投影 -> 分头 -> 调用给定 attention 实现 -> 合并 -> 线性输出
    适配 self-attn 与 cross-attn：
      - 允许 Tq != Tk（例如 decoder->encoder 交叉注意力）
    输入/输出: (B, S, T, d)
    """
    def __init__(self, d_size: int, nheads: int, attention_impl, dropout: float):
        super().__init__()
        assert d_size % nheads == 0
        self.d_k = d_size // nheads
        self.nheads = nheads
        self.q_proj = nn.Linear(d_size, d_size)
        self.k_proj = nn.Linear(d_size, d_size)
        self.v_proj = nn.Linear(d_size, d_size)
        self.o_proj = nn.Linear(d_size, d_size)
        self.dropout = nn.Dropout(dropout)
        self.attn = attention_impl  # 例如 T_attention 或 S_attention

    def forward(
        self,
        query: torch.Tensor,  # (B,S,Tq,D)
        key: torch.Tensor,    # (B,S,Tk,D)
        value: torch.Tensor,  # (B,S,Tk,D)
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # 对每个张量**分别**用自己的 (B,S,T) 做分头，避免把 Tq 当成 Tk
        def proj_reshape(x: torch.Tensor, proj: nn.Linear):
            y = proj(x)                      # (B,S,T,D)
            B, S, T, _ = y.shape
            y = y.view(B, S, T, self.nheads, self.d_k).permute(0, 3, 1, 2, 4)  # (B,h,S,T,d_k)
            return y, (B, S, T)

        q, (Bq, Sq, Tq) = proj_reshape(query, self.q_proj)
        k, (_,  _,  Tk) = proj_reshape(key,   self.k_proj)
        v, (_,  _,  Tv) = proj_reshape(value, self.v_proj)
        # 这里允许 Tq != Tk/Tv，注意力实现需返回与 q 对齐的形状 (B,h,S,Tq,d_k)

        x = self.attn(q, k, v, mask=mask, dropout=self.dropout)  # 期望 (B,h,S,Tq,d_k)
        Bh, Hh, Sh, Th, Dk = x.shape
        x = x.permute(0, 2, 3, 1, 4).contiguous().view(Bh, Sh, Th, Hh * Dk)  # (B,S,Tq,D)
        return self.o_proj(x)                                # (B,S,T,D)


# -------------------------
# Encoder / Decoder Blocks
# -------------------------
class GeoformerEncoderLayer(nn.Module):
    def __init__(self, config: GeoformerConfig):
        super().__init__()
        d = config.d_size
        self.dropout = config.dropout
        self.ln1 = nn.LayerNorm(d, eps=config.layer_norm_eps)
        self.ln2 = nn.LayerNorm(d, eps=config.layer_norm_eps)
        self.time_attn  = MultiHeadTSAttention(d, config.nheads, T_attention, config.dropout)
        self.space_attn = MultiHeadTSAttention(d, config.nheads, S_attention, config.dropout)
        self.ff = nn.Sequential(
            nn.Linear(d, config.dim_feedforward),
            nn.ReLU(),
            nn.Linear(config.dim_feedforward, d),
        )
        self.drop = nn.Dropout(config.dropout)

    def ts_attn(self, x, mask: Optional[torch.Tensor]):
        t = self.time_attn(x, x, x, mask)   # (B,S,T,d)
        s = self.space_attn(t, t, t, mask)  # (B,S,T,d)
        return s

    def forward(self, x, mask: Optional[torch.Tensor] = None):
        # x: (B,S,T,d)
        x = self.ln1(x + self.drop(self.ts_attn(x, mask)))
        x = self.ln2(x + self.drop(self.ff(x)))
        return x


class GeoformerDecoderLayer(nn.Module):
    def __init__(self, config: GeoformerConfig):
        super().__init__()
        d = config.d_size
        self.ln1 = nn.LayerNorm(d, eps=config.layer_norm_eps)
        self.ln2 = nn.LayerNorm(d, eps=config.layer_norm_eps)
        self.ln3 = nn.LayerNorm(d, eps=config.layer_norm_eps)

        self.self_time  = MultiHeadTSAttention(d, config.nheads, T_attention, config.dropout)
        self.self_space = MultiHeadTSAttention(d, config.nheads, S_attention, config.dropout)
        self.cross_attn = MultiHeadTSAttention(d, config.nheads, T_attention, config.dropout)  # 对 encoder 输出做 T-attn
        self.ff = nn.Sequential(
            nn.Linear(d, config.dim_feedforward),
            nn.ReLU(),
            nn.Linear(config.dim_feedforward, d),
        )
        self.drop = nn.Dropout(config.dropout)

    def divided_ts(self, x, mask):
        t = self.self_time(x, x, x, mask)
        s = self.self_space(t, t, t, None)
        return s

    def forward(self, x, memory, tgt_mask: Optional[torch.Tensor], mem_mask: Optional[torch.Tensor]):
        # Self (T+S)
        x = self.ln1(x + self.drop(self.divided_ts(x, tgt_mask)))
        # Cross (with encoder)
        x = self.ln2(x + self.drop(self.cross_attn(x, memory, memory, mem_mask)))
        # FF
        x = self.ln3(x + self.drop(self.ff(x)))
        return x


class GeoformerEncoder(nn.Module):
    def __init__(self, config: GeoformerConfig):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(GeoformerEncoderLayer(config)) for _ in range(config.num_encoder_layers)])

    def forward(self, x, mask: Optional[torch.Tensor] = None):
        for layer in self.layers:
            x = layer(x, mask)
        return x


class GeoformerDecoder(nn.Module):
    def __init__(self, config: GeoformerConfig):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(GeoformerDecoderLayer(config)) for _ in range(config.num_decoder_layers)])

    def forward(self, x, memory, tgt_mask: Optional[torch.Tensor], mem_mask: Optional[torch.Tensor]):
        for layer in self.layers:
            x = layer(x, memory, tgt_mask, mem_mask)
        return x


# -------------------------
# Outputs
# -------------------------
@dataclass
class GeoformerOutput(ModelOutput):
    prediction: torch.Tensor                      # (B, T_out, C, H, W)
    encoder_last_hidden_state: Optional[torch.Tensor] = None  # (B,S,T_in,d)
    decoder_last_hidden_state: Optional[torch.Tensor] = None  # (B,S,T_out,d)


# -------------------------
# PreTrained + Model
# -------------------------
class GeoformerPreTrainedModel(PreTrainedModel):
    config_class = GeoformerConfig
    base_model_prefix = "geoformer"
    _no_split_modules = ["GeoformerEncoderLayer", "GeoformerDecoderLayer"]  # 让 device_map 在层级处切分

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.xavier_uniform_(module.weight)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)


class GeoformerModel(GeoformerPreTrainedModel):
    """
    等价于你原来的 Geoformer（含 encode/decode、两套 embedding、patch unfold/fold），
    但符合 Transformers 的 save/load/auto-shard 规范。
    """
    def __init__(self, config: GeoformerConfig):
        super().__init__(config)

        # 计算单个 patch cube 展平后的通道维
        if config.needtauxy:
            cube_dim = (config.input_channels + 2) * config.patch_size[0] * config.patch_size[1]
        else:
            cube_dim = config.input_channels * config.patch_size[0] * config.patch_size[1]
        self.cube_dim = cube_dim

        self.predictor_emb  = GeoformerEmbeddings(
            cube_dim=cube_dim, d_size=config.d_size,
            emb_spatial_size=config.emb_spatial_size,
            max_len=config.input_length,
            layer_norm_eps=config.layer_norm_eps,
        )
        self.predictand_emb = GeoformerEmbeddings(
            cube_dim=cube_dim, d_size=config.d_size,
            emb_spatial_size=config.emb_spatial_size,
            max_len=config.output_length,
            layer_norm_eps=config.layer_norm_eps,
        )

        self.encoder = GeoformerEncoder(config)
        self.decoder = GeoformerDecoder(config)
        self.linear_out = nn.Linear(config.d_size, cube_dim)

        self.post_init()

    # --- public API ---
    @staticmethod
    def causal_mask(sz: int, device=None, dtype=None):
        # True 表示被mask（不允许看未来）
        mask = torch.triu(torch.ones(sz, sz, device=device, dtype=torch.bool), diagonal=1)
        return mask

    def encode(self, predictor: torch.Tensor, in_mask: Optional[torch.Tensor] = None):
        """
        predictor: (B, T_in, C, H, W)
        return:   (B, S, T_in, d)
        """
        B, T, C, H, W = predictor.shape
        ps0, ps1 = self.config.patch_size
        x = unfold_func(predictor, self.config.patch_size)              # (B, T*C*ps0*ps1, H/ps0, W/ps1)
        x = x.reshape(B, T, self.cube_dim, -1).permute(0, 3, 1, 2)     # (B, S, T, cube_dim)
        x = self.predictor_emb(x)                                      # (B, S, T, d)
        return self.encoder(x, in_mask)                                # (B, S, T, d)

    def _decode_once(self, tokens: torch.Tensor, memory: torch.Tensor,
                     tgt_mask: Optional[torch.Tensor], mem_mask: Optional[torch.Tensor],
                     H: int, W: int):
        """
        tokens: (B, T_cur, C, H, W)  ->  (B, T_cur, C, H, W)
        """
        B, T, C, _, _ = tokens.shape
        ps0, ps1 = self.config.patch_size
        S = (H // ps0) * (W // ps1)

        x = unfold_func(tokens, self.config.patch_size)                 # (B, T*C*ps0*ps1, H/ps0, W/ps1)
        x = x.reshape(B, T, self.cube_dim, S).permute(0, 3, 1, 2)      # (B, S, T, cube_dim)
        x = self.predictand_emb(x)                                     # (B, S, T, d)
        x = self.decoder(x, memory, tgt_mask, mem_mask)                # (B, S, T, d)

        x = self.linear_out(x).permute(0, 2, 3, 1)                     # (B, T, cube_dim, S)
        x = x.reshape(B, T, self.cube_dim, H // ps0, W // ps1)
        x = fold_func(x, output_size=(H, W), kernel_size=self.config.patch_size)  # (B, T, C, H, W)
        return x

    def forward(
        self,
        predictor: torch.Tensor,                # (B, T_in, C, H, W)
        predictand: Optional[torch.Tensor]=None,# (B, T_out, C, H, W)  训练时可给 teacher-forcing 序列
        in_mask: Optional[torch.Tensor]=None,   # 编码器时间mask（下三角或 None）
        enout_mask: Optional[torch.Tensor]=None,# 交叉注意力 mask（可选）
        teacher_forcing_ratio: float = 0.0,     # 与原 sv_ratio 等价（0~1）
        autoregressive: bool = True,            # 推理时逐步生成
        return_dict: bool = True,
    ) -> GeoformerOutput:
        """
        训练：
            - 如给出 predictand（目标序列），将使用 teacher forcing（按概率采样）+ 一次性解码输出全序列
        推理：
            - 不给 predictand，则从 predictor 的最后一帧开始自回归生成 output_length 步
        """
        B, T_in, C, H, W = predictor.shape
        enc = self.encode(predictor, in_mask)   # (B,S,T_in,d)

        # Decoder mask: 下三角因果掩码（长度取当前解码长度）
        def get_causal_mask(t):
            return self.causal_mask(t, device=predictor.device)

        if self.training and predictand is not None:
            # 训练：先用 (pred_last, gt[:-1]) 做一次并行解码，得到 outvar_pred
            concat_in = torch.cat([predictor[:, -1:], predictand[:, :-1]], dim=1)  # (B, T_out, C, H, W)
            out_mask = get_causal_mask(concat_in.size(1))
            outvar_pred = self._decode_once(concat_in, enc, out_mask, enout_mask, H, W)

            # Scheduled Sampling（teacher forcing）
            if teacher_forcing_ratio > 1e-7:
                with torch.no_grad():
                    bern = torch.bernoulli(
                        torch.full((B, self.config.output_length - 1, 1, 1, 1), teacher_forcing_ratio, device=predictor.device)
                    )
                mixed = bern * predictand[:, :-1] + (1 - bern) * outvar_pred[:, :-1]
            else:
                mixed = outvar_pred[:, :-1]

            tokens = torch.cat([predictor[:, -1:], mixed], dim=1)  # (B, T_out, C, H, W)
            out_mask = get_causal_mask(tokens.size(1))
            final_pred = self._decode_once(tokens, enc, out_mask, enout_mask, H, W)

            dec_last = None  # 可按需返回 decoder hidden
            return GeoformerOutput(
                prediction=final_pred, encoder_last_hidden_state=enc, decoder_last_hidden_state=dec_last
            )

        # 推理：自回归生成
        if predictand is None:
            assert autoregressive, "Inference without labels requires autoregressive=True."
            cur = predictor[:, -1:]  # (B,1,C,H,W)
            outs = []
            for step in range(self.config.output_length):
                out_mask = get_causal_mask(cur.size(1))
                pred = self._decode_once(cur, enc, out_mask, enout_mask, H, W)  # (B,t,C,H,W)
                next_frame = pred[:, -1:]  # 取最新一步
                outs.append(next_frame)
                cur = torch.cat([cur, next_frame], dim=1)
            final = torch.cat(outs, dim=1)
            return GeoformerOutput(prediction=final, encoder_last_hidden_state=enc, decoder_last_hidden_state=None)

        # 评估：给完整 predictand，一次性并行解码
        out_mask = get_causal_mask(predictand.size(1))
        outvar_pred = self._decode_once(predictand, enc, out_mask, enout_mask, H, W)
        return GeoformerOutput(prediction=outvar_pred, encoder_last_hidden_state=enc, decoder_last_hidden_state=None)
