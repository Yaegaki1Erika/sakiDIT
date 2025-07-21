from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F

from diffusers.utils import logging
from diffusers.models.attention_processor import Attention
from .draft_attention import Draft_Attention
# from sageattention import sageattn

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def apply_rotary_emb(
        x: torch.Tensor,
        freqs_cis: Union[torch.Tensor, Tuple[torch.Tensor]],
        use_real: bool = True,
        use_real_unbind_dim: int = -1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if use_real:
        cos, sin = freqs_cis  # [S, D]
        cos = cos[None, None]
        sin = sin[None, None]
        cos, sin = cos.to(x.device), sin.to(x.device)

        if use_real_unbind_dim == -1:
            x_real, x_imag = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)  # [B, S, H, D//2]
            x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(3)
        elif use_real_unbind_dim == -2:
            x_real, x_imag = x.reshape(*x.shape[:-1], 2, -1).unbind(-2)  # [B, S, H, D//2]
            x_rotated = torch.cat([-x_imag, x_real], dim=-1)
        else:
            raise ValueError(f"`use_real_unbind_dim={use_real_unbind_dim}` but should be -1 or -2.")

        out = (x.float() * cos + x_rotated.float() * sin).to(x.dtype)

        return out
    else:
        x_rotated = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        freqs_cis = freqs_cis.unsqueeze(2)
        x_out = torch.view_as_real(x_rotated * freqs_cis).flatten(3)

        return x_out.type_as(x)


class BaseAttnProcessor:
    def __init__(self):
        self.draft_attn = Draft_Attention(
            pool_h=9,
            pool_w=16,
            latent_h=45,
            latent_w=80,
            visual_len=25200,
            text_len=226,
            sparsity_ratio=0.7,
            batch_size=2  # 确保索引构造正确
        )

    def __call__(
            self,
            attn: Attention,
            hidden_states: torch.Tensor,
            encoder_hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        text_seq_length = encoder_hidden_states.size(1)

        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        batch_size, sequence_length, _ = hidden_states.shape
        # print("encoder_hidden_states:", encoder_hidden_states.shape)
        # print("hidden_states:", hidden_states.shape)

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)    

        # Apply RoPE if needed
        if image_rotary_emb is not None:

            query[:, :, text_seq_length:] = apply_rotary_emb(query[:, :, text_seq_length:], image_rotary_emb)
            if not attn.is_cross_attention:
                key[:, :, text_seq_length:] = apply_rotary_emb(key[:, :, text_seq_length:], image_rotary_emb)
        # F.scaled_dot_product_attention = sageattn
        # 在外部初始化
        # B, H, S, D = query.shape
        # print("b is",B)
        B = 2
        S = 25426
        q = query.transpose(1, 2).reshape(-1, attn.heads, head_dim)
        k = key.transpose(1, 2).reshape(-1, attn.heads, head_dim)
        v = value.transpose(1, 2).reshape(-1, attn.heads, head_dim)

        

        cu_seqlens = torch.tensor([0, 25426, 50852], dtype=torch.int32, device=q.device)
        # 举例: cu_seqlens = [0, 25426, 50852]

        x = self.draft_attn(
            q=q,
            k=k,
            v=v,
            attn_mask=attention_mask,
            causal=False,
            drop_rate=0.0,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_kv=cu_seqlens,
            max_seqlen_q=25426,
            max_seqlen_kv=25426,
            batch_size=2,
            sparsity_ratio=0.7,
            block_sparse_attention=True,
        )

        # ✅ 保留视觉部分（跳过 text_len）
        # hidden_states = x.view(B, S, -1)[:, self.draft_attn.text_len:, :]

        # hidden_states = F.scaled_dot_product_attention(
        #     query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        # )
        # 假设 x.shape = [B, S, H, D]，恢复之后
        H = attn.heads
        D = head_dim
        x = x.view(B, S, H, D)

        # text token 位置
        text_start = self.draft_attn.visual_len
        text_end = text_start + self.draft_attn.text_len
        text_range = slice(text_start, text_end)

        q_reshaped = q.view(B, S, H, D)
        k_reshaped = k.view(B, S, H, D)
        v_reshaped = v.view(B, S, H, D)

        # 抽出 text 区间
        text_start = self.draft_attn.visual_len
        text_end = text_start + self.draft_attn.text_len
        text_range = slice(text_start, text_end)

        # 抽取并转换维度
        q_text = q_reshaped[:, text_range, :, :].transpose(1, 2)      # [B, H, text_len, D]
        k_all = k_reshaped.transpose(1, 2)                             # [B, H, total_len, D]
        v_all = v_reshaped.transpose(1, 2)                             # [B, H, total_len, D]

        # dense attention
        text_output = F.scaled_dot_product_attention(
            q_text, k_all, v_all, dropout_p=0.0, is_causal=False
        )  # [B, H, text_len, D]

        # 恢复形状 [B, text_len, H, D]
        text_output = text_output.transpose(1, 2)

        # 写入回 x
        x[:, text_range, :, :] = text_output
        
        # flatten head
        hidden_states = x.view(batch_size, -1, attn.heads * head_dim).transpose(1, 2)  # [B, C, S]
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        encoder_hidden_states, hidden_states = hidden_states.split(
            [text_seq_length, hidden_states.size(1) - text_seq_length], dim=1
        )
        return hidden_states, encoder_hidden_states