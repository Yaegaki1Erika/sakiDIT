from typing import Any, Dict, Optional, Tuple, Union

import torch
from torch import nn

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import PeftAdapterMixin
from diffusers.utils import USE_PEFT_BACKEND, logging, scale_lora_layers, unscale_lora_layers
from diffusers.utils.torch_utils import maybe_allow_in_graph
from diffusers.models.attention import Attention, FeedForward
from diffusers.models.embeddings import get_3d_sincos_pos_embed, TimestepEmbedding, Timesteps
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import AdaLayerNorm
from .attn_processor import BaseAttnProcessor

from .pab import enable_pab, if_broadcast_spatial,if_broadcast_mlp_test
from .teacache import get_should_calc,enable_teacache

accumulated_rel_l1_distance=0
previous_modulated_input=None
sort_attn=0
attn_hidden_states_global=[None]*37
attn_encoder_hidden_states_global=[None]*37
logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


import torch

import torch

def quantize_int4(tensor):
    orig_shape = tensor.shape
    flat = tensor.flatten()

    scale = flat.abs().max() / 7.0
    scale = max(scale, 1e-8)
    qt = (flat / scale).round().clamp(-8, 7).to(torch.int8)

    # 打包两个 int4 → uint8
    if qt.numel() % 2 != 0:
        qt = torch.cat([qt, torch.zeros(1, dtype=torch.int8, device=qt.device)])  # 补1个0

    qt_packed = (qt[::2] & 0x0F) | ((qt[1::2] & 0x0F) << 4)
    return qt_packed.to(torch.uint8), scale, orig_shape


def dequantize_int4(qt_packed, scale, orig_shape):
    total_elems = torch.prod(torch.tensor(orig_shape)).item()
    unpacked_len = qt_packed.numel() * 2

    # 解码成 int4
    low = (qt_packed & 0x0F).to(torch.int8)
    high = ((qt_packed >> 4) & 0x0F).to(torch.int8)
    low[low >= 8] -= 16
    high[high >= 8] -= 16

    qt = torch.empty(unpacked_len, dtype=torch.int8, device=qt_packed.device)
    qt[::2] = low
    qt[1::2] = high

    # 截断多余补的元素，reshape
    qt = qt[:total_elems]
    return (qt.to(torch.float16) * scale).reshape(orig_shape)


def print_tensor_info(name, tensor):
    print(f"{name}: shape={tensor.shape}, mean={tensor.mean():.4f}, std={tensor.std():.4f}, min={tensor.min():.4f}, max={tensor.max():.4f}")


def quantize_tensor_dynamic(tensor):
    min_val = tensor.min()
    max_val = tensor.max()
    scale = 255.0 / (max_val - min_val + 1e-5)
    qtensor = ((tensor - min_val) * scale).round().clamp(0, 255).to(torch.uint8)
    return qtensor, min_val, scale

def dequantize_tensor_dynamic(qtensor, min_val, scale):
    return (qtensor.to(torch.float16) / scale) + min_val


class BaseLayerNormZero(nn.Module):
    def __init__(
            self,
            conditioning_dim: int,
            embedding_dim: int,
            elementwise_affine: bool = True,
            eps: float = 1e-5,
            bias: bool = True,
    ) -> None:
        super().__init__()

        self.silu = nn.SiLU()
        self.linear = nn.Linear(conditioning_dim, 6 * embedding_dim, bias=bias)
        self.norm = nn.LayerNorm(embedding_dim, eps=eps, elementwise_affine=elementwise_affine)

    def forward(
            self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, temb: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        shift, scale, gate, enc_shift, enc_scale, enc_gate = self.linear(self.silu(temb)).chunk(6, dim=1)
        hidden_states = self.norm(hidden_states) * (1 + scale)[:, None, :] + shift[:, None, :]
        encoder_hidden_states = self.norm(encoder_hidden_states) * (1 + enc_scale)[:, None, :] + enc_shift[:, None, :]
        return hidden_states, encoder_hidden_states, gate[:, None, :], enc_gate[:, None, :]


class BasePatchEmbed(nn.Module):
    def __init__(
            self,
            patch_size: int = 2,
            in_channels: int = 16,
            embed_dim: int = 1920,
            text_embed_dim: int = 4096,
            bias: bool = True,
            sample_width: int = 90,
            sample_height: int = 60,
            sample_frames: int = 49,
            temporal_compression_ratio: int = 4,
            max_text_seq_length: int = 226,
            spatial_interpolation_scale: float = 1.875,
            temporal_interpolation_scale: float = 1.0,
            use_positional_embeddings: bool = True,
            use_learned_positional_embeddings: bool = True,
    ) -> None:
        super().__init__()

        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.sample_height = sample_height
        self.sample_width = sample_width
        self.sample_frames = sample_frames
        self.temporal_compression_ratio = temporal_compression_ratio
        self.max_text_seq_length = max_text_seq_length
        self.spatial_interpolation_scale = spatial_interpolation_scale
        self.temporal_interpolation_scale = temporal_interpolation_scale
        self.use_positional_embeddings = use_positional_embeddings
        self.use_learned_positional_embeddings = use_learned_positional_embeddings

        self.proj = nn.Conv2d(
            in_channels, embed_dim, kernel_size=(patch_size, patch_size), stride=patch_size, bias=bias
        )
        self.text_proj = nn.Linear(text_embed_dim, embed_dim)

        if use_positional_embeddings or use_learned_positional_embeddings:
            persistent = use_learned_positional_embeddings
            pos_embedding = self._get_positional_embeddings(sample_height, sample_width, sample_frames)
            self.register_buffer("pos_embedding", pos_embedding, persistent=persistent)
        self.cache = None

    def _get_positional_embeddings(
            self, sample_height: int, sample_width: int, sample_frames: int, device: Optional[torch.device] = None
    ) -> torch.Tensor:
        post_patch_height = sample_height // self.patch_size
        post_patch_width = sample_width // self.patch_size
        post_time_compression_frames = (sample_frames - 1) // self.temporal_compression_ratio + 1
        num_patches = post_patch_height * post_patch_width * post_time_compression_frames

        pos_embedding = get_3d_sincos_pos_embed(
            self.embed_dim,
            (post_patch_width, post_patch_height),
            post_time_compression_frames,
            self.spatial_interpolation_scale,
            self.temporal_interpolation_scale,
            device=device,
            output_type="pt",
        )
        pos_embedding = pos_embedding.flatten(0, 1)
        joint_pos_embedding = pos_embedding.new_zeros(
            1, self.max_text_seq_length + num_patches, self.embed_dim, requires_grad=False
        )
        joint_pos_embedding.data[:, self.max_text_seq_length:].copy_(pos_embedding)

        return joint_pos_embedding

    def forward(self, text_embeds: torch.Tensor, image_embeds: torch.Tensor):
        r"""
        Args:
            text_embeds (`torch.Tensor`):
                Input text embeddings. Expected shape: (batch_size, seq_length, embedding_dim).
            image_embeds (`torch.Tensor`):
                Input image embeddings. Expected shape: (batch_size, num_frames, channels, height, width).
        """
        text_embeds = self.text_proj(text_embeds)

        batch_size, num_frames, channels, height, width = image_embeds.shape
        image_embeds = image_embeds.reshape(-1, channels, height, width)
        image_embeds = self.proj(image_embeds)
        image_embeds = image_embeds.view(batch_size, num_frames, *image_embeds.shape[1:])
        image_embeds = image_embeds.flatten(3).transpose(2, 3)  # [batch, num_frames, height x width, channels]
        image_embeds = image_embeds.flatten(1, 2)  # [batch, num_frames x height x width, channels]

        embeds = torch.cat(
            [text_embeds, image_embeds], dim=1
        ).contiguous()  # [batch, seq_length + num_frames x height x width, channels]

        if self.use_positional_embeddings or self.use_learned_positional_embeddings:

            pre_time_compression_frames = (num_frames - 1) * self.temporal_compression_ratio + 1

            if (
                    self.sample_height != height
                    or self.sample_width != width
                    or self.sample_frames != pre_time_compression_frames
            ):
                if (
                        not self.cache
                        or (
                        isinstance(self.cache, list) and (self.cache[0] != height
                                                          or self.cache[1] != width
                                                          or self.cache[2] != pre_time_compression_frames)
                )
                ):
                    pos_embedding = self._get_positional_embeddings(
                        height, width, pre_time_compression_frames, device=embeds.device
                    )
                    self.cache = [height, width, pre_time_compression_frames, pos_embedding]
                else:
                    pos_embedding = self.cache[3]
            else:
                pos_embedding = self.pos_embedding

            pos_embedding = pos_embedding.to(dtype=embeds.dtype)
            embeds = embeds + pos_embedding

        return embeds


@maybe_allow_in_graph
class BaseBlock(nn.Module):
    def __init__(
            self,
            dim: int,
            num_attention_heads: int,
            attention_head_dim: int,
            time_embed_dim: int,
            dropout: float = 0.0,
            activation_fn: str = "gelu-approximate",
            attention_bias: bool = False,
            qk_norm: bool = True,
            norm_elementwise_affine: bool = True,
            norm_eps: float = 1e-5,
            final_dropout: bool = True,
            ff_inner_dim: Optional[int] = None,
            ff_bias: bool = True,
            attention_out_bias: bool = True,
            block_idx: int = 0,
    ):
        super().__init__()

        # 1. Self Attention
        self.norm1 = BaseLayerNormZero(time_embed_dim, dim, norm_elementwise_affine, norm_eps, bias=True)

        self.attn1 = Attention(
            query_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            qk_norm="layer_norm" if qk_norm else None,
            eps=1e-6,
            bias=attention_bias,
            out_bias=attention_out_bias,
            processor=BaseAttnProcessor(),
        )

        # 2. Feed Forward
        self.norm2 = BaseLayerNormZero(time_embed_dim, dim, norm_elementwise_affine, norm_eps, bias=True)

        self.ff = FeedForward(
            dim,
            dropout=dropout,
            activation_fn=activation_fn,
            final_dropout=final_dropout,
            inner_dim=ff_inner_dim,
            bias=ff_bias,
        )
        # pab
        # self.attn_count = 0
        # self.mlp_count = 0
        self.last_attn = None
        self.block_idx = block_idx
        self.mlp=None

    def forward(
            self,
            hidden_states: torch.Tensor,
            encoder_hidden_states: torch.Tensor,
            temb: torch.Tensor,
            image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
            timestep=None,
            attention_kwargs: Optional[Dict[str, Any]] = None,
            attn_count1: int = 0,
            mlp_count1: int = 0,
    ) -> torch.Tensor:
        # print(self.block_idx)
        # global attn_hidden_states_global,attn_encoder_hidden_states_global
        text_seq_length = encoder_hidden_states.size(1)
        attention_kwargs = attention_kwargs or {}

        # norm & modulate
        norm_hidden_states, norm_encoder_hidden_states, gate_msa, enc_gate_msa = self.norm1(
            hidden_states, encoder_hidden_states, temb
        )
        #print(enable_pab())
        #print("saki!!")
        if enable_pab():
            broadcast_attn, _ = if_broadcast_spatial(int(timestep[0]), attn_count1)
        if enable_pab() and broadcast_attn and self.block_idx<37:
            #print("saki!!!")
            # attn_hidden_states = attn_hidden_states_global[self.block_idx]
            # attn_encoder_hidden_states = attn_encoder_hidden_states_global[self.block_idx]
            # attn_hidden_states_global[self.block_idx] = None
            # attn_encoder_hidden_states_global[self.block_idx] = None
            # attn_hidden_states, attn_encoder_hidden_states = self.last_attn
            attn_hidden_states, attn_encoder_hidden_states = [
            dequantize_tensor_dynamic(*item) for item in self.last_attn
            ]
        else:
            attn_hidden_states, attn_encoder_hidden_states = self.attn1(
                hidden_states=norm_hidden_states,
                encoder_hidden_states=norm_encoder_hidden_states,
                image_rotary_emb=image_rotary_emb,
                **attention_kwargs,
            )
            if enable_pab() and self.block_idx<37:
                self.last_attn = tuple(quantize_tensor_dynamic(t) for t in (attn_hidden_states, attn_encoder_hidden_states))
                # self.last_attn = (attn_hidden_states, attn_encoder_hidden_states)
                # attn_hidden_states_global[self.block_idx] = attn_hidden_states
                # attn_encoder_hidden_states_global[self.block_idx] = attn_encoder_hidden_states


        # attn_hidden_states, attn_encoder_hidden_states = self.attn1(
        #     hidden_states=norm_hidden_states,
        #     encoder_hidden_states=norm_encoder_hidden_states,
        #     image_rotary_emb=image_rotary_emb,
            
        # )
        hidden_states = hidden_states + gate_msa * attn_hidden_states
        encoder_hidden_states = encoder_hidden_states + enc_gate_msa * attn_encoder_hidden_states

        # norm & modulate
        norm_hidden_states, norm_encoder_hidden_states, gate_ff, enc_gate_ff = self.norm2(
            hidden_states, encoder_hidden_states, temb
        )


        if enable_pab():
            broadcast_mlp, _ = if_broadcast_mlp_test(int(timestep[0]), mlp_count1)
        if enable_pab() and broadcast_mlp and self.block_idx<28:
            if self.block_idx<2:
                ff_output = dequantize_int4(*self.mlp)
            elif self.block_idx<28:
                ff_output = dequantize_tensor_dynamic(*self.mlp)            
        else:

            # feed-forward
            norm_hidden_states = torch.cat([norm_encoder_hidden_states, norm_hidden_states], dim=1)
            ff_output = self.ff(norm_hidden_states)
            if enable_pab():
                if self.block_idx<2:
                    qt_packed, scale, shape = quantize_int4(ff_output)
                    self.mlp = (qt_packed, scale, shape)
                elif self.block_idx<28:
                    self.mlp = quantize_tensor_dynamic(ff_output)


        hidden_states = hidden_states + gate_ff * ff_output[:, text_seq_length:]
        encoder_hidden_states = encoder_hidden_states + enc_gate_ff * ff_output[:, :text_seq_length]

        return hidden_states, encoder_hidden_states


class BaseTransformer3DModel(ModelMixin, ConfigMixin, PeftAdapterMixin):
    _supports_gradient_checkpointing = True
    _no_split_modules = ["BaseBlock", "BasePatchEmbed"]

    @register_to_config
    def __init__(
            self,
            num_attention_heads: int = 30,
            attention_head_dim: int = 64,
            in_channels: int = 16,
            out_channels: Optional[int] = 16,
            flip_sin_to_cos: bool = True,
            freq_shift: int = 0,
            time_embed_dim: int = 512,
            ofs_embed_dim: Optional[int] = None,
            text_embed_dim: int = 4096,
            num_layers: int = 30,
            dropout: float = 0.0,
            attention_bias: bool = True,
            sample_width: int = 90,
            sample_height: int = 60,
            sample_frames: int = 49,
            patch_size: int = 2,
            temporal_compression_ratio: int = 4,
            max_text_seq_length: int = 226,
            activation_fn: str = "gelu-approximate",
            timestep_activation_fn: str = "silu",
            norm_elementwise_affine: bool = True,
            norm_eps: float = 1e-5,
            spatial_interpolation_scale: float = 1.875,
            temporal_interpolation_scale: float = 1.0,
            use_rotary_positional_embeddings: bool = False,
            use_learned_positional_embeddings: bool = False,
            patch_bias: bool = True,
    ):
        super().__init__()
        inner_dim = num_attention_heads * attention_head_dim

        # 1. Patch embedding
        self.patch_embed = BasePatchEmbed(
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=inner_dim,
            text_embed_dim=text_embed_dim,
            bias=patch_bias,
            sample_width=sample_width,
            sample_height=sample_height,
            sample_frames=sample_frames,
            temporal_compression_ratio=temporal_compression_ratio,
            max_text_seq_length=max_text_seq_length,
            spatial_interpolation_scale=spatial_interpolation_scale,
            temporal_interpolation_scale=temporal_interpolation_scale,
            use_positional_embeddings=not use_rotary_positional_embeddings,
            use_learned_positional_embeddings=use_learned_positional_embeddings,
        )
        self.embedding_dropout = nn.Dropout(dropout)

        self.time_proj = Timesteps(inner_dim, flip_sin_to_cos, freq_shift)
        self.time_embedding = TimestepEmbedding(inner_dim, time_embed_dim, timestep_activation_fn)

        # 3. Define spatio-temporal transformers blocks
        self.transformer_blocks = nn.ModuleList(
            [
                BaseBlock(
                    dim=inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    time_embed_dim=time_embed_dim,
                    dropout=dropout,
                    activation_fn=activation_fn,
                    attention_bias=attention_bias,
                    norm_elementwise_affine=norm_elementwise_affine,
                    norm_eps=norm_eps,
                    block_idx=i,
                )
                for i in range(num_layers)
            ]
        )
        self.norm_final = nn.LayerNorm(inner_dim, norm_eps, norm_elementwise_affine)

        # 4. Output blocks
        self.norm_out = AdaLayerNorm(
            embedding_dim=time_embed_dim,
            output_dim=2 * inner_dim,
            norm_elementwise_affine=norm_elementwise_affine,
            norm_eps=norm_eps,
            chunk_dim=1,
        )

        output_dim = patch_size * patch_size * out_channels
        self.proj_out = nn.Linear(inner_dim, output_dim)
        self.gradient_checkpointing = False
        
        # self.previous_residual = None
        # self.previous_residual_encoder = None

    @property
    def attn_processors(self) -> Dict[str, BaseAttnProcessor]:
        processors = {}

        def fn_recursive_add_processors(name: str, module: torch.nn.Module, processors: Dict[str, BaseAttnProcessor]):
            if hasattr(module, "get_processor"):
                processors[f"{name}.processor"] = module.get_processor()

            for sub_name, child in module.named_children():
                fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)

            return processors

        for name, module in self.named_children():
            fn_recursive_add_processors(name, module, processors)

        return processors

    def set_attn_processor(self, processor: Union[BaseAttnProcessor, Dict[str, BaseAttnProcessor]]):
        count = len(self.attn_processors.keys())

        if isinstance(processor, dict) and len(processor) != count:
            raise ValueError(
                f"A dict of processors was passed, but the number of processors {len(processor)} does not match the"
                f" number of attention layers: {count}. Please make sure to pass {count} processor classes."
            )

        def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor):
            if hasattr(module, "set_processor"):
                if not isinstance(processor, dict):
                    module.set_processor(processor)
                else:
                    module.set_processor(processor.pop(f"{name}.processor"))

            for sub_name, child in module.named_children():
                fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)

        for name, module in self.named_children():
            fn_recursive_attn_processor(name, module, processor)


    def forward(
            self,
            hidden_states: torch.Tensor,
            encoder_hidden_states: torch.Tensor,
            timestep: Union[int, float, torch.LongTensor],
            timestep_cond: Optional[torch.Tensor] = None,
            image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
            attention_kwargs: Optional[Dict[str, Any]] = None,
            return_dict: bool = True,
            delta_cache_flag: int = 0, # 1 记录 delta_cache；2 应用 delta_cache；0 不记录也不应用
            delta_cache_start: int = 0, # 要通过 cache skip 的第一个 block
            delta_cache_end: int = 4,   # 要通过 cache skip 的最后一个 block
            cache = None,
            cnt = 0,

    ):
        global accumulated_rel_l1_distance
        global previous_modulated_input
        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if attention_kwargs is not None and attention_kwargs.get("scale", None) is not None:
                logger.warning(
                    "Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective."
                )
        
        batch_size, num_frames, channels, height, width = hidden_states.shape

        # 1. Time embedding
        timesteps = timestep
        t_emb = self.time_proj(timesteps)

        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=hidden_states.dtype)
        emb = self.time_embedding(t_emb, timestep_cond)

        # 2. Patch embedding
        hidden_states = self.patch_embed(encoder_hidden_states, hidden_states)
        hidden_states = self.embedding_dropout(hidden_states)
        
        text_seq_length = encoder_hidden_states.shape[1]
        encoder_hidden_states = hidden_states[:, :text_seq_length]
        hidden_states = hidden_states[:, text_seq_length:]
        # should_calc=True
        if cnt == 0 or cnt == 29:
            should_calc=True
            accumulated_rel_l1_distance = 0
        else:
            accumulated_rel_l1_distance,should_calc=get_should_calc(cnt,accumulated_rel_l1_distance,emb,previous_modulated_input)
            if should_calc ==True:
                accumulated_rel_l1_distance=0
        
        previous_modulated_input = emb
        
            # ori_hidden_states = hidden_states.clone()
            # ori_encoder_hidden_states = encoder_hidden_states.clone()
        # print_tensor_info("encoder_hidden_states:before teacache",encoder_hidden_states)
        # print_tensor_info("hidden_states:before teacache",hidden_states)
        if enable_teacache and not should_calc:
            # print("use cache: ",cnt)
            hidden_states =hidden_states+ cache["previous_residual"]
            encoder_hidden_states =encoder_hidden_states+ cache["previous_residual_encoder"]
        else:
            if enable_teacache and should_calc:
                ori_hidden_states=hidden_states
                ori_encoder_hidden_states=encoder_hidden_states
            # 3. Transformer blocks
            for i, block in enumerate(self.transformer_blocks):
                block = self.transformer_blocks[i]
                
                if i == delta_cache_start and delta_cache_flag == 1:
                    hidden_states_before = hidden_states
                    encoder_hidden_states_before = encoder_hidden_states
                    
                if i == delta_cache_start and delta_cache_flag == 2:
                    hidden_states += cache["hidden_states"]
                    encoder_hidden_states += cache["encoder_hidden_states"]
                    i = delta_cache_end + 1
                    break

                if torch.is_grad_enabled() and self.gradient_checkpointing:
                    hidden_states, encoder_hidden_states = self._gradient_checkpointing_func(
                        block,
                        hidden_states,
                        encoder_hidden_states,
                        emb,
                        image_rotary_emb,
                        attention_kwargs,
                    )
                else:
                    hidden_states, encoder_hidden_states = block(
                        hidden_states=hidden_states,
                        encoder_hidden_states=encoder_hidden_states,
                        temb=emb,
                        image_rotary_emb=image_rotary_emb,
                        timestep=timesteps if enable_pab() else None,
                        attention_kwargs=attention_kwargs,
                        attn_count1=cnt,
                        mlp_count1=cnt,
                    )

                if i == delta_cache_end and delta_cache_flag == 1:
                    cache["hidden_states"] = hidden_states - hidden_states_before
                    cache["encoder_hidden_states"] = encoder_hidden_states - encoder_hidden_states_before
                    # print("[DEBUG] delta_cache hidden_states norm:", cache["hidden_states"].abs().mean().item())
                    # print("[DEBUG] delta_cache encoder_hidden_states norm:", cache["encoder_hidden_states"].abs().mean().item())
            if enable_teacache and should_calc:
                cache["previous_residual"] = hidden_states - ori_hidden_states
                cache["previous_residual_encoder"] = encoder_hidden_states - ori_encoder_hidden_states

        hidden_states = self.norm_final(hidden_states)

        # 4. Final block
        hidden_states = self.norm_out(hidden_states, temb=emb)
        hidden_states = self.proj_out(hidden_states)

        # 5. Unpatchify
        p = self.config.patch_size
        output = hidden_states.reshape(batch_size, num_frames, height // p, width // p, -1, p, p)
        output = output.permute(0, 1, 4, 2, 5, 3, 6).flatten(5, 6).flatten(3, 4)

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)
        
        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)
