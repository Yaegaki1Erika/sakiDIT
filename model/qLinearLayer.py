import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import gemm_int8
from functools import partial
from .quant import quantize_tensor, quantize_tensor_channel_group
from diffusers.models.attention import Attention as DiffAttn
from diffusers.models.attention import FeedForward as DiffFF

def find_linear_layers(
    module: nn.Module,
    prefix: str = "",
    *,
    scope=("all",),          # ("attention", "ffn", "all")
):
    """
    返回 { '层路径': Linear对象 } 的字典，可按作用域过滤：
      scope = ("attention",) —— 仅找 Attention 中的 q/k/v/o 投影
      scope = ("ffn",)       —— 仅找 FeedForward / MLP 中的两层
      scope = ("all",)       —— 模型里所有 Linear
    """
    found = {}
    for name, child in module.named_children():
        path = f"{prefix}.{name}" if prefix else name

        if isinstance(child, nn.Linear):
            ok = False
            if "all" in scope:
                ok = True
            
            # 用路径字符串来粗判更简单：
            if (
                ok or
                ("attention" in scope and ".attn" in path.lower())
                or ("ffn" in scope and (".ff" in path.lower() or ".mlp" in path.lower()))
            ):
                found[path] = child

        # 继续向下递归
        found.update(find_linear_layers(child, path, scope=scope))
    return found

def replace_linear_with_quant(m, q_state_dict, prefix = ""):
    """
    遍历模型 m，把所有 nn.Linear 换成 QuantLinear。
    q_state_dict 必须包含同名的 weight_q / scale / zero 键。
    """
    for name, child in list(m.named_children()):
        key = f"{prefix}.{name}" if prefix else name
        if isinstance(child, nn.Linear):
            wq = q_state_dict[key + "_weight_q"]
            sc = q_state_dict[key + "_scale"]
            ze = q_state_dict[key + "_zero"]
            # 创建新的量化层，复制 bias（如果你保留了 bias）
            qlin = QuantLinear(child.in_features, child.out_features,
                               bias=child.bias is not None,
                               weight_q=wq, scale=sc, zero=ze)
            if child.bias is not None:
                qlin.bias.data.copy_(child.bias.data)
            setattr(m, name, qlin)        # 真正替换
            del child                     # 释放 float 权重显存
        else:
            replace_linear_with_quant(child, q_state_dict, key)
            
def replace_linear_skeleton(mod):
    """递归把 nn.Linear 换成空壳 QuantLinear，不依赖 state_dict。"""
    for name, child in list(mod.named_children()):
        if isinstance(child, nn.Linear):
            qlin = QuantLinear(child.in_features,
                               child.out_features,
                               bias=child.bias is not None)
            setattr(mod, name, qlin)
        else:
            replace_linear_skeleton(child)
            
@torch.no_grad()
def quantize_core(
        w: torch.tensor, 
        n_bits, group_size, tiling, sym, clip_ratio=1.0, 
        quant_type="int", quant_method="max"
    ) -> torch.tensor:
    savedShape = w.shape
    w = w.squeeze()
    if not w.is_contiguous():
        w = w.contiguous()
    assert w.is_contiguous(), "tensor should be continous for bitsandbytes kernel."

    if tiling > 0:
        assert False, "16x16 Block-wise Quantization is abandoned in this project."

    if group_size > 0:
        assert w.shape[-1] % group_size == 0
        w = w.reshape(-1, group_size) # row-major order

    assert w.dim() == 2, "Weight format should be: [num_groups, group_size]"
    assert n_bits < 16    
    
    assert quant_type == "int", "Options should be in [int, fp]"
    if quant_method == "max":
        if sym:
            w_max = w.abs().amax(dim=-1, keepdim=True).clamp(min=1e-5)
        else:
            w_max = w.amax(dim=-1, keepdim=True)
            w_min = w.amin(dim=-1, keepdim=True)

        if sym:
            q_max = (2**(n_bits-1)-1)
            q_min = (-2**(n_bits-1))
            if clip_ratio < 1.0:
                w_max = w_max * clip_ratio
            scales = w_max / q_max
            base = torch.zeros_like(scales)
        else:
            q_max = (2**(n_bits)-1)
            q_min = (0)
            if clip_ratio < 1.0:
                w_max *= clip_ratio
                w_min *= clip_ratio
            scales = (w_max-w_min).clamp(min=1e-5) / q_max
            base = torch.round(-w_min/scales).clamp_(min=q_min, max=q_max)
        w = (torch.clamp(torch.round(w / scales) + base, q_min, q_max) - base) * scales
    
    else:
        raise NotImplementedError
    
    return w.reshape(savedShape)
            
def quantize_activation(x: torch.tensor) -> torch.tensor:    
    qFunction = partial(
        quantize_core, 
        n_bits=8, 
        group_size=128, 
        tiling=0, 
        sym=False,
        clip_ratio=1.0,
        quant_type="int"
    )

    savedShape = x.shape
    x = x.reshape(-1, savedShape[-1])
    assert (savedShape[-1]) % 128 == 0
    
    x = qFunction(x)

    return x.view(savedShape)

class QuantLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True,
                 weight_q=None, scale=None, zero=None, group_size=128):
        super().__init__()
        self.in_features, self.out_features, self.group_size = in_features, out_features, group_size

        # 只保留 bias（如果有），权重改成 buffer
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

        # int8 权重 + 量化参数
        self.num_groups = Ng = math.ceil(in_features / group_size)        # 分组数
        if weight_q is not None:
            self.register_buffer("weight_q", weight_q)
        else:
            self.register_buffer("weight_q", torch.empty(out_features, in_features, dtype=torch.int8))
        if scale is not None:
            self.register_buffer("scale", scale)
        else:
            self.register_buffer("scale", torch.empty(out_features, Ng, dtype=torch.float16))
        if zero is not None:
            self.register_buffer("zero", zero)
        else:
            self.register_buffer("zero", torch.empty(out_features, Ng, dtype=torch.float16))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (N, in_features) or (batch..., in_features)

        步骤：
        1.  reshape uint8 权重  → (O, Ng, G)
        2.  broadcast 反量化  → float32
        3.  还原形状         → (O, I)
        4.  F.linear
        """
        # breakpoint()
        # quantize_activation(x)
        O, I, Ng, G = self.out_features, self.in_features, self.num_groups, self.group_size

        padded_I = Ng * G                    # = ceil(I/G) * G
        need_pad = I != padded_I             # 是否有余数
        w_q = (self.weight_q.to(torch.int16) & 0xFF).to(torch.uint8)   # 0…255
        # (O, I) → (O, padded_I)  (如果 I 本来就整除，会直接返回原 tensor)
        if need_pad:
            w_q = F.pad(w_q, (0, padded_I - I))  # 右侧 0 pad
        else:
            w_q = w_q

        # (O, padded_I) → (O, Ng, G)
        w_q = w_q.view(O, Ng, G)

        # 反量化: (O, Ng, 1) broadcast → (O, Ng, G)
        w_fp = self.scale[:, :, None] * (w_q.to(torch.float16) - self.zero[:, :, None])

        # 还原 → (O, padded_I) 并按需裁掉多余列
        w_fp = w_fp.view(O, padded_I)
        if need_pad:
            w_fp = w_fp[:, :I]

        return F.linear(x, w_fp, self.bias)
    
    # def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (N, in_features) or (batch..., in_features)

        步骤：
        1.  量化权重 → int8
        2.  broadcast 反量化  → float32
        3.  还原形状         → (O, I)
        4.  F.linear
        """
        # gemm_int8.matmul(a, b, alpha=1.0)  # Returns bfloat16 tensor of (a @ b.t()) * alpha

    def extra_repr(self):   # 方便 print(model)
        return (f"in_features={self.in_features}, out_features={self.out_features}, "
                f"int8_weight_shape={tuple(self.weight_q.shape)}")


def find_qlinear_layers(module, name=''):
    if type(module) == QLinearLayer:
        if module.enable_quant:
            return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_qlinear_layers(
            child, name=name + '.' + name1 if name != '' else name1
        ))
    return res

class QLinearLayer(nn.Module):
    def __init__(
        self,
        originalLayer: nn.Linear,
        args,
        enable_quant: bool = True
    ):
        super().__init__()
        self.args = args
        self.register_buffer('weight', originalLayer.weight.data)
        self.enable_quant = enable_quant # whether to allow quant on weights, default True
        if originalLayer.bias is not None:
            self.register_buffer('bias', originalLayer.bias.data)
        else:
            self.bias = None
        self.quantized = False
        
    @torch.no_grad()
    def forward(self, x):
        y = torch.functional.F.linear(x, self.weight, self.bias)
        return y
    
    def to(self, *args, **kwargs):
        super(QLinearLayer, self).to(*args, **kwargs)
        self.weight = self.weight.to(*args, **kwargs)
        return self
    
    @torch.no_grad()
    def quant(self):
        if self.args.wbits >= 16:
            return

        self.weight = quantize_tensor_channel_group(
            self.weight.clone(), 
            n_bits=self.args.wbits,
            exponential=self.args.exponential, 
            sym=self.args.w_sym,
            group_size=self.args.weight_group_size,
            channel_group=self.args.weight_channel_group,
            clip_ratio=self.args.w_clip_ratio,
            tiling=self.args.tiling,
            quant_type=self.args.quant_type,
            quant_method=self.args.quant_method,
        )

        self.quantized = True
        return
    
    def extra_repr(self):
        return f'wbit={self.args.wbits}, sym={self.args.w_sym}, group_size={self.args.weight_group_size}, channel_group={self.args.weight_channel_group}, quantized={self.quantized}'