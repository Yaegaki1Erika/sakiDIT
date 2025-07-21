from vae import *
import torch
import torch_tensorrt
from pathlib import Path

# 加载预训练模型
model_dir = Path("/root/eval/dataset/Base-I2V")
vae = AutoencoderKLBase.from_pretrained(
    model_dir.as_posix(),
    torch_dtype=torch.float16,
    subfolder="vae",
)

# 设置模型为评估模式并将其转到 CUDA
model = vae.eval().cuda()
model = model.encoder  # 获取解码器


inputs = torch_tensorrt.Input(
    min_shape=[1, 3, 8, 240, 128],   
    opt_shape=[1, 3, 8, 240, 360],    
    max_shape=[1, 3, 9, 240, 384],    
    dtype=torch.float16
)


# 使用 Torch-TensorRT 编译模型
trt_gm = torch_tensorrt.compile(model, inputs=[inputs])  # 这里的 inputs 传入的是一个列表

# 模型推理
input_tensor = torch.randn(1, 3, 8, 240, 128, dtype=torch.float16).cuda()  # 用随机数据来推理
output = trt_gm(input_tensor)
print("saki")
print(output)
