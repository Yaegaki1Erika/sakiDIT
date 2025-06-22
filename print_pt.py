import torch

model = torch.load("transformer_quantized_full.pt", weights_only=False)

for name, param in model.named_parameters():
    print(name, param.dtype)

