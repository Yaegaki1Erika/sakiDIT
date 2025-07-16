import torch
from model.conv import conv_forward

def test_conv_forward():
    # ✅ 所有 Tensor 转为 CUDA（Triton 只能处理 GPU 数据）
    device = torch.device("cuda")
    
    x = torch.randn(2, 3, 16, 16, device=device)  # 输入
    w = torch.randn(5, 3, 3, 3, device=device)    # 权重
    bias = torch.randn(5, device=device)         # 偏置

    # 调用你的卷积函数
    output = conv_forward(
        x,
        w,
        bias,
        stride=(2, 2),
        padding=(1, 1),
        dilation=(1, 1)
    )

    # 验证基本属性
    assert output is not None, "Output should not be None"
    assert output.dim() == 4, "Output should be 4D tensor"

    # 验证输出尺寸
    expected_h = (16 + 2*1 - 1*(3-1) - 1)//2 + 1
    expected_w = (16 + 2*1 - 1*(3-1) - 1)//2 + 1
    assert output.shape == (2, 5, expected_h, expected_w), \
        f"Wrong output shape. Expected (2,5,{expected_h},{expected_w}), got {output.shape}"

    # 测试不支持的功能
    assert conv_forward(x, w, bias, groups=2) is None, \
        "groups != 1 should return None"

    assert conv_forward(x, w, bias, transposed=True) is None, \
        "transposed=True should return None"

    print("✅ All tests passed!")

if __name__ == "__main__":
    test_conv_forward()
