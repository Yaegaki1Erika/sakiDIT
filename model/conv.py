import torch
import triton
import triton.language as tl

def conv_heuristics():
    configs = [
        triton.Config({
            "BLOCK_M": 128,
            "BLOCK_N": 128,
            "BLOCK_K": 32
        },
                      num_stages=2,
                      num_warps=8),
        triton.Config({
            "BLOCK_M": 256,
            "BLOCK_N": 64,
            "BLOCK_K": 32
        },
                      num_stages=2,
                      num_warps=8),
        triton.Config({
            "BLOCK_M": 256,
            "BLOCK_N": 32,
            "BLOCK_K": 32
        },
                      num_stages=4,
                      num_warps=4),
        triton.Config({
            "BLOCK_M": 256,
            "BLOCK_N": 32,
            "BLOCK_K": 64
        },
                      num_stages=4,
                      num_warps=4),
        triton.Config({
            "BLOCK_M": 256,
            "BLOCK_N": 16,
            "BLOCK_K": 32
        },
                      num_stages=4,
                      num_warps=2),
        triton.Config({
            "BLOCK_M": 64,
            "BLOCK_N": 128,
            "BLOCK_K": 32
        },
                      num_stages=4,
                      num_warps=8),
        triton.Config({
            "BLOCK_M": 128,
            "BLOCK_N": 64,
            "BLOCK_K": 32
        },
                      num_stages=4,
                      num_warps=4),
        triton.Config({
            "BLOCK_M": 64,
            "BLOCK_N": 64,
            "BLOCK_K": 32
        },
                      num_stages=4,
                      num_warps=4),
        triton.Config({
            "BLOCK_M": 128,
            "BLOCK_N": 16,
            "BLOCK_K": 32
        },
                      num_stages=4,
                      num_warps=4),
        triton.Config({
            "BLOCK_M": 128,
            "BLOCK_N": 128,
            "BLOCK_K": 128
        },
                      num_stages=3,
                      num_warps=8),
        triton.Config({
            "BLOCK_M": 256,
            "BLOCK_N": 64,
            "BLOCK_K": 128
        },
                      num_stages=3,
                      num_warps=8),
        triton.Config({
            "BLOCK_M": 256,
            "BLOCK_N": 32,
            "BLOCK_K": 128
        },
                      num_stages=4,
                      num_warps=4),
        triton.Config({
            "BLOCK_M": 64,
            "BLOCK_N": 128,
            "BLOCK_K": 128
        },
                      num_stages=4,
                      num_warps=4),
        triton.Config({
            "BLOCK_M": 128,
            "BLOCK_N": 64,
            "BLOCK_K": 128
        },
                      num_stages=4,
                      num_warps=4),
        triton.Config({
            "BLOCK_M": 128,
            "BLOCK_N": 32,
            "BLOCK_K": 64
        },
                      num_stages=4,
                      num_warps=2),
        triton.Config({
            "BLOCK_M": 64,
            "BLOCK_N": 64,
            "BLOCK_K": 64
        },
                      num_stages=4,
                      num_warps=2),
        # triton.Config(
        #     {"BLOCK_M": 128, "BLOCK_N": 16, "BLOCK_K": 64}, num_stages=4, num_warps=2
        # ),
    ]
    key = [
        "BATCH",
        "IN_C",
        "IN_H",
        "IN_W",
        "KERNEL_N",
        "KERNEL_H",
        "KERNEL_W",
        "OUT_H",
        "OUT_W",
        # parameters of conv
        "stride_h",
        "stride_w",
        "padding_h",
        "padding_w",
        "dilation_h",
        "dilation_w",
        "output_padding_h",
        "output_padding_w",
        "groups",
    ]
    prune_configs_by = {
        "early_config_prune": early_config_prune,
        "perf_model": estimate_conv_time,
        "top_k": 10,
    }
    return triton.autotune(configs, key, prune_configs_by=prune_configs_by)



class _conv:
    kernel = _kernel_delta_x_hwc

    # for the contigous order of w ptr, what"s the corresponding
    # ptr changes for x in a sliding window
    @staticmethod
    def _delta_x_ptr_hwc(
        IN_C,
        KERNEL_H,
        KERNEL_W,
        dilation_h,
        dilation_w,
        stride_wc,
        stride_wh,
        stride_ww,
        stride_xc,
        stride_xh,
        stride_xw,
        device,
    ):
        # get the order of axes in w, innermost dimension outward
        stride_w_3d = [stride_wc, stride_wh, stride_ww]
        order = sorted(range(len(stride_w_3d)), key=stride_w_3d.__getitem__)
        window_size = IN_C * KERNEL_H * KERNEL_W

        r_window = torch.arange(0, window_size, 1, device=device)
        window_unpack = _unpack(r_window, order, [IN_C, KERNEL_H, KERNEL_W])
        window_unpack_c = window_unpack[order[0]]
        window_unpack_h = window_unpack[order[1]]
        window_unpack_w = window_unpack[order[2]]
        r_dilation_h = dilation_h * window_unpack_h
        r_dilation_w = dilation_w * window_unpack_w
        r_inc = window_unpack_c
        # delta_x = (
        #     r_dilation_h * stride_xh + r_dilation_w * stride_xw + r_inc * stride_xc
        # )
        # return delta_x
        return (
            r_dilation_h,
            r_dilation_w,
            r_inc,
        )

    @staticmethod
    def _delta_x_ptr(
        IN_C,
        KERNEL_H,
        KERNEL_W,
        dilation_h,
        dilation_w,
        stride_wc,
        stride_wh,
        stride_ww,
        stride_xc,
        stride_xh,
        stride_xw,
        device,
    ):
        # get the order of axes in w, innermost dimension outward
        stride_w_3d = [stride_wc, stride_wh, stride_ww]
        order = sorted(range(len(stride_w_3d)), key=stride_w_3d.__getitem__)
        window_size = IN_C * KERNEL_H * KERNEL_W

        r_window = torch.arange(0, window_size, 1, device=device)
        window_unpack = _unpack(r_window, order, [IN_C, KERNEL_H, KERNEL_W])
        window_unpack_c = window_unpack[order[0]]
        window_unpack_h = window_unpack[order[1]]
        window_unpack_w = window_unpack[order[2]]
        r_dilation_h = dilation_h * window_unpack_h
        r_dilation_w = dilation_w * window_unpack_w
        r_inc = window_unpack_c
        delta_x = (r_dilation_h * stride_xh + r_dilation_w * stride_xw +
                   r_inc * stride_xc)
        return delta_x

    @staticmethod
    def _call(
        x,
        w,
        bias,
        stride,
        padding,
        dilation,
        transposed,
        output_padding,
        groups,
    ):
        # Q: should we check x, w, bias dtypes?
        device = x.device
        # input shapes
        shape_x = x.shape
        shape_w = w.shape
        shape_bias = bias.shape if bias is not None else None

        # indicies for the layout
        xn, xc, xh, xw = 0, 1, 2, 3
        yn, yc, yh, yw = 0, 1, 2, 3
        wn, wc, wh, ww = 0, 1, 2, 3

        # out_channel, in_channel, kernel_height, kernel_width
        kernel_size = [shape_w[wh], shape_w[ww]]
        input_size = [shape_x[xh], shape_x[xw]]
        assert (not shape_bias or shape_bias[0] == shape_w[wn]
                ), f"bias shape did not match{shape_bias} != {shape_w[wn]}"
        in_channel = shape_w[wc] * groups

        assert shape_x[
            xc] % groups == 0, "in_channels must be divisible by groups"
        assert shape_w[
            wn] % groups == 0, "out_channels must be divisible by groups"
        assert (shape_x[xc] == in_channel
                ), f"in_channel did not match {shape_x[xc]} != {in_channel}"

        assert (len(stride) == len(padding) == len(dilation) ==
                len(output_padding) == len(kernel_size) == len(input_size))

        # output shape
        shape_y = [0] * 4
        shape_y[yn] = shape_x[xn]
        shape_y[yc] = shape_w[wn]
        shape_y[yh] = (input_size[0] + 2 * padding[0] - dilation[0] *
                       (kernel_size[0] - 1) - 1 +
                       stride[0]) // stride[0] + 2 * output_padding[0]
        shape_y[yw] = (input_size[1] + 2 * padding[1] - dilation[1] *
                       (kernel_size[1] - 1) - 1 +
                       stride[1]) // stride[1] + 2 * output_padding[1]

        BATCH = shape_x[xn]
        IN_C = shape_x[xc]
        IN_H = shape_x[xh]
        IN_W = shape_x[xw]
        KERNEL_N = shape_w[wn]
        KERNEL_H = shape_w[wh]
        KERNEL_W = shape_w[ww]
        OUT_H = shape_y[yh]
        OUT_W = shape_y[yw]

        # get strides for tensors
        stride_x = x.stride()
        stride_w = w.stride()
        with_bias = bias is not None
        if with_bias:
            bias = bias.contiguous()
        else:
            bias = None

        # output layout should be the same as x
        if stride_x[xc] < stride_x[xh] and stride_x[xc] < stride_x[xw]:
            memory_format = torch.channels_last
        else:
            memory_format = torch.contiguous_format
        # allocate output
        y = torch.empty(shape_y,
                        device=device,
                        dtype=x.dtype,
                        memory_format=memory_format)
        stride_y = y.stride()

        # allocate tmp
        # WINDOW_SIZE = KERNEL_H * KERNEL_W * IN_C
        # tmp_x = torch.empty((BATCH * OUT_H * OUT_W, WINDOW_SIZE), device=device, dtype=x.dtype)
        # tmp_w = torch.empty((WINDOW_SIZE, KERNEL_N), device=device, dtype=w.dtype)
        # accumulator types
        x_dtype = x.dtype
        if x_dtype in (
                torch.float32,
                torch.bfloat16,
        ):
            ACC_TYPE = tl.float32
        elif x_dtype in (torch.float16, ):
            ACC_TYPE = tl.float16
        elif x_dtype in (torch.float64, ):
            ACC_TYPE = tl.float64
        else:
            ACC_TYPE = tl.int32
        # ACC_TYPE = (tl.float32 if x.dtype in [
        #     torch.float16, torch.bfloat16, torch.float32
        # ] else tl.int32)
        # if stride_x[xc] == 1 and stride_x > 1 and stride_y > 1:
        CONV1X1_NHWC = False
        if stride_x[xc] == 1 and KERNEL_H == 1 and KERNEL_W == 1:
            CONV1X1_NHWC = True
        #  do we need delta x ptr for h, w, c dimension each or not
        DELTA_X_PTR_HWC = (False if
                           ((padding[0] == 0 and padding[1] == 0) or
                            (KERNEL_H == 1 and KERNEL_W == 1)) else True)
        if not CONV1X1_NHWC:
            if DELTA_X_PTR_HWC:
                delta_xh, delta_xw, delta_xc = _conv._delta_x_ptr_hwc(
                    IN_C,
                    KERNEL_H,
                    KERNEL_W,
                    dilation[0],
                    dilation[1],
                    stride_w[wc],
                    stride_w[wh],
                    stride_w[ww],
                    stride_x[xc],
                    stride_x[xh],
                    stride_x[xw],
                    device,
                )
            else:
                delta_x = _conv._delta_x_ptr(
                    IN_C,
                    KERNEL_H,
                    KERNEL_W,
                    dilation[0],
                    dilation[1],
                    stride_w[wc],
                    stride_w[wh],
                    stride_w[ww],
                    stride_x[xc],
                    stride_x[xh],
                    stride_x[xw],
                    device,
                )
        else:
            delta_x = None
            delta_xh, delta_xw, delta_xc = None, None, None

        # launch kernel, 2-dim, batch*h*w, kernel
        def grid(META):
            return (
                triton.cdiv(BATCH * OUT_H * OUT_W, META["BLOCK_M"]),
                triton.cdiv(KERNEL_N, META["BLOCK_N"]),
            )

        # conv1x1 or padding==0
        if CONV1X1_NHWC or not DELTA_X_PTR_HWC:
            _kernel_delta_x[grid](
                x,
                w,
                bias,
                y,
                # stride nchw for x,w,y tensor
                stride_x[xn],
                stride_x[xc],
                stride_x[xh],
                stride_x[xw],
                stride_w[wn],
                stride_w[wc],
                stride_w[wh],
                stride_w[ww],
                stride_y[yn],
                stride_y[yc],
                stride_y[yh],
                stride_y[yw],
                # pointer inc for x
                delta_x,
                # Tensor dimensions
                BATCH,
                IN_C,
                IN_H,
                IN_W,
                KERNEL_N,
                KERNEL_H,
                KERNEL_W,
                OUT_H,
                OUT_W,
                # conv parameters
                stride[0],
                stride[1],
                padding[0],
                padding[1],
                dilation[0],
                dilation[1],
                output_padding[0],
                output_padding[1],
                groups,
                # Metaparameters
                ACC_TYPE=ACC_TYPE,
                CONV1X1_NHWC=CONV1X1_NHWC,
                # BLOCK_M=128,
                # BLOCK_N=32,
                # BLOCK_K=32,
                GROUP_H=1,
                WITH_BIAS=with_bias,
            )
        # need to know ptr update for each dimension to check if
        # the sliding window is out of bounds
        else:
            # kernel = _kernel_delta_x_hwc
            _kernel_delta_x_hwc[grid](
                x,
                w,
                bias,
                y,
                # stride nchw for x,w,y tensor
                stride_x[xn],
                stride_x[xc],
                stride_x[xh],
                stride_x[xw],
                stride_w[wn],
                stride_w[wc],
                stride_w[wh],
                stride_w[ww],
                stride_y[yn],
                stride_y[yc],
                stride_y[yh],
                stride_y[yw],
                # pointer inc for x
                delta_xh,
                delta_xw,
                delta_xc,
                # Tensor dimensions
                BATCH,
                IN_C,
                IN_H,
                IN_W,
                KERNEL_N,
                KERNEL_H,
                KERNEL_W,
                OUT_H,
                OUT_W,
                # conv parameters
                stride[0],
                stride[1],
                padding[0],
                padding[1],
                dilation[0],
                dilation[1],
                output_padding[0],
                output_padding[1],
                groups,
                # Metaparameters
                ACC_TYPE=ACC_TYPE,
                CONV1X1_NHWC=CONV1X1_NHWC,
                # BLOCK_M=128,
                # BLOCK_N=32,
                # BLOCK_K=32,
                GROUP_H=1,
                WITH_BIAS=with_bias,
            )

        # if bias is not None:
        #     if len(bias.shape) == 1:
        #         bias = bias.reshape([1, bias.shape[0], 1, 1])
        #     y += bias
        return y

    @staticmethod
    def forward(
            x,
            w,
            bias,
            stride=(1, 1),
            padding=(0, 0),
            dilation=(1, 1),
            transposed=False,
            output_padding=(0, 0),
            groups=1,
    ):
        if groups != 1:
            print(f"Do not support groups = {groups}")
            return
        if transposed:
            print("Do not support transposed")
        return _conv._call(
            x,
            w,
            bias,
            stride,
            padding,
            dilation,
            transposed,
            output_padding,
            groups,
        )

