import cv2
import math
import sys
import torch
import numpy as np
from PIL import Image
from pathlib import Path
# from accelerate import init_empty_weights
# from safetensors.torch import load_file
from diffusers.utils import export_to_video, load_image
from transformers import T5EncoderModel, T5Tokenizer
from model.pipeline import BaseImageToVideoPipeline
from model.scheduler import BaseDPMScheduler
from model.transformer import BaseTransformer3DModel
from model.vae import AutoencoderKLBase
# import torch_tensorrt
# from model.qLinearLayer import replace_linear_skeleton
# from torchao.quantization import autoquant
# torch.cuda.set_per_process_memory_fraction(40 / 96, device=0)
# import time
total_vedio_num=0
vae_batch_size=4
def load_video(video_path: str, new_fps: int = 8):
    cap = cv2.VideoCapture(video_path)
    old_fps = int(cap.get(cv2.CAP_PROP_FPS))
    step = max(1, math.ceil(old_fps / new_fps))

    cnt = 0
    frames = []
    while cap.isOpened():
        cnt += 1
        ret, frame = cap.read()
        if not ret:
            break
        if cnt % step != 0:
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)

    cap.release()
    return frames


def resize_and_crop_image(image, target_size):
    w, h = image.size

    if (h > w and target_size[0] < target_size[1]) or (h < w and target_size[0] > target_size[1]):
        target_size = target_size[::-1]

    scale = min(target_size[0] / h, target_size[1] / w)
    new_size = (int(w * scale), int(h * scale))
    resized_image = image.resize(new_size, Image.LANCZOS)
    start_x = (resized_image.size[0] - target_size[1]) // 2
    start_y = (resized_image.size[1] - target_size[0]) // 2
    cropped_image = resized_image.crop((start_x, start_y, start_x + target_size[1], start_y + target_size[0]))

    return cropped_image


def resize_and_crop_video(video, target_size):
    resized_frames = []

    for frame in video:
        h, w, _ = frame.shape

        if (h > w and target_size[0] < target_size[1]) or (h < w and target_size[0] > target_size[1]):
            target_size = target_size[::-1]

        scale = min(target_size[0] / h, target_size[1] / w)
        new_size = (int(w * scale), int(h * scale))
        pil_image = Image.fromarray(frame)
        resized_image = pil_image.resize(new_size, Image.LANCZOS)
        start_x = (resized_image.size[0] - target_size[1]) // 2
        start_y = (resized_image.size[1] - target_size[0]) // 2
        cropped_image = resized_image.crop((start_x, start_y, start_x + target_size[1], start_y + target_size[0]))
        resized_frames.append(np.array(cropped_image))

    return resized_frames


@torch.inference_mode()
def i2v():
    assert len(sys.argv) == 5

    video_input_dir = Path(sys.argv[1])
    image_input_dir = Path(sys.argv[2])
    output_dir = Path(sys.argv[3])
    model_dir = Path(sys.argv[4])

    # 
    

    # h, w
    image_size = (720, 1280)
    base_height = 720
    base_width = 1280
    num_frames = 25

    negative_prompt = "Generate videos rife with blurry, indistinct visuals lacking clarity, where characters and objects exhibit exaggerated, mismatched proportions, leading to a chaotic spatial incoherence. Twist everyday interactions into disturbing spectacles, with an emphasis on the horrifying deformation of hands—fingers that are disgustingly elongated, shortened, or contorted into nightmarish configurations. These hands should look utterly alien, with joints bending in impossible ways to evoke a deep sense of horror and discomfort. The footage should include abrupt appearances and disappearances of elements, disrupting continuity. Transitions and movements should be jarring, compounded by shaky, unsteady camera work and unattractive compositions that fail to please aesthetically. Expect gross anatomical distortions with twisted bodies, hands, and faces, alongside strange, illogical interactions that defy spatial logic and physical reality, all culminating in a visually displeasing and disorienting viewing experience."

    data = []
    for p in video_input_dir.glob('*.mp4'):
        ipath = image_input_dir / (p.stem + '.jpg')
        video = load_video(p.as_posix())
        image = load_image(ipath.as_posix())

        resized_frames = resize_and_crop_video(video, image_size)
        video_array_init = np.stack(resized_frames, axis=0)[:num_frames]
        video_array = video_array_init.transpose(0, 3, 1, 2)
        video_tensor = torch.from_numpy(video_array)
        video_tensor = video_tensor.unsqueeze(0)
        video_tensor = (video_tensor - 127.5) / 127.5

        image = resize_and_crop_image(image, image_size)
        image_array = np.array(image).transpose(2, 0, 1)
        image_array = (image_array - 127.5) / 127.5
        image_tensor = torch.from_numpy(image_array).type_as(video_tensor)
        image_tensor = image_tensor.unsqueeze(0).unsqueeze(0)
        video_tensor[:, 0, :, :, :] = image_tensor[:, 0, :, :, :]
        data.append((p.stem, video_tensor))

    # make sure the order, fixed seed
    data = sorted(data, key=lambda x: x[0])

    device = torch.device('cuda:0')
    dtype = torch.float16
    generator = torch.Generator(device=device).manual_seed(42)

    tokenizer = T5Tokenizer.from_pretrained(
        model_dir.as_posix(),
        trust_remote_code=True,
        subfolder="tokenizer",
    )

    text_encoder = T5EncoderModel.from_pretrained(
        model_dir.as_posix(),
        torch_dtype=dtype,
        subfolder="text_encoder",
    )
    text_encoder.eval()

    vae = AutoencoderKLBase.from_pretrained(
        model_dir.as_posix(),
        torch_dtype=dtype,
        subfolder="vae",
    )
    vae.eval()
    # vae = torch.compile(vae, mode="reduce-overhead", fullgraph=False)
    # vae=torch.compile(vae)

    transformer = BaseTransformer3DModel.from_pretrained(
        model_dir.as_posix(),
        torch_dtype=dtype,
        subfolder="transformer",
    )
    transformer.eval()

    # ckpt = "./cti/diffusion_pytorch_model.safetensors"

    # # 1. 零显存创建网络骨架
    # with init_empty_weights():
    #     config = BaseTransformer3DModel.load_config(
    #         "./cti/config.json"
    #     )
    #     transformer  = BaseTransformer3DModel.from_config(config)

    # # 2. 把 nn.Linear → QuantLinear
    # for block in transformer.transformer_blocks:
    #     replace_linear_skeleton(block)

    # # 3. 把量化权重真正 load 进来
    # state_dict = load_file(ckpt)
    # missing, unexpected = transformer.load_state_dict(state_dict, strict=False, assign=True)   # buffers 已匹配
    # assert not missing and not unexpected, f"state_dict mismatch: {missing=}, {unexpected=}"

    # transformer.to("cuda").eval()

    scheduler = BaseDPMScheduler(
        clip_sample=False,
        prediction_type='v_prediction',
        rescale_betas_zero_snr=True,
        snr_shift_scale=1.0,
        timestep_spacing='trailing',
    )

    pipe = BaseImageToVideoPipeline(
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        vae=vae,
        transformer=transformer,
        scheduler=scheduler,
    )
    
    prompt = ""
    pipe.vae.enable_tiling()
    pipe.to(dtype=dtype, device=device)
    # pipe.vae.to(dtype=dtype,device=device)
    # pipe.text_encoder.to(dtype=dtype, device=device)
    # pipe.transformer = autoquant(pipe.transformer, error_on_unseen=False)
    # torch.save(transformer.state_dict(), "transformer_quantized.pth")
    # torch.save(pipe.transformer, "transformer_quantized_quto.pt")
    round_times = []
    stem_batch = []
    target_i=0
    global total_vedio_num,vae_batch_size
    total_vedio_num=len(data)
    tmp_stem=0
    for i, (stem, da) in enumerate(data):
        stem_batch.append(stem)

        # if i!=target_i:
        #     continue
        # da=da.to(dtype=dtype, device=device)
        # if 1 <= i <= 4:
        #     t0 = time.perf_counter()
        videos = pipe(
            pose_video=da,
            height=image_size[0],
            width=image_size[1],
            base_height=base_height,
            base_width=base_width,
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_videos_per_prompt=1,
            num_inference_steps=30, # num_inference_steps不可更改，否则为0分
            num_frames=num_frames,
            guidance_scale=6,
            generator=generator,
            use_dynamic_cfg=True,
            total_vedio_num=total_vedio_num,
        )
        # if not saved:
        #     # 一次性保存结构和权重
        #     torch.save(pipe.transformer, "transformer_quantized_full.pt")
        if (i!=0 and (i+1)%vae_batch_size==0) or i+1==total_vedio_num:
            # print("saki111")
            for video, stem in zip(videos, stem_batch):
                # print("saki!!!")
                stem_now=stem_batch[tmp_stem]
                tmp_stem+=1
                export_to_video(video[0], (output_dir / f'{stem_now}.mp4').as_posix(), fps=8)
        # export_to_video(video, (output_dir / f'{stem}.mp4').as_posix(), fps=8)
        # if 1 <= i <= 4:
        #     t1 = time.perf_counter()
        #     round_times.append(t1 - t0)

    # if round_times:
    #     print(f"Rounds 2–4 total time: {sum(round_times):.2f}s")
    #     for idx, t in enumerate(round_times, start=2):
    #         print(f"Round {idx} time: {t:.2f}s")



if __name__ == '__main__':
    i2v()