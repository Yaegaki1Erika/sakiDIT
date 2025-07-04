import cv2
import math
import sys
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from diffusers.utils import export_to_video, load_image
from transformers import T5EncoderModel, T5Tokenizer
from model.pipeline import BaseImageToVideoPipeline
from model.scheduler import BaseDPMScheduler
from model.transformer import BaseTransformer3DModel
from model.vae import AutoencoderKLBase
import torch.cuda.nvtx as nvtx
# from torchao.quantization import autoquant

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

    # 默认配置，不可以修改
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

    transformer = BaseTransformer3DModel.from_pretrained(
        model_dir.as_posix(),
        torch_dtype=dtype,
        subfolder="transformer",
    )
    # transformer = torch.load("transformer_quantized_full.pt", weights_only=False)

    transformer.eval()

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

    target_i=0
    for i, (stem, da) in enumerate(data):
        # da=da.to(dtype=dtype, device=device)
        video = pipe(
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
        ).frames[0]
        # if not saved:
        #     # 一次性保存结构和权重
        #     torch.save(pipe.transformer, "transformer_quantized_full.pt")

        export_to_video(video, (output_dir / f'{stem}.mp4').as_posix(), fps=8)


if __name__ == '__main__':
    i2v()