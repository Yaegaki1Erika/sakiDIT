import torch
import cv2
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torch.nn.functional as F

# 加载 CLIP 模型
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").eval().to(device)
clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# 提取视频的 CLIP 特征（抽帧 + 平均池化）
def extract_clip_video_feature(video_path, frame_interval=30):
    cap = cv2.VideoCapture(video_path)
    features = []
    idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if idx % frame_interval == 0:
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            inputs = clip_proc(images=img, return_tensors="pt").to(device)
            with torch.no_grad():
                feat = clip_model.get_image_features(**inputs)  # shape: [1, 512]
            features.append(feat)
        idx += 1

    cap.release()
    if len(features) == 0:
        raise ValueError("No frames extracted.")
    features = torch.cat(features, dim=0)  # shape: [N, 512]
    return features.mean(dim=0)  # shape: [512]

# 计算两个向量的余弦相似度
def compute_similarity(feat1, feat2):
    feat1 = F.normalize(feat1, dim=0)
    feat2 = F.normalize(feat2, dim=0)
    return torch.dot(feat1, feat2).item()

# 主函数
if __name__ == "__main__":
    video1_path = "./gtt/1.mp4"  # 替换成你的视频路径
    video2_path = "./output_3s/1.mp4"

    print("Extracting features for Video 1...")
    feat1 = extract_clip_video_feature(video1_path)

    print("Extracting features for Video 2...")
    feat2 = extract_clip_video_feature(video2_path)

    score = compute_similarity(feat1, feat2)
    print(f"\n🎯 Cosine similarity between the two videos: {score:.4f}")
