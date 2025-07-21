import os
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

# 提取视频特征
def extract_features_from_directory(directory_path, video_names):
    features = {}
    for video_name in video_names:
        video_path = os.path.join(directory_path, f"{video_name}.mp4")
        print(f"Extracting features for {video_path}...")
        feat = extract_clip_video_feature(video_path)
        features[video_name] = feat
    return features

# 主函数
if __name__ == "__main__":
    gtt_dir = "./gtt"  # gtt 目录路径
    output_3s_dir = "./output_3s"  # output_3s 目录路径

    # 视频文件名列表（假设两个目录的文件名相同）
    video_names = ["1", "2", "3", "4", "5"]

    # 提取每个视频的特征
    print("Extracting features for gtt videos...")
    gtt_features = extract_features_from_directory(gtt_dir, video_names)

    print("Extracting features for output_3s videos...")
    output_3s_features = extract_features_from_directory(output_3s_dir, video_names)
    scores = []
    # 计算每一对视频的相似度
    for video_name in video_names:
        feat1 = gtt_features[video_name]
        feat2 = output_3s_features[video_name]
        score = compute_similarity(feat1, feat2)
        scores.append(score)
        print(f"\n🎯 Cosine similarity between {video_name} from gtt and output_3s: {score:.4f}")
    if scores:
        avg_score = sum(scores) / len(scores)
        print(f"\n📊 Average cosine similarity over {len(scores)} videos: {avg_score:.4f}")
    # 计算两个集合的平均特征相似度
    feat1_avg = torch.stack(list(gtt_features.values())).mean(dim=0)
    feat2_avg = torch.stack(list(output_3s_features.values())).mean(dim=0)
    score = compute_similarity(feat1_avg, feat2_avg)
    print(f"\n🎯 Cosine similarity between the average features of the two video sets: {score:.4f}")
