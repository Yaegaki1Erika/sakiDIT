#!/bin/bash

# 输入视频文件夹
if [ -z "$VIDEO_INPUT_DIR" ]; then
  VIDEO_INPUT_DIR="dataset/videos"
fi
# 输入图片文件夹
if [ -z "$IMAGE_INPUT_DIR" ]; then
  IMAGE_INPUT_DIR="dataset/images"
fi
# 输出视频文件夹
if [ -z "$OUPUT_DIR" ]; then
  OUPUT_DIR="output_3s"
fi
# 模型权重文件夹
if [ -z "$WEIGHTS" ]; then
  WEIGHTS="dataset/Base-I2V"
fi

which python

MODEL_PATH=${WEIGHTS}

mkdir -p $OUPUT_DIR

#运行推理脚本
python base.py \
  $VIDEO_INPUT_DIR \
  $IMAGE_INPUT_DIR \
  $OUPUT_DIR \
  $MODEL_PATH
