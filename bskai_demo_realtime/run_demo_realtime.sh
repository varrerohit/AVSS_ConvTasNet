#!/bin/bash

# Example execution script for real-time ConvTasNet
/Users/rohit/miniconda3/envs/avss/bin/python demo_realtime_convtasnet.py \
    --input ./input_videos/mix_main.mp4 \
    --output ./outputs/mix_main_convtasnet \
    --sdk-path /Users/rohit/Desktop/SDKs/BSocial/BM.BSocial-3.1.0-2025-11-11-16-47-build-8f1f318-darwin-arm64 \
    --license-path /Users/rohit/Desktop/SDKs/licence_offline_rnd.bskai \
    --config ../config/train.yml \
    --audio-weights ../checkpoints/lrs2_model.ckpt \
    --video-weights ../model/video_resnet18.pt
