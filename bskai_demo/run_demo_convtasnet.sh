#!/bin/bash

# Example execution script
/Users/rohit/miniconda3/envs/avss/bin/python demo_convtasnet.py \
    --input ../../dolphin/bskai_demo/demo_videos/mix_main.mp4 \
    --sdk-path /Users/rohit/Desktop/SDKs/BSocial/BM.BSocial-3.1.0-2025-11-11-16-47-build-8f1f318-darwin-arm64 \
    --license-path /Users/rohit/Desktop/SDKs/licence_offline_rnd.bskai \
    --audio-weights ../checkpoints/Wujian-Model-Baseline/epoch=115.ckpt \
    --video-weights ../model/video_resnet18.pt
