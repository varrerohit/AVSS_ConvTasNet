#!/bin/bash

# Example execution script
python demo_convtasnet.py \
    --input ../../demo1/mix.mp4 \
    --sdk-path /Users/rohit/Desktop/BMBAutomotive \
    --license-path /Users/rohit/Desktop/BMBAutomotive/license.key \
    --audio-weights ../checkpoints/Wujian-Model-Baseline/epoch=115.ckpt \
    --video-weights ../model/Wujian_Model/video_resnet18.pt
