# AV-ConvTasNet: Time Domain Audio Visual Speech Separation

> **⚠️ CRITICAL WARNING: Official Pretrained Weights Issue**
>
> The pretrained model links provided in the official repository are currently incorrect. The Google Drive links for LRS2, LRS3, and VoxCeleb2 point to **CTCNet** weights instead of **AV-ConvTasNet** weights. 
>
> As a result, loading those weights into this architecture will fail or result in pure noise. Our inference scripts in `bskai_demo/` and `bskai_demo_realtime/` will explicitly throw a `RuntimeError` if you attempt to use these mismatched weights.
>
> **Relevant Links:**
> - **Official Repo:** [JusperLee/AV-ConvTasNet](https://github.com/JusperLee/AV-ConvTasNet)
> - **Issue Tracker:** [Track resolution or report here](https://github.com/JusperLee/AV-ConvTasNet/issues)

---

## Project Overview

This repository contains an implementation of the Time Domain Audio Visual Speech Separation (AV-ConvTasNet) algorithm. It utilizes the **BlueSkeye API** for robust face tracking and detection, enabling target speaker extraction from monaural mixtures using visual cues.

## Inference Scripts

We provide two primary inference demos optimized for Apple Silicon (MPS) and general deployment:

1.  **Offline / Post-Processing:** `bskai_demo/demo_convtasnet.py`
    - Designed for high-accuracy processing of pre-recorded video files.
2.  **Real-Time / Streaming:** `bskai_demo_realtime/demo_realtime_convtasnet.py`
    - Optimized for low-latency live feed simulation.

For detailed technical specifications, architecture diagrams, and hyperparameter decisions, please refer to [INFERENCE_SPECS.md](./INFERENCE_SPECS.md).

## Usage

To run the demos, ensure you have the BlueSkeye SDK installed and configured. 

### Offline Demo
```bash
cd bskai_demo
./run_demo_convtasnet.sh
```

### Real-Time Demo
```bash
cd bskai_demo_realtime
./run_demo_realtime.sh
```

## Data Preparation & Training

Because the official pretrained weights are currently unavailable, you may need to train the model from scratch using the provided scripts:
```bash
cd Trainer
python train.py --opt config/train.yml
```

---
*Note: This is a modified version of the AV-ConvTasNet repository integrated with BlueSkeye tracking pipelines.*
