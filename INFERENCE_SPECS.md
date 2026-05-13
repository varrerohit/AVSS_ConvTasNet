# AV-ConvTasNet Inference Specifications

## Model Architecture
- **Type:** Decoupled (Video Feature Extractor + Audio-Visual Separator)
- **Video Class:** `model.video_model.video` (front3d + ResNet18)
- **Audio Class:** `model.av_model.AV_model`
- **Fusion:** Concat + Projection in `AV_model`.

## Hyperparameters
- **Audio Sample Rate:** 8,000 Hz (Requires resampling from 16kHz/44.1kHz)
- **Mouth ROI Size:** 112 x 112 pixels
- **Visual Channels:** 3 (RGB)
- **Window Size:** 4.0 seconds
- **Hop Size:** 2.0 seconds (50% overlap)
- **Target Video FPS:** 25.0

## Preprocessing & Normalization
- **Visual Alignment:** Affine transformation followed by `CenterCrop` to 112x112.
- **Normalization:** Specific mean/std normalization: `(img - 0.361858) / 0.147485`.
- **Audio:** Time-domain raw waveform.

## Weight Loading
- **Format:** PyTorch `.ckpt` or `.pt`
- **Video Model:** Requires `update_parameter` to map `front3D` and `resnet` keys to the `front3d` prefix.
- **Audio Model:** Requires `load_state_dict_in` to strip `av_model.` prefix from keys.

## Inference Strategy
- **Execution:** Feed mouth ROI to `videomodel` to get embeddings, then feed embeddings + audio to `audiomodel`.
- **Overlap-Add:** Hann window cross-fading for 4-second chunks.

## ⚠️ Known Issue: Official Pretrained Weights
As of May 2026, the pretrained model links provided in the **[Official AV-ConvTasNet Repository](https://github.com/JusperLee/AV-ConvTasNet#usage)** are incorrect. 

The Google Drive links provided by the author for LRS2, LRS3, and VoxCeleb2 point to **CTCNet** checkpoints rather than AV-ConvTasNet checkpoints. Because CTCNet uses Spatial Pyramid Pooling Depthwise (`spp_dw`) layers and the Asteroid framework, the keys in those `.ckpt` files (e.g., `masker.video_block.video.0.spp_dw.0.conv.weight`) do not match the `AV_model` architecture defined in this repository (which expects keys like `video.conv1d_list.0.dconv.weight`).

If you attempt to load the currently linked weights, `load_state_dict_in` will explicitly throw a `RuntimeError` to prevent the model from executing with random noise.

To track the resolution of this issue or to obtain the correct weights, please refer to the corresponding GitHub issue: 
**[Issue: Incorrect Pretrained Models linked in README](https://github.com/JusperLee/AV-ConvTasNet/issues)** *(Link placeholder until issue is officially filed).*

Currently, to use this specific model, you must train it from scratch using the provided `Trainer/train.py` script. Alternatively, you can use the **Swift-Net** or **Dolphin** models provided in the adjacent directories, as their pretrained weights are accurate and functioning.
