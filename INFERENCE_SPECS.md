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
