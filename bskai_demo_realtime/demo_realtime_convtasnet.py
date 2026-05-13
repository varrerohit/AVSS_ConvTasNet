"""
Real-time Audio-Visual Speech Separation using AV-ConvTasNet and BlueSkeye SDK.

This script demonstrates a real-time inference pipeline. It continuously captures
frames from a video, tracks and aligns speaker faces using the BlueSkeye SDK,
and pushes the cropped mouth sequences and audio chunks to the AV-ConvTasNet
model. It uses a sliding-window overlap-add methodology to reconstruct the audio.
"""

import warnings
warnings.filterwarnings("ignore")
import os
import argparse
import sys
import gc
import time
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import yaml
import torchaudio
import soundfile as sf
from collections import deque
from skimage import transform as tf

# Add AV-ConvTasNet to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.av_model import AV_model
from model.video_model import video

# --- APPLE SILICON MPS PATCH ---
# Custom pooling wrappers to avoid MPS backend errors on Mac
_orig_adaptive_avg_pool1d = F.adaptive_avg_pool1d
_orig_adaptive_avg_pool2d = F.adaptive_avg_pool2d

def _mps_safe_adaptive_avg_pool1d(input, output_size):
    """Safely executes adaptive average pooling 1D."""
    if input.device.type == 'mps':
        return _orig_adaptive_avg_pool1d(input.cpu(), output_size).to(input.device)
    return _orig_adaptive_avg_pool1d(input, output_size)

def _mps_safe_adaptive_avg_pool2d(input, output_size):
    """Safely executes adaptive average pooling 2D."""
    if input.device.type == 'mps':
        return _orig_adaptive_avg_pool2d(input.cpu(), output_size).to(input.device)
    return _orig_adaptive_avg_pool2d(input, output_size)

F.adaptive_avg_pool1d = _mps_safe_adaptive_avg_pool1d
F.adaptive_avg_pool2d = _mps_safe_adaptive_avg_pool2d

# --- SETTINGS ---
WINDOW_SIZE_SEC = 4.0  
HOP_SIZE_SEC = 2.0     
TARGET_FPS = 25.0      
AUDIO_SAMPLE_RATE = 8000 
MOUTH_CROP_HEIGHT = 112 
MOUTH_CROP_WIDTH = 112  

# ==========================================
# MODEL LOAD UTILS
# ==========================================
def load_state_dict_in(model, pretrained_dict):
    """
    Loads specific keys from a pretrained state dictionary into the model.
    """
    model_dict = model.state_dict()
    update_dict = {}
    for k, v in pretrained_dict.items():
        if 'av_model' in k:
            update_dict[k[9:]] = v
    model_dict.update(update_dict)
    model.load_state_dict(model_dict)
    return model

def update_parameter(model, pretrained_dict):
    """
    Updates the video frontend (ResNet18) parameters and freezes them.
    """
    model_dict = model.state_dict()
    update_dict = {}
    for k, v in pretrained_dict.items():
        if 'front3D' in k:
            k_parts = k.split('.')
            k_ = 'front3d.conv3d.'+k_parts[1]+'.'+k_parts[2]
            update_dict[k_] = v
        if 'resnet' in k:
            k_ = 'front3d.'+k
            update_dict[k_] = v
    model_dict.update(update_dict)
    model.load_state_dict(model_dict)
    for p in model.parameters():
        p.requires_grad = False
    return model

def CenterCrop(batch_img, size):
    """
    Applies a center crop to a batch of video frames.
    
    Args:
        batch_img (np.ndarray): Tensor of shape (B, D, H, W).
        size (tuple): Target (Height, Width).
        
    Returns:
        np.ndarray: Cropped sequence.
    """
    h, w = batch_img[0][0].shape[0], batch_img[0][0].shape[1]
    th, tw = size
    img = np.zeros((len(batch_img), len(batch_img[0]), th, tw))
    for i in range(len(batch_img)):
        w1 = int(round((w - tw))/2.)
        h1 = int(round((h - th))/2.)
        img[i] = batch_img[i, :, h1:h1+th, w1:w1+tw]
    return img

def ColorNormalize(batch_img):
    """
    Standardizes image values to a specific mean and standard deviation.
    """
    mean = 0.361858
    std = 0.147485
    batch_img = (batch_img - mean) / std
    return batch_img

def init_sdk(sdk_path, license_path):
    """
    Initializes the BlueSkeye SDK for real-time tracking.
    """
    sys.path.insert(0, os.path.join(sdk_path, "python"))
    if "BSocial" in sdk_path:
        from BMBSocial import BMBSocialAPI, BSocialImageType
        api = BMBSocialAPI()
        image_type = BSocialImageType.BGR
    else:
        from BMBAutomotive import BMBAutomotiveAPI, BAutomotiveImageType
        api = BMBAutomotiveAPI()
        image_type = BAutomotiveImageType.BGR

    api.load_licence_key(license_path)
    if api.init() != 0:
        raise RuntimeError("Unable to init an instance of the SDK.")
    return api, image_type

def bb_intersection_over_union(boxA, boxB):
    """
    Calculates IoU for continuous bounding box tracking.
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    if float(boxAArea + boxBArea - interArea) == 0:
        return 0.0
    return interArea / float(boxAArea + boxBArea - interArea)

def warp_img(src, dst, img, std_size):
    """Warps an image via similarity transform based on landmarks."""
    tform = tf.estimate_transform('similarity', src, dst) 
    warped = tf.warp(img, inverse_map=tform.inverse, output_shape=std_size) 
    warped = warped * 255 
    warped = warped.astype('uint8')
    return warped, tform

def apply_transform(transform, img, std_size):
    """Applies an existing transform to an image."""
    warped = tf.warp(img, inverse_map=transform.inverse, output_shape=std_size)
    warped = warped * 255 
    warped = warped.astype('uint8')
    return warped

def cut_patch(img, landmarks, height, width, threshold=5):
    """Safely extracts a rectangular region centered around landmarks."""
    center_x, center_y = np.mean(landmarks, axis=0)
    center_y = max(height, min(img.shape[0] - height, center_y))
    center_x = max(width, min(img.shape[1] - width, center_x))
    cutted_img = np.copy(img[ int(round(center_y) - round(height)): int(round(center_y) + round(height)),
                         int(round(center_x) - round(width)): int(round(center_x) + round(width))])
    return cutted_img

def process_single_chunk_isolated(audiomodel, videomodel, mix_chunk, roi_chunk, device):
    """
    Processes a single time-window of audio and video while enforcing strict 
    memory cleanup to avoid accumulation in the MPS/CUDA graph.
    
    Args:
        audiomodel: AV-ConvTasNet model.
        videomodel: Video ResNet model.
        mix_chunk: Audio tensor.
        roi_chunk: Video NumPy array.
        device: Target execution device.
        
    Returns:
        np.ndarray: Detached NumPy array of the output audio chunk.
    """
    # Audio prep
    mix_input = mix_chunk[None].to(device)
    
    # Video prep (crop, normalize, reshape)
    mouth = roi_chunk[None]
    mouth = CenterCrop(mouth, (112, 112))
    mouth = ColorNormalize(mouth)
    B, D, H, W = mouth.shape
    mouth = np.reshape(mouth, (B, 1, D, H, W))
    mouth = torch.from_numpy(mouth).type(torch.float32).to(device)
    
    # Forward pass
    mouth_emb = videomodel(mouth)
    est_sources = audiomodel(mix_input, mouth_emb)
    
    # Extract to pure CPU format
    numpy_result = est_sources.detach().cpu().numpy().copy()
    
    # Immediate garbage collection of tensors
    del est_sources
    del mouth_emb
    del mix_input
    del mouth
    
    return numpy_result

def process_realtime(input_video, output_dir, number_of_speakers, sdk_path, license_path, config_path, audio_weights, video_weights, device_type):
    """
    Simulates real-time inference by maintaining a rolling buffer of frames.
    
    Args:
        input_video (str): Path to input video file.
        output_dir (str): Directory for output media.
        number_of_speakers (int): Number of speakers.
        sdk_path (str): BlueSkeye SDK directory.
        license_path (str): SDK license path.
        config_path (str): Architecture config path.
        audio_weights (str): Audio model weights.
        video_weights (str): Video model weights.
        device_type (str): Requested execution device.
    """
    if device_type == 'mps' and torch.backends.mps.is_available():
        device = torch.device('mps')
    elif device_type == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"[System] AI Device: {device}")
    
    api, image_type = init_sdk(sdk_path, license_path)
    api.reset()
    api.set_min_process_time(0)
    api.set_inference_increment_enabled(False)
    api.set_log_level(0)

    # 1. Model Initialization
    print(f"[System] Loading architecture config from: {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    videomodel = video(**config['video_model'])
    if video_weights and os.path.exists(video_weights):
        pretrain = torch.load(video_weights, map_location='cpu', weights_only=False)['model_state_dict']
        videomodel = update_parameter(videomodel, pretrain)
    videomodel.to(device)
    videomodel.eval()
    
    audiomodel = AV_model(**config['AV_model'])
    if audio_weights and os.path.exists(audio_weights):
        state_dict = torch.load(audio_weights, map_location='cpu', weights_only=False)
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        audiomodel = load_state_dict_in(audiomodel, state_dict)
    audiomodel.to(device)
    audiomodel.eval()
    
    # 2. Template matching initialization
    try:
        mean_face_landmarks = np.load(os.path.join(os.path.dirname(__file__), '../bskai_demo/assets/20words_mean_face.npy'))
    except:
        print("[Warning] mean_face.npy not found. Mouth crops may be unaligned.")
        mean_face_landmarks = None

    # 3. Media Pipeline Setup
    global TARGET_FPS
    from moviepy import VideoFileClip
    video_clip = VideoFileClip(input_video)
    
    cap = cv2.VideoCapture(input_video)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps > 0:
        TARGET_FPS = video_fps
        print(f"[System] Dynamic FPS allocation: {TARGET_FPS} fps")
    
    if video_clip.fps != TARGET_FPS:
        print(f"[Warning] Video FPS is {video_clip.fps}, expected {TARGET_FPS}. Sync might be off.")

    os.makedirs(output_dir, exist_ok=True)
    temp_audio_wav = os.path.join(output_dir, "temp_in.wav")
    video_clip.audio.write_audiofile(temp_audio_wav, fps=AUDIO_SAMPLE_RATE, nbytes=2, codec='pcm_s16le', logger=None)
    mix_audio, sr = torchaudio.load(temp_audio_wav)
    mix_audio = mix_audio.mean(dim=0)
    
    if sr != AUDIO_SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(sr, AUDIO_SAMPLE_RATE)
        mix_audio = resampler(mix_audio)
        sr = AUDIO_SAMPLE_RATE
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video_writers = []
    temp_vids = []
    
    for s in range(number_of_speakers):
        v_name = os.path.join(output_dir, f"temp_vid_s{s+1}.mp4")
        temp_vids.append(v_name)
        vw = cv2.VideoWriter(v_name, fourcc, TARGET_FPS, (frame_width, frame_height))
        out_video_writers.append(vw)

    window_size = int(WINDOW_SIZE_SEC * AUDIO_SAMPLE_RATE)
    hop_size = int(HOP_SIZE_SEC * AUDIO_SAMPLE_RATE)
    max_rolling_frames = int(WINDOW_SIZE_SEC * TARGET_FPS) + 10 
    
    hann_window = torch.hann_window(window_size)
    
    last_boxes = [None] * number_of_speakers
    last_lnds = [None] * number_of_speakers
    
    audio_out_buffers = [torch.zeros(len(mix_audio) + window_size) for _ in range(number_of_speakers)]
    weights_out_buffers = [torch.zeros(len(mix_audio) + window_size) for _ in range(number_of_speakers)]
    
    stablePntsIDs = [33, 36, 39, 42, 45]
    
    # Data queues
    rolling_mouths = [deque(maxlen=max_rolling_frames) for _ in range(number_of_speakers)]
    
    # Lookahead buffer for landmark smoothing
    window_margin = 12
    lookahead_frames = []
    lookahead_lnds = [[] for _ in range(number_of_speakers)]
    lookahead_boxes = [[] for _ in range(number_of_speakers)]
    last_trans = [None] * number_of_speakers
    
    print(f"\n[System] Starting Real-time Inference Simulation for {number_of_speakers} speakers...")
    
    frames_read = 0
    frames_processed_total = 0
    frames_since_last_inference = 0
    processing_time_acc = 0.0
    fps_display = 0.0
    
    current_start_audio_idx = 0
    current_end_audio_idx = current_start_audio_idx + window_size
    next_inference_frame = int(current_end_audio_idx / AUDIO_SAMPLE_RATE * TARGET_FPS)
    
    rolling_frames_bgr = deque(maxlen=max_rolling_frames)
    rolling_boxes = [deque(maxlen=max_rolling_frames) for _ in range(number_of_speakers)]
    
    def process_buffered_frame(cur_frame_rgb, cur_boxes, cur_lnds, is_draining=False):
        """
        Pulls a single frame from the queue, processes mouth extraction, and
        dispatches inference to the model if the temporal window threshold is met.
        """
        nonlocal frames_processed_total, frames_since_last_inference, current_start_audio_idx, current_end_audio_idx, next_inference_frame, processing_time_acc, fps_display
        
        for s in range(number_of_speakers):
            cur_box = cur_boxes[s]
            cur_lnd = cur_lnds[s]
            
            # Smoothing
            if not is_draining:
                smoothed_lnd = np.mean(lookahead_lnds[s], axis=0)
            else:
                smoothed_lnd = cur_lnd 
                
            # Face extraction
            if mean_face_landmarks is not None and np.sum(cur_lnd) > 0:
                if not is_draining:
                    trans_frame, trans = warp_img(smoothed_lnd[stablePntsIDs, :], mean_face_landmarks[stablePntsIDs, :], cur_frame_rgb, (256, 256))
                    last_trans[s] = trans
                else:
                    if last_trans[s] is not None:
                        trans = last_trans[s]
                        trans_frame = apply_transform(trans, cur_frame_rgb, (256, 256))
                    else:
                        trans_frame, trans = warp_img(cur_lnd[stablePntsIDs, :], mean_face_landmarks[stablePntsIDs, :], cur_frame_rgb, (256, 256))
                        
                trans_landmarks = trans(cur_lnd)
                mouth_crop = cut_patch(trans_frame, trans_landmarks[48:68], MOUTH_CROP_HEIGHT//2, MOUTH_CROP_WIDTH//2)
                if mouth_crop.size == 0 or mouth_crop.shape[0] != MOUTH_CROP_HEIGHT or mouth_crop.shape[1] != MOUTH_CROP_WIDTH:
                     mouth_crop = np.zeros((MOUTH_CROP_HEIGHT, MOUTH_CROP_WIDTH, 3), dtype=np.uint8)
            else:
                mouth_crop = np.zeros((MOUTH_CROP_HEIGHT, MOUTH_CROP_WIDTH, 3), dtype=np.uint8)
                
            mouth_gray = cv2.cvtColor(mouth_crop, cv2.COLOR_BGR2GRAY)
            rolling_mouths[s].append(mouth_gray)
            rolling_boxes[s].append(cur_box)
            
        cur_frame_bgr = cv2.cvtColor(cur_frame_rgb, cv2.COLOR_RGB2BGR)
        rolling_frames_bgr.append(cur_frame_bgr)
        
        frames_processed_total += 1
        frames_since_last_inference += 1
        
        # Inference threshold met
        if frames_processed_total == next_inference_frame:
            t2 = time.time()
            
            if current_end_audio_idx <= len(mix_audio):
                win_audio = mix_audio[current_start_audio_idx:current_end_audio_idx]
            else:
                pad_size = current_end_audio_idx - len(mix_audio)
                win_audio = torch.cat((mix_audio[current_start_audio_idx:], torch.zeros(pad_size)))
            
            start_frame = int(current_start_audio_idx / AUDIO_SAMPLE_RATE * TARGET_FPS)
            end_frame = frames_processed_total
            req_frames = end_frame - start_frame
            
            for s in range(number_of_speakers):
                win_mouths = np.stack(list(rolling_mouths[s])[-req_frames:], axis=0)
                
                with torch.no_grad():
                    est_audio = process_single_chunk_isolated(audiomodel, videomodel, win_audio, win_mouths, device)
                    est_audio_tensor = torch.from_numpy(est_audio)

                if est_audio_tensor.ndim == 3:
                    est_audio_tensor = est_audio_tensor[0]
                elif est_audio_tensor.ndim == 1:
                    est_audio_tensor = est_audio_tensor[None, :]

                # Reconstruct output using Hanning overlap-add
                valid_len = min(window_size, audio_out_buffers[s].size(0) - current_start_audio_idx)
                if valid_len > 0:
                    est_data = est_audio_tensor[0, :valid_len]
                    audio_out_buffers[s][current_start_audio_idx:current_start_audio_idx+valid_len] += est_data * hann_window[:valid_len]
                    weights_out_buffers[s][current_start_audio_idx:current_start_audio_idx+valid_len] += hann_window[:valid_len]
                
            if device.type == 'mps':
                torch.mps.empty_cache()
            gc.collect()
            
            t3 = time.time()
            processing_time_acc += (t3 - t2)
            
            if processing_time_acc > 0:
                fps_display = frames_since_last_inference / processing_time_acc
                
            print(f"\rProcessing Chunk... End-to-End FPS: {fps_display:.2f} (Frame {frames_processed_total})", end="")
            
            processing_time_acc = 0.0
            frames_since_last_inference = 0
            
            current_start_audio_idx += hop_size
            current_end_audio_idx = current_start_audio_idx + window_size
            next_inference_frame = int(current_end_audio_idx / AUDIO_SAMPLE_RATE * TARGET_FPS)

        # Output video writing
        for s in range(number_of_speakers):
            f_img = cur_frame_bgr.copy()
            box = rolling_boxes[s][-1]
            if box is not None:
                cv2.rectangle(f_img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255), 4)
            cv2.putText(f_img, f"End-to-End FPS: {fps_display:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            out_video_writers[s].write(f_img)

    # 4. Main capturing loop
    while cap.isOpened():
        t0 = time.time()
        ret, frame_bgr = cap.read()
        if not ret: break
        
        api.set_image(frame_bgr, image_type, False)
        api.run()
        
        try:
            predictions = api.get_predictions()
            if not isinstance(predictions, list): predictions = [predictions]
        except:
            predictions = []
            
        detected_boxes = []
        detected_lnds = []
        
        for pred in predictions:
            x = np.array([pred.landmarks.x[k] for k in range(68)])
            y = np.array([pred.landmarks.y[k] for k in range(68)])
            p_lnd = np.column_stack((x, y))
            
            x_min, x_max = np.min(x), np.max(x)
            y_min, y_max = np.min(y), np.max(y)
            width, height = x_max - x_min, y_max - y_min
            width_center, height_center = (x_max + x_min) / 2, (y_max + y_min) / 2
            square_width = int(max(width, height) * 1.5)
            p_box = [width_center - square_width/2, height_center - square_width/2, width_center + square_width/2, height_center + square_width/2]
            
            detected_boxes.append(p_box)
            detected_lnds.append(p_lnd)
            
        speaker_boxes = [None] * number_of_speakers
        speaker_lnds = [None] * number_of_speakers
        
        # IoU tracking algorithm
        if frames_read == 0:
            for s in range(number_of_speakers):
                if s < len(detected_boxes):
                    speaker_boxes[s] = detected_boxes[s]
                    speaker_lnds[s] = detected_lnds[s]
        else:
            matched_speakers = set()
            for b_idx, box in enumerate(detected_boxes):
                iou_scores = []
                for s in range(number_of_speakers):
                    if s in matched_speakers:
                        iou_scores.append(-1)
                    else:
                        if last_boxes[s] is not None:
                            iou_scores.append(bb_intersection_over_union(box, last_boxes[s]))
                        else:
                            iou_scores.append(-1)
                
                if iou_scores and max(iou_scores) > 0:
                    best_speaker = iou_scores.index(max(iou_scores))
                    speaker_boxes[best_speaker] = box
                    speaker_lnds[best_speaker] = detected_lnds[b_idx]
                    matched_speakers.add(best_speaker)
                    
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        
        for s in range(number_of_speakers):
            if speaker_boxes[s] is None:
                speaker_boxes[s] = last_boxes[s] if last_boxes[s] is not None else [0,0,10,10]
                speaker_lnds[s] = last_lnds[s] if last_lnds[s] is not None else np.zeros((68,2))
                
            last_boxes[s] = speaker_boxes[s]
            last_lnds[s] = speaker_lnds[s]
            
        lookahead_frames.append(frame_rgb)
        for s in range(number_of_speakers):
            lookahead_lnds[s].append(speaker_lnds[s])
            lookahead_boxes[s].append(speaker_boxes[s])
            
        t1 = time.time()
        processing_time_acc += (t1 - t0)

        # Buffer control
        if len(lookahead_frames) == window_margin:
            cur_frame_rgb = lookahead_frames.pop(0)
            cur_boxes = [lookahead_boxes[s].pop(0) for s in range(number_of_speakers)]
            cur_lnds = [lookahead_lnds[s].pop(0) for s in range(number_of_speakers)]
            process_buffered_frame(cur_frame_rgb, cur_boxes, cur_lnds, is_draining=False)

        frames_read += 1

    # 5. Pipeline Draining
    while len(lookahead_frames) > 0:
        cur_frame_rgb = lookahead_frames.pop(0)
        cur_boxes = [lookahead_boxes[s].pop(0) for s in range(number_of_speakers)]
        cur_lnds = [lookahead_lnds[s].pop(0) for s in range(number_of_speakers)]
        
        t0 = time.time()
        process_buffered_frame(cur_frame_rgb, cur_boxes, cur_lnds, is_draining=True)
        processing_time_acc += (time.time() - t0)

    # Process remaining audio chunk
    if frames_since_last_inference > 0:
        if current_end_audio_idx <= len(mix_audio):
            win_audio = mix_audio[current_start_audio_idx:current_end_audio_idx]
        else:
            pad_size = current_end_audio_idx - len(mix_audio)
            if pad_size < len(mix_audio[current_start_audio_idx:]):
                win_audio = torch.cat((mix_audio[current_start_audio_idx:], torch.zeros(pad_size)))
            else:
                rem = mix_audio[current_start_audio_idx:]
                win_audio = torch.cat((rem, torch.zeros(window_size - len(rem))))
        
        start_frame = int(current_start_audio_idx / AUDIO_SAMPLE_RATE * TARGET_FPS)
        end_frame = frames_processed_total
        req_frames = end_frame - start_frame
        
        if req_frames > 0:
            for s in range(number_of_speakers):
                valid_mouths = list(rolling_mouths[s])[-req_frames:]
                if len(valid_mouths) < req_frames:
                    pad_mouths = [np.zeros_like(valid_mouths[0])] * (req_frames - len(valid_mouths))
                    valid_mouths = pad_mouths + valid_mouths
                win_mouths = np.stack(valid_mouths, axis=0)
                
                with torch.no_grad():
                    est_audio = process_single_chunk_isolated(audiomodel, videomodel, win_audio, win_mouths, device)
                    est_audio_tensor = torch.from_numpy(est_audio)

                if est_audio_tensor.ndim == 3:
                    est_audio_tensor = est_audio_tensor[0]
                elif est_audio_tensor.ndim == 1:
                    est_audio_tensor = est_audio_tensor[None, :]

                valid_len = min(window_size, audio_out_buffers[s].size(0) - current_start_audio_idx)
                if valid_len > 0:
                    est_data = est_audio_tensor[0, :valid_len]
                    audio_out_buffers[s][current_start_audio_idx:current_start_audio_idx+valid_len] += est_data * hann_window[:valid_len]
                    weights_out_buffers[s][current_start_audio_idx:current_start_audio_idx+valid_len] += hann_window[:valid_len]


    cap.release()
    for s in range(number_of_speakers):
        out_video_writers[s].release()
    
    print("\n[System] Saving complete WAV files and merging with Video...")
    from moviepy import VideoFileClip, AudioFileClip
    
    # Final normalization and merge
    for s in range(number_of_speakers):
        final_wav_path = os.path.join(output_dir, f"speaker_{s+1}_realtime_output.wav")
        final_mp4_path = os.path.join(output_dir, f"speaker_{s+1}_realtime_output.mp4")
        
        final_audio = audio_out_buffers[s][:len(mix_audio)] / (weights_out_buffers[s][:len(mix_audio)] + 1e-8)
        sf.write(final_wav_path, final_audio.numpy(), AUDIO_SAMPLE_RATE)
        
        final_vid = VideoFileClip(temp_vids[s])
        final_aud = AudioFileClip(final_wav_path)
        
        set_audio_fn = getattr(final_vid, "set_audio", getattr(final_vid, "with_audio", None))
        final_vid = set_audio_fn(final_aud)
        final_vid.write_videofile(final_mp4_path, codec='libx264', audio_codec='aac', logger=None)
        
        final_vid.close()
        final_aud.close()
        
        if os.path.exists(temp_vids[s]): os.remove(temp_vids[s])
        
    if os.path.exists(temp_audio_wav): os.remove(temp_audio_wav)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Real-time Speaker Separation using AV-ConvTasNet")
    parser.add_argument('-i', '--input', type=str, required=True, help='Path to input video')
    parser.add_argument('-o', '--output', type=str, required=True, help='Path to output directory')
    parser.add_argument('-s', '--speakers', type=int, default=2, help='Number of speakers')
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML')
    parser.add_argument('--audio-weights', type=str, required=True, help='Path to AV-ConvTasNet weights')
    parser.add_argument('--video-weights', type=str, required=True, help='Path to ResNet weights')
    parser.add_argument("-sdk", "--sdk-path", required=True, help='Path to BlueSkeye SDK')
    parser.add_argument("-l", "-lic", "--license-path", required=True, help='Path to SDK License')
    parser.add_argument("--device", type=str, default='cpu', help='Device for inference')
    args = parser.parse_args()
    
    process_realtime(args.input, args.output, args.speakers, args.sdk_path, args.license_path, args.config, args.audio_weights, args.video_weights, args.device)
