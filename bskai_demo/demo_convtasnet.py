"""
Offline Audio-Visual Speech Separation using AV-ConvTasNet and BlueSkeye SDK.

This script separates speech from multiple speakers in a video. It uses the BlueSkeye
SDK for face tracking, extracts mouth regions, processes the video stream with a ResNet18
frontend, and uses the AV-ConvTasNet model to estimate the isolated audio sources.
"""

import warnings
warnings.filterwarnings("ignore")
import os
import argparse
import sys
import gc
import torch
import torch.nn.functional as F
import yaml
import numpy as np
import cv2
from PIL import Image
from moviepy import VideoFileClip, AudioFileClip, ImageSequenceClip
from collections import deque                                                 
from skimage import transform as tf
import torchaudio

# Add AV-ConvTasNet to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model.av_model import AV_model
from model.video_model import video

# Prevent python from writing .pyc files
sys.dont_write_bytecode = True

# ==========================================
# HYPERPARAMETERS & SETTINGS
# ==========================================
WINDOW_SIZE_SEC = 4.0  # AV-ConvTasNet segment length is 4.0s
HOP_SIZE_SEC = 2.0     
TARGET_FPS = 25.0      
AUDIO_SAMPLE_RATE = 8000 
FACE_CROP_SIZE = 224   
MOUTH_CROP_HEIGHT = 112 
MOUTH_CROP_WIDTH = 112  

# --- APPLE SILICON MPS PATCH ---
# Fallbacks for pooling layers to support Apple Silicon MPS
_orig_adaptive_avg_pool1d = F.adaptive_avg_pool1d
_orig_adaptive_avg_pool2d = F.adaptive_avg_pool2d

def _mps_safe_adaptive_avg_pool1d(input, output_size):
    """Safe adaptive_avg_pool1d wrapper for MPS."""
    if input.device.type == 'mps':
        return _orig_adaptive_avg_pool1d(input.cpu(), output_size).to(input.device)
    return _orig_adaptive_avg_pool1d(input, output_size)

def _mps_safe_adaptive_avg_pool2d(input, output_size):
    """Safe adaptive_avg_pool2d wrapper for MPS."""
    if input.device.type == 'mps':
        return _orig_adaptive_avg_pool2d(input.cpu(), output_size).to(input.device)
    return _orig_adaptive_avg_pool2d(input, output_size)

F.adaptive_avg_pool1d = _mps_safe_adaptive_avg_pool1d
F.adaptive_avg_pool2d = _mps_safe_adaptive_avg_pool2d

# ==========================================
# MODEL LOAD UTILS
# ==========================================
def load_state_dict_in(model, pretrained_dict):
    """
    Loads pretrained weights into the AV-ConvTasNet model by filtering specific keys.
    
    Args:
        model (nn.Module): The target AV model.
        pretrained_dict (dict): The loaded state dictionary.
        
    Returns:
        nn.Module: The model with loaded weights.
    """
    model_dict = model.state_dict()
    update_dict = {}
    for k, v in pretrained_dict.items():
        if 'av_model' in k:
            update_dict[k[9:]] = v
        # Fallback for keys that match exactly
        elif k in model_dict:
            update_dict[k] = v
            
    if len(update_dict) == 0:
        raise RuntimeError("CRITICAL ERROR: No matching keys found in the audio checkpoint! The model is randomly initialized, which will result in noise. Please verify you have the correct AV-ConvTasNet weights.")
        
    model_dict.update(update_dict)
    model.load_state_dict(model_dict)
    print(f"[System] Successfully loaded {len(update_dict)}/{len(model_dict)} parameters into AV_model.")
    return model

def update_parameter(model, pretrained_dict):
    """
    Updates the video ResNet18 parameters with pretrained weights, parsing nested keys.
    
    Args:
        model (nn.Module): The video embedding model.
        pretrained_dict (dict): The loaded state dictionary.
        
    Returns:
        nn.Module: The updated model with frozen parameters.
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
    
    # Freeze video parameters during inference
    for p in model.parameters():
        p.requires_grad = False
    return model

# ==========================================
# PREPROCESSING UTILS
# ==========================================
def CenterCrop(batch_img, size):
    """
    Crops the center of a batched image sequence.
    
    Args:
        batch_img (np.ndarray): Image tensor (B, D, H, W).
        size (tuple): Target (Height, Width).
        
    Returns:
        np.ndarray: Centered crop.
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
    Normalizes image values to have zero mean and specific variance.
    
    Args:
        batch_img (np.ndarray): Image tensor.
        
    Returns:
        np.ndarray: Normalized image tensor.
    """
    mean = 0.361858
    std = 0.147485
    batch_img = (batch_img - mean) / std
    return batch_img

def init_sdk(sdk_path, license_path):
    """
    Initializes the BlueSkeye SDK.
    
    Args:
        sdk_path (str): Path to SDK.
        license_path (str): Path to SDK license.
        
    Returns:
        tuple: (api_instance, image_type_constant)
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

def linear_interpolate(landmarks, start_idx, stop_idx):
    """Linearly interpolates landmarks between indices."""
    start_landmarks = landmarks[start_idx]
    stop_landmarks = landmarks[stop_idx]
    delta = stop_landmarks - start_landmarks
    for idx in range(1, stop_idx-start_idx):
        landmarks[start_idx+idx] = start_landmarks + idx/float(stop_idx-start_idx) * delta
    return landmarks

def warp_img(src, dst, img, std_size):
    """Applies a similarity transform to warp the image based on landmarks."""
    tform = tf.estimate_transform('similarity', src, dst) 
    warped = tf.warp(img, inverse_map=tform.inverse, output_shape=std_size) 
    warped = warped * 255 
    warped = warped.astype('uint8')
    return warped, tform

def apply_transform(transform, img, std_size):
    """Applies a pre-computed transform to an image."""
    warped = tf.warp(img, inverse_map=transform.inverse, output_shape=std_size)
    warped = warped * 255 
    warped = warped.astype('uint8')
    return warped

def cut_patch(img, landmarks, height, width, threshold=5):
    """Crops a patch around the center of the given landmarks."""
    center_x, center_y = np.mean(landmarks, axis=0)
    if center_y - height < 0: center_y = height                                                    
    if center_y - height < 0 - threshold: raise Exception('too much bias in height')                           
    if center_x - width < 0: center_x = width                                                     
    if center_x - width < 0 - threshold: raise Exception('too much bias in width')                            
    if center_y + height > img.shape[0]: center_y = img.shape[0] - height                                     
    if center_y + height > img.shape[0] + threshold: raise Exception('too much bias in height')                           
    if center_x + width > img.shape[1]: center_x = img.shape[1] - width                                      
    if center_x + width > img.shape[1] + threshold: raise Exception('too much bias in width')                            
                                                                             
    cutted_img = np.copy(img[ int(round(center_y) - round(height)): int(round(center_y) + round(height)),
                         int(round(center_x) - round(width)): int(round(center_x) + round(width))])
    return cutted_img

def convert_bgr2gray(data):
    """Converts a sequence of BGR images to grayscale."""
    return np.stack([cv2.cvtColor(_, cv2.COLOR_BGR2GRAY) for _ in data], axis=0)

def save2npz(filename, data=None):
    """Saves data array to a compressed NPZ file."""
    assert data is not None, "data is {}".format(data)
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    np.savez_compressed(filename, data=data)

def read_video(filename):
    """Yields RGB frames from a video file."""
    cap = cv2.VideoCapture(filename)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    cap.release()

def bb_intersection_over_union(boxA, boxB):
    """Calculates Intersection over Union for speaker tracking."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

# ==========================================
# 1. FACE DETECTION (STREAMING MEMORY)
# ==========================================
def detectface(video_input_path, output_path, detect_every_N_frame, scalar_face_detection, number_of_speakers, api, image_type):
    """
    Tracks and extracts faces using the BlueSkeye SDK.
    
    Args:
        video_input_path (str): Input video path.
        output_path (str): Output directory.
        detect_every_N_frame (int): Detection interval.
        scalar_face_detection (float): Bounding box scaler.
        number_of_speakers (int): Number of speakers.
        api: SDK instance.
        image_type: SDK image format.
        
    Returns:
        str: CSV file path linking video to speakers.
    """
    print('[System] Running Tracking on Apple Silicon via SDK')
    os.makedirs(os.path.join(output_path, 'faces'), exist_ok=True)
    os.makedirs(os.path.join(output_path, 'landmark'), exist_ok=True)

    landmarks_dic = {i: [] for i in range(number_of_speakers)}
    faces_dic = {i: [] for i in range(number_of_speakers)}
    boxes_dic = {i: [] for i in range(number_of_speakers)}

    for i, frame_arr in enumerate(read_video(video_input_path)):
        print(f'\rTracking frame: {i + 1}', end='')
        
        frame = Image.fromarray(frame_arr)
        frame_bgr = cv2.cvtColor(frame_arr, cv2.COLOR_RGB2BGR)
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
            lnd = np.column_stack((x, y))
            
            x_min, x_max = np.min(x), np.max(x)
            y_min, y_max = np.min(y), np.max(y)
            width, height = x_max - x_min, y_max - y_min
            width_center, height_center = (x_max + x_min) / 2, (y_max + y_min) / 2
            square_width = int(max(width, height) * scalar_face_detection)
            box = [width_center - square_width/2, height_center - square_width/2, width_center + square_width/2, height_center + square_width/2]
            
            detected_boxes.append(box)
            detected_lnds.append(lnd)

        if i == 0:
            for j in range(number_of_speakers):
                if j < len(detected_boxes):
                    box = detected_boxes[j]
                    lnd = detected_lnds[j]
                else:
                    box = detected_boxes[-1] if detected_boxes else [0,0,10,10]
                    lnd = detected_lnds[-1] if detected_lnds else np.zeros((68,2))

                face = frame.crop((box[0], box[1], box[2], box[3])).resize((FACE_CROP_SIZE, FACE_CROP_SIZE))
                
                scale_x = FACE_CROP_SIZE / (max(1, box[2] - box[0]))
                scale_y = FACE_CROP_SIZE / (max(1, box[3] - box[1]))
                preds = np.zeros_like(lnd)
                preds[:, 0] = (lnd[:, 0] - box[0]) * scale_x
                preds[:, 1] = (lnd[:, 1] - box[1]) * scale_y
                
                faces_dic[j].append(face)
                landmarks_dic[j].append(preds)
                boxes_dic[j].append(box)
        else:
            # Re-identify speakers using IoU between frames
            matched_speakers = set()
            speaker_boxes = [None] * number_of_speakers
            speaker_lnds = [None] * number_of_speakers
            
            for b_idx, box in enumerate(detected_boxes):
                iou_scores = []
                for speaker_id in range(number_of_speakers):
                    if speaker_id in matched_speakers:
                        iou_scores.append(-1) 
                    else:
                        last_box = boxes_dic[speaker_id][-1]
                        iou_scores.append(bb_intersection_over_union(box, last_box))
                
                if iou_scores and max(iou_scores) > 0:  
                    best_speaker = iou_scores.index(max(iou_scores))
                    speaker_boxes[best_speaker] = box
                    speaker_lnds[best_speaker] = detected_lnds[b_idx]
                    matched_speakers.add(best_speaker)
            
            for speaker_id in range(number_of_speakers):
                if speaker_boxes[speaker_id] is not None:
                    box = speaker_boxes[speaker_id]
                    lnd = speaker_lnds[speaker_id]
                    face = frame.crop((box[0], box[1], box[2], box[3])).resize((FACE_CROP_SIZE, FACE_CROP_SIZE))
                    
                    scale_x = FACE_CROP_SIZE / (max(1, box[2] - box[0]))
                    scale_y = FACE_CROP_SIZE / (max(1, box[3] - box[1]))
                    preds = np.zeros_like(lnd)
                    preds[:, 0] = (lnd[:, 0] - box[0]) * scale_x
                    preds[:, 1] = (lnd[:, 1] - box[1]) * scale_y
                    
                    faces_dic[speaker_id].append(face)
                    landmarks_dic[speaker_id].append(preds)
                    boxes_dic[speaker_id].append(box)
                else:
                    box = boxes_dic[speaker_id][-1]
                    face = frame.crop((box[0], box[1], box[2], box[3])).resize((FACE_CROP_SIZE, FACE_CROP_SIZE))
                    
                    faces_dic[speaker_id].append(face)
                    landmarks_dic[speaker_id].append(landmarks_dic[speaker_id][-1])
                    boxes_dic[speaker_id].append(box)
    
    print("\n[System] Video writing step...")
    for s in range(number_of_speakers):
        cap = cv2.VideoCapture(video_input_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or TARGET_FPS
        
        tracked_video_path = os.path.join(output_path, f'video_tracked{s+1}.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(tracked_video_path, fourcc, fps, (width, height))
        
        idx = 0
        while cap.isOpened():
            ret, frame_bgr = cap.read()
            if not ret or idx >= len(boxes_dic[s]):
                break
            box = [int(v) for v in boxes_dic[s][idx]]
            cv2.rectangle(frame_bgr, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 6)
            writer.write(frame_bgr)
            idx += 1
            
        cap.release()
        writer.release()

    for i in range(number_of_speakers):    
        save2npz(os.path.join(output_path, 'landmark', 'speaker' + str(i+1)+'.npz'), data=landmarks_dic[i])
        face_frames = [np.array(f) for f in faces_dic[i]]
        if face_frames:
            face_clip = ImageSequenceClip(face_frames, fps=TARGET_FPS)
            face_video_path = os.path.join(output_path, 'faces', 'speaker' + str(i+1) + '.mp4')
            face_clip.write_videofile(face_video_path, codec='libx264', audio=False, logger=None)
            face_clip.close()

    parts = video_input_path.split('/')
    video_name = parts[-1][:-4]
    os.makedirs(os.path.join(output_path, 'filename_input'), exist_ok=True)
    csvfile = open(os.path.join(output_path, 'filename_input', str(video_name) + '.csv'), 'w')
    for i in range(number_of_speakers):
        csvfile.write('speaker' + str(i+1)+ ',0\n')
    csvfile.close()
    return os.path.join(output_path, 'filename_input', str(video_name) + '.csv')

# ==========================================
# 2. MOUTH CROP 
# ==========================================
def crop_patch_logic(mean_face_landmarks, video_pathname, landmarks, window_margin, start_idx, stop_idx, crop_height, crop_width, STD_SIZE=(256, 256)):
    """
    Main logic to crop aligned mouth patches.
    
    Args:
        mean_face_landmarks (np.ndarray): Mean face template for affine alignment.
        video_pathname (str): Input video path.
        landmarks (list): Detected landmarks for each frame.
        window_margin (int): Frame smoothing window length.
        start_idx (int): Mouth landmark start index (48).
        stop_idx (int): Mouth landmark end index (68).
        crop_height (int): Target crop height.
        crop_width (int): Target crop width.
        STD_SIZE (tuple): Base warp dimensions.
        
    Returns:
        np.ndarray: Cropped sequence (T, H, W, 3).
    """
    stablePntsIDs = [33, 36, 39, 42, 45]
    q_frame, q_landmarks = deque(), deque()
    sequence = []
    trans = None
    
    clean_landmarks = []
    for lnd in landmarks:
        if lnd is None: continue
        if isinstance(lnd, list) or (isinstance(lnd, np.ndarray) and lnd.ndim == 3):
            clean_landmarks.append(lnd[0])
        else:
            clean_landmarks.append(lnd)
            
    frame_gen = read_video(video_pathname)
    
    for frame_idx, frame in enumerate(frame_gen):
        if frame_idx >= len(clean_landmarks): break 
            
        q_landmarks.append(clean_landmarks[frame_idx])
        q_frame.append(frame)
        
        # Sliding window smoothing for stable warping
        if len(q_frame) == window_margin:
            smoothed_landmarks = np.mean(q_landmarks, axis=0)
            cur_landmarks = q_landmarks.popleft()
            cur_frame = q_frame.popleft()
            trans_frame, trans = warp_img(smoothed_landmarks[stablePntsIDs, :], mean_face_landmarks[stablePntsIDs, :], cur_frame, STD_SIZE)
            trans_landmarks = trans(cur_landmarks)
            sequence.append(cut_patch(trans_frame, trans_landmarks[start_idx:stop_idx], crop_height//2, crop_width//2))
            
    if len(sequence) == 0 and len(q_frame) > 0:
        smoothed_landmarks = np.mean(q_landmarks, axis=0)
        cur_landmarks = q_landmarks.popleft()
        cur_frame = q_frame.popleft()
        trans_frame, trans = warp_img(smoothed_landmarks[stablePntsIDs, :], mean_face_landmarks[stablePntsIDs, :], cur_frame, STD_SIZE)
        trans_landmarks = trans(cur_landmarks)
        sequence.append(cut_patch(trans_frame, trans_landmarks[start_idx:stop_idx], crop_height//2, crop_width//2))

    while q_frame:
        cur_frame = q_frame.popleft()
        if trans is not None:
            trans_frame = apply_transform(trans, cur_frame, STD_SIZE)
            trans_landmarks = trans(q_landmarks.popleft())
            sequence.append(cut_patch(trans_frame, trans_landmarks[start_idx:stop_idx], crop_height//2, crop_width//2))
        
    return np.array(sequence) if len(sequence) > 0 else None

def landmarks_interpolate(landmarks):
    """Fills missing landmarks using linear interpolation."""
    valid_frames_idx = [idx for idx, _ in enumerate(landmarks) if _ is not None]
    if not valid_frames_idx: return None
    for idx in range(1, len(valid_frames_idx)):
        if valid_frames_idx[idx] - valid_frames_idx[idx-1] == 1:
            continue
        else:
            landmarks = linear_interpolate(landmarks, valid_frames_idx[idx-1], valid_frames_idx[idx])
    valid_frames_idx = [idx for idx, _ in enumerate(landmarks) if _ is not None]
    if valid_frames_idx:
        landmarks[:valid_frames_idx[0]] = [landmarks[valid_frames_idx[0]]] * valid_frames_idx[0]
        landmarks[valid_frames_idx[-1]:] = [landmarks[valid_frames_idx[-1]]] * (len(landmarks) - valid_frames_idx[-1])
    return landmarks

def crop_mouth(video_direc, landmark_direc, filename_path, save_direc, convert_gray=False, testset_only=False):
    """
    Orchestrates the mouth ROI extraction for all speakers.
    
    Args:
        video_direc (str): Source video directory.
        landmark_direc (str): Source landmark directory.
        filename_path (str): Video ID registry list.
        save_direc (str): Output directory for ROIs.
        convert_gray (bool): Convert cropped mouth to grayscale.
        testset_only (bool): If True, process test set only.
    """
    lines = open(filename_path).read().splitlines()
    lines = list(filter(lambda x: 'test' in x, lines)) if testset_only else lines

    mean_face_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets', '20words_mean_face.npy')
    try:
        mean_face_landmarks = np.load(mean_face_path)
    except FileNotFoundError:
        print("[Warning] mean_face.npy not found in assets, using basic crop.")
        mean_face_landmarks = None

    for filename_idx, line in enumerate(lines):
        filename, person_id = line.split(',')
        print('idx: {} \tProcessing.\t{}'.format(filename_idx, filename))

        video_pathname = os.path.join(video_direc, filename+'.mp4')
        landmarks_pathname = os.path.join(landmark_direc, filename+'.npz')
        dst_pathname = os.path.join( save_direc, filename+'.npz')

        multi_sub_landmarks = np.load(landmarks_pathname, allow_pickle=True)['data']
        landmarks = [None] * len(multi_sub_landmarks)
        for frame_idx in range(len(landmarks)):
            try:
                landmarks[frame_idx] = multi_sub_landmarks[frame_idx]
            except (IndexError, TypeError):
                continue

        preprocessed_landmarks = landmarks_interpolate(landmarks)
        if not preprocessed_landmarks: continue

        if mean_face_landmarks is not None:
            sequence = crop_patch_logic(mean_face_landmarks, video_pathname, preprocessed_landmarks, 12, 48, 68, MOUTH_CROP_HEIGHT, MOUTH_CROP_WIDTH)
        else:
            # Fallback for basic unaligned crop if no mean face template
            frame_gen = read_video(video_pathname)
            sequence = []
            for frame_idx, frame in enumerate(frame_gen):
                if frame_idx >= len(preprocessed_landmarks): break
                seq_frame = cut_patch(frame, preprocessed_landmarks[frame_idx][48:68], MOUTH_CROP_HEIGHT//2, MOUTH_CROP_WIDTH//2)
                sequence.append(seq_frame)
            sequence = np.array(sequence)

        assert sequence is not None, "cannot crop from {}.".format(filename)

        data = convert_bgr2gray(sequence) if convert_gray else sequence[...,::-1]
        save2npz(dst_pathname, data=data)

# ==========================================
# 3. ISOLATED INFERENCE FUNCTION 
# ==========================================
def process_single_chunk_isolated(audiomodel, videomodel, mix_chunk, roi_chunk, device):
    """
    Executes model inference for a single window isolated from the main graph memory.
    
    This function forces local variables to be destroyed immediately upon return.
    It returns a pure NumPy array, severing all hidden ties to the PyTorch Graph,
    which is essential for Apple Silicon MPS stability.
    
    Args:
        audiomodel (nn.Module): AV-ConvTasNet separation model.
        videomodel (nn.Module): Video ResNet18 extraction model.
        mix_chunk (torch.Tensor): Audio mixture segment.
        roi_chunk (np.ndarray): Video ROI segment.
        device (torch.device): Compute device.
        
    Returns:
        np.ndarray: Evaluated speech representation.
    """
    # 1. Send tensors to GPU
    mix_input = mix_chunk[None].to(device)  # [1, T]
    
    # AV-ConvTasNet expects video shape: (B, 1, D, H, W)
    # roi_chunk shape is (T, H, W)
    mouth = roi_chunk[None]
    mouth = CenterCrop(mouth, (112, 112))
    mouth = ColorNormalize(mouth)
    B, D, H, W = mouth.shape
    mouth = np.reshape(mouth, (B, 1, D, H, W))
    mouth = torch.from_numpy(mouth).type(torch.float32).to(device)
    
    # 2. Run Forward Pass
    mouth_emb = videomodel(mouth)
    est_sources = audiomodel(mix_input, mouth_emb)
    
    # 3. Pull back to CPU
    numpy_result = est_sources.detach().cpu().numpy().copy()
    
    # 4. Explicit Memory Cleanup
    del est_sources
    del mouth_emb
    del mix_input
    del mouth
    
    return numpy_result

# ==========================================
# 4. VIDEO PIPELINE 
# ==========================================
def convert_video_fps(input_file, output_file, target_fps=25):
    """Ensures input video matches target extraction FPS."""
    video = VideoFileClip(input_file)
    if video.fps != target_fps:
        video.write_videofile(output_file, fps=target_fps, codec='libx264', audio_codec='aac', logger=None)
    else:
        import shutil
        shutil.copy2(input_file, output_file)
    video.close()

def extract_audio(video_file, audio_output_file, sample_rate=8000):
    """Extracts raw WAV audio from MP4."""
    video = VideoFileClip(video_file)
    video.audio.write_audiofile(audio_output_file, fps=sample_rate, nbytes=2, codec='pcm_s16le', logger=None)
    video.close()

def merge_video_audio(video_file, audio_file, output_file):
    """Creates final video with isolated audio track."""
    video = VideoFileClip(video_file)
    audio = AudioFileClip(audio_file)
    set_audio_fn = getattr(video, "set_audio", getattr(video, "with_audio", None))
    final_video = set_audio_fn(audio)
    final_video.write_videofile(output_file, codec='libx264', audio_codec='aac', logger=None)
    video.close()
    audio.close()
    final_video.close()

def process_video(input_file, output_path, number_of_speakers, detect_every_N_frame, scalar_face_detection, config_path, audio_weights, video_weights, sdk_path, license_path):
    """
    Main execution pipeline for AV-ConvTasNet offline inference.
    
    Args:
        input_file (str): Input MP4 video path.
        output_path (str): Output save directory.
        number_of_speakers (int): Number of sources to extract.
        detect_every_N_frame (int): SDK detection parameter.
        scalar_face_detection (float): Face crop box multiplier.
        config_path (str): AV-ConvTasNet YAML architecture.
        audio_weights (str): Weights for audio component.
        video_weights (str): Weights for ResNet video component.
        sdk_path (str): BlueSkeye SDK directory.
        license_path (str): BlueSkeye license file.
        
    Returns:
        list: File paths of separated videos.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.backends.mps.is_available():
        # Fallback to CPU if MPS gives issues, AV-ConvTasNet might have 3D convolutions not supported on MPS
        device = torch.device('cpu') 
    print(f"[System] AI Device: {device}")
    
    api, image_type = init_sdk(sdk_path, license_path)
    api.reset()
    api.set_min_process_time(0)
    api.set_inference_increment_enabled(False)
    api.set_log_level(0)
    
    os.makedirs(output_path, exist_ok=True)
    temp_25fps_file = os.path.join(output_path, 'temp_25fps.mp4')
    convert_video_fps(input_file, temp_25fps_file, target_fps=TARGET_FPS)
    
    # 1. SDK Video Tracking
    filename_path = detectface(temp_25fps_file, output_path, detect_every_N_frame, scalar_face_detection, number_of_speakers, api, image_type)
    
    # 2. Audio Extraction
    audio_output = os.path.join(output_path, 'audio.wav')
    extract_audio(temp_25fps_file, audio_output, sample_rate=AUDIO_SAMPLE_RATE)
    
    # Copy assets for landmark alignment
    dolphin_assets = '/Users/rohit/Desktop/avss/dolphin/bskai_demo/assets'
    local_assets = os.path.join(os.path.dirname(__file__), 'assets')
    if not os.path.exists(local_assets) and os.path.exists(dolphin_assets):
        import shutil
        shutil.copytree(dolphin_assets, local_assets)
        
    # 3. Mouth Extraction (Grayscale for AV-ConvTasNet frontend)
    crop_mouth(os.path.join(output_path, "faces"), 
               os.path.join(output_path, "landmark"), 
               filename_path, 
               os.path.join(output_path, "mouthroi"), 
               convert_gray=True, 
               testset_only=False)
    
    # 4. Architecture Instantiation
    print(f"[System] Loading architecture config from: {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize Video Model (ResNet18)
    videomodel = video(**config['video_model'])
    if video_weights and os.path.exists(video_weights):
        if os.path.getsize(video_weights) < 1000:
            print(f"[Warning] Video pretrain file '{video_weights}' appears to be a Git LFS pointer. Skipping video weights load.")
        else:
            pretrain = torch.load(video_weights, map_location='cpu', weights_only=False)['model_state_dict']
            videomodel = update_parameter(videomodel, pretrain)
    videomodel.to(device)
    videomodel.eval()
    
    # Initialize Audio Model (AV-ConvTasNet)
    audiomodel = AV_model(**config['AV_model'])
    
    if audio_weights and os.path.exists(audio_weights):
        print(f"[System] Loading audio weights from: {audio_weights}")
        state_dict = torch.load(audio_weights, map_location='cpu', weights_only=False)
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        audiomodel = load_state_dict_in(audiomodel, state_dict)
    else:
        print("[System] No audio weights provided or found.")
        
    audiomodel.to(device)
    audiomodel.eval()
    
    # 5. Segmented Inference Loop (Overlap-Add)
    with torch.no_grad():
        for i in range(number_of_speakers):
            mouth_roi = np.load(os.path.join(output_path, "mouthroi", f"speaker{i+1}.npz"))["data"]
            
            mix, sr = torchaudio.load(audio_output)
            mix = mix.mean(dim=0) 
            
            # Resample if needed
            if sr != AUDIO_SAMPLE_RATE:
                resampler = torchaudio.transforms.Resample(sr, AUDIO_SAMPLE_RATE)
                mix = resampler(mix)
                sr = AUDIO_SAMPLE_RATE

            window_size = int(WINDOW_SIZE_SEC * sr)
            hop_size = int(HOP_SIZE_SEC * sr)
            all_estimates = []
            
            # Sliding window execution
            start_idx = 0
            while start_idx < len(mix):
                end_idx = min(start_idx + window_size, len(mix))
                window_mix = mix[start_idx:end_idx]
                
                # Zero padding if at the very end
                if len(window_mix) < window_size:
                    pad = torch.zeros(window_size - len(window_mix))
                    window_mix = torch.cat((window_mix, pad))
                    
                # Match temporal space for video frames
                start_frame = int(start_idx / sr * TARGET_FPS)
                end_frame = int(end_idx / sr * TARGET_FPS)
                end_frame = min(end_frame, len(mouth_roi))
                window_mouth_roi = mouth_roi[start_frame:end_frame]
                
                if len(window_mouth_roi) == 0:
                    start_idx += hop_size
                    continue
                
                # Execute isolation logic
                isolated_numpy_result = process_single_chunk_isolated(
                    audiomodel, videomodel, window_mix, window_mouth_roi, device
                )
                
                if isolated_numpy_result.ndim == 3:
                    isolated_numpy_result = isolated_numpy_result[0]
                elif isolated_numpy_result.ndim == 1:
                    isolated_numpy_result = isolated_numpy_result[None, :]
                
                all_estimates.append({
                    'start': start_idx,
                    'end': start_idx + len(window_mix),
                    'estimate': isolated_numpy_result 
                })
                
                del window_mix
                del window_mouth_roi
                
                if device.type == 'mps':
                    torch.mps.empty_cache()
                gc.collect() 
                
                start_idx += hop_size
                if start_idx >= len(mix): break
            
            # 6. Overlap-Add Reconstruction
            output_length = len(mix)
            merged_output = torch.zeros(1, output_length)
            weights = torch.zeros(output_length)
            
            for est in all_estimates:
                window_len = est['end'] - est['start']
                if window_len > window_size:
                    window_len = window_size
                hann_window = torch.hann_window(window_len)
                
                cpu_tensor_chunk = torch.from_numpy(est['estimate'])
                
                valid_len = min(window_len, output_length - est['start'])
                if valid_len <= 0: continue
                
                # Handling outputs shape to match
                est_data = cpu_tensor_chunk[0, :valid_len]
                if est_data.shape[0] < valid_len:
                    valid_len = est_data.shape[0]

                merged_output[0, est['start']:est['start']+valid_len] += est_data[:valid_len] * hann_window[:valid_len]
                weights[est['start']:est['start']+valid_len] += hann_window[:valid_len]
            
            merged_output[:, weights > 0] /= weights[weights > 0]
            
            # Save the final estimated audio
            torchaudio.save(os.path.join(output_path, f"speaker{i+1}_est.wav"), merged_output, sr)

    # 7. Merge final videos
    output_files = []
    for i in range(number_of_speakers):
        video_input = os.path.join(output_path, f"video_tracked{i+1}.mp4")
        audio_input = os.path.join(output_path, f"speaker{i+1}_est.wav")
        video_output = os.path.join(output_path, f"s{i+1}.mp4")
        
        merge_video_audio(video_input, audio_input, video_output)
        output_files.append(video_output)
    
    if os.path.exists(temp_25fps_file): os.remove(temp_25fps_file)
    return output_files

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Video Speaker Separation via AV-ConvTasNet')
    parser.add_argument('--input', '-i', type=str, required=True, help='Path to input video file')
    parser.add_argument('--output', '-o', type=str, default=None, help='Output directory path')
    parser.add_argument('--speakers', '-s', type=int, default=2, help='Number of speakers (default: 2)')
    parser.add_argument('--detect-every-n', type=int, default=8, help='Detect faces every N frames')
    parser.add_argument('--face-scale', type=float, default=1.5, help='Face bounding box scale factor')
    
    parser.add_argument('--config', type=str, default="../config/train.yml", help='Local YAML Config path')
    parser.add_argument('--audio-weights', type=str, default=None, help='Local Weights path for AV-ConvTasNet')
    parser.add_argument('--video-weights', type=str, default=None, help='Local Weights path for video_resnet18')
    
    parser.add_argument("-sdk", "--sdk-path", required=True, help="Path to B-Automotive SDK")
    parser.add_argument("-l", "-lic", "--license-path", required=True, help="Path to SDK License")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' does not exist")
        exit(1)
    
    if args.output is None:
        input_basename = os.path.splitext(os.path.basename(args.input))[0]
        args.output = os.path.join(os.path.dirname(args.input), input_basename + "_output")
    
    output_files = process_video(
        input_file=args.input,
        output_path=args.output,
        number_of_speakers=args.speakers,
        detect_every_N_frame=args.detect_every_n,
        scalar_face_detection=args.face_scale,
        config_path=args.config,
        audio_weights=args.audio_weights,
        video_weights=args.video_weights,
        sdk_path=args.sdk_path,
        license_path=args.license_path
    )
    
    print("\nProcessing completed!")
    for i, output_file in enumerate(output_files):
        print(f"  Speaker {i+1}: {output_file}")
