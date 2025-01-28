# Copyright (2025) Bytedance Ltd. and/or its affiliates 

# Licensed under the Apache License, Version 2.0 (the "License"); 
# you may not use this file except in compliance with the License. 
# You may obtain a copy of the License at 

#     http://www.apache.org/licenses/LICENSE-2.0 

# Unless required by applicable law or agreed to in writing, software 
# distributed under the License is distributed on an "AS IS" BASIS, 
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
# See the License for the specific language governing permissions and 
# limitations under the License. 
import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision.transforms import Compose
import cv2
from tqdm import tqdm
import numpy as np
import gc
import os
import json

from .dinov2 import DINOv2
from .dpt_temporal import DPTHeadTemporal
from .util.transform import Resize, NormalizeImage, PrepareForNet

# infer settings, do not change
INFER_LEN = 32
OVERLAP = 10
KEYFRAMES = [0,12,24,25,26,27,28,29,30,31]
INTERP_LEN = 8

class VideoDepthAnything(nn.Module):
    def __init__(
        self,
        encoder='vitl',
        features=256, 
        out_channels=[256, 512, 1024, 1024], 
        use_bn=False, 
        use_clstoken=False,
        num_frames=32,
        pe='ape'
    ):
        super(VideoDepthAnything, self).__init__()

        self.intermediate_layer_idx = {
            'vits': [2, 5, 8, 11],
            'vitl': [4, 11, 17, 23]
        }
        
        self.encoder = encoder
        self.pretrained = DINOv2(model_name=encoder)

        self.head = DPTHeadTemporal(self.pretrained.embed_dim, features, use_bn, out_channels=out_channels, use_clstoken=use_clstoken, num_frames=num_frames, pe=pe)

    def forward(self, x):
        B, T, C, H, W = x.shape
        patch_h, patch_w = H // 14, W // 14
        features = self.pretrained.get_intermediate_layers(x.flatten(0,1), self.intermediate_layer_idx[self.encoder], return_class_token=True)
        depth = self.head(features, patch_h, patch_w, T)
        depth = F.interpolate(depth, size=(H, W), mode="bilinear", align_corners=True)
        depth = F.relu(depth)
        # Fix reversed scale by inverting depth:
        depth = 1.0 / (depth + 1e-6)
        return depth.squeeze(1).unflatten(0, (B, T)) # return shape [B, T, H, W]
    
    def infer_video_depth(
        self, video_path, output_dir, target_fps, input_size=518, device='cuda',
        poses_data=None, augmented_json_path=None
    ):
        os.makedirs(output_dir, exist_ok=True)
        
        cap = cv2.VideoCapture(video_path)
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        fps = cap.get(cv2.CAP_PROP_FPS) if target_fps == -1 else target_fps
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        ratio = max(frame_height, frame_width) / min(frame_height, frame_width)
        if ratio > 1.78:
            input_size = int(input_size * 1.777 / ratio)
            input_size = round(input_size / 14) * 14

        transform = Compose([
            Resize(
                width=input_size,
                height=input_size,
                resize_target=False,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method='lower_bound',
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ])

        frame_step = INFER_LEN - OVERLAP
        frame_buffer = []
        pre_input = None
        current_frame_idx = 0  # Initialize here
        last_saved_frame = -1
        frames_since_clear = 0
        CLEAR_INTERVAL = 256  # Reduced interval

        # Prepare structure for augmented poses
        augmented_poses = {
            "metadata": poses_data["metadata"] if poses_data else {},
            "frames": {}
        }

        # Set PyTorch memory settings
        torch.cuda.set_per_process_memory_fraction(0.85)
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()

        pbar = tqdm(total=total_frames)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_buffer.append(frame)
            pbar.update(1)

            if len(frame_buffer) == INFER_LEN or not ret:
                # Pad buffer if needed
                while len(frame_buffer) < INFER_LEN:
                    frame_buffer.append(frame_buffer[-1].copy())

                cur_list = []
                for frame in frame_buffer:
                    processed = transform({'image': frame.astype(np.float32) / 255.0})['image']
                    cur_list.append(torch.from_numpy(processed).unsqueeze(0).unsqueeze(0))
                    del processed
                
                cur_input = torch.cat(cur_list, dim=1).to(device)
                del cur_list

                if pre_input is not None:
                    cur_input[:, :OVERLAP, ...] = pre_input[:, KEYFRAMES, ...]

                with torch.no_grad():
                    depth = self.forward(cur_input)
                    depth = F.interpolate(depth.flatten(0,1).unsqueeze(1), 
                                       size=(frame_height, frame_width), 
                                       mode='bilinear', 
                                       align_corners=True)
                    
                    for i in range(depth.shape[0]):
                        frame_idx = current_frame_idx + i
                        if frame_idx >= total_frames or frame_idx <= last_saved_frame:
                            continue
                        # Grab 2D pose data, sample depth
                        str_idx = str(frame_idx)
                        if poses_data and str_idx in poses_data["frames"]:
                            depth_map = depth[i][0].cpu().numpy()
                            augmented_poses["frames"][str_idx] = []
                            for pose in poses_data["frames"][str_idx]:
                                new_pose = pose.copy()
                                new_pose["joints2d"] = []
                                for (x, y) in pose["joints2d"]:
                                    xi, yi = int(round(x)), int(round(y))
                                    if 0 <= xi < frame_width and 0 <= yi < frame_height:
                                        d_val = float(depth_map[yi, xi])
                                    else:
                                        d_val = 0.0
                                    new_pose["joints2d"].append([x, y, d_val])
                                augmented_poses["frames"][str_idx].append(new_pose)

                        last_saved_frame = frame_idx

                pre_input = cur_input.clone()
                del cur_input
                del depth

                frames_since_clear += len(frame_buffer) - OVERLAP
                if frames_since_clear >= CLEAR_INTERVAL:
                    # Safe memory clearing
                    if pre_input is not None:
                        pre_input_cpu = pre_input.cpu()
                        del pre_input
                        torch.cuda.empty_cache()
                        gc.collect()
                        pre_input = pre_input_cpu.to(device)
                        del pre_input_cpu
                    else:
                        torch.cuda.empty_cache()
                        gc.collect()
                    frames_since_clear = 0

                current_frame_idx += frame_step  # Update using frame_step
                frame_buffer = frame_buffer[frame_step:]

        cap.release()
        pbar.close()
        
        # Final cleanup
        if pre_input is not None:
            del pre_input
        torch.cuda.empty_cache()
        gc.collect()

        # Write augmented poses to JSON
        if poses_data and augmented_json_path:
            with open(augmented_json_path, 'w') as f:
                json.dump(augmented_poses, f, indent=4)

        return fps
