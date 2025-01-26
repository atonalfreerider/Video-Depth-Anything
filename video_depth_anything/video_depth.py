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
        depth = self.head(features, patch_h, patch_w, T)[0]
        depth = F.interpolate(depth, size=(H, W), mode="bilinear", align_corners=True)
        depth = F.relu(depth)
        # Fix reversed scale by inverting depth:
        depth = 1.0 / (depth + 1e-6)
        return depth.squeeze(1).unflatten(0, (B, T)) # return shape [B, T, H, W]
    
    def infer_video_depth(self, video_path, output_dir, target_fps, input_size=518, device='cuda'):
        os.makedirs(output_dir, exist_ok=True)
        
        cap = cv2.VideoCapture(video_path)
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        fps = cap.get(cv2.CAP_PROP_FPS) if target_fps == -1 else target_fps
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        ratio = max(frame_height, frame_width) / min(frame_height, frame_width)
        if ratio > 1.78:  # we recommend to process video with ratio smaller than 16:9 due to memory limitation
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
        pre_input = None
        frames_since_clear = 0
        CLEAR_INTERVAL = 256
        
        # Track all frames for 1:1 correspondence
        all_frames = []
        frame_count = 0

        # Set PyTorch memory settings
        torch.cuda.set_per_process_memory_fraction(0.85)
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()

        # First pass: read all frames
        pbar = tqdm(total=total_frames, desc="Reading frames")
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            all_frames.append(frame)
            frame_count += 1
            pbar.update(1)
        cap.release()
        pbar.close()

        # Second pass: process frames ensuring 1:1 correspondence
        pbar = tqdm(total=frame_count, desc="Processing depth")
        saved_frame_count = 0
        
        for batch_idx, start_idx in enumerate(range(0, frame_count, frame_step)):
            end_idx = min(start_idx + INFER_LEN, frame_count)
            current_batch = all_frames[start_idx:end_idx]
            
            # Pad batch if needed
            while len(current_batch) < INFER_LEN:
                current_batch.append(current_batch[-1].copy())

            cur_list = []
            for frame in current_batch:
                processed = transform({'image': frame.astype(np.float32) / 255.0})['image']
                cur_list.append(torch.from_numpy(processed).unsqueeze(0).unsqueeze(0))
                del processed
            
            cur_input = torch.cat(cur_list, dim=1).to(device)
            del cur_list

            # Apply overlap for temporal consistency (but don't save overlapped frames)
            if pre_input is not None:
                cur_input[:, :OVERLAP, ...] = pre_input[:, KEYFRAMES, ...]

            with torch.no_grad():
                depth = self.forward(cur_input)
                depth = F.interpolate(depth.flatten(0,1).unsqueeze(1), 
                                   size=(frame_height, frame_width), 
                                   mode='bilinear', 
                                   align_corners=True)
                
                # For first batch, save all frames
                # For subsequent batches, skip the overlapped frames
                save_start = OVERLAP if batch_idx > 0 else 0
                save_end = min(end_idx - start_idx, depth.shape[0])
                
                for i in range(save_start, save_end):
                    frame_idx = start_idx + i
                    if frame_idx < frame_count:
                        depth_frame = depth[i][0].cpu().numpy()
                        np.savez_compressed(
                            os.path.join(output_dir, f'depth_{saved_frame_count:06d}.npz'),
                            depth=depth_frame
                        )
                        saved_frame_count += 1
                        pbar.update(1)

            pre_input = cur_input.clone()
            del cur_input
            del depth

            frames_since_clear += len(current_batch)
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

        pbar.close()
        
        # Final cleanup
        if pre_input is not None:
            del pre_input
        del all_frames
        torch.cuda.empty_cache()
        gc.collect()
        
        print(f"Processed {frame_count} input frames, saved {saved_frame_count} depth maps")
        return fps
