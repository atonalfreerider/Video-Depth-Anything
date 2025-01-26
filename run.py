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
import argparse
import os
import torch
import gc

from video_depth_anything.video_depth import VideoDepthAnything

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Video Depth Anything')
    parser.add_argument('--input_video', type=str, default='./assets/example_videos/davis_rollercoaster.mp4')
    parser.add_argument('--output_dir', type=str, default='./outputs')
    parser.add_argument('--input_size', type=int, default=518)
    parser.add_argument('--max_res', type=int, default=1080)
    parser.add_argument('--encoder', type=str, default='vits', choices=['vits', 'vitl'])
    parser.add_argument('--target_fps', type=int, default=-1, help='target fps of the input video, -1 means the original fps')
    parser.add_argument('--fp32', action='store_true', help='model infer with torch.float32, default is torch.float16')
    parser.add_argument('--grayscale', action='store_true', help='do not apply colorful palette')
    parser.add_argument('--save_npz', action='store_true', help='save depths as npz')
    parser.add_argument('--save_exr', action='store_true', help='save depths as exr')

    args = parser.parse_args()

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create depth output directory
    depth_output_dir = os.path.join(args.output_dir, 'depths')
    os.makedirs(depth_output_dir, exist_ok=True)

    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    }

    video_depth_anything = VideoDepthAnything(**model_configs[args.encoder])
    video_depth_anything.load_state_dict(
        torch.load(
            f'./checkpoints/video_depth_anything_{args.encoder}.pth',
            map_location='cpu',
            weights_only=True  # Add safe loading parameter
        ),
        strict=True
    )
    video_depth_anything = video_depth_anything.to(DEVICE).eval()

    fps = video_depth_anything.infer_video_depth(
        args.input_video, 
        depth_output_dir,
        args.target_fps, 
        input_size=args.input_size, 
        device=DEVICE
    )

    # Save FPS information
    with open(os.path.join(depth_output_dir, 'metadata.txt'), 'w') as f:
        f.write(f'fps: {fps}\n')

    # Clear final memory
    del video_depth_anything
    torch.cuda.empty_cache()
    gc.collect()
