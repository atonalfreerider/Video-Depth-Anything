import os
import numpy as np
import argparse
import imageio
from matplotlib import cm
from tqdm import tqdm
import glob
import cv2
import json

def create_depth_visualization(input_dir, output_path, max_frames=-1, augmented_poses_json=None):
    # Get all npz files and sort them
    depth_files = sorted(glob.glob(os.path.join(input_dir, 'depth_*.npz')))
    
    # Validate frame sequence
    frame_numbers = [int(os.path.basename(f).split('_')[1].split('.')[0]) for f in depth_files]
    expected_sequence = list(range(min(frame_numbers), max(frame_numbers) + 1))
    
    if frame_numbers != expected_sequence:
        print("Warning: Frame sequence is not continuous!")
        print(f"Found {len(frame_numbers)} frames, expected {len(expected_sequence)}")
        print(f"First missing frames: {set(expected_sequence) - set(frame_numbers)}")
        
        # Ensure we have a continuous sequence
        valid_files = []
        last_frame = -1
        for f, num in zip(depth_files, frame_numbers):
            if num != last_frame + 1 and last_frame != -1:
                print(f"Gap detected between frames {last_frame} and {num}")
            valid_files.append(f)
            last_frame = num
        depth_files = valid_files

    if max_frames > 0:
        depth_files = depth_files[:max_frames]
    
    # Read metadata for FPS
    fps = 30  # default
    metadata_path = os.path.join(input_dir, 'metadata.txt')
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            for line in f:
                if line.startswith('fps:'):
                    fps = float(line.split(':')[1].strip())

    # Initialize video writer
    writer = imageio.get_writer(output_path, fps=fps, macro_block_size=1, 
                              codec='libx264', ffmpeg_params=['-crf', '18'])
    
    # Get colormap
    colormap = np.array(cm.get_cmap('inferno').colors)
    
    # Find global min/max for consistent visualization
    print("Computing depth range...")
    depths = []
    sample_files = depth_files[:min(100, len(depth_files))]
    for f in tqdm(sample_files):
        depth = np.load(f)['depth']
        depths.append(depth)
    depths = np.stack(depths)
    d_min, d_max = depths.min(), depths.max()
    del depths  # Free memory
    
    poses_data = None
    if augmented_poses_json and os.path.exists(augmented_poses_json):
        with open(augmented_poses_json, 'r') as f:
            poses_data = json.load(f)

    # Process each depth map
    print("Creating visualization...")
    last_frame_num = -1
    for depth_file in tqdm(depth_files):
        # Check frame number continuity
        frame_num = int(os.path.basename(depth_file).split('_')[1].split('.')[0])
        if last_frame_num != -1 and frame_num != last_frame_num + 1:
            print(f"Warning: Frame jump from {last_frame_num} to {frame_num}")
        
        # Load depth map
        depth = np.load(depth_file)['depth']
        
        # Normalize and colorize
        depth_norm = ((depth - d_min) / (d_max - d_min) * 255).astype(np.uint8)
        depth_vis = (colormap[depth_norm] * 255).astype(np.uint8)
        
        # Convert to BGR for cv2 drawing
        depth_bgr = cv2.cvtColor(depth_vis, cv2.COLOR_RGBA2BGR)

        # Overlay keypoints if we have them
        if poses_data:
            frame_num = int(os.path.basename(depth_file).split('_')[1].split('.')[0])
            str_idx = str(frame_num)
            if "frames" in poses_data and str_idx in poses_data["frames"]:
                for pose in poses_data["frames"][str_idx]:
                    for (x, y, d_val) in pose["joints2d"]:
                        xi, yi = int(round(x)), int(round(y))
                        if 0 <= xi < depth_bgr.shape[1] and 0 <= yi < depth_bgr.shape[0]:
                            cv2.circle(depth_bgr, (xi, yi), 4, (0, 255, 0), -1)
                            cv2.putText(
                                depth_bgr, f"{d_val:.2f}m", (xi+5, yi-5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
                            )

        writer.append_data(cv2.cvtColor(depth_bgr, cv2.COLOR_BGR2RGB))
        
        last_frame_num = frame_num
    
    writer.close()
    print(f"Visualization saved to: {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Depth Map Visualization')
    parser.add_argument('--input_dir', type=str, required=True, 
                        help='Directory containing depth_*.npz files')
    parser.add_argument('--output_path', type=str, required=True,
                        help='Output video path (e.g., output.mp4)')
    parser.add_argument('--max_frames', type=int, default=-1,
                        help='Maximum number of frames to process (-1 for all)')
    parser.add_argument('--augmented_poses_json', type=str, default=None,
                        help='Path to the augmented poses JSON file')
    
    args = parser.parse_args()
    create_depth_visualization(
        args.input_dir,
        args.output_path,
        args.max_frames,
        augmented_poses_json=args.augmented_poses_json
    )
