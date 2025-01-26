import os
import numpy as np
import sys
import imageio
from matplotlib import cm
from tqdm import tqdm
import glob

def create_depth_visualization(input_dir, output_path):
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
    colormap = np.array(cm.get_cmap('inferno_r').colors)
    
    # Find global min/max for consistent visualization
    print("Computing depth range...")
    all_mins = []
    all_maxs = []
    all_percentiles = []
    
    # Sample more frames and use percentiles for robustness
    sample_step = max(1, len(depth_files) // 200)  # Sample ~200 frames evenly
    sample_files = depth_files[::sample_step]
    
    for f in tqdm(sample_files):
        depth = np.load(f)['depth']
        # Remove invalid values
        valid_depth = depth[np.isfinite(depth) & (depth > 0)]
        if len(valid_depth) > 0:
            all_mins.append(np.percentile(valid_depth, 1))  # 1st percentile
            all_maxs.append(np.percentile(valid_depth, 99))  # 99th percentile
            all_percentiles.extend(np.percentile(valid_depth, [5, 25, 75, 95]))
    
    # Use robust statistics
    if all_mins and all_maxs:
        d_min = np.percentile(all_mins, 5)  # Conservative minimum
        d_max = np.percentile(all_maxs, 95)  # Conservative maximum
    else:
        d_min, d_max = 0.1, 10.0  # Fallback values
    
    print(f"Depth range: {d_min:.3f} to {d_max:.3f}")
    
    # Process each depth map
    print("Creating visualization...")
    for depth_file in tqdm(depth_files):
        # Load depth map
        depth = np.load(depth_file)['depth']
        
        # Handle invalid values
        depth = np.where(np.isfinite(depth) & (depth > 0), depth, d_max)
        
        # Normalize with clamping
        depth_clamped = np.clip(depth, d_min, d_max)
        depth_norm = ((depth_clamped - d_min) / (d_max - d_min) * 255).astype(np.uint8)
        depth_vis = (colormap[depth_norm] * 255).astype(np.uint8)
        
        writer.append_data(depth_vis)
    
    writer.close()
    print(f"Visualization saved to: {output_path}")

if __name__ == '__main__':
    input_dir = sys.argv[1]
    output_path = os.path.join(os.path.dirname(input_dir), 'depth.mp4')
    create_depth_visualization(input_dir, output_path)
