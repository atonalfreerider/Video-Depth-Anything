import argparse
import numpy as np
import os
import json
from tqdm import tqdm
from glob import glob
from video_depth_anything.depth_analysis import analyze_depth_frame

def process_depth_maps(depth_dir: str, frame_height: int, frame_width: int, output_json: str):
    """Process all depth maps in a directory and save analysis results."""
    # Debug prints
    print(f"Looking for depth maps in: {os.path.abspath(depth_dir)}")
    
    # Get all depth map files - updated pattern to match 'frame_*.npz'
    depth_files = sorted(glob(os.path.join(depth_dir, 'frame_*.npz')))
    print(f"Found {len(depth_files)} depth map files")
    
    if len(depth_files) == 0:
        # List directory contents for debugging
        print("\nDirectory contents:")
        for f in os.listdir(depth_dir):
            print(f"  {f}")
        raise ValueError(f"No depth map files found in {depth_dir}")

    # Prepare structure for analysis results
    analysis_results = {
        "metadata": {
            "frame_height": frame_height,
            "frame_width": frame_width,
            "total_frames": len(depth_files),
            "depth_files": depth_files  # Add this for debugging
        },
        "frames": {}
    }

    # Process each depth map
    for depth_file in tqdm(depth_files, desc="Analyzing depth maps"):
        try:
            # Extract frame number from filename
            frame_idx = int(os.path.basename(depth_file).split('_')[1].split('.')[0])
            
            # Load depth map
            depth_data = np.load(depth_file)
            print(f"\nDepth file keys: {depth_data.files}")  # Debug print
            depth_map = depth_data['depth']
            
            # Analyze depth map
            analysis_result = analyze_depth_frame(
                depth_map,
                frame_height,
                frame_width,
                visualize=True
            )
            
            # Store results
            analysis_results["frames"][str(frame_idx)] = [{
                "depth_analysis": analysis_result
            }]
            
        except Exception as e:
            print(f"\nError processing {depth_file}: {str(e)}")
            continue
    
    # Save results
    with open(output_json, 'w') as f:
        json.dump(analysis_results, f, indent=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyze saved depth maps')
    parser.add_argument('--depth_dir', type=str, required=True, help='Directory containing depth map NPZ files')
    parser.add_argument('--frame_height', type=int, required=True, help='Original video frame height')
    parser.add_argument('--frame_width', type=int, required=True, help='Original video frame width')
    parser.add_argument('--output_json', type=str, required=True, help='Output path for analysis results')
    
    args = parser.parse_args()
    
    # Ensure depth_dir exists
    if not os.path.exists(args.depth_dir):
        raise ValueError(f"Directory does not exist: {args.depth_dir}")
    
    process_depth_maps(
        args.depth_dir,
        args.frame_height,
        args.frame_width,
        args.output_json
    )
