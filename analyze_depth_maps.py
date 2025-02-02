import argparse
import numpy as np
import os
import json
from tqdm import tqdm
from glob import glob
from video_depth_anything.depth_analysis import analyze_depth_frame

def process_depth_maps(depth_dir: str, poses_json: str = None, output_json: str = None, debug: bool = False):
    """Process all depth maps in a directory and save analysis results."""
    print(f"Looking for depth maps in: {os.path.abspath(depth_dir)}")
    
    # Load video metadata
    with open(os.path.join(depth_dir, 'video_metadata.json'), 'r') as f:
        metadata = json.load(f)
    
    frame_height = metadata['frame_height']
    frame_width = metadata['frame_width']
    
    # Load poses if provided
    poses_data = None
    if poses_json and os.path.exists(poses_json):
        with open(poses_json, 'r') as f:
            poses_data = json.load(f)

    # Get all depth map files
    depth_files = sorted(glob(os.path.join(depth_dir, 'depth_*.npz')))
    print(f"Found {len(depth_files)} depth map files")
    
    if len(depth_files) == 0:
        print("\nDirectory contents:")
        for f in os.listdir(depth_dir):
            print(f"  {f}")
        raise ValueError(f"No depth map files found in {depth_dir}")

    # Prepare output structure
    output_data = {
        "metadata": metadata,
        "frames": {}
    }

    def safe_calculate_y(slope, intercept, frame_height):
        """Safely calculate y coordinate from slope and intercept"""
        try:
            if abs(slope) < 1e-6:  # Nearly horizontal line
                return frame_height // 2
            y = int((-intercept) / slope)
            return min(max(y, 0), frame_height - 1)
        except:
            return frame_height // 2

    # Process each depth map
    for depth_file in tqdm(depth_files, desc="Processing depth maps", dynamic_ncols=True):
        try:
            frame_idx = int(os.path.basename(depth_file).split('_')[1].split('.')[0])
            str_idx = str(frame_idx)
            
            # Load depth map silently
            with np.load(depth_file) as depth_data:
                depth_map = depth_data['depth']
            
            # Analyze depth map with minimal output
            analysis_result = analyze_depth_frame(
                depth_map,
                frame_height,
                frame_width,
                visualize=False,  # Only visualize in debug mode
                debug=debug
            )
            
            # Initialize frame data
            output_data["frames"][str_idx] = []
            
            if poses_data and str_idx in poses_data["frames"]:
                for pose in poses_data["frames"][str_idx]:
                    new_pose = pose.copy()
                    new_pose["joints2d"] = []
                    
                    # Add depth to joints with boundary checking
                    for (x, y) in pose["joints2d"]:
                        xi = min(max(int(round(x)), 0), frame_width - 1)
                        yi = min(max(int(round(y)), 0), frame_height - 1)
                        d_val = float(depth_map[yi, xi])
                        new_pose["joints2d"].append([x, y, d_val])
                    
                    # Add structure lines if analysis was successful
                    if analysis_result["success"]:
                        wall_analysis = analysis_result["wall_analysis"]
                        
                        # Add wall lines with depth values and boundary checking
                        wall_lines = []
                        if wall_analysis["type"] == "double_wall":
                            # Left wall line
                            left_eq = wall_analysis["left_equation"]
                            x1, y1 = 0, 0
                            x2 = min(int(wall_analysis["corner"]["x"]), frame_width - 1)
                            y2 = safe_calculate_y(left_eq["slope"], left_eq["intercept"], frame_height)
                            
                            if y2 is not None:  # Only add wall if y calculation succeeded
                                d1 = float(depth_map[y1, x1])
                                d2 = wall_analysis["corner"]["depth"]
                                wall_lines.append({
                                    "type": "left_wall",
                                    "start": [x1, y1, d1],
                                    "end": [x2, y2, d2]
                                })
                            
                                # Right wall line - only add if left wall was successful
                                right_eq = wall_analysis["right_equation"]
                                x1 = x2  # Use previous end point
                                y1 = y2
                                x2 = frame_width - 1
                                y2 = safe_calculate_y(right_eq["slope"], right_eq["intercept"], frame_height)
                                
                                if y2 is not None:
                                    d1 = wall_analysis["corner"]["depth"]
                                    d2 = float(depth_map[y2, x2])
                                    wall_lines.append({
                                        "type": "right_wall",
                                        "start": [x1, y1, d1],
                                        "end": [x2, y2, d2]
                                    })
                        else:
                            # Single wall line
                            eq = wall_analysis["equation"]
                            x1, y1 = 0, 0
                            x2 = frame_width - 1
                            y2 = safe_calculate_y(eq["slope"], eq["intercept"], frame_height)
                            
                            if y2 is not None:
                                d1 = float(depth_map[y1, x1])
                                d2 = float(depth_map[y2, x2])
                                wall_lines.append({
                                    "type": "wall",
                                    "start": [x1, y1, d1],
                                    "end": [x2, y2, d2]
                                })
                        
                        if wall_lines:  # Only add structure_lines if we have valid walls
                            new_pose["structure_lines"] = {
                                "wall_lines": wall_lines
                            }
                    
                    output_data["frames"][str_idx].append(new_pose)
            else:
                # Store just the wall analysis
                if analysis_result["success"]:
                    output_data["frames"][str_idx].append({
                        "depth_analysis": {
                            "success": True,
                            "wall_analysis": analysis_result["wall_analysis"]
                        }
                    })
                else:
                    output_data["frames"][str_idx].append({
                        "depth_analysis": {
                            "success": False,
                            "error": analysis_result.get("error", "Unknown error")
                        }
                    })
                
        except Exception as e:
            print(f"\nError frame {frame_idx}: {str(e)}")
            continue

        # Periodically save results to avoid data loss
        if frame_idx % 1000 == 0 and output_json:
            with open(output_json + '.temp', 'w') as f:
                json.dump(output_data, f)
    
    # Final save
    if output_json:
        if os.path.exists(output_json + '.temp'):
            os.rename(output_json + '.temp', output_json)
        else:
            with open(output_json, 'w') as f:
                json.dump(output_data, f, indent=4)
    
    return output_data

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyze saved depth maps')
    parser.add_argument('--depth_dir', type=str, required=True, 
                      help='Directory containing depth map NPZ files')
    parser.add_argument('--poses_json', type=str, default=None,
                      help='Optional path to poses JSON file')
    parser.add_argument('--output_json', type=str, required=True,
                      help='Output path for analysis results')
    parser.add_argument('--debug', action='store_true',
                      help='Enable debug visualizations')
    
    args = parser.parse_args()
    
    # Ensure depth_dir exists
    if not os.path.exists(args.depth_dir):
        raise ValueError(f"Directory does not exist: {args.depth_dir}")
    
    process_depth_maps(args.depth_dir, args.poses_json, args.output_json, args.debug)
