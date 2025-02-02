import argparse
import cv2
import numpy as np
import json
import os
from tqdm import tqdm
from pathlib import Path
from video_depth_anything.depth_analysis import analyze_depth_frame
from scipy.interpolate import UnivariateSpline

JOINT_CONNECTIONS = [
    (0, 1), (0, 2), (0, 3),  # Pelvis to hips and spine1
    (1, 4), (4, 7), (7, 10),  # Left leg
    (2, 5), (5, 8), (8, 11),  # Right leg
    (3, 6), (6, 9), (9, 12),  # Spine
    (12, 13), (12, 14), (12, 15),  # Neck to collars and head
    (13, 16), (16, 18), (18, 20), (20, 22),  # Left arm
    (14, 17), (17, 19), (19, 21), (21, 23)   # Right arm
]

def draw_skeleton(frame, joints, connections=JOINT_CONNECTIONS, color=(0, 255, 0), thickness=2):
    """Draw skeleton connections on frame."""
    for joint1_idx, joint2_idx in connections:
        if joint1_idx >= len(joints) or joint2_idx >= len(joints):
            continue
        joint1 = joints[joint1_idx]
        joint2 = joints[joint2_idx]
        pt1 = (int(joint1[0]), int(joint1[1]))
        pt2 = (int(joint2[0]), int(joint2[1]))
        cv2.line(frame, pt1, pt2, color, thickness)
    
    # Draw joints
    for joint in joints:
        cv2.circle(frame, (int(joint[0]), int(joint[1])), 4, color, -1)

def draw_walls(frame, wall_lines, color=(255, 0, 0), thickness=2):
    """Draw wall lines on frame."""
    for wall in wall_lines:
        start = wall["start"]
        end = wall["end"]
        pt1 = (int(start[0]), int(start[1]))
        pt2 = (int(end[0]), int(end[1]))
        cv2.line(frame, pt1, pt2, color, thickness)
        
        # Draw depth points
        depth_color = (0, 0, 255)  # Red for depth points
        cv2.circle(frame, pt1, 4, depth_color, -1)
        cv2.circle(frame, pt2, 4, depth_color, -1)

def draw_intersection_lines(frame, wall_analysis, floor_spline, height, width, color=(0, 255, 255), thickness=2):
    """Draw floor-wall intersection lines and their linear regression."""
    try:
        scale_factor = height / np.max(floor_spline.get_knots())
        y_range = np.arange(height)
        floor_depths = floor_spline(y_range) * scale_factor
        intersection_points = []
        
        if wall_analysis["type"] == "double_wall":
            corner_x = int(wall_analysis["corner"]["x"])
            
            # Find left wall intersections
            left_x_range = np.arange(0, corner_x)
            left_eq = wall_analysis["left_equation"]
            left_intersections = []
            
            # Scan from bottom up for each x position
            for x in left_x_range:
                wall_depth = (left_eq["slope"] * x + left_eq["intercept"]) * scale_factor
                # Find intersections starting from bottom of frame
                matches = y_range[np.abs(floor_depths - wall_depth) < 1.0]
                if len(matches) > 0:
                    y = int(matches[0])  # Take first (lowest) intersection
                    left_intersections.append([x, y])
                    # Draw intersection point
                    cv2.circle(frame, (int(x), height - y), 3, color, -1)  # Flip y-coordinate
            
            # Find right wall intersections
            right_x_range = np.arange(corner_x, width)
            right_eq = wall_analysis["right_equation"]
            right_intersections = []
            
            for x in right_x_range:
                wall_depth = (right_eq["slope"] * x + right_eq["intercept"]) * scale_factor
                matches = y_range[np.abs(floor_depths - wall_depth) < 1.0]
                if len(matches) > 0:
                    y = int(matches[0])  # Take first (lowest) intersection
                    right_intersections.append([x, y])
                    # Draw intersection point
                    cv2.circle(frame, (int(x), height - y), 3, color, -1)  # Flip y-coordinate
            
            # Fit and draw regression lines with proper y-coordinate orientation
            if len(left_intersections) > 1:
                left_points = np.array(left_intersections)
                # Flip y-coordinates for regression
                left_points[:, 1] = height - left_points[:, 1]
                left_slope, left_intercept = np.polyfit(left_points[:, 0], left_points[:, 1], 1)
                left_line_x = np.array([0, corner_x])
                left_line_y = left_slope * left_line_x + left_intercept
                cv2.line(frame, 
                        (int(left_line_x[0]), int(left_line_y[0])),
                        (int(left_line_x[1]), int(left_line_y[1])),
                        (255, 0, 0), thickness)
            
            if len(right_intersections) > 1:
                right_points = np.array(right_intersections)
                # Flip y-coordinates for regression
                right_points[:, 1] = height - right_points[:, 1]
                right_slope, right_intercept = np.polyfit(right_points[:, 0], right_points[:, 1], 1)
                right_line_x = np.array([corner_x, width-1])
                right_line_y = right_slope * right_line_x + right_intercept
                cv2.line(frame,
                        (int(right_line_x[0]), int(right_line_y[0])),
                        (int(right_line_x[1]), int(right_line_y[1])),
                        (255, 0, 0), thickness)
            
            # Draw room corner line
            cv2.line(frame, (corner_x, 0), (corner_x, height-1), (0, 255, 0), thickness)
            
            # Store intersection data with proper y-coordinates
            return {
                "left_intersections": [[x, height-y] for x, y in left_intersections] if left_intersections else None,
                "right_intersections": [[x, height-y] for x, y in right_intersections] if right_intersections else None,
                "left_regression": {"slope": float(left_slope), "intercept": float(left_intercept)} if len(left_intersections) > 1 else None,
                "right_regression": {"slope": float(right_slope), "intercept": float(right_intercept)} if len(right_intersections) > 1 else None,
                "corner_x": corner_x
            }
            
        else:  # Single wall
            eq = wall_analysis["equation"]
            intersections = []
            
            for x in range(width):
                wall_depth = (eq["slope"] * x + eq["intercept"]) * scale_factor
                matches = y_range[np.abs(floor_depths - wall_depth) < 1.0]
                if len(matches) > 0:
                    y = matches[-1]
                    intersections.append([x, y])
                    # Draw intersection point
                    cv2.circle(frame, (int(x), int(y)), 3, color, -1)
            
            # Fit and draw regression line
            if len(intersections) > 1:
                points = np.array(intersections)
                slope, intercept = np.polyfit(points[:, 0], points[:, 1], 1)
                line_y1 = int(slope * 0 + intercept)
                line_y2 = int(slope * (width-1) + intercept)
                cv2.line(frame, (0, line_y1), (width-1, line_y2), (255, 0, 0), thickness)
            
            # Store intersection data
            return {
                "intersections": intersections if len(intersections) > 0 else None,
                "regression": {"slope": float(slope), "intercept": float(intercept)} if len(intersections) > 1 else None
            }

    except Exception as e:
        print(f"Error drawing intersection lines: {e}")
        return None

def convert_to_json_serializable(obj):
    """Convert numpy types to JSON serializable Python types."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    return obj

def create_visualization(
    video_path: str,
    depth_dir: str,
    poses_json: str = None,
    output_video: str = None,
    output_analysis: str = None,
    depth_alpha: float = 0.3,
    line_alpha: float = 0.7
):
    """
    Create visualization video and analysis JSON.
    
    Args:
        video_path: Path to original video
        depth_dir: Directory containing depth maps
        poses_json: Optional path to poses JSON
        output_video: Path for visualization video output
        output_analysis: Path for analysis JSON output
        depth_alpha: Transparency of depth overlay
        line_alpha: Transparency of line overlays
    """
    # Load poses if provided
    poses_data = None
    if poses_json and os.path.exists(poses_json):
        with open(poses_json, 'r') as f:
            poses_data = json.load(f)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create colormap for depth visualization
    colormap = cv2.COLORMAP_MAGMA
    
    # Initialize analysis data structure
    analysis_data = {
        "metadata": {
            "frame_height": height,
            "frame_width": width,
            "fps": fps,
            "total_frames": frame_count
        },
        "frames": {}
    }
    
    with tqdm(total=frame_count, desc="Creating visualization") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
            str_idx = str(frame_idx)
            
            # Load and process depth map
            depth_file = Path(depth_dir) / f'depth_{frame_idx:06d}.npz'
            try:
                if depth_file.exists():
                    with np.load(depth_file) as depth_data:
                        depth_map = depth_data['depth']
                    
                    # Create overlay for depth and lines
                    overlay = frame.copy()
                    
                    # Analyze depth map
                    analysis_result = analyze_depth_frame(
                        depth_map,
                        height,
                        width,
                        visualize=False,
                        debug=False
                    )
                    
                    # Draw skeleton if poses are available
                    if poses_data and str_idx in poses_data["frames"]:
                        for pose in poses_data["frames"][str_idx]:
                            # Get 2D joints
                            joints2d = pose.get("joints2d", [])
                            if joints2d:
                                joints = [j[:2] for j in joints2d]  # Only use x,y coordinates
                                draw_skeleton(overlay, joints)
                    
                    # Draw wall lines and intersections
                    if analysis_result["success"]:
                        wall_analysis = analysis_result["wall_analysis"]
                        floor_analysis = analysis_result["floor_analysis"]
                        
                        # Create floor spline for intersection calculation
                        y_points = np.array(floor_analysis["points"]["y"])
                        depth_points = np.array(floor_analysis["points"]["depths"])
                        floor_spline = UnivariateSpline(y_points, depth_points, k=3, s=0.5)
                        
                        # Draw wall lines
                        wall_lines = []
                        if wall_analysis["type"] == "double_wall":
                            left_eq = wall_analysis["left_equation"]
                            right_eq = wall_analysis["right_equation"]
                            corner = wall_analysis["corner"]
                            
                            # Left wall
                            wall_lines.append({
                                "type": "left_wall",
                                "start": [0, 0, float(depth_map[0, 0])],
                                "end": [corner["x"], 0, corner["depth"]]
                            })
                            
                            # Right wall
                            wall_lines.append({
                                "type": "right_wall",
                                "start": [corner["x"], 0, corner["depth"]],
                                "end": [width-1, 0, float(depth_map[0, width-1])]
                            })
                        else:
                            # Single wall
                            wall_lines.append({
                                "type": "wall",
                                "start": [0, 0, float(depth_map[0, 0])],
                                "end": [width-1, 0, float(depth_map[0, width-1])]
                            })
                        
                        # Create combined analysis for intersection drawing
                        combined_analysis = analysis_result["wall_analysis"]
                        combined_analysis["floor_analysis"] = analysis_result["floor_analysis"]
                        
                        # Draw wall lines first
                        draw_walls(overlay, wall_lines)
                        
                        # Then draw intersection lines using complete analysis
                        intersection_data = draw_intersection_lines(overlay, wall_analysis, floor_spline, height, width)
                        if intersection_data:
                            analysis_result["intersection_analysis"] = intersection_data
                    
                    # Final blend
                    frame = cv2.addWeighted(overlay, line_alpha, frame, 1 - line_alpha, 0)
                    
                    # Add depth visualization
                    if depth_map is not None:
                        # Invert depth map so close objects are hot colors
                        depth_map = 1.0 - depth_map  # Invert the values
                        depth_map = ((depth_map - depth_map.min()) / 
                                   (depth_map.max() - depth_map.min()) * 255).astype(np.uint8)
                        depth_colored = cv2.applyColorMap(depth_map, colormap)
                        if depth_colored.shape[:2] != (height, width):
                            depth_colored = cv2.resize(depth_colored, (width, height))
                        frame = cv2.addWeighted(depth_colored, depth_alpha, frame, 1 - depth_alpha, 0)
                    
                    # Update analysis data with serializable types
                    if analysis_result["success"]:
                        analysis_result = convert_to_json_serializable(analysis_result)
                    analysis_data["frames"][str(frame_idx)] = analysis_result
            
            except Exception as e:
                print(f"Error processing frame {frame_idx}: {e}")
            
            out.write(frame)
            pbar.update(1)
    
    # Save analysis data with proper type conversion
    if output_analysis:
        try:
            # Convert entire analysis data to JSON serializable format
            serializable_data = convert_to_json_serializable(analysis_data)
            with open(output_analysis, 'w') as f:
                json.dump(serializable_data, f, indent=4)
        except Exception as e:
            print(f"Error saving analysis data: {str(e)}")
            # Try to save without pretty printing as fallback
            with open(output_analysis, 'w') as f:
                json.dump(serializable_data, f)
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create visualization video with poses and depth")
    parser.add_argument("--video", required=True, help="Path to original video")
    parser.add_argument("--depth_dir", required=True, help="Directory containing depth NPZ files")
    parser.add_argument("--poses_json", help="Path to poses JSON file")
    parser.add_argument("--output_video", required=True, help="Output visualization video path")
    parser.add_argument("--output_analysis", required=True, help="Output analysis JSON path")
    parser.add_argument("--depth_alpha", type=float, default=0.3, help="Depth map blend alpha")
    parser.add_argument("--line_alpha", type=float, default=1.0, help="Line overlay blend alpha")
    
    args = parser.parse_args()
    
    create_visualization(
        args.video,
        args.depth_dir,
        args.poses_json,
        args.output_video,
        args.output_analysis,
        args.depth_alpha,
        args.line_alpha
    )
