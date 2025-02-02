import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.stats import linregress, zscore
from typing import Dict, List, Tuple, Optional
from .depth_visualization import plot_depth_analysis

def find_deepest_points(depth_map: np.ndarray, axis: int) -> Tuple[np.ndarray, np.ndarray]:
    """Find deepest points along specified axis."""
    indices = np.arange(depth_map.shape[1-axis])
    deepest_points = np.max(depth_map, axis=axis)
    return indices, deepest_points

def remove_outliers(x: np.ndarray, y: np.ndarray, threshold: float = 2.0) -> Tuple[np.ndarray, np.ndarray]:
    """Remove outliers using z-score on the depth values."""
    z_scores = np.abs(zscore(y))
    mask = z_scores < threshold
    return x[mask], y[mask]

def fit_floor_spline(height: int, depth_map: np.ndarray, wall_analysis: Dict) -> Tuple[UnivariateSpline, np.ndarray, np.ndarray]:
    """
    Fit floor curve that follows floor points while maintaining monotonicity.
    Only uses points below the corner depth for fitting and visualization.
    """
    y_coords, depths = find_deepest_points(depth_map, 1)
    
    # Get corner y position and filter points strictly below it
    corner_y = find_corner_y(wall_analysis, depth_map)
    floor_mask = y_coords >= corner_y
    y_floor = y_coords[floor_mask]
    depths_floor = depths[floor_mask]
    
    # Get corner depth for filtering
    corner_depth = wall_analysis['corner']['depth']
    
    # Remove points that are deeper than corner depth with more lenient threshold
    valid_depth_mask = depths_floor < (corner_depth * 0.95)  # 95% of corner depth
    y_floor = y_floor[valid_depth_mask]
    depths_floor = depths_floor[valid_depth_mask]
    
    if len(y_floor) < 4:  # Need more points for higher degree spline
        raise ValueError("Not enough valid floor points after depth filtering")
    
    # Sort points for spline fitting
    sort_idx = np.argsort(y_floor)
    y_sorted = y_floor[sort_idx]
    depths_sorted = depths_floor[sort_idx]
    
    # Create weights that more strongly favor near-camera points
    norm_y = (y_sorted - y_sorted.min()) / (y_sorted.max() - y_sorted.min())
    weights = np.exp(-norm_y)  # Increased weight difference
    
    # Add extra points to guide the curve's behavior at endpoints
    y_extra = np.concatenate([
        [y_sorted[0] - 5],  # Point before start
        y_sorted,
        [y_sorted[-1] + 5]  # Point after end
    ])
    
    depths_extra = np.concatenate([
        [depths_sorted[0] * 1.02],  # Slightly higher at start
        depths_sorted,
        [depths_sorted[-1] * 0.98]  # Slightly lower at end
    ])
    
    weights_extra = np.concatenate([
        [weights[0]],
        weights,
        [weights[-1]]
    ])
    
    # Fit spline with higher degree and lower smoothing
    initial_spline = UnivariateSpline(
        y_extra,
        depths_extra,
        w=weights_extra,
        k=3,  # Higher degree for more flexibility
        s=1.0  # Lower smoothing factor
    )
    
    # Sample points more densely
    y_dense = np.linspace(y_sorted[0], y_sorted[-1], 200)
    depths_dense = initial_spline(y_dense)
    
    # Ensure monotonicity with smoother transition
    depths_monotonic = np.zeros_like(depths_dense)
    depths_monotonic[0] = depths_dense[0]
    window_size = 5
    
    for i in range(1, len(depths_dense)):
        # Use windowed minimum for smoother transitions
        start_idx = max(0, i - window_size)
        depths_monotonic[i] = min(
            depths_dense[i],
            np.min(depths_monotonic[start_idx:i]) * 1.001  # Allow slight increase
        )
    
    # Final spline with reduced smoothing
    final_spline = UnivariateSpline(y_dense, depths_monotonic, k=3, s=0.5)
    
    return final_spline, y_floor, depths_floor

def find_max_depth_point(depth_map: np.ndarray) -> Tuple[int, float]:
    """Find x coordinate of maximum depth point."""
    x_coords, depths = find_deepest_points(depth_map, 0)
    max_depth_idx = np.argmax(depths)
    return x_coords[max_depth_idx], depths[max_depth_idx]

def fit_wall_segments(width: int, depth_map: np.ndarray) -> Dict:
    """Fit two segments split at maximum depth point with outlier removal."""
    try:
        x_coords, depths = find_deepest_points(depth_map, 0)

        # Check for valid depth values first
        if np.all(depths == 0) or np.all(np.isnan(depths)):
            raise ValueError("Invalid depth values")

        # Handle constant depth case
        if np.allclose(depths, depths[0], rtol=1e-5, atol=1e-5):
            return {
                "type": "single_wall",
                "equation": {
                    "slope": 0.0,
                    "intercept": float(depths[0])
                },
                "points": {
                    "x": [0, width-1],
                    "depths": [float(depths[0]), float(depths[0])]
                },
                "corner": {
                    "x": float(width//2),
                    "depth": float(depths[0])
                }
            }

        try:
            x_coords_clean, depths_clean = remove_outliers(x_coords, depths)
        except:
            # If outlier removal fails, use original points
            x_coords_clean, depths_clean = x_coords, depths
        
        if len(x_coords_clean) < 3:
            # Not enough points, use simple max depth point
            max_depth_idx = np.argmax(depths)
            return {
                "type": "single_wall",
                "equation": {
                    "slope": 0.0,
                    "intercept": float(depths[max_depth_idx])
                },
                "points": {
                    "x": [0, width-1],
                    "depths": [float(depths[max_depth_idx])] * 2
                },
                "corner": {
                    "x": float(x_coords[max_depth_idx]),
                    "depth": float(depths[max_depth_idx])
                }
            }
        
        max_depth_idx = np.argmax(depths_clean)
        max_depth_x = x_coords_clean[max_depth_idx]
        max_depth = depths_clean[max_depth_idx]
        
        # Ensure we don't divide by zero in slope calculations
        def safe_linregress(x, y):
            try:
                if len(set(x)) <= 1:  # All x values are the same
                    return 0.0, float(np.mean(y)), 1.0
                if len(set(y)) <= 1:  # All y values are the same
                    return 0.0, float(y[0]), 1.0
                slope, intercept, r_value, _, _ = linregress(x, y)
                if np.isnan(slope) or np.isinf(slope):
                    return 0.0, float(np.mean(y)), 0.0
                return float(slope), float(intercept), float(r_value)
            except:
                return 0.0, float(np.mean(y)), 0.0

        # Single wall case
        if max_depth_x <= 10 or max_depth_x >= (width - 10):
            slope, intercept, r_value = safe_linregress(x_coords_clean, depths_clean)
            return {
                "type": "single_wall",
                "equation": {
                    "slope": slope,
                    "intercept": intercept
                },
                "points": {
                    "x": x_coords_clean.tolist(),
                    "depths": depths_clean.tolist()
                },
                "corner": {
                    "x": float(max_depth_x),
                    "depth": float(max_depth)
                }
            }
        
        # Double wall case
        left_mask = x_coords_clean <= max_depth_x
        right_mask = x_coords_clean >= max_depth_x
        
        left_slope, left_intercept, left_r = safe_linregress(
            x_coords_clean[left_mask], 
            depths_clean[left_mask]
        )
        
        right_slope, right_intercept, right_r = safe_linregress(
            x_coords_clean[right_mask], 
            depths_clean[right_mask]
        )
        
        # Safe intersection calculation
        if abs(left_slope - right_slope) < 1e-6:
            x_int = max_depth_x
        else:
            x_int = (right_intercept - left_intercept) / (left_slope - right_slope)
            if np.isnan(x_int) or np.isinf(x_int) or x_int < 0 or x_int >= width:
                x_int = max_depth_x
        
        return {
            "type": "double_wall",
            "wall_intersection_x": float(x_int),
            "corner": {
                "x": float(x_int),
                "depth": float(max_depth)
            },
            "left_equation": {
                "slope": left_slope,
                "intercept": left_intercept
            },
            "right_equation": {
                "slope": right_slope,
                "intercept": right_intercept
            },
            "points": {
                "x": x_coords_clean.tolist(),
                "depths": depths_clean.tolist()
            }
        }
        
    except Exception as e:
        print(f"Wall fitting error: {str(e)}")
        # Ultimate fallback - return a flat wall
        flat_depth = float(np.mean(depth_map[depth_map > 0]) if np.any(depth_map > 0) else 1.0)
        return {
            "type": "single_wall",
            "equation": {
                "slope": 0.0,
                "intercept": flat_depth
            },
            "points": {
                "x": [0, width-1],
                "depths": [flat_depth, flat_depth]
            },
            "corner": {
                "x": float(width//2),
                "depth": flat_depth
            }
        }

def find_corner_y(wall_analysis: Dict, depth_map: np.ndarray) -> int:
    """Find the y-coordinate of the corner(s) for floor point filtering."""
    height = depth_map.shape[0]
    
    if wall_analysis['type'] == 'double_wall':
        x_int = int(wall_analysis['wall_intersection_x'])
        # Get depth profile at intersection
        depth_profile = depth_map[:, x_int]
    else:
        # For single wall, use the edge where wall is detected
        if wall_analysis['equation']['slope'] > 0:
            depth_profile = depth_map[:, 0]  # Left edge
        else:
            depth_profile = depth_map[:, -1]  # Right edge
    
    # Find where depth starts decreasing significantly (moving up from bottom)
    gradients = np.gradient(depth_profile)
    corner_candidates = np.where(np.abs(gradients) > np.std(gradients))[0]
    
    if len(corner_candidates) > 0:
        # Take the lowest significant change point
        corner_y = corner_candidates[-1]
        return min(max(corner_y, height // 4), height * 3 // 4)  # Constrain to middle half
    
    return height // 2  # Default to middle if no clear corner found

def analyze_depth_frame(depth_map: np.ndarray, frame_height: int, frame_width: int, 
                       visualize: bool = False, debug: bool = False) -> Dict:
    """
    Analyze a single depth frame for floor and wall structure.
    
    Args:
        depth_map: Input depth map
        frame_height: Height of frame
        frame_width: Width of frame
        visualize: Whether to save visualization plots
        debug: Whether to show interactive debug visualizations
    """
    try:
        # Normalize depth map to 0-1 range if not already normalized
        if depth_map.max() > 1.0 or depth_map.min() < 0.0:
            depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
        
        # First analyze walls to find corner position
        wall_analysis = fit_wall_segments(frame_width, depth_map)
        corner_y = find_corner_y(wall_analysis, depth_map)
        
        # Get floor points and filter based on corner_y
        y_coords, depths = find_deepest_points(depth_map, 1)
        floor_mask = y_coords >= corner_y
        y_coords_filtered = y_coords[floor_mask]
        depths_filtered = depths[floor_mask]
        
        # Only process if we have enough floor points
        if len(y_coords_filtered) < 3:
            raise ValueError("Not enough floor points found")
            
        # Pass wall analysis to floor spline fitting
        floor_spline, y_coords_clean, depths_monotonic = fit_floor_spline(
            frame_height, depth_map, wall_analysis
        )
        
        # Only run visualization if either visualize or debug is True
        if visualize or debug:
            try:
                plot_depth_analysis(
                    depth_map,
                    y_coords_clean,
                    depths_monotonic,
                    np.array(wall_analysis['points']['x']),
                    np.array(wall_analysis['points']['depths']),
                    floor_spline,
                    wall_analysis,
                    debug=debug  # Pass debug flag to visualization function
                )
            except Exception as viz_error:
                print(f"Visualization error: {str(viz_error)}")
        
        return {
            "success": True,
            "wall_analysis": wall_analysis,
            "floor_analysis": {
                "corner_y": int(corner_y),
                "points": {
                    "y": y_coords_clean.tolist(),
                    "depths": depths_monotonic.tolist()
                }
            }
        }
    except Exception as e:
        print(f"Analysis error: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }
