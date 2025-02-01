import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress
from typing import List, Tuple

def find_intersection_points(floor_spline, wall_eq, x_range, scale_factor, height):
    """Find all intersection points between floor and wall."""
    y_range = np.arange(height)
    floor_depths = floor_spline(y_range) * scale_factor
    wall_depths = (wall_eq['slope'] * x_range + wall_eq['intercept']) * scale_factor
    
    intersections = []
    for x_idx, x in enumerate(x_range):
        wall_depth = wall_depths[x_idx]
        # Find y values where floor depth matches wall depth
        depth_diff = np.abs(floor_depths - wall_depth)
        y_matches = y_range[depth_diff < (height * 0.01)]  # 1% tolerance
        
        for y in y_matches:
            if 0 <= y < height:
                intersections.append((x, y))
    
    return intersections

def plot_depth_analysis(depth_map: np.ndarray, y_coords: np.ndarray, depths_y: np.ndarray, 
                       x_coords: np.ndarray, depths_x: np.ndarray, 
                       floor_spline, wall_analysis: dict):
    """Create debug visualization plots for depth analysis."""
    try:
        plt.ion()  # Turn on interactive mode
        plt.close('all')
        fig = plt.figure(figsize=(15, 5))
        
        height, width = depth_map.shape
        
        # Normalize depths to frame height
        max_depth = max(np.max(depths_y), np.max(depths_x))
        scale_factor = height / max_depth
        depths_y_scaled = depths_y * scale_factor
        depths_x_scaled = depths_x * scale_factor
        
        # Plot 1: Floor spline
        ax1 = fig.add_subplot(131)
        ax1.scatter(y_coords, depths_y_scaled, c='blue', alpha=0.5, label='Data points')
        y_smooth = np.linspace(0, height-1, 200)
        spline_values = floor_spline(y_smooth) * scale_factor
        ax1.plot(y_smooth, spline_values, 'r-', label='Spline fit')
        ax1.set_xlabel('Y coordinate')
        ax1.set_ylabel('Depth (scaled)')
        ax1.set_title('Floor Depth Profile')
        ax1.set_xlim(0, height-1)
        ax1.set_ylim(0, height)
        ax1.legend()
        
        # Plot 2: Wall segments with corner point
        ax2 = fig.add_subplot(132)
        ax2.scatter(x_coords, depths_x_scaled, c='blue', alpha=0.5, label='Data points')
        
        if wall_analysis['type'] == 'double_wall':
            x_int = wall_analysis['corner']['x']
            corner_depth_scaled = wall_analysis['corner']['depth'] * scale_factor
            
            # Plot left segment
            x_left = np.array([0, x_int])
            left_eq = wall_analysis['left_equation']
            y_left = (left_eq['slope'] * x_left + left_eq['intercept']) * scale_factor
            ax2.plot(x_left, y_left, 'r-', alpha=0.7, label='Left wall')
            
            # Plot right segment
            x_right = np.array([x_int, width-1])
            right_eq = wall_analysis['right_equation']
            y_right = (right_eq['slope'] * x_right + right_eq['intercept']) * scale_factor
            ax2.plot(x_right, y_right, 'g-', alpha=0.7, label='Right wall')
            
            # Highlight corner point
            ax2.scatter([x_int], [corner_depth_scaled], c='red', s=100, zorder=5, 
                       marker='*', label='Corner point')
        else:
            eq = wall_analysis['equation']
            x = np.array([0, width-1])
            y = (eq['slope'] * x + eq['intercept']) * scale_factor
            ax2.plot(x, y, 'r-', alpha=0.7, label='Wall')
            
            # For single wall, show corner point
            corner_x = wall_analysis['corner']['x']
            corner_depth_scaled = wall_analysis['corner']['depth'] * scale_factor
            ax2.scatter([corner_x], [corner_depth_scaled], c='red', s=100, zorder=5, 
                       marker='*', label='Corner point')
        
        ax2.set_xlabel('X coordinate')
        ax2.set_ylabel('Depth (scaled)')
        ax2.set_title('Wall Depth Profile')
        ax2.legend()
        
        # Plot 3: Depth map with lines overlay and intersection points
        ax3 = fig.add_subplot(133)
        ax3.imshow(depth_map, cmap='viridis', origin='upper')
        
        if wall_analysis['type'] == 'double_wall':
            x_int = wall_analysis['corner']['x']
            ax3.axvline(x=x_int, color='yellow', linestyle='--', alpha=0.5)
            
            # Find and plot left wall intersections
            left_x_range = np.arange(0, x_int)
            left_intersections = find_intersection_points(
                floor_spline, 
                wall_analysis['left_equation'],
                left_x_range,
                scale_factor,
                height
            )
            
            # Find and plot right wall intersections
            right_x_range = np.arange(x_int, width)
            right_intersections = find_intersection_points(
                floor_spline,
                wall_analysis['right_equation'],
                right_x_range,
                scale_factor,
                height
            )
            
            # Plot intersection points
            if left_intersections:
                left_x, left_y = zip(*left_intersections)
                ax3.scatter(left_x, left_y, c='blue', s=20, alpha=0.5, label='Left intersections')
                # Fit line through left intersection points
                if len(left_intersections) > 1:
                    left_slope, left_intercept, _, _, _ = linregress(left_x, left_y)
                    left_line_x = np.array([0, x_int])
                    left_line_y = left_slope * left_line_x + left_intercept
                    ax3.plot(left_line_x, left_line_y, 'r-', linewidth=2)
            
            if right_intersections:
                right_x, right_y = zip(*right_intersections)
                ax3.scatter(right_x, right_y, c='green', s=20, alpha=0.5, label='Right intersections')
                # Fit line through right intersection points
                if len(right_intersections) > 1:
                    right_slope, right_intercept, _, _, _ = linregress(right_x, right_y)
                    right_line_x = np.array([x_int, width-1])
                    right_line_y = right_slope * right_line_x + right_intercept
                    ax3.plot(right_line_x, right_line_y, 'g-', linewidth=2)
            
        else:
            # Single wall case
            x_range = np.arange(width)
            intersections = find_intersection_points(
                floor_spline,
                wall_analysis['equation'],
                x_range,
                scale_factor,
                height
            )
            
            if intersections:
                int_x, int_y = zip(*intersections)
                ax3.scatter(int_x, int_y, c='blue', s=20, alpha=0.5, label='Intersections')
                # Fit line through intersection points
                if len(intersections) > 1:
                    slope, intercept, _, _, _ = linregress(int_x, int_y)
                    line_x = np.array([0, width-1])
                    line_y = slope * line_x + intercept
                    ax3.plot(line_x, line_y, 'r-', linewidth=2)
        
        ax3.set_title('Depth Map with Wall Lines')
        ax3.legend()
        
        plt.tight_layout()
        fig.canvas.draw()
        
        # Wait for keypress
        while True:
            if plt.waitforbuttonpress():
                break
        
        plt.close(fig)
        
    except Exception as e:
        print(f"Error in visualization: {str(e)}")
        plt.close('all')
    finally:
        plt.close('all')
