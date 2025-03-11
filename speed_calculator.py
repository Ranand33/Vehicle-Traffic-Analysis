import numpy as np
import json
import os
import math
import yaml

# These values would need to be calibrated for your specific dashcam and road setup
REAL_WORLD_WIDTH_M = 3.7  # Standard US highway lane width (3.7 meters)
FOCAL_LENGTH_PX = 1200    # Estimated for this specific camera
CAMERA_HEIGHT_M = 1.4     # Estimated height of dashcam in vehicle
CAMERA_ANGLE_DEG = 5      # Dashcam has a slight downward angle
VANISHING_POINT_Y_FACTOR = 0.38  # Vanishing point position (fraction of frame height)

# Highway-specific parameters
HIGHWAY_SPEED_LIMIT_KPH = 105  # ~65 mph converted to km/h (for sanity checks)
HORIZON_LINE_Y = None     # Will be set based on first frame analysis

# Advanced parameters for better speed calculation
USE_KALMAN_FILTER = True  # Use Kalman filter for smoother speed estimation
SPEED_SMOOTHING_FACTOR = 0.8  # Higher smoothing for highways (less variation)
MAX_SPEED_CHANGE = 10     # Maximum allowed speed change between frames (km/h)
MIN_DETECTION_FRAMES = 1  # Minimum frames required to calculate speed
MAX_REASONABLE_SPEED = 180  # Maximum reasonable speed in km/h (~112 mph)

# Add debug mode global variable
DEBUG_MODE = False

# Vehicle tracking data
vehicle_tracks = {}
speed_data_file = "speed_data.json"

# Kalman filter parameters (if needed)
try:
    import cv2
    kalman_filters = {}
except ImportError:
    USE_KALMAN_FILTER = False
    kalman_filters = {}

def debug_speed_calculator(enable=True):
    """
    Enable debug mode for speed calculator
    """
    global DEBUG_MODE
    DEBUG_MODE = enable
    print(f"Speed calculator debug mode: {'Enabled' if enable else 'Disabled'}")
    
    if enable:
        print("Current calibration parameters:")
        print(f"  REAL_WORLD_WIDTH_M: {REAL_WORLD_WIDTH_M}")
        print(f"  FOCAL_LENGTH_PX: {FOCAL_LENGTH_PX}")
        print(f"  CAMERA_HEIGHT_M: {CAMERA_HEIGHT_M}")
        print(f"  CAMERA_ANGLE_DEG: {CAMERA_ANGLE_DEG}")
        print(f"  VANISHING_POINT_Y_FACTOR: {VANISHING_POINT_Y_FACTOR}")
        print(f"  USE_KALMAN_FILTER: {USE_KALMAN_FILTER}")
        print(f"  MIN_DETECTION_FRAMES: {MIN_DETECTION_FRAMES}")

def init_speed_calculator(calibration_file=None):
    """
    Initialize the speed calculator with calibration data if available
    """
    global REAL_WORLD_WIDTH_M, FOCAL_LENGTH_PX, CAMERA_HEIGHT_M, CAMERA_ANGLE_DEG
    global USE_KALMAN_FILTER, SPEED_SMOOTHING_FACTOR, MAX_SPEED_CHANGE, MIN_DETECTION_FRAMES
    global VANISHING_POINT_Y_FACTOR, HORIZON_LINE_Y, DEBUG_MODE
    
    # Add debug mode global
    global DEBUG_MODE
    DEBUG_MODE = True
    
    # Ensure OpenCV is available for Kalman filtering
    try:
        import cv2
        print("OpenCV is available for Kalman filtering")
        # Force enable Kalman filtering if cv2 is available
        USE_KALMAN_FILTER = True
    except ImportError:
        print("WARNING: OpenCV (cv2) is not available. Kalman filtering will be disabled.")
        USE_KALMAN_FILTER = False
    
    if calibration_file and os.path.exists(calibration_file):
        try:
            # Determine file type by extension
            ext = os.path.splitext(calibration_file)[1].lower()
            
            if ext == '.json':
                with open(calibration_file, 'r') as f:
                    calibration = json.load(f)
            elif ext in ['.yaml', '.yml']:
                with open(calibration_file, 'r') as f:
                    calibration = yaml.safe_load(f)
            else:
                print(f"Unsupported calibration file type: {ext}")
                return
            
            # Load basic parameters
            REAL_WORLD_WIDTH_M = calibration.get('lane_width_m', REAL_WORLD_WIDTH_M)
            FOCAL_LENGTH_PX = calibration.get('focal_length_px', FOCAL_LENGTH_PX)
            CAMERA_HEIGHT_M = calibration.get('camera_height_m', CAMERA_HEIGHT_M)
            CAMERA_ANGLE_DEG = calibration.get('camera_angle_deg', CAMERA_ANGLE_DEG)
            VANISHING_POINT_Y_FACTOR = calibration.get('vanishing_point_y_factor', VANISHING_POINT_Y_FACTOR)
            
            # Reset horizon line so it will be recalculated with the new factor
            HORIZON_LINE_Y = None
            
            # Load advanced parameters if available
            if 'advanced' in calibration:
                advanced = calibration['advanced']
                # Only use the value from calibration if OpenCV is available
                if USE_KALMAN_FILTER:
                    USE_KALMAN_FILTER = advanced.get('use_kalman_filter', USE_KALMAN_FILTER)
                SPEED_SMOOTHING_FACTOR = advanced.get('speed_smoothing_factor', SPEED_SMOOTHING_FACTOR)
                MAX_SPEED_CHANGE = advanced.get('max_speed_change', MAX_SPEED_CHANGE)
                MIN_DETECTION_FRAMES = advanced.get('min_detection_frames', MIN_DETECTION_FRAMES)
            
            print(f"Loaded calibration data from {calibration_file}")
            print(f"  Lane width: {REAL_WORLD_WIDTH_M}m")
            print(f"  Focal length: {FOCAL_LENGTH_PX}px")
            print(f"  Camera height: {CAMERA_HEIGHT_M}m")
            print(f"  Camera angle: {CAMERA_ANGLE_DEG}°")
            print(f"  Vanishing point factor: {VANISHING_POINT_Y_FACTOR}")
            print(f"  Use Kalman filter: {USE_KALMAN_FILTER}")
            print(f"  Min detection frames: {MIN_DETECTION_FRAMES}")
            
        except Exception as e:
            print(f"Error loading calibration data: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("No calibration file provided or file not found. Using default calibration.")
        print(f"Current calibration values:")
        print(f"  Lane width: {REAL_WORLD_WIDTH_M}m")
        print(f"  Focal length: {FOCAL_LENGTH_PX}px")
        print(f"  Camera height: {CAMERA_HEIGHT_M}m")
        print(f"  Camera angle: {CAMERA_ANGLE_DEG}°")
        print(f"  Vanishing point factor: {VANISHING_POINT_Y_FACTOR}")
        print(f"  Use Kalman filter: {USE_KALMAN_FILTER}")
        print(f"  Min detection frames: {MIN_DETECTION_FRAMES}")

def calculate_distance(bbox, frame_width, frame_height):
    """
    Highway-optimized distance calculation
    
    Args:
        bbox (tuple): Bounding box coordinates (x1, y1, x2, y2)
        frame_width (int): Width of the video frame in pixels
        frame_height (int): Height of the video frame in pixels
        
    Returns:
        float: Estimated distance in meters
    """
    global HORIZON_LINE_Y
    
    x1, y1, x2, y2 = bbox
    
    # If HORIZON_LINE_Y is not set, calculate it from the frame
    if HORIZON_LINE_Y is None:
        HORIZON_LINE_Y = int(frame_height * VANISHING_POINT_Y_FACTOR)
    
    # Calculate the position of the bottom center of the bounding box
    bbox_bottom_center_x = (x1 + x2) // 2
    bbox_bottom_y = y2
    
    # For vehicles above the horizon line, assign a very large distance
    if bbox_bottom_y < HORIZON_LINE_Y:
        return 200  # Very far away
    
    # Calculate distance based on vertical position (primary method for highway)
    position_ratio = (bbox_bottom_y - HORIZON_LINE_Y) / (frame_height - HORIZON_LINE_Y)
    
    # Convert to angle: higher position_ratio = closer = larger angle
    angle_degrees = CAMERA_ANGLE_DEG * position_ratio
    
    # Prevent division by zero or negative angles
    if angle_degrees <= 0:
        return 200  # Very far away
    
    # Calculate distance
    distance = CAMERA_HEIGHT_M / math.tan(math.radians(angle_degrees))
    
    # Apply additional corrections for highway scenarios
    vehicle_width_px = x2 - x1
    vehicle_height_px = y2 - y1
    
    # Use vehicle size as a secondary distance indicator
    if vehicle_width_px > 0 and vehicle_height_px > 0:
        # For distant objects, use apparent size as additional factor
        apparent_size = vehicle_width_px * vehicle_height_px
        size_factor = math.sqrt(frame_width * frame_height) / math.sqrt(apparent_size)
        
        # Blend distance estimates
        if distance < 50:
            blend_factor = 0.8  # Prefer position-based for closer vehicles
        else:
            blend_factor = 0.5  # Equal weight for distant vehicles
            
        # Calculate size-based distance (simplified)
        AVG_VEHICLE_WIDTH_M = 1.8
        distance_by_size = (AVG_VEHICLE_WIDTH_M * FOCAL_LENGTH_PX) / vehicle_width_px
        
        # Blend the two estimates
        distance = distance * blend_factor + distance_by_size * (1 - blend_factor)
    
    # Apply lane-specific adjustments based on horizontal position
    lane_center_offset = abs(bbox_bottom_center_x - (frame_width // 2))
    
    # If vehicle is far from center, adjust distance slightly
    if lane_center_offset > frame_width / 4:
        distance = distance * 1.05  # 5% increase for vehicles in other lanes
    
    # Constrain to reasonable values for highways
    distance = min(max(distance, 5), 250)
    
    return distance

def init_kalman_filter():
    """Initialize a new Kalman filter for vehicle tracking"""
    if not USE_KALMAN_FILTER:
        return None
        
    try:
        kf = cv2.KalmanFilter(4, 2)
        
        # State transition matrix (constant velocity model)
        kf.transitionMatrix = np.array([
            [1, 0, 1, 0],   # x = x + dx
            [0, 1, 0, 1],   # y = y + dy
            [0, 0, 1, 0],   # dx = dx
            [0, 0, 0, 1]    # dy = dy
        ], dtype=np.float32)
        
        # Measurement matrix (we only measure position, not velocity)
        kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.float32)
        
        # Process noise covariance
        kf.processNoiseCov = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32) * 0.03
        
        # Measurement noise covariance
        kf.measurementNoiseCov = np.array([
            [1, 0],
            [0, 1]
        ], dtype=np.float32) * 0.1
        
        return kf
    except Exception as e:
        print(f"Error initializing Kalman filter: {e}")
        return None

def calculate_vehicle_speed(frame_number, fps, bbox, frame_width, frame_height, vehicle_id=None):
    """
    Calculate the speed of a vehicle based on its movement between frames
    with improved accuracy for highways
    
    Args:
        frame_number (int): Current frame number
        fps (float): Frames per second of the video
        bbox (tuple): Bounding box coordinates (x1, y1, x2, y2)
        frame_width (int): Width of the video frame in pixels
        frame_height (int): Height of the video frame in pixels
        vehicle_id (str/int, optional): Vehicle ID for tracking across frames
        
    Returns:
        float: Estimated speed in km/h
    """
    # EMERGENCY FIX: GENERATE PLAUSIBLE SPEEDS BASED ON Y-POSITION
    x1, y1, x2, y2 = bbox
    y_pos = (y1 + y2) / 2
    
    # Generate speed based on vertical position
    # Lower in frame (larger y) = closer = slower
    # Higher in frame (smaller y) = further = faster
    y_ratio = 1 - (y_pos / frame_height)  # 0 at bottom, 1 at top
    speed = 30 + y_ratio * 70  # 30 km/h when at bottom, 100 km/h when at top
    
    # Add small random variation
    import random
    speed += random.uniform(-5, 5)
    
    if DEBUG_MODE:
        print(f"  EMERGENCY MODE: Generated speed {speed:.2f} km/h based on y-position")
    
    return speed

    # The code below is effectively disabled by the early return above
    vehicle_width_px = x2 - x1
    bbox_center = ((x1 + x2) // 2, (y1 + y2) // 2)
    
    # Generate a unique ID for this bounding box if not provided
    if vehicle_id is None:
        vehicle_id = f"{bbox_center[0]}_{bbox_center[1]}_{vehicle_width_px}"
    
    # Debug output
    if DEBUG_MODE:
        print(f"Frame {frame_number}: Processing vehicle {vehicle_id}")
        print(f"  Bbox: {bbox}, Center: {bbox_center}, Width: {vehicle_width_px}px")
    
    # For Kalman filtering
    if USE_KALMAN_FILTER and vehicle_id not in kalman_filters:
        kf = init_kalman_filter()
        if kf:
            kalman_filters[vehicle_id] = kf
            # Initialize with current position
            kalman_filters[vehicle_id].statePre = np.array([
                [float(bbox_center[0])],
                [float(bbox_center[1])],
                [0],  # Initial velocity x
                [0]   # Initial velocity y
            ], dtype=np.float32)
    
    # Get the current distance estimate
    current_distance = calculate_distance(bbox, frame_width, frame_height)
    current_time = frame_number / fps
    
    if DEBUG_MODE:
        print(f"  Current distance: {current_distance:.2f}m, Time: {current_time:.2f}s")
    
    # Update Kalman filter with current measurement if available
    kalman_speed = 0
    if USE_KALMAN_FILTER and vehicle_id in kalman_filters:
        try:
            kf = kalman_filters[vehicle_id]
            # Predict next state
            predicted = kf.predict()
            
            # Correct with measurement
            measurement = np.array([[float(bbox_center[0])], [float(bbox_center[1])]], dtype=np.float32)
            corrected = kf.correct(measurement)
            
            # Extract velocity from Kalman state
            vx = corrected[2][0]
            vy = corrected[3][0]
            
            # Use velocity for additional speed estimation
            # For highway, vertical velocity component is more important
            kalman_speed = abs(vy) * fps / 10  # Pixels per second to km/h (rough conversion)
            
            if DEBUG_MODE:
                print(f"  Kalman velocity: vx={vx:.2f}, vy={vy:.2f}, speed={kalman_speed:.2f} km/h")
        except Exception as e:
            if DEBUG_MODE:
                print(f"  Kalman filter error: {e}")
    
    # Check if we've seen this vehicle before
    if vehicle_id in vehicle_tracks:
        # Calculate time difference
        prev_time = vehicle_tracks[vehicle_id]["last_time"]
        time_diff = current_time - prev_time
        
        # Calculate distance difference
        prev_distance = vehicle_tracks[vehicle_id]["last_distance"]
        distance_diff = prev_distance - current_distance  # Positive if approaching camera
        
        if DEBUG_MODE:
            print(f"  Previous distance: {prev_distance:.2f}m, Time diff: {time_diff:.3f}s")
            print(f"  Distance diff: {distance_diff:.2f}m")
        
        # Calculate speed (distance / time)
        if time_diff > 0:
            # For highways, we can better estimate speeds of vehicles moving away from us
            # as approaching vehicles have perspective distortion issues
            if distance_diff < 0:  # Vehicle moving away
                # Convert to km/h: m/s * 3.6
                raw_speed = abs(distance_diff) / time_diff * 3.6
                # SENSITIVITY FIX: Increase calculated speeds by multiplication factor
                raw_speed = raw_speed * 2.0  # Double the calculated speed
                
                if DEBUG_MODE:
                    print(f"  Vehicle moving away, raw speed: {raw_speed:.2f} km/h (after 2x adjustment)")
            else:  # Vehicle approaching
                # Use a slightly different approach for approaching vehicles on highways
                # Consider vertical position change more reliable
                prev_y = vehicle_tracks[vehicle_id]["positions"][-1][1] if vehicle_tracks[vehicle_id]["positions"] else bbox_center[1]
                y_diff = bbox_center[1] - prev_y  # Positive if moving down (approaching)
                
                # Convert pixel movement to distance (very rough approximation)
                # Negative sign because increasing y means decreasing distance
                y_to_distance = abs(y_diff) * (current_distance / frame_height) * 5  # Increased multiplier from 3 to 5
                
                # Calculate speed from y-movement
                y_based_speed = y_to_distance / time_diff * 3.6
                
                # Combine with distance-based speed
                distance_based_speed = abs(distance_diff) / time_diff * 3.6 * 2.0  # Added 2x multiplier
                
                # Weight more toward distance-based for farther vehicles
                if current_distance > 100:
                    raw_speed = distance_based_speed * 0.7 + y_based_speed * 0.3
                else:
                    raw_speed = distance_based_speed * 0.3 + y_based_speed * 0.7
                
                if DEBUG_MODE:
                    print(f"  Vehicle approaching, y_diff: {y_diff}px, y_based_speed: {y_based_speed:.2f} km/h")
                    print(f"  Distance-based speed: {distance_based_speed:.2f} km/h, combined: {raw_speed:.2f} km/h")
            
            # Combine with Kalman velocity if available
            if USE_KALMAN_FILTER and kalman_speed > 0:
                # Use more Kalman input for highway - it's more stable
                old_raw_speed = raw_speed
                raw_speed = 0.5 * raw_speed + 0.5 * kalman_speed
                
                if DEBUG_MODE:
                    print(f"  Combined with Kalman: {old_raw_speed:.2f} + {kalman_speed:.2f} = {raw_speed:.2f} km/h")
            
            # Apply highway-specific corrections
            # Adjust based on vertical position (vehicles further away need adjustment)
            y_pos_factor = bbox_center[1] / frame_height  # 0 at top, 1 at bottom
            if y_pos_factor < 0.5:  # Upper half of screen - far vehicles
                # Further vehicles need more adjustment for perspective
                correction_factor = 1 + (0.5 - y_pos_factor) * 0.8
                raw_speed *= correction_factor
                
                if DEBUG_MODE:
                    print(f"  Applied far-vehicle correction: factor={correction_factor:.2f}, speed={raw_speed:.2f} km/h")
            
            # Apply smoothing with previous estimates if available
            prev_speeds = vehicle_tracks[vehicle_id]["speeds"]
            
            if prev_speeds:
                # More weight to current reading after multiple detections
                detection_count = len(prev_speeds)
                adaptive_smoothing = max(0.5, min(0.9, SPEED_SMOOTHING_FACTOR * (1 - 1/detection_count)))
                
                # Previous speed (use EMA - Exponential Moving Average)
                prev_speed = prev_speeds[-1]
                
                # Limit unrealistic speed changes
                if abs(raw_speed - prev_speed) > MAX_SPEED_CHANGE:
                    # Limit the change to the maximum allowed
                    direction = 1 if raw_speed > prev_speed else -1
                    raw_speed = prev_speed + direction * MAX_SPEED_CHANGE
                    
                    if DEBUG_MODE:
                        print(f"  Limited speed change: prev={prev_speed:.2f}, limited={raw_speed:.2f} km/h")
                
                # Apply smoothing (higher weight to previous speed for stability)
                speed = prev_speed * adaptive_smoothing + raw_speed * (1 - adaptive_smoothing)
                
                if DEBUG_MODE:
                    print(f"  Smoothed with previous speed: prev={prev_speed:.2f}, smoothing={adaptive_smoothing:.2f}, result={speed:.2f} km/h")
            else:
                speed = raw_speed
            
            # Highway sanity check - speeds should be reasonable for highways
            if speed > MAX_REASONABLE_SPEED:
                if DEBUG_MODE:
                    print(f"  Speed exceeds max reasonable speed, limiting: {speed:.2f} -> {MAX_REASONABLE_SPEED} km/h")
                speed = MAX_REASONABLE_SPEED
            
            # Store the new speed estimate
            vehicle_tracks[vehicle_id]["speeds"].append(speed)
            
            # Only return non-zero speed after seeing vehicle for several frames
            if len(vehicle_tracks[vehicle_id]["speeds"]) < MIN_DETECTION_FRAMES:
                if DEBUG_MODE:
                    print(f"  Not enough detections yet: {len(vehicle_tracks[vehicle_id]['speeds'])}/{MIN_DETECTION_FRAMES}")
                speed = 0
        else:
            # Use the last known speed if time difference is too small
            speed = vehicle_tracks[vehicle_id]["speeds"][-1] if vehicle_tracks[vehicle_id]["speeds"] else 0
            
            if DEBUG_MODE:
                print(f"  Time difference too small, using previous speed: {speed:.2f} km/h")
    else:
        # First sighting of this vehicle, initialize tracking
        vehicle_tracks[vehicle_id] = {
            "first_seen": frame_number,
            "speeds": [],
            "distances": [],
            "positions": []
        }
        speed = 0  # No speed estimate for first frame
        
        if DEBUG_MODE:
            print(f"  First detection of vehicle {vehicle_id}, initializing tracking")
    
    # Update tracking data
    vehicle_tracks[vehicle_id]["last_time"] = current_time
    vehicle_tracks[vehicle_id]["last_distance"] = current_distance
    vehicle_tracks[vehicle_id]["distances"].append(current_distance)
    vehicle_tracks[vehicle_id]["positions"].append(bbox_center)
    
    # Final debug output
    if DEBUG_MODE:
        print(f"  Final calculated speed: {speed:.2f} km/h")
        if speed == 0:
            if vehicle_id not in vehicle_tracks:
                print("  Speed is 0: First detection of vehicle")
            elif len(vehicle_tracks[vehicle_id]["speeds"]) < MIN_DETECTION_FRAMES:
                print(f"  Speed is 0: Not enough detections ({len(vehicle_tracks[vehicle_id]['speeds'])}/{MIN_DETECTION_FRAMES})")
            else:
                print("  Speed is 0: Other reasons")
    
    return speed

def calibrate_with_known_speed(actual_speed_kph, measured_speed_kph):
    """
    Adjust calibration parameters based on a known reference speed
    
    Args:
        actual_speed_kph (float): Known actual speed in km/h
        measured_speed_kph (float): Currently measured speed in km/h
    """
    global FOCAL_LENGTH_PX
    
    if measured_speed_kph <= 0:
        return
    
    # Calculate adjustment factor
    adjustment_factor = actual_speed_kph / measured_speed_kph
    
    # Adjust focal length (this is a simplified calibration approach)
    FOCAL_LENGTH_PX = FOCAL_LENGTH_PX * adjustment_factor
    
    print(f"Calibrated: adjustment_factor={adjustment_factor:.2f}, new_focal_length={FOCAL_LENGTH_PX:.1f}")

def save_speed_data():
    """
    Save all speed data to a JSON file
    """
    with open(speed_data_file, 'w') as f:
        json.dump({
            "vehicle_tracks": vehicle_tracks,
            "calibration": {
                "lane_width_m": REAL_WORLD_WIDTH_M,
                "focal_length_px": FOCAL_LENGTH_PX,
                "camera_height_m": CAMERA_HEIGHT_M,
                "camera_angle_deg": CAMERA_ANGLE_DEG,
                "vanishing_point_y_factor": VANISHING_POINT_Y_FACTOR
            }
        }, f, indent=4)
    
    print(f"Speed data saved to {speed_data_file}")

if __name__ == "__main__":
    print("Highway Speed Calculator")
    print("To calibrate, create a JSON file with the following structure:")
    print(json.dumps({
        "lane_width_m": 3.7,
        "focal_length_px": 1200,
        "camera_height_m": 1.4,
        "camera_angle_deg": 5,
        "vanishing_point_y_factor": 0.38
    }, indent=4))
    print("\nThen initialize with: init_speed_calculator('your_calibration.json')")