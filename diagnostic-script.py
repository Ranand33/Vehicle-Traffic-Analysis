#!/usr/bin/env python3
"""
Diagnostic tool for traffic analysis system
Checks the entire pipeline and identifies speed calculation issues
"""
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import cv2
import time
from pathlib import Path

# Import our modules
from speed_calculator import calculate_vehicle_speed, init_speed_calculator, debug_speed_calculator

def check_speed_calculator():
    """Test the speed calculator with synthetic data"""
    print("\n=== Testing Speed Calculator ===")
    
    # Enable debug mode
    debug_speed_calculator(True)
    
    # Create synthetic frame data
    frame_width = 1920
    frame_height = 1080
    fps = 30
    
    # Simulate a vehicle moving across frames
    print("\nSimulating vehicle movement across 10 frames...")
    
    # Track if any speeds were calculated
    any_speed_calculated = False
    
    # Run 10 frames with a vehicle moving upward (away from camera)
    for frame in range(1, 11):
        # Make the vehicle move up (y decreases) - simulating moving away
        y_pos = 800 - frame * 20  # Start at y=800, move up by 20px each frame
        
        # Create a bounding box - format is (x1, y1, x2, y2)
        bbox = (800, y_pos, 1000, y_pos + 100)
        
        # Test speed calculation
        vehicle_id = "test_vehicle_1"
        speed = calculate_vehicle_speed(
            frame_number=frame,
            fps=fps,
            bbox=bbox,
            frame_width=frame_width,
            frame_height=frame_height,
            vehicle_id=vehicle_id
        )
        
        print(f"Frame {frame}: Speed = {speed:.2f} km/h")
        
        if speed > 0:
            any_speed_calculated = True
    
    if any_speed_calculated:
        print("\n✅ SUCCESS: Speed calculator generated non-zero speed values")
    else:
        print("\n❌ ERROR: Speed calculator did not generate any non-zero speed values")
    
    return any_speed_calculated

def analyze_vehicle_data(data_file):
    """Analyze vehicle data file for speed calculation issues"""
    print(f"\n=== Analyzing Vehicle Data: {data_file} ===")
    
    try:
        with open(data_file, 'r') as f:
            data = json.load(f)
        
        # Check if we have any vehicles
        if 'vehicles' not in data or not data['vehicles']:
            print("❌ ERROR: No vehicles in data file")
            return False
        
        total_vehicles = len(data['vehicles'])
        print(f"Total vehicles detected: {total_vehicles}")
        
        # Check speeds
        vehicles_with_speed_data = 0
        vehicles_with_nonzero_speed = 0
        total_speed_entries = 0
        nonzero_speed_entries = 0
        
        for vehicle in data['vehicles']:
            if 'speeds' in vehicle and vehicle['speeds']:
                vehicles_with_speed_data += 1
                
                # Count speed entries
                speed_entries = len(vehicle['speeds'])
                total_speed_entries += speed_entries
                
                # Count non-zero speeds
                nonzero_speeds = sum(1 for s in vehicle['speeds'] if s.get('speed_kph', 0) > 0)
                nonzero_speed_entries += nonzero_speeds
                
                if nonzero_speeds > 0:
                    vehicles_with_nonzero_speed += 1
                    
                # Print details for diagnostic purposes
                print(f"Vehicle {vehicle['id']}: {speed_entries} speed entries, {nonzero_speeds} non-zero entries")
                
                # Print the first 3 speed entries if they exist
                if speed_entries > 0:
                    print("  First speed entries:")
                    for i, speed in enumerate(vehicle['speeds'][:3]):
                        print(f"    Frame {speed.get('frame', 'N/A')}: {speed.get('speed_kph', 0):.2f} km/h")
        
        # Print summary
        print("\nSpeed Statistics:")
        print(f"Vehicles with speed data: {vehicles_with_speed_data}/{total_vehicles} ({vehicles_with_speed_data/total_vehicles*100:.1f}%)")
        print(f"Vehicles with non-zero speed: {vehicles_with_nonzero_speed}/{total_vehicles} ({vehicles_with_nonzero_speed/total_vehicles*100:.1f}%)")
        
        if total_speed_entries > 0:
            print(f"Total speed entries: {total_speed_entries}")
            print(f"Non-zero speed entries: {nonzero_speed_entries}/{total_speed_entries} ({nonzero_speed_entries/total_speed_entries*100:.1f}%)")
        
        # Determine success
        if vehicles_with_nonzero_speed > 0:
            print("\n✅ SUCCESS: Some vehicles have non-zero speed values")
            return True
        else:
            print("\n❌ ERROR: No vehicles have non-zero speed values")
            
            # Diagnostic help
            if vehicles_with_speed_data > 0:
                print("\nDiagnostic Info:")
                print("  - Speed entries exist but all values are zero")
                print("  - This suggests vehicles are tracked but speed calculation is failing")
                print("  - Check MIN_DETECTION_FRAMES in speed_calculator.py")
                print("  - Check calibration parameters match your video")
            else:
                print("\nDiagnostic Info:")
                print("  - No speed entries exist")
                print("  - This suggests vehicles aren't being tracked across frames")
                print("  - Check tracking code in vehicle_recognition.py")
            
            return False
    
    except Exception as e:
        print(f"❌ ERROR analyzing vehicle data: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_calibration_variations():
    """Test different calibration values to see which work"""
    print("\n=== Testing Calibration Variations ===")
    
    # Create a synthetic test with different calibration values
    import speed_calculator as sc
    
    # Enable debug mode
    debug_speed_calculator(True)
    
    # Store original values
    orig_focal_length = sc.FOCAL_LENGTH_PX
    orig_camera_height = sc.CAMERA_HEIGHT_M
    orig_camera_angle = sc.CAMERA_ANGLE_DEG
    orig_vanishing_point = sc.VANISHING_POINT_Y_FACTOR
    
    # Create synthetic frame data
    frame_width = 1920
    frame_height = 1080
    fps = 30
    
    # Variations to test
    variations = [
        {"name": "Original", "focal_length": orig_focal_length, "camera_height": orig_camera_height, 
         "camera_angle": orig_camera_angle, "vanishing_point": orig_vanishing_point},
        {"name": "Higher focal length", "focal_length": orig_focal_length*1.5, "camera_height": orig_camera_height, 
         "camera_angle": orig_camera_angle, "vanishing_point": orig_vanishing_point},
        {"name": "Lower camera height", "focal_length": orig_focal_length, "camera_height": orig_camera_height*0.7, 
         "camera_angle": orig_camera_angle, "vanishing_point": orig_vanishing_point},
        {"name": "Higher camera angle", "focal_length": orig_focal_length, "camera_height": orig_camera_height, 
         "camera_angle": orig_camera_angle*1.5, "vanishing_point": orig_vanishing_point},
        {"name": "Lower vanishing point", "focal_length": orig_focal_length, "camera_height": orig_camera_height, 
         "camera_angle": orig_camera_angle, "vanishing_point": max(0.1, orig_vanishing_point*0.7)}
    ]
    
    results = []
    
    # Test each variation
    for var in variations:
        print(f"\nTesting: {var['name']}")
        print(f"  focal_length={var['focal_length']}, camera_height={var['camera_height']}, "
              f"camera_angle={var['camera_angle']}, vanishing_point={var['vanishing_point']}")
        
        # Set parameters
        sc.FOCAL_LENGTH_PX = var['focal_length']
        sc.CAMERA_HEIGHT_M = var['camera_height']
        sc.CAMERA_ANGLE_DEG = var['camera_angle']
        sc.VANISHING_POINT_Y_FACTOR = var['vanishing_point']
        sc.HORIZON_LINE_Y = None  # Reset to recalculate
        
        speeds = []
        
        # Run 10 frames with a vehicle moving upward (away from camera)
        for frame in range(1, 11):
            # Make vehicle move up (y decreases)
            y_pos = 800 - frame * 20  # Start at y=800, move up by 20px each frame
            
            # Create a bounding box
            bbox = (800, y_pos, 1000, y_pos + 100)
            
            # Calculate speed
            vehicle_id = f"test_vehicle_{var['name']}"
            speed = calculate_vehicle_speed(
                frame_number=frame,
                fps=fps,
                bbox=bbox,
                frame_width=frame_width,
                frame_height=frame_height,
                vehicle_id=vehicle_id
            )
            
            speeds.append(speed)
        
        # Analyze results
        nonzero_speeds = sum(1 for s in speeds if s > 0)
        avg_speed = np.mean([s for s in speeds if s > 0]) if nonzero_speeds > 0 else 0
        
        print(f"  Non-zero speeds: {nonzero_speeds}/10")
        print(f"  Average speed: {avg_speed:.2f} km/h")
        
        results.append({
            "name": var['name'],
            "nonzero_speeds": nonzero_speeds,
            "avg_speed": avg_speed
        })
    
    # Restore original values
    sc.FOCAL_LENGTH_PX = orig_focal_length
    sc.CAMERA_HEIGHT_M = orig_camera_height
    sc.CAMERA_ANGLE_DEG = orig_camera_angle
    sc.VANISHING_POINT_Y_FACTOR = orig_vanishing_point
    sc.HORIZON_LINE_Y = None
    
    # Print summary
    print("\nCalibration Test Results:")
    print("-" * 60)
    print(f"{'Variation':<25} {'Non-zero Speeds':<15} {'Average Speed':<15}")
    print("-" * 60)
    
    for result in results:
        print(f"{result['name']:<25} {result['nonzero_speeds']}/10{'':<9} {result['avg_speed']:.2f} km/h")
    
    # Identify best variation
    best_variation = max(results, key=lambda x: (x['nonzero_speeds'], x['avg_speed']))
    
    print("\nRecommended calibration variation:")
    print(f"  {best_variation['name']}")
    
    # Generate calibration file for best variation
    best_index = next(i for i, r in enumerate(results) if r['name'] == best_variation['name'])
    best_params = variations[best_index]
    
    calibration_data = {
        "lane_width_m": 3.7,
        "focal_length_px": best_params['focal_length'],
        "camera_height_m": best_params['camera_height'],
        "camera_angle_deg": best_params['camera_angle'],
        "vanishing_point_y_factor": best_params['vanishing_point'],
        "advanced": {
            "use_kalman_filter": True,
            "speed_smoothing_factor": 0.8,
            "max_speed_change": 10,
            "min_detection_frames": 1
        }
    }
    
    with open('recommended_calibration.json', 'w') as f:
        json.dump(calibration_data, f, indent=4)
    
    print(f"Recommended calibration saved to recommended_calibration.json")

def main():
    """Run the diagnostic checks"""
    print("Traffic Analysis System Diagnostic Tool")
    print("======================================\n")
    
    # Step 1: Check speed calculator with synthetic data
    speed_calculator_ok = check_speed_calculator()
    
    # Step 2: Check vehicle data file if it exists
    vehicle_data_path = "output/data/vehicle_data.json"
    vehicle_data_ok = False
    
    if os.path.exists(vehicle_data_path):
        vehicle_data_ok = analyze_vehicle_data(vehicle_data_path)
    else:
        print(f"\n⚠️ WARNING: Vehicle data file not found at {vehicle_data_path}")
    
    # Step 3: Test calibration variations
    test_calibration_variations()
    
    # Print overall summary
    print("\n=== Diagnostic Summary ===")
    print(f"Speed calculator: {'✅ OK' if speed_calculator_ok else '❌ Failed'}")
    
    if os.path.exists(vehicle_data_path):
        print(f"Vehicle data: {'✅ OK' if vehicle_data_ok else '❌ Failed'}")
    else:
        print("Vehicle data: ⚠️ Not checked (file not found)")
    
    print("\nRecommendations:")
    if not speed_calculator_ok:
        print("1. Modify speed_calculator.py to uncomment the forced test speed lines")
        print("   This will verify the pipeline can handle non-zero speeds")
    
    if os.path.exists("recommended_calibration.json"):
        print("2. Try using the recommended_calibration.json file:")
        print("   python run-traffic-analysis.py --video your_video.mp4 --calibration recommended_calibration.json --debug")
    
    print("3. Check for exceptions in the console output during video processing")
    print("4. Verify you're running with the --debug flag for detailed logging")
    print("5. Try a different video file to rule out video-specific issues")

if __name__ == "__main__":
    main()