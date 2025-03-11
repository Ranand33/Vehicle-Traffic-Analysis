#!/usr/bin/env python3

import os
import argparse
import subprocess
import time
import sys
import json
from pathlib import Path
from datetime import datetime

def print_section(title):
    """Print a formatted section header"""
    print("\n" + "="*80)
    print(title)
    print("="*80)

def run_command(cmd, description):
    """Run a command with proper error handling and timing"""
    print(f"Running: {' '.join(cmd)}")
    start_time = time.time()
    
    try:
        # Use subprocess.Popen to capture output in real-time
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        # Print output in real-time
        for line in process.stdout:
            print(line.rstrip())
        
        # Wait for process to complete and get return code
        return_code = process.wait()
        
        if return_code != 0:
            print(f"Error: {description} failed with return code {return_code}")
            return False
        
        elapsed_time = time.time() - start_time
        print(f"{description} completed in {elapsed_time/60:.2f} minutes")
        return True
    
    except Exception as e:
        print(f"Error executing {description}: {str(e)}")
        return False

def verify_calibration_file(calibration_file):
    """Verify and validate the calibration file"""
    if not calibration_file:
        return False
    
    if not os.path.exists(calibration_file):
        print(f"Warning: Calibration file {calibration_file} not found")
        return False
    
    try:
        with open(calibration_file, 'r') as f:
            # Try to load and validate the calibration data
            calibration = json.load(f)
            
            # Check for required fields
            required_fields = ['lane_width_m', 'focal_length_px', 'camera_height_m', 
                              'camera_angle_deg', 'vanishing_point_y_factor']
            
            missing_fields = [field for field in required_fields if field not in calibration]
            
            if missing_fields:
                print(f"Warning: Calibration file missing required fields: {', '.join(missing_fields)}")
                return False
            
            print("Calibration file verified successfully")
            return True
    
    except Exception as e:
        print(f"Error validating calibration file: {str(e)}")
        return False

def create_default_calibration(output_path):
    """Create a default calibration file if none exists"""
    default_calibration = {
        "lane_width_m": 3.7,
        "focal_length_px": 1200,
        "camera_height_m": 1.4,
        "camera_angle_deg": 5,
        "vanishing_point_y_factor": 0.38,
        "advanced": {
            "use_kalman_filter": True,
            "speed_smoothing_factor": 0.8,
            "max_speed_change": 10,
            "min_detection_frames": 1
        }
    }
    
    try:
        with open(output_path, 'w') as f:
            json.dump(default_calibration, f, indent=4)
        print(f"Created default calibration file at {output_path}")
        return True
    except Exception as e:
        print(f"Error creating default calibration file: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Run complete traffic analysis from video to report')
    parser.add_argument('--video', required=True, type=str, help='Path to input dashcam video')
    parser.add_argument('--output', default='output', type=str, help='Output directory')
    parser.add_argument('--vehicle_model', default='yolov8n.pt', type=str, help='Path to YOLOv8 vehicle detector model')
    parser.add_argument('--make_model', default=None, type=str, help='Path to vehicle make/model recognition model')
    parser.add_argument('--calibration', default=None, type=str, help='Path to speed calibration file')
    parser.add_argument('--confidence', default=0.5, type=float, help='Detection confidence threshold')
    parser.add_argument('--skip_video', action='store_true', help='Skip video processing (use existing data)')
    parser.add_argument('--debug', action='store_true', help='Enable debug output for vehicle detection and speed calculation')
    parser.add_argument('--emergency_speed', action='store_true', help='Force use of emergency speed calculation mode')
    parser.add_argument('--create_default_calibration', action='store_true', help='Create a default calibration file if none exists')
    
    args = parser.parse_args()
    
    # Verify that video file exists
    if not args.skip_video and not os.path.exists(args.video):
        print(f"Error: Video file {args.video} not found")
        return 1
    
    # Create default calibration file if requested
    if args.create_default_calibration and not args.calibration:
        default_calibration_path = 'default_calibration.json'
        create_default_calibration(default_calibration_path)
        args.calibration = default_calibration_path
    
    # Verify calibration file
    if args.calibration:
        calibration_valid = verify_calibration_file(args.calibration)
        if not calibration_valid:
            print("Warning: Calibration file validation failed. Using default values.")
    else:
        print("Warning: No calibration file specified. Speed calculations may be inaccurate.")
        print("Consider creating a calibration.json file for your specific video.")
        print("You can use --create_default_calibration to create a starting point.")
    
    # Create output directory structure
    output_dir = Path(args.output)
    data_dir = output_dir / 'data'
    analysis_dir = output_dir / 'analysis'
    log_dir = output_dir / 'logs'
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(analysis_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Set paths for files
    video_output = data_dir / 'output_video.mp4'
    data_output = data_dir / 'vehicle_data.json'
    log_file = log_dir / f'run_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
    
    # Log basic information
    with open(log_file, 'w') as f:
        f.write(f"Traffic Analysis Run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Video: {args.video}\n")
        f.write(f"Calibration: {args.calibration if args.calibration else 'None'}\n")
        f.write(f"Vehicle Model: {args.vehicle_model}\n")
        f.write(f"Make/Model Recognition: {args.make_model if args.make_model else 'Disabled'}\n")
        f.write(f"Confidence Threshold: {args.confidence}\n")
        f.write(f"Debug Mode: {args.debug}\n")
        f.write(f"Emergency Speed Mode: {args.emergency_speed}\n\n")
    
    # Step 1: Process the video (unless skipped)
    if not args.skip_video:
        print_section(f"STEP 1: Processing video {args.video}")
        
        cmd = [
            'python', 'vehicle_recognition.py',
            '--input', args.video,
            '--output', str(data_dir),
            '--vehicle_model', args.vehicle_model,
            '--confidence', str(args.confidence)
        ]
        
        if args.make_model:
            cmd.extend(['--make_model', args.make_model])
        
        if args.calibration:
            cmd.extend(['--calibration', args.calibration])
        
        if args.debug:
            cmd.append('--debug')
            
        if args.emergency_speed:
            cmd.append('--emergency_speed')
        
        # Run the vehicle recognition system
        if not run_command(cmd, "Video processing"):
            print("Video processing failed. Aborting analysis.")
            return 1
    else:
        print("Skipping video processing (using existing data)")
    
    # Step 2: Run data science analysis
    print_section("STEP 2: Running data science analysis")
    
    if not data_output.exists():
        print(f"Error: Data file {data_output} not found. Please process the video first.")
        return 1
    
    # Run the data analysis
    cmd = [
        'python', 'data-science-analysis.py',
        '--data', str(data_output),
        '--output', str(analysis_dir)
    ]
    
    if args.debug:
        cmd.append('--debug')
    
    if not run_command(cmd, "Data analysis"):
        print("Data analysis failed.")
        return 1
    
    # Print summary of outputs
    print_section("ANALYSIS COMPLETE")
    print(f"Processed video: {args.video}")
    print(f"Output directory: {output_dir}")
    print(f"Video output: {video_output}")
    print(f"Data output: {data_output}")
    print(f"Analysis report: {analysis_dir}/traffic_analysis_report.md")
    print(f"Visualizations: {analysis_dir}")
    print(f"Log file: {log_file}")
    print("="*80)
    
    # Check if any vehicles had speeds calculated
    try:
        with open(data_output, 'r') as f:
            data = json.load(f)
            vehicles_with_speed = 0
            total_vehicles = len(data.get('vehicles', []))
            
            for vehicle in data.get('vehicles', []):
                has_speed = any(speed.get('speed_kph', 0) > 0 for speed in vehicle.get('speeds', []))
                if has_speed:
                    vehicles_with_speed += 1
            
            print(f"\nVehicle Speed Statistics:")
            print(f"  Total vehicles detected: {total_vehicles}")
            
            if total_vehicles > 0:
                print(f"  Vehicles with calculated speeds: {vehicles_with_speed} ({vehicles_with_speed/total_vehicles*100:.1f}%)")
                
                if args.emergency_speed:
                    print("  NOTE: Emergency speed calculation mode was used.")
                    print("        Speeds are estimated based on vehicle position in frame.")
                
                if vehicles_with_speed == 0:
                    print("\nWARNING: No vehicles had speed calculated!")
                    print("Suggestions:")
                    print("  1. Try running with --emergency_speed flag to force position-based speed estimation")
                    print("  2. Check that the calibration.json values match your video")
                    print("  3. Increase tracking duration by ensuring vehicles stay in frame longer")
                    print("  4. Run with --debug flag to see detailed speed calculation information")
                elif vehicles_with_speed < total_vehicles * 0.5 and not args.emergency_speed:
                    print("\nNote: Less than half of tracked vehicles had speed calculated.")
                    print("This may be normal for vehicles that appear briefly in frame.")
                    print("Consider using --emergency_speed for more consistent results.")
    except Exception as e:
        print(f"Error analyzing speed statistics: {str(e)}")
    
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nProcess interrupted by user. Exiting.")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        sys.exit(1)