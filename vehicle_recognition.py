import cv2
import numpy as np
import os
import time
import math
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import stats
from ultralytics import YOLO
import json
from datetime import datetime
from speed_calculator import calculate_vehicle_speed, init_speed_calculator, debug_speed_calculator
import argparse
from pathlib import Path

class VehicleAnalyzer:
    def __init__(self, input_video, output_dir="output", vehicle_detector_model="yolov8n.pt", 
                 make_model_detector_model=None, confidence_threshold=0.5, calibration_file=None,
                 debug_mode=False):
        """
        Initialize the Vehicle Analyzer for processing dashcam videos and analyzing traffic patterns
        
        Args:
            input_video (str): Path to input dashcam video
            output_dir (str): Directory to save output files
            vehicle_detector_model (str): Path to YOLOv8 vehicle detection model
            make_model_detector_model (str): Path to vehicle make/model recognition model
            confidence_threshold (float): Confidence threshold for detections
            calibration_file (str): Path to speed calibration file
            debug_mode (bool): Enable extra debug output
        """
        self.input_video = input_video
        self.output_dir = output_dir
        self.output_video = os.path.join(output_dir, "output_video.mp4")
        self.output_data = os.path.join(output_dir, "vehicle_data.json")
        self.confidence_threshold = confidence_threshold
        self.debug_mode = debug_mode
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Define vehicle classes (COCO dataset)
        self.vehicle_classes = {
            2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck',
            # Additional vehicle classes if needed
            9: 'traffic light', 
        }
        
        # Initialize speed calculator
        if calibration_file:
            init_speed_calculator(calibration_file)
            # Enable debug mode for speed calculator
            debug_speed_calculator(True)
        else:
            # Still initialize with defaults
            init_speed_calculator(None)
            debug_speed_calculator(debug_mode)
        
        # Load models
        print("Loading detection models...")
        self.vehicle_detector = YOLO(vehicle_detector_model)
        
        # Load make/model recognizer if provided
        self.make_model_recognizer = None
        if make_model_detector_model and os.path.exists(make_model_detector_model):
            print(f"Loading make/model recognizer from {make_model_detector_model}")
            self.make_model_recognizer = YOLO(make_model_detector_model)
            
            # Load class mapping file if it exists (in same directory as model)
            model_dir = os.path.dirname(make_model_detector_model)
            mapping_file = os.path.join(model_dir, "class_mapping.yaml")
            self.class_mapping = {}
            
            if os.path.exists(mapping_file):
                import yaml
                with open(mapping_file, 'r') as f:
                    self.class_mapping = yaml.safe_load(f)
                print(f"Loaded {len(self.class_mapping)} make/model classes")
            else:
                print("Warning: No class mapping file found. Make/model names will be class indices.")
        else:
            print("No make/model recognizer specified or model file not found")
            self.make_model_recognizer = None
            self.class_mapping = {}
        
        # Data collection variables
        self.frame_data = []  # For collecting per-frame data
        self.vehicle_data = {
            "video_info": {
                "filename": input_video,
                "processed_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            },
            "vehicles": []
        }
        
        # Tracking variables
        self.tracked_vehicles = {}
        self.next_vehicle_id = 0
        
        # Status reporting variables
        self.total_speed_entries = 0
        self.nonzero_speed_entries = 0
    
    def process_video(self):
        """Process the video and collect data for analysis"""
        # Open the input video
        cap = cv2.VideoCapture(self.input_video)
        if not cap.isOpened():
            print(f"Error: Could not open video {self.input_video}")
            return False
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Update video info
        self.vehicle_data["video_info"].update({
            "fps": fps,
            "resolution": f"{width}x{height}",
            "total_frames": total_frames
        })
        
        # Print video info
        print(f"Processing video: {self.input_video}")
        print(f"Resolution: {width}x{height}, FPS: {fps}, Total frames: {total_frames}")
        
        # Create video writer for output
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self.output_video, fourcc, fps, (width, height))
        
        # Process each frame
        frame_count = 0
        start_time = time.time()
        
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process the frame
                frame_count += 1
                timestamp = frame_count / fps
                
                # Detect vehicles
                vehicle_results = self.vehicle_detector(frame)
                
                # For current frame statistics
                frame_vehicles = []
                current_speeds = []
                vehicle_count = 0
                
                # Get detections
                detections = []
                if len(vehicle_results) > 0:
                    # Process each detection
                    for result in vehicle_results:
                        boxes = result.boxes.data.cpu().numpy()
                        for box in boxes:
                            x1, y1, x2, y2, conf, class_id = box
                            class_id = int(class_id)
                            
                            # Check if this is a vehicle class and meets confidence threshold
                            if class_id in self.vehicle_classes and conf >= self.confidence_threshold:
                                detections.append((x1, y1, x2, y2, conf, class_id))
                
                # Process each detection
                for detection in detections:
                    x1, y1, x2, y2, conf, class_id = detection
                    vehicle_count += 1
                    
                    # Convert to integers for OpenCV
                    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                    
                    # Skip if bounding box is invalid
                    if x1 >= x2 or y1 >= y2 or x1 < 0 or y1 < 0 or x2 > width or y2 > height:
                        continue
                    
                    # Extract vehicle ROI
                    vehicle_roi = frame[y1:y2, x1:x2]
                    
                    if vehicle_roi.size == 0:
                        continue
                    
                    # Recognize make and model if we have a recognizer
                    vehicle_make = "Unknown"
                    vehicle_model = "Unknown"
                    
                    if self.make_model_recognizer:
                        try:
                            make_model_results = self.make_model_recognizer(vehicle_roi)
                            
                            if len(make_model_results) > 0 and len(make_model_results[0].boxes) > 0:
                                # Get the highest confidence result
                                mm_boxes = make_model_results[0].boxes
                                mm_confidences = mm_boxes.conf.cpu().numpy()
                                
                                if len(mm_confidences) > 0:
                                    best_idx = np.argmax(mm_confidences)
                                    mm_class_id = int(mm_boxes.cls[best_idx].item())
                                    mm_conf = float(mm_confidences[best_idx])
                                    
                                    # Map class ID to make/model name
                                    if mm_class_id in self.class_mapping:
                                        class_name = self.class_mapping[mm_class_id]
                                        if "_" in class_name:
                                            vehicle_make, vehicle_model = class_name.split("_", 1)
                                        else:
                                            vehicle_make = class_name
                                    else:
                                        vehicle_make = f"Class_{mm_class_id}"
                        except Exception as e:
                            if self.debug_mode:
                                print(f"Error during make/model recognition: {e}")

                    # Track the vehicle
                    vehicle_center = ((x1 + x2) // 2, (y1 + y2) // 2)
                    vehicle_id = None
                    vehicle_width = x2 - x1
                    vehicle_height = y2 - y1
                    vehicle_area = vehicle_width * vehicle_height

                    # Improved tracking based on center position and size
                    for v_id, v_info in self.tracked_vehicles.items():
                        # Skip vehicles not seen in recent frames (more than 5 frames ago)
                        if frame_count - v_info.get("last_seen_frame", 0) > 5:
                            continue
                            
                        prev_center = v_info["last_center"]
                        # Calculate normalized distance (based on vehicle size)
                        dist = np.sqrt((vehicle_center[0] - prev_center[0])**2 + 
                                      (vehicle_center[1] - prev_center[1])**2)
                        
                        # Adjust threshold based on vehicle size and position
                        # Larger/closer vehicles can move more pixels between frames
                        size_factor = math.sqrt(vehicle_width * vehicle_height) / 100
                        position_factor = y2 / height  # Lower in frame = larger threshold
                        threshold = 40 + 50 * size_factor + 30 * position_factor  # Increased thresholds
                        
                        # Check if centers are close enough
                        if dist < threshold:
                            vehicle_id = v_id
                            if self.debug_mode:
                                print(f"Frame {frame_count}: Matched vehicle {v_id} (dist={dist:.1f}, threshold={threshold:.1f})")
                            break

                    # If no match was found, create a new vehicle ID
                    if vehicle_id is None:
                        # New vehicle
                        vehicle_id = self.next_vehicle_id
                        self.next_vehicle_id += 1
                        self.tracked_vehicles[vehicle_id] = {
                            "first_seen": frame_count,
                            "last_seen_frame": frame_count,
                            "make": vehicle_make,
                            "model": vehicle_model,
                            "vehicle_class": self.vehicle_classes.get(class_id, f"Class_{class_id}"),
                            "speeds": [],
                            "positions": [],
                            "last_center": vehicle_center
                        }
                        # Add to our vehicle data list
                        self.vehicle_data["vehicles"].append({
                            "id": vehicle_id,
                            "make": vehicle_make,
                            "model": vehicle_model,
                            "vehicle_class": self.vehicle_classes.get(class_id, f"Class_{class_id}"),
                            "first_seen_frame": frame_count,
                            "first_seen_time": timestamp,
                            "speeds": []
                        })
                        
                        if self.debug_mode:
                            print(f"Frame {frame_count}: Created new vehicle ID {vehicle_id}")
                    else:
                        # Update existing vehicle
                        self.tracked_vehicles[vehicle_id]["last_center"] = vehicle_center
                        self.tracked_vehicles[vehicle_id]["last_seen_frame"] = frame_count
                        # Update make/model if previously unknown
                        if self.tracked_vehicles[vehicle_id]["make"] == "Unknown" and vehicle_make != "Unknown":
                            self.tracked_vehicles[vehicle_id]["make"] = vehicle_make
                            self.tracked_vehicles[vehicle_id]["model"] = vehicle_model
                            # Update in vehicle_data as well
                            for vehicle in self.vehicle_data["vehicles"]:
                                if vehicle["id"] == vehicle_id:
                                    vehicle["make"] = vehicle_make
                                    vehicle["model"] = vehicle_model
                    
                    # Calculate vehicle speed using our enhanced speed calculator
                    vehicle_speed = calculate_vehicle_speed(
                        frame_count, 
                        fps, 
                        (x1, y1, x2, y2), 
                        width, 
                        height,
                        vehicle_id
                    )
                    
                    # Track speed statistics
                    self.total_speed_entries += 1
                    if vehicle_speed > 0:
                        self.nonzero_speed_entries += 1
                        if self.debug_mode and frame_count % 10 == 0:
                            nonzero_pct = (self.nonzero_speed_entries / self.total_speed_entries) * 100
                            print(f"Speed stats: {self.nonzero_speed_entries}/{self.total_speed_entries} non-zero speeds ({nonzero_pct:.1f}%)")
                    
                    # Only include valid speeds
                    if 0 < vehicle_speed < 200:  # Reasonable speed range in km/h
                        current_speeds.append(vehicle_speed)
                    
                    # Update vehicle data
                    self.tracked_vehicles[vehicle_id]["speeds"].append(vehicle_speed)
                    self.tracked_vehicles[vehicle_id]["positions"].append((x1, y1, x2, y2))
                    
                    # Update speeds in JSON data
                    for vehicle in self.vehicle_data["vehicles"]:
                        if vehicle["id"] == vehicle_id:
                            vehicle["speeds"].append({
                                "frame": frame_count,
                                "time": timestamp,
                                "speed_kph": vehicle_speed
                            })
                    
                    # Add to current frame data
                    frame_vehicles.append({
                        "id": vehicle_id,
                        "make": vehicle_make,
                        "model": vehicle_model,
                        "vehicle_class": self.vehicle_classes.get(class_id, f"Class_{class_id}"),
                        "speed": vehicle_speed,
                        "position": (x1, y1, x2, y2)
                    })
                    
                    # Draw bounding box and information on the frame
                    # Use color based on speed (green for fast, yellow for medium, red for slow)
                    if vehicle_speed >= 80:
                        color = (0, 255, 0)  # Green
                    elif vehicle_speed >= 40:
                        color = (0, 255, 255)  # Yellow
                    else:
                        color = (0, 0, 255)  # Red
                        
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Enhanced label with more visible speed
                    speed_str = f"{vehicle_speed:.1f} km/h"
                    label = f"ID:{vehicle_id} {vehicle_make} {vehicle_model}"
                    
                    # Draw background for label
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                    cv2.rectangle(frame, (x1, y1 - 30), (x1 + label_size[0], y1), (0, 0, 0), -1)
                    
                    # Draw label text
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    
                    # Draw speed with larger, more visible font and contrasting background
                    speed_size = cv2.getTextSize(speed_str, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                    cv2.rectangle(frame, (x1, y1 - 55), (x1 + speed_size[0], y1 - 30), (0, 0, 0), -1)
                    cv2.putText(frame, speed_str, (x1, y1 - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Calculate average speed for this frame
                avg_speed = np.mean(current_speeds) if current_speeds else 0
                
                # Add overlay with stats
                cv2.rectangle(frame, (10, 10), (350, 110), (0, 0, 0), -1)  # Background
                cv2.putText(frame, f"Frame: {frame_count}/{total_frames}", (15, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                cv2.putText(frame, f"Vehicles: {vehicle_count}", (15, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                cv2.putText(frame, f"Avg Speed: {avg_speed:.1f} km/h", (15, 70), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                
                # Add speed stats to overlay
                nonzero_pct = (self.nonzero_speed_entries / self.total_speed_entries) * 100 if self.total_speed_entries > 0 else 0
                cv2.putText(frame, f"Speed calculations: {self.nonzero_speed_entries}/{self.total_speed_entries} ({nonzero_pct:.1f}%)", 
                            (15, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                
                # Write the processed frame to output video
                out.write(frame)
                
                # Store frame data for analysis
                self.frame_data.append({
                    "frame": frame_count,
                    "time": timestamp,
                    "vehicle_count": vehicle_count,
                    "avg_speed": avg_speed,
                    "vehicles": frame_vehicles
                })
                
                # Print progress
                if frame_count % 100 == 0 or frame_count == total_frames:
                    elapsed_time = time.time() - start_time
                    frames_per_second = frame_count / elapsed_time if elapsed_time > 0 else 0
                    estimated_time = (total_frames - frame_count) / frames_per_second if frames_per_second > 0 else 0
                    print(f"Processed {frame_count}/{total_frames} frames ({(frame_count/total_frames*100):.1f}%) "
                          f"at {frames_per_second:.1f} fps. Est. time remaining: {estimated_time:.1f} seconds")
                    
                    # Print speed stats periodically
                    nonzero_pct = (self.nonzero_speed_entries / self.total_speed_entries) * 100 if self.total_speed_entries > 0 else 0
                    print(f"Speed calculations: {self.nonzero_speed_entries}/{self.total_speed_entries} non-zero speeds ({nonzero_pct:.1f}%)")
        
        except Exception as e:
            print(f"Error processing video: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Release resources
            cap.release()
            out.release()
            
            # Save the vehicle data to a JSON file
            with open(self.output_data, 'w') as f:
                json.dump(self.vehicle_data, f, indent=4)
            
            print(f"Processing complete. Output saved to {self.output_video}")
            print(f"Vehicle data saved to {self.output_data}")
            
            # Print final speed stats
            if self.total_speed_entries > 0:
                nonzero_pct = (self.nonzero_speed_entries / self.total_speed_entries) * 100
                print(f"Final speed statistics: {self.nonzero_speed_entries}/{self.total_speed_entries} non-zero speeds ({nonzero_pct:.1f}%)")
            
            return True
    
    def analyze_data(self):
        """Analyze the collected data and generate insights"""
        if not self.frame_data:
            print("No data to analyze. Please process a video first.")
            return False
        
        print("Analyzing vehicle data...")
        
        # Create DataFrame for analysis
        df = pd.DataFrame([
            {"frame": fd["frame"], 
             "time": fd["time"], 
             "vehicle_count": fd["vehicle_count"], 
             "avg_speed": fd["avg_speed"]}
            for fd in self.frame_data
        ])
        
        # Save raw data to CSV
        csv_path = os.path.join(self.output_dir, "frame_data.csv")
        df.to_csv(csv_path, index=False)
        print(f"Raw frame data saved to {csv_path}")
        
        # Only analyze frames with vehicles
        df_with_vehicles = df[df['vehicle_count'] > 0]
        
        # Handle empty dataset
        if len(df_with_vehicles) == 0:
            print("No frames with vehicles detected.")
            return False
        
        # Calculate correlation between vehicle count and average speed
        # Filter out frames with zero speed to get meaningful correlations
        df_with_speed = df_with_vehicles[df_with_vehicles['avg_speed'] > 0]
        
        if len(df_with_speed) >= 2:  # Need at least 2 points for correlation
            correlation, p_value = stats.pearsonr(df_with_speed['vehicle_count'], 
                                                  df_with_speed['avg_speed'])
            
            print(f"Correlation Analysis:")
            print(f"Pearson correlation coefficient: {correlation:.3f}")
            print(f"P-value: {p_value:.5f}")
            
            if p_value < 0.05:
                correlation_significance = "statistically significant"
            else:
                correlation_significance = "not statistically significant"
            
            if correlation < -0.5:
                correlation_description = "strong negative"
            elif correlation < -0.3:
                correlation_description = "moderate negative"
            elif correlation < -0.1:
                correlation_description = "weak negative"
            elif correlation < 0.1:
                correlation_description = "very weak or no"
            elif correlation < 0.3:
                correlation_description = "weak positive"
            elif correlation < 0.5:
                correlation_description = "moderate positive"
            else:
                correlation_description = "strong positive"
            
            correlation_summary = (
                f"There is a {correlation_description} correlation ({correlation:.3f}) between "
                f"the number of vehicles and their average speed, which is {correlation_significance}."
            )
        else:
            # Default values if not enough data points
            correlation = 0
            p_value = 1
            correlation_summary = "Insufficient data for correlation analysis."
            
        print(correlation_summary)
        
        # Generate visualizations
        self._generate_visualizations(df, correlation, p_value, correlation_summary)
        
        # Generate summary statistics
        summary_stats = {
            "frames_analyzed": len(df),
            "frames_with_vehicles": len(df_with_vehicles),
            "frames_with_speed_data": len(df_with_speed),
            "total_unique_vehicles": len(self.vehicle_data["vehicles"]),
            "avg_vehicle_count": df['vehicle_count'].mean(),
            "max_vehicle_count": df['vehicle_count'].max(),
            "avg_speed_overall": df_with_speed['avg_speed'].mean() if len(df_with_speed) > 0 else 0,
            "max_speed": df_with_speed['avg_speed'].max() if len(df_with_speed) > 0 else 0,
            "correlation": correlation,
            "p_value": p_value,
            "correlation_summary": correlation_summary,
            "total_speed_measurements": self.total_speed_entries,
            "nonzero_speed_measurements": self.nonzero_speed_entries,
            "speed_calculation_rate": (self.nonzero_speed_entries / self.total_speed_entries * 100) if self.total_speed_entries > 0 else 0
        }
        
        # Add vehicle make/model statistics if available
        if self.make_model_recognizer:
            # Count occurrences of each make and model
            makes = [v["make"] for v in self.vehicle_data["vehicles"] if v["make"] != "Unknown"]
            models = [f"{v['make']} {v['model']}" for v in self.vehicle_data["vehicles"] 
                      if v["make"] != "Unknown" and v["model"] != "Unknown"]
            
            if makes:
                make_counts = pd.Series(makes).value_counts().to_dict()
                summary_stats["top_makes"] = dict(sorted(make_counts.items(), 
                                                         key=lambda x: x[1], reverse=True)[:5])
            
            if models:
                model_counts = pd.Series(models).value_counts().to_dict()
                summary_stats["top_models"] = dict(sorted(model_counts.items(), 
                                                          key=lambda x: x[1], reverse=True)[:5])
        
        # Save summary statistics
        summary_path = os.path.join(self.output_dir, "analysis_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary_stats, f, indent=4)
        
        print(f"Analysis summary saved to {summary_path}")
        return True
    
    def _generate_visualizations(self, df, correlation, p_value, correlation_summary):
        """Generate visualization plots"""
        # Create visualizations directory
        vis_dir = os.path.join(self.output_dir, "visualizations")
        os.makedirs(vis_dir, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set(font_scale=1.2)
        
        # 1. Time series of vehicle count
        plt.figure(figsize=(12, 6))
        plt.plot(df['time'], df['vehicle_count'], color='blue', linewidth=1)
        plt.title('Number of Vehicles Over Time')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Vehicle Count')
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, "vehicle_count_time_series.png"), dpi=300)
        plt.close()
        
        # 2. Time series of average speed
        plt.figure(figsize=(12, 6))
        df_speed = df[df['avg_speed'] > 0]  # Filter out zero speeds
        if len(df_speed) > 0:
            plt.plot(df_speed['time'], df_speed['avg_speed'], color='green', linewidth=1)
            
            # Add annotation with count of non-zero speeds
            non_zero_pct = len(df_speed) / len(df) * 100
            plt.annotate(
                f'Non-zero speeds: {len(df_speed)} ({non_zero_pct:.1f}% of frames)',
                xy=(0.05, 0.95),
                xycoords='axes fraction',
                fontsize=12,
                bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8)
            )
            
            plt.title('Average Vehicle Speed Over Time (Non-Zero Only)')
            plt.xlabel('Time (seconds)')
            plt.ylabel('Average Speed (km/h)')
            plt.tight_layout()
            plt.savefig(os.path.join(vis_dir, "avg_speed_time_series.png"), dpi=300)
        plt.close()
        
        # 3. Scatter plot of vehicle count vs. average speed with regression line
        plt.figure(figsize=(10, 8))
        # Only use frames with non-zero speeds for correlation scatter
        df_with_speed = df[df['avg_speed'] > 0]
        if len(df_with_speed) > 0:
            ax = sns.regplot(
                x='vehicle_count', 
                y='avg_speed', 
                data=df_with_speed, 
                scatter_kws={'alpha':0.4}, 
                line_kws={'color':'red'}
            )
            plt.title('Correlation Between Vehicle Count and Average Speed')
            plt.xlabel('Number of Vehicles')
            plt.ylabel('Average Speed (km/h)')
            
            # Add correlation annotation
            corr_text = f"Correlation: {correlation:.3f} (p={p_value:.3f})"
            plt.annotate(corr_text, xy=(0.05, 0.95), xycoords='axes fraction', 
                        backgroundcolor='white', fontsize=12)
            
            plt.tight_layout()
            plt.savefig(os.path.join(vis_dir, "correlation_scatter.png"), dpi=300)
        plt.close()
        
        # 4. Histogram of vehicle counts
        plt.figure(figsize=(10, 6))
        sns.histplot(df['vehicle_count'], kde=True, color='blue')
        plt.title('Distribution of Vehicle Counts')
        plt.xlabel('Number of Vehicles per Frame')
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, "vehicle_count_histogram.png"), dpi=300)
        plt.close()
        
        # 5. Histogram of average speeds
        plt.figure(figsize=(10, 6))
        df_speed = df[df['avg_speed'] > 0]
        if len(df_speed) > 0:
            sns.histplot(df_speed['avg_speed'], kde=True, color='green')
            plt.title('Distribution of Average Speeds')
            plt.xlabel('Average Speed (km/h)')
            plt.ylabel('Frequency')
            plt.tight_layout()
            plt.savefig(os.path.join(vis_dir, "speed_histogram.png"), dpi=300)
        plt.close()
        
        # 6. Summary visualization with key findings
        plt.figure(figsize=(12, 8))
        plt.text(0.5, 0.95, "Vehicle Traffic Analysis Summary", ha='center', fontsize=20, fontweight='bold')
        
        # Add summary statistics
        summary_text = [
            f"Total frames analyzed: {len(df)}",
            f"Frames with vehicles: {len(df[df['vehicle_count'] > 0])}",
            f"Maximum vehicles in a frame: {df['vehicle_count'].max()}",
            f"Average vehicles per frame: {df['vehicle_count'].mean():.2f}",
            f"Average speed: {df_speed['avg_speed'].mean():.2f} km/h" if len(df_speed) > 0 else "No speed data",
            f"Maximum speed: {df_speed['avg_speed'].max():.2f} km/h" if len(df_speed) > 0 else "No speed data",
            f"Speed calculations: {self.nonzero_speed_entries}/{self.total_speed_entries} ({self.nonzero_speed_entries/self.total_speed_entries*100:.1f}%)" if self.total_speed_entries > 0 else "No speed data",
            "",
            "Correlation Analysis:",
            correlation_summary,
        ]
        
        y_pos = 0.85
        for line in summary_text:
            plt.text(0.5, y_pos, line, ha='center', fontsize=14)
            y_pos -= 0.05
        
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, "analysis_summary.png"), dpi=300)
        plt.close()
        
        if hasattr(self, 'make_model_recognizer') and self.make_model_recognizer:
            # 7. Bar chart of vehicle makes
            makes = [v["make"] for v in self.vehicle_data["vehicles"] if v["make"] != "Unknown"]
            if makes:
                plt.figure(figsize=(12, 8))
                make_counts = pd.Series(makes).value_counts().nlargest(10)
                sns.barplot(x=make_counts.values, y=make_counts.index, palette='viridis')
                plt.title('Top 10 Vehicle Makes Detected')
                plt.xlabel('Count')
                plt.ylabel('Make')
                plt.tight_layout()
                plt.savefig(os.path.join(vis_dir, "top_makes_bar_chart.png"), dpi=300)
                plt.close()
        
        print(f"Visualizations saved to {vis_dir}")

def main():
    parser = argparse.ArgumentParser(description='Vehicle Analysis and Traffic Pattern Detection')
    parser.add_argument('--input', type=str, required=True, help='Input video file')
    parser.add_argument('--output', type=str, default='output', help='Output directory')
    parser.add_argument('--vehicle_model', type=str, default='yolov8n.pt', help='YOLOv8 vehicle detection model')
    parser.add_argument('--make_model', type=str, default=None, help='Vehicle make/model recognition model')
    parser.add_argument('--confidence', type=float, default=0.5, help='Detection confidence threshold')
    parser.add_argument('--calibration', type=str, default=None, help='Speed calibration file')
    parser.add_argument('--analysis_only', action='store_true', help='Run analysis on existing data without processing video')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')
    
    args = parser.parse_args()
    
    # Create the analyzer
    analyzer = VehicleAnalyzer(
        input_video=args.input,
        output_dir=args.output,
        vehicle_detector_model=args.vehicle_model,
        make_model_detector_model=args.make_model,
        confidence_threshold=args.confidence,
        calibration_file=args.calibration,
        debug_mode=args.debug
    )
    
    # If not analysis only, process the video
    if not args.analysis_only:
        if not analyzer.process_video():
            print("Video processing failed")
            return
    else:
        # Check if data exists
        data_file = os.path.join(args.output, "vehicle_data.json")
        if not os.path.exists(data_file):
            print(f"Data file {data_file} not found. Please process a video first.")
            return
        
        # Load existing data
        with open(data_file, 'r') as f:
            analyzer.vehicle_data = json.load(f)
            
            # Count speed entries for statistics
            total_entries = 0
            nonzero_entries = 0
            for vehicle in analyzer.vehicle_data.get('vehicles', []):
                if 'speeds' in vehicle:
                    speeds = vehicle['speeds']
                    total_entries += len(speeds)
                    nonzero_entries += sum(1 for s in speeds if s.get('speed_kph', 0) > 0)
            
            analyzer.total_speed_entries = total_entries
            analyzer.nonzero_speed_entries = nonzero_entries
    
    # Analyze the data
    analyzer.analyze_data()

if __name__ == "__main__":
    main()