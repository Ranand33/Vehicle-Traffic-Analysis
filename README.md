# Vehicle-Traffic-Analysis
## Description
I'm trying to create a comprehensive traffic analysis system using computer vision to detect vehicles, track movement, calculate speeds, and identify vehicle makes and models from video footage. Please suggest changes to improve the system.

## Features
Vehicle Detection: YOLOv8-based detection of cars, trucks, buses, and motorcycles
Vehicle Tracking: Adaptive tracking algorithm with perspective awareness
Speed Calculation: Position-based velocity estimation without specialized equipment
Make/Model Recognition: Custom-trained model for vehicle brand identification
Statistical Analysis: Comprehensive data processing and visualization pipeline

# Usage 
Install all dependencies and run this command on a terminal
`python run-traffic-analysis.py --video YourVideo.mp4 --make_model (this is optional) /YourPath/best.pt --calibration recommended_calibration.json`
