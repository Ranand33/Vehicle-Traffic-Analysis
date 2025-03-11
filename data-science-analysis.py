import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
import json
import os
import argparse
from pathlib import Path
import warnings
import time  # For timestamps in reports
from datetime import datetime
warnings.filterwarnings('ignore')

class TrafficDataAnalyzer:
    """
    Advanced analysis of traffic data from vehicle recognition system with
    focus on correlation between vehicle density and speed
    """
    
    def __init__(self, data_file, output_dir="analysis_output", debug=False):
        """
        Initialize the analyzer with data from vehicle recognition system
        
        Args:
            data_file (str): Path to JSON data file from vehicle recognition system
            output_dir (str): Directory to save analysis outputs
            debug (bool): Enable debug output
        """
        self.data_file = data_file
        self.output_dir = output_dir
        self.debug = debug
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load data
        self.raw_data = self._load_data()
        
        # Processed data frames
        self.frame_df = None
        self.vehicle_df = None
        self.make_model_df = None
        self.speed_df = None  # New dataframe specifically for speed analysis
        
        # Analysis results
        self.correlation_results = {}
        self.regression_results = None
        self.hourly_patterns = None
        self.speed_analysis_results = {}  # New container for speed-specific analysis
        
        # Debug information
        if self.debug:
            print(f"Initialized TrafficDataAnalyzer with data from {data_file}")
            print(f"Output will be saved to {output_dir}")
    
    def _load_data(self):
        """Load and validate the data file"""
        try:
            with open(self.data_file, 'r') as f:
                data = json.load(f)
                
            # Minimal validation
            if 'vehicles' not in data or 'video_info' not in data:
                raise ValueError("Invalid data format: missing 'vehicles' or 'video_info'")
            
            if self.debug:
                print(f"Successfully loaded data file with {len(data['vehicles'])} vehicles")
                print(f"Video info: {data['video_info']['filename']}, {data['video_info'].get('resolution', 'unknown resolution')}")
                
                # Check for speed data
                speed_counts = 0
                valid_speed_counts = 0
                for vehicle in data['vehicles']:
                    if 'speeds' in vehicle:
                        speed_counts += len(vehicle['speeds'])
                        valid_speed_counts += sum(1 for s in vehicle['speeds'] if s.get('speed_kph', 0) > 0)
                
                print(f"Found {speed_counts} speed measurements, {valid_speed_counts} with non-zero values")
                
            return data
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    def process_data(self):
        """Process raw data into analysis-ready DataFrames"""
        if not self.raw_data:
            print("No data to process")
            return False
        
        # Extract video information
        video_info = self.raw_data['video_info']
        fps = video_info.get('fps', 30)
        
        # Process frame-level data if available
        if 'frame_data' in self.raw_data:
            self.frame_df = pd.DataFrame(self.raw_data['frame_data'])
        else:
            # Create frame-level data from vehicle detections
            print("Creating frame-level data from vehicle detections...")
            self._reconstruct_frame_data(fps)
        
        # Process vehicle-level data
        vehicle_data = []
        speed_data = []  # For detailed speed analysis
        
        for vehicle in self.raw_data['vehicles']:
            vehicle_id = vehicle['id']
            make = vehicle.get('make', 'Unknown')
            model = vehicle.get('model', 'Unknown')
            vehicle_class = vehicle.get('vehicle_class', 'Unknown')
            first_seen_frame = vehicle.get('first_seen_frame', 0)
            first_seen_time = vehicle.get('first_seen_time', 0)
            
            # Extract and process speed data
            speeds = vehicle.get('speeds', [])
            if speeds:
                # Filter out zero speeds for statistics - they likely indicate initialization or tracking issues
                valid_speeds = [s.get('speed_kph', 0) for s in speeds if s.get('speed_kph', 0) > 0]
                
                if valid_speeds:
                    avg_speed = np.mean(valid_speeds)
                    max_speed = np.max(valid_speeds)
                    min_speed = np.min(valid_speeds)
                    speed_std = np.std(valid_speeds) if len(valid_speeds) > 1 else 0
                    
                    # Add to vehicle data
                    vehicle_data.append({
                        'vehicle_id': vehicle_id,
                        'make': make,
                        'model': model,
                        'vehicle_class': vehicle_class,
                        'first_seen_frame': first_seen_frame,
                        'first_seen_time': first_seen_time,
                        'avg_speed': avg_speed,
                        'max_speed': max_speed,
                        'min_speed': min_speed,
                        'speed_std': speed_std,
                        'valid_speed_count': len(valid_speeds),
                        'total_speed_measurements': len(speeds),
                        'detection_count': len(speeds)
                    })
                    
                    # Add detailed speed data for advanced analysis
                    for speed_entry in speeds:
                        if speed_entry.get('speed_kph', 0) > 0:  # Only include valid speeds
                            speed_data.append({
                                'vehicle_id': vehicle_id,
                                'frame': speed_entry.get('frame', 0),
                                'time': speed_entry.get('time', 0),
                                'speed_kph': speed_entry.get('speed_kph', 0),
                                'make': make,
                                'model': model,
                                'vehicle_class': vehicle_class
                            })
                else:
                    # Add vehicle with no valid speed data
                    vehicle_data.append({
                        'vehicle_id': vehicle_id,
                        'make': make,
                        'model': model,
                        'vehicle_class': vehicle_class,
                        'first_seen_frame': first_seen_frame,
                        'first_seen_time': first_seen_time,
                        'avg_speed': 0,
                        'max_speed': 0,
                        'min_speed': 0,
                        'speed_std': 0,
                        'valid_speed_count': 0,
                        'total_speed_measurements': len(speeds),
                        'detection_count': len(speeds)
                    })
            else:
                # Add vehicle with no speed data
                vehicle_data.append({
                    'vehicle_id': vehicle_id,
                    'make': make,
                    'model': model,
                    'vehicle_class': vehicle_class,
                    'first_seen_frame': first_seen_frame,
                    'first_seen_time': first_seen_time,
                    'avg_speed': 0,
                    'max_speed': 0,
                    'min_speed': 0,
                    'speed_std': 0,
                    'valid_speed_count': 0,
                    'total_speed_measurements': 0,
                    'detection_count': 0
                })
        
        # Create vehicle DataFrame
        self.vehicle_df = pd.DataFrame(vehicle_data)
        
        # Create speed DataFrame if we have data
        if speed_data:
            self.speed_df = pd.DataFrame(speed_data)
        else:
            self.speed_df = pd.DataFrame(columns=['vehicle_id', 'frame', 'time', 'speed_kph', 'make', 'model', 'vehicle_class'])
            print("Warning: No valid speed data found in the dataset")
        
        # Create make/model analysis DataFrame
        if len(self.vehicle_df) > 0:
            make_model_counts = self.vehicle_df.groupby(['make', 'model']).size().reset_index(name='count')
            
            # Only include vehicles with valid speed data for speed statistics
            valid_speed_df = self.vehicle_df[self.vehicle_df['valid_speed_count'] > 0]
            
            if len(valid_speed_df) > 0:
                make_model_speeds = valid_speed_df.groupby(['make', 'model'])['avg_speed'].mean().reset_index()
                self.make_model_df = pd.merge(make_model_counts, make_model_speeds, on=['make', 'model'], how='left')
            else:
                # If no valid speed data, create dataframe without speed info
                self.make_model_df = make_model_counts
                self.make_model_df['avg_speed'] = np.nan
        
        # Print statistics about the processed data
        total_vehicles = len(self.vehicle_df)
        vehicles_with_speed = len(self.vehicle_df[self.vehicle_df['valid_speed_count'] > 0])
        
        print(f"Processed {len(self.frame_df) if self.frame_df is not None else 0} frames and {total_vehicles} vehicles")
        print(f"Vehicles with valid speed data: {vehicles_with_speed}/{total_vehicles} ({vehicles_with_speed/total_vehicles*100:.1f}% if tracked)")
        
        if self.debug and len(self.vehicle_df) > 0:
            print("\nVehicle Data Summary:")
            print(f"  Average detection count per vehicle: {self.vehicle_df['detection_count'].mean():.1f}")
            
            if vehicles_with_speed > 0:
                speed_df = self.vehicle_df[self.vehicle_df['valid_speed_count'] > 0]
                print(f"  Average speed: {speed_df['avg_speed'].mean():.1f} km/h")
                print(f"  Speed range: {speed_df['min_speed'].min():.1f} - {speed_df['max_speed'].max():.1f} km/h")
                print(f"  Average valid speed measurements per vehicle: {speed_df['valid_speed_count'].mean():.1f}")
        
        return True
    
    def _reconstruct_frame_data(self, fps):
        """Reconstruct frame-level data from vehicle detections"""
        frame_data = {}
        
        # Determine the total frame count
        max_frame = 0
        for vehicle in self.raw_data['vehicles']:
            speeds = vehicle.get('speeds', [])
            for speed in speeds:
                frame = speed.get('frame', 0)
                max_frame = max(max_frame, frame)
        
        # Initialize frame data dictionary
        for frame in range(1, max_frame + 1):
            frame_data[frame] = {
                'frame': frame,
                'time': frame / fps,
                'vehicle_count': 0,
                'vehicle_ids': [],
                'speeds': []
            }
        
        # Fill in vehicle data for each frame
        for vehicle in self.raw_data['vehicles']:
            vehicle_id = vehicle['id']
            speeds = vehicle.get('speeds', [])
            
            for speed in speeds:
                frame = speed.get('frame', 0)
                speed_kph = speed.get('speed_kph', 0)
                
                if frame in frame_data:
                    frame_data[frame]['vehicle_count'] += 1
                    frame_data[frame]['vehicle_ids'].append(vehicle_id)
                    if speed_kph > 0:
                        frame_data[frame]['speeds'].append(speed_kph)
        
        # Calculate average speed for each frame
        for frame, data in frame_data.items():
            data['avg_speed'] = np.mean(data['speeds']) if data['speeds'] else 0
        
        # Convert to DataFrame
        self.frame_df = pd.DataFrame(list(frame_data.values()))
    
    def analyze_correlation(self):
        """Analyze correlation between vehicle count and average speed"""
        if self.frame_df is None or len(self.frame_df) == 0:
            print("No frame data available for correlation analysis")
            return False
        
        print("Analyzing correlation between vehicle count and average speed...")
        
        # Filter to frames with vehicles and valid speeds
        valid_frames = self.frame_df[(self.frame_df['vehicle_count'] > 0) & (self.frame_df['avg_speed'] > 0)]
        
        if len(valid_frames) == 0:
            print("No valid frames with both vehicles and speed data")
            return False
        
        # Calculate Pearson correlation
        correlation, p_value = stats.pearsonr(valid_frames['vehicle_count'], valid_frames['avg_speed'])
        
        # Calculate Spearman rank correlation (better for non-linear relationships)
        spearman_corr, spearman_p = stats.spearmanr(valid_frames['vehicle_count'], valid_frames['avg_speed'])
        
        # Store results
        self.correlation_results = {
            'pearson_correlation': correlation,
            'pearson_p_value': p_value,
            'spearman_correlation': spearman_corr,
            'spearman_p_value': spearman_p,
            'n_frames': len(valid_frames),
            'significance': p_value < 0.05,
            'relationship': self._interpret_correlation(correlation)
        }
        
        print(f"Correlation analysis complete:")
        print(f"  Pearson correlation: {correlation:.3f} (p={p_value:.5f})")
        print(f"  Spearman correlation: {spearman_corr:.3f} (p={spearman_p:.5f})")
        print(f"  Relationship: {self._interpret_correlation(correlation)}")
        
        return True
    
    def _interpret_correlation(self, correlation):
        """Interpret correlation coefficient in words"""
        if correlation < -0.7:
            return "strong negative"
        elif correlation < -0.3:
            return "moderate negative"
        elif correlation < -0.1:
            return "weak negative"
        elif correlation < 0.1:
            return "negligible or no"
        elif correlation < 0.3:
            return "weak positive"
        elif correlation < 0.7:
            return "moderate positive"
        else:
            return "strong positive"
    
    def run_regression_analysis(self):
        """Run regression analysis to model the relationship between vehicle count and speed"""
        if self.frame_df is None or len(self.frame_df) == 0:
            print("No frame data available for regression analysis")
            return False
        
        print("Running regression analysis...")
        
        # Filter to frames with vehicles and valid speeds
        valid_frames = self.frame_df[(self.frame_df['vehicle_count'] > 0) & (self.frame_df['avg_speed'] > 0)].copy()
        
        if len(valid_frames) == 0:
            print("No valid frames with both vehicles and speed data")
            return False
        
        # Add squared term to test for non-linear relationship
        valid_frames['vehicle_count_squared'] = valid_frames['vehicle_count'] ** 2
        
        # Simple linear regression
        X = sm.add_constant(valid_frames['vehicle_count'])
        y = valid_frames['avg_speed']
        linear_model = sm.OLS(y, X).fit()
        
        # Polynomial regression with squared term
        X_quad = sm.add_constant(valid_frames[['vehicle_count', 'vehicle_count_squared']])
        quad_model = sm.OLS(y, X_quad).fit()
        
        # Compare models using AIC
        linear_aic = linear_model.aic
        quad_aic = quad_model.aic
        
        # Store better model
        better_model = "linear" if linear_aic <= quad_aic else "quadratic"
        
        # Store regression results
        self.regression_results = {
            'linear_model': {
                'slope': linear_model.params[1],
                'intercept': linear_model.params[0],
                'r_squared': linear_model.rsquared,
                'p_value': linear_model.f_pvalue,
                'aic': linear_aic,
                'summary': str(linear_model.summary())
            },
            'quadratic_model': {
                'linear_term': quad_model.params[1],
                'quadratic_term': quad_model.params[2],
                'intercept': quad_model.params[0],
                'r_squared': quad_model.rsquared,
                'p_value': quad_model.f_pvalue,
                'aic': quad_aic,
                'summary': str(quad_model.summary())
            },
            'better_model': better_model
        }
        
        # Create predicted values
        max_vehicles = valid_frames['vehicle_count'].max()
        vehicle_range = np.arange(1, max_vehicles + 1)
        
        if better_model == "linear":
            formula = f"y = {linear_model.params[0]:.2f} + {linear_model.params[1]:.2f}x"
            predicted = linear_model.params[0] + linear_model.params[1] * vehicle_range
        else:
            formula = (f"y = {quad_model.params[0]:.2f} + {quad_model.params[1]:.2f}x + "
                      f"{quad_model.params[2]:.4f}x²")
            predicted = (quad_model.params[0] + quad_model.params[1] * vehicle_range + 
                         quad_model.params[2] * vehicle_range**2)
        
        self.regression_results['formula'] = formula
        self.regression_results['vehicle_range'] = vehicle_range.tolist()
        self.regression_results['predicted_speeds'] = predicted.tolist()
        
        print(f"Regression analysis complete:")
        print(f"  Better model: {better_model}")
        print(f"  R-squared: {self.regression_results[better_model+'_model']['r_squared']:.3f}")
        print(f"  Formula: {formula}")
        
        return True
    
    def analyze_speed_patterns(self):
        """Analyze speed patterns and distribution"""
        if self.speed_df is None or len(self.speed_df) == 0:
            print("No speed data available for analysis")
            return False
        
        print("Analyzing speed patterns...")
        
        # Basic speed statistics
        speed_stats = {
            'count': len(self.speed_df),
            'mean': self.speed_df['speed_kph'].mean(),
            'median': self.speed_df['speed_kph'].median(),
            'std': self.speed_df['speed_kph'].std(),
            'min': self.speed_df['speed_kph'].min(),
            'max': self.speed_df['speed_kph'].max(),
            'percentiles': {
                '25': self.speed_df['speed_kph'].quantile(0.25),
                '50': self.speed_df['speed_kph'].quantile(0.5),
                '75': self.speed_df['speed_kph'].quantile(0.75),
                '90': self.speed_df['speed_kph'].quantile(0.9),
                '95': self.speed_df['speed_kph'].quantile(0.95)
            }
        }
        
        # Analyze speed by vehicle class
        class_speed = self.speed_df.groupby('vehicle_class')['speed_kph'].agg(['mean', 'std', 'min', 'max', 'count']).reset_index()
        
        # Analyze speed changes over time
        # Group by time segments for trend analysis
        time_segments = pd.cut(self.speed_df['time'], bins=10)
        time_speed = self.speed_df.groupby(time_segments)['speed_kph'].agg(['mean', 'std', 'min', 'max', 'count']).reset_index()
        
        # Store results
        self.speed_analysis_results = {
            'basic_stats': speed_stats,
            'class_speed': class_speed.to_dict('records'),
            'time_speed': time_speed.to_dict('records'),
        }
        
        # Print summary of findings
        print(f"Speed analysis complete:")
        print(f"  Average speed: {speed_stats['mean']:.2f} km/h")
        print(f"  Speed range: {speed_stats['min']:.2f} - {speed_stats['max']:.2f} km/h")
        print(f"  Standard deviation: {speed_stats['std']:.2f} km/h")
        
        if len(class_speed) > 1:
            print("\n  Speed by vehicle class:")
            for _, row in class_speed.iterrows():
                print(f"    {row['vehicle_class']}: {row['mean']:.2f} km/h (count: {row['count']})")
        
        return True
    
    def analyze_make_model_trends(self):
        """Analyze trends by vehicle make and model"""
        if self.make_model_df is None or len(self.make_model_df) == 0:
            print("No make/model data available for analysis")
            return False
        
        print("Analyzing vehicle make/model trends...")
        
        # Require at least 3 makes
        if self.make_model_df['make'].nunique() < 3:
            print("Not enough different makes for meaningful analysis")
            return False
        
        # Filter out Unknown makes and models
        filtered_df = self.make_model_df[
            (self.make_model_df['make'] != 'Unknown') & 
            (self.make_model_df['model'] != 'Unknown')
        ]
        
        if len(filtered_df) == 0:
            print("No valid make/model data after filtering")
            return False
        
        # Get top makes by count
        top_makes = filtered_df.groupby('make')['count'].sum().sort_values(ascending=False).head(5).index.tolist()
        
        # Filter to top makes and calculate statistics
        top_make_data = filtered_df[filtered_df['make'].isin(top_makes)]
        
        # Only include speed data if available
        if 'avg_speed' in filtered_df.columns and not filtered_df['avg_speed'].isna().all():
            make_stats = top_make_data.groupby('make').agg({
                'count': 'sum',
                'avg_speed': 'mean'
            }).reset_index()
            
            # Get top models by count
            top_models = filtered_df.sort_values('count', ascending=False).head(10)
            
            # Get fastest and slowest models (with at least 3 observations)
            valid_speed_models = filtered_df[~filtered_df['avg_speed'].isna() & (filtered_df['count'] >= 3)]
            
            if len(valid_speed_models) > 0:
                fastest_models = valid_speed_models.sort_values('avg_speed', ascending=False).head(5)
                slowest_models = valid_speed_models.sort_values('avg_speed').head(5)
            else:
                fastest_models = pd.DataFrame()
                slowest_models = pd.DataFrame()
        else:
            # Without speed data, just do count analysis
            make_stats = top_make_data.groupby('make').agg({
                'count': 'sum'
            }).reset_index()
            
            # Get top models by count
            top_models = filtered_df.sort_values('count', ascending=False).head(10)
            fastest_models = pd.DataFrame()
            slowest_models = pd.DataFrame()
        
        # Store results
        self.make_model_results = {
            'top_makes': make_stats.to_dict('records'),
            'top_models': top_models.to_dict('records'),
            'fastest_models': fastest_models.to_dict('records') if len(fastest_models) > 0 else [],
            'slowest_models': slowest_models.to_dict('records') if len(slowest_models) > 0 else []
        }
        
        return True
    
    def analyze_time_patterns(self):
        """Analyze patterns over time"""
        if self.frame_df is None or len(self.frame_df) == 0:
            print("No frame data available for time pattern analysis")
            return False
        
        print("Analyzing time patterns...")
        
        # Add time segments
        self.frame_df['segment_id'] = pd.cut(
            self.frame_df['time'], 
            bins=10, 
            labels=False
        )
        
        # Group by segment and calculate statistics
        segment_stats = self.frame_df.groupby('segment_id').agg({
            'vehicle_count': ['mean', 'max'],
            'avg_speed': ['mean', 'max', 'min']
        }).reset_index()
        
        # Flatten multi-level columns
        segment_stats.columns = ['_'.join(col).strip('_') for col in segment_stats.columns.values]
        
        # Get time range for each segment
        segment_times = self.frame_df.groupby('segment_id')['time'].agg(['min', 'max']).reset_index()
        segment_stats = pd.merge(segment_stats, segment_times, on='segment_id')
        
        # Add segment label
        segment_stats['segment_label'] = segment_stats.apply(
            lambda row: f"{row['min']:.1f}s-{row['max']:.1f}s", axis=1
        )
        
        # Find peak vehicle and speed segments
        peak_vehicle_segment = segment_stats.loc[segment_stats['vehicle_count_mean'].idxmax()]
        
        # Check if we have valid speed data
        if not segment_stats['avg_speed_mean'].isna().all():
            peak_speed_segment = segment_stats.loc[segment_stats['avg_speed_mean'].idxmax()]
        else:
            # Create a placeholder with the same structure but NaN values
            peak_speed_segment = segment_stats.iloc[0].copy()
            peak_speed_segment[:] = np.nan
        
        # Identify trends
        vehicle_trend = self._identify_trend(segment_stats['vehicle_count_mean'].tolist())
        
        # Only analyze speed trend if we have valid speed data
        if not segment_stats['avg_speed_mean'].isna().all():
            speed_values = [v for v in segment_stats['avg_speed_mean'].tolist() if not np.isnan(v)]
            speed_trend = self._identify_trend(speed_values)
        else:
            speed_trend = "no data"
        
        # Store results
        self.time_patterns = {
            'segment_stats': segment_stats.to_dict('records'),
            'peak_vehicle_segment': peak_vehicle_segment.to_dict(),
            'peak_speed_segment': peak_speed_segment.to_dict(),
            'vehicle_trend': vehicle_trend,
            'speed_trend': speed_trend
        }
        
        return True
    
    def _identify_trend(self, values):
        """Identify the trend in a series of values"""
        # Filter out NaN values
        values = [v for v in values if not np.isnan(v)]
        
        if len(values) < 2:
            return "insufficient data"
        
        # Linear regression to find slope
        x = np.arange(len(values))
        slope, _, r_value, p_value, _ = stats.linregress(x, values)
        
        # Interpret slope
        if p_value > 0.05:
            return "no significant trend"
        elif slope > 0:
            return "increasing" if r_value > 0.5 else "slightly increasing"
        else:
            return "decreasing" if r_value > 0.5 else "slightly decreasing"
    
    def generate_visualizations(self):
        """Generate visualizations of the analysis results"""
        print("Generating visualizations...")
        
        # Create visualization subdirectory
        vis_dir = os.path.join(self.output_dir, "visualizations")
        os.makedirs(vis_dir, exist_ok=True)
        
        self._create_correlation_plot()
        self._create_regression_plot()
        self._create_time_series_plots()
        self._create_make_model_plots()
        self._create_speed_plots()
        self._create_summary_visualization()
        
        return True
    
    def _create_correlation_plot(self):
        """Create correlation scatter plot"""
        if self.frame_df is None or len(self.frame_df) == 0:
            return
        
        # Filter frames with vehicles and valid speeds
        valid_frames = self.frame_df[(self.frame_df['vehicle_count'] > 0) & (self.frame_df['avg_speed'] > 0)]
        
        if len(valid_frames) > 0:
            plt.figure(figsize=(10, 8))
            
            # Create scatter plot with density heatmap
            sns.jointplot(
                x='vehicle_count', 
                y='avg_speed', 
                data=valid_frames, 
                kind='hex',
                height=8,
                marginal_kws={'color': 'blue'},
                cmap='viridis'
            )
            
            plt.title('Relationship Between Vehicle Count and Average Speed', fontsize=16, y=1.05)
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'visualizations', 'correlation_hexbin_plot.png'), dpi=300)
            plt.close()
            
            # Create scatter plot with regression line
            plt.figure(figsize=(10, 8))
            ax = sns.regplot(
                x='vehicle_count', 
                y='avg_speed', 
                data=valid_frames, 
                scatter_kws={'alpha': 0.4, 's': 50}, 
                line_kws={'color': 'red', 'lw': 2}
            )
            
            # Add correlation info
            if self.correlation_results:
                corr = self.correlation_results.get('pearson_correlation', 0)
                p_val = self.correlation_results.get('pearson_p_value', 1)
                relationship = self.correlation_results.get('relationship', 'unknown')
                
                plt.annotate(
                    f"Correlation: {corr:.3f} (p={p_val:.3f})\n"
                    f"Relationship: {relationship}",
                    xy=(0.05, 0.95),
                    xycoords='axes fraction',
                    bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8),
                    fontsize=12
                )
            
            plt.title('Correlation Between Vehicle Count and Average Speed', fontsize=16)
            plt.xlabel('Number of Vehicles', fontsize=14)
            plt.ylabel('Average Speed (km/h)', fontsize=14)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'visualizations', 'correlation_scatter_plot.png'), dpi=300)
            plt.close()
    
    def _create_regression_plot(self):
        """Create regression model visualization"""
        if not self.regression_results:
            return
        
        plt.figure(figsize=(12, 8))
        
        # Filter frames with vehicles and valid speeds
        valid_frames = self.frame_df[(self.frame_df['vehicle_count'] > 0) & (self.frame_df['avg_speed'] > 0)]
        
        if len(valid_frames) == 0:
            return
        
        # Scatter plot
        plt.scatter(
            valid_frames['vehicle_count'], 
            valid_frames['avg_speed'],
            alpha=0.4,
            s=50,
            c='blue',
            label='Observed Data'
        )
        
        # Get model data
        vehicle_range = np.array(self.regression_results['vehicle_range'])
        predicted = np.array(self.regression_results['predicted_speeds'])
        formula = self.regression_results['formula']
        better_model = self.regression_results['better_model']
        r_squared = self.regression_results[better_model + '_model']['r_squared']
        
        # Plot regression line
        plt.plot(
            vehicle_range,
            predicted,
            'r-',
            linewidth=3,
            label=f"Regression Model (R² = {r_squared:.3f})"
        )
        
        # Confidence intervals (approximation)
        if better_model == 'linear':
            # For linear model, we can use predicted +/- std error
            std_error = np.std(valid_frames['avg_speed'] - (
                self.regression_results['linear_model']['intercept'] + 
                self.regression_results['linear_model']['slope'] * valid_frames['vehicle_count']
            ))
            
            plt.fill_between(
                vehicle_range,
                predicted - 1.96 * std_error,
                predicted + 1.96 * std_error,
                alpha=0.2,
                color='red',
                label='95% Confidence Interval'
            )
        
        # Add formula annotation
        plt.annotate(
            formula,
            xy=(0.05, 0.05),
            xycoords='axes fraction',
            bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8),
            fontsize=14
        )
        
        plt.title(f'Regression Analysis: {better_model.capitalize()} Model', fontsize=16)
        plt.xlabel('Number of Vehicles', fontsize=14)
        plt.ylabel('Average Speed (km/h)', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'visualizations', 'regression_model_plot.png'), dpi=300)
        plt.close()
    
    def _create_time_series_plots(self):
        """Create time series visualizations"""
        if self.frame_df is None or len(self.frame_df) == 0:
            return
        
        # Time series of vehicle count
        plt.figure(figsize=(14, 7))
        plt.plot(
            self.frame_df['time'],
            self.frame_df['vehicle_count'],
            'b-',
            linewidth=1.5,
            alpha=0.7
        )
        
        # Add smoothed trend line
        window_size = max(5, len(self.frame_df) // 50)  # Adaptive window size
        if len(self.frame_df) > window_size:
            smoothed = self.frame_df['vehicle_count'].rolling(window=window_size, center=True).mean()
            plt.plot(
                self.frame_df['time'],
                smoothed,
                'r-',
                linewidth=2.5,
                label=f'Moving Average (window={window_size})'
            )
        
        plt.title('Vehicle Count Over Time', fontsize=16)
        plt.xlabel('Time (seconds)', fontsize=14)
        plt.ylabel('Number of Vehicles', fontsize=14)
        plt.grid(True, alpha=0.3)
        if len(self.frame_df) > window_size:
            plt.legend(fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'visualizations', 'vehicle_count_time_series.png'), dpi=300)
        plt.close()
        
        # Time series of average speed
        plt.figure(figsize=(14, 7))
        
        # Filter frames with valid speeds
        speed_frames = self.frame_df[self.frame_df['avg_speed'] > 0]
        
        if len(speed_frames) > 0:
            plt.plot(
                speed_frames['time'],
                speed_frames['avg_speed'],
                'g-',
                linewidth=1.5,
                alpha=0.7
            )
            
            # Add smoothed trend line
            window_size = max(5, len(speed_frames) // 50)  # Adaptive window size
            if len(speed_frames) > window_size:
                smoothed = speed_frames['avg_speed'].rolling(window=window_size, center=True).mean()
                plt.plot(
                    speed_frames['time'],
                    smoothed,
                    'r-',
                    linewidth=2.5,
                    label=f'Moving Average (window={window_size})'
                )
            
            plt.title('Average Vehicle Speed Over Time', fontsize=16)
            plt.xlabel('Time (seconds)', fontsize=14)
            plt.ylabel('Average Speed (km/h)', fontsize=14)
            plt.grid(True, alpha=0.3)
            if len(speed_frames) > window_size:
                plt.legend(fontsize=12)
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'visualizations', 'avg_speed_time_series.png'), dpi=300)
            plt.close()
        
        # Combined plot with dual axes
        fig, ax1 = plt.subplots(figsize=(14, 7))
        
        # Vehicle count on left axis
        color1 = 'tab:blue'
        ax1.set_xlabel('Time (seconds)', fontsize=14)
        ax1.set_ylabel('Number of Vehicles', color=color1, fontsize=14)
        ax1.plot(self.frame_df['time'], self.frame_df['vehicle_count'], color=color1, alpha=0.7)
        ax1.tick_params(axis='y', labelcolor=color1)
        
        # Average speed on right axis
        if len(speed_frames) > 0:
            ax2 = ax1.twinx()
            color2 = 'tab:green'
            ax2.set_ylabel('Average Speed (km/h)', color=color2, fontsize=14)
            ax2.plot(speed_frames['time'], speed_frames['avg_speed'], color=color2, alpha=0.7)
            ax2.tick_params(axis='y', labelcolor=color2)
        
        plt.title('Vehicle Count and Average Speed Over Time', fontsize=16)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'visualizations', 'combined_time_series.png'), dpi=300)
        plt.close()
    
    def _create_speed_plots(self):
        """Create speed-specific visualizations"""
        # Ensure we have the speed dataframe
        if self.speed_df is None or len(self.speed_df) == 0:
            return
        
        # 1. Speed histogram
        plt.figure(figsize=(12, 7))
        sns.histplot(self.speed_df['speed_kph'], kde=True, bins=20)
        plt.title('Distribution of Vehicle Speeds', fontsize=16)
        plt.xlabel('Speed (km/h)', fontsize=14)
        plt.ylabel('Frequency', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'visualizations', 'speed_histogram.png'), dpi=300)
        plt.close()
        
        # 2. Speed by vehicle class
        if 'vehicle_class' in self.speed_df.columns and self.speed_df['vehicle_class'].nunique() > 1:
            plt.figure(figsize=(12, 8))
            sns.boxplot(x='vehicle_class', y='speed_kph', data=self.speed_df)
            plt.title('Speed Distribution by Vehicle Class', fontsize=16)
            plt.xlabel('Vehicle Class', fontsize=14)
            plt.ylabel('Speed (km/h)', fontsize=14)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'visualizations', 'speed_by_class.png'), dpi=300)
            plt.close()
        
        # 3. Speed over time with individual vehicles
        if len(self.speed_df) > 0:
            plt.figure(figsize=(14, 8))
            
            # Create a colormap based on vehicle ID
            vehicle_ids = self.speed_df['vehicle_id'].unique()
            colors = plt.cm.tab20(np.linspace(0, 1, min(20, len(vehicle_ids))))
            color_map = {id: colors[i % len(colors)] for i, id in enumerate(vehicle_ids)}
            
            # Plot speed points for each vehicle
            for vehicle_id in vehicle_ids[:20]:  # Limit to 20 vehicles to avoid overcrowding
                vehicle_data = self.speed_df[self.speed_df['vehicle_id'] == vehicle_id]
                plt.plot(vehicle_data['time'], vehicle_data['speed_kph'], 'o-', 
                         alpha=0.7, linewidth=1, markersize=4, 
                         color=color_map[vehicle_id], label=f'Vehicle {vehicle_id}')
            
            # Add overall trend line if we have enough data points
            if len(self.speed_df) > 10:
                # Sort by time for proper line plotting
                trend_data = self.speed_df.sort_values('time')
                plt.plot(trend_data['time'], trend_data['speed_kph'].rolling(window=10, min_periods=1).mean(), 
                         'r-', linewidth=3, label='Overall Trend')
            
            plt.title('Vehicle Speeds Over Time', fontsize=16)
            plt.xlabel('Time (seconds)', fontsize=14)
            plt.ylabel('Speed (km/h)', fontsize=14)
            plt.grid(True, alpha=0.3)
            
            # Only show legend if we have a reasonable number of vehicles
            if len(vehicle_ids) <= 10:
                plt.legend(fontsize=10, loc='best')
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'visualizations', 'speed_time_by_vehicle.png'), dpi=300)
            plt.close()
        
        # 4. Speed heatmap (time vs speed)
        if len(self.speed_df) >= 10:
            plt.figure(figsize=(14, 8))
            
            # Create time and speed bins
            time_bins = np.linspace(self.speed_df['time'].min(), self.speed_df['time'].max(), 20)
            speed_bins = np.linspace(self.speed_df['speed_kph'].min(), self.speed_df['speed_kph'].max(), 20)
            
            # Create 2D histogram
            hist, xedges, yedges = np.histogram2d(
                self.speed_df['time'], 
                self.speed_df['speed_kph'], 
                bins=[time_bins, speed_bins]
            )
            
            # Plot heatmap
            plt.imshow(
                hist.T, 
                origin='lower', 
                aspect='auto',
                extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                cmap='viridis'
            )
            
            plt.colorbar(label='Number of Vehicles')
            plt.title('Speed Distribution Over Time', fontsize=16)
            plt.xlabel('Time (seconds)', fontsize=14)
            plt.ylabel('Speed (km/h)', fontsize=14)
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'visualizations', 'speed_time_heatmap.png'), dpi=300)
            plt.close()
    
    def _create_make_model_plots(self):
        """Create visualizations of make/model analysis"""
        if self.make_model_df is None or len(self.make_model_df) == 0:
            return
        
        # Filter out Unknown makes and models
        filtered_df = self.make_model_df[
            (self.make_model_df['make'] != 'Unknown') & 
            (self.make_model_df['model'] != 'Unknown')
        ]
        
        if len(filtered_df) == 0:
            return
        
        # Top makes bar chart
        plt.figure(figsize=(12, 8))
        make_counts = filtered_df.groupby('make')['count'].sum().sort_values(ascending=False).head(10)
        
        sns.barplot(x=make_counts.values, y=make_counts.index, palette='viridis')
        plt.title('Top 10 Vehicle Makes', fontsize=16)
        plt.xlabel('Count', fontsize=14)
        plt.ylabel('Make', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'visualizations', 'top_makes_bar_chart.png'), dpi=300)
        plt.close()
        
        # Top models bar chart
        plt.figure(figsize=(12, 8))
        top_models = filtered_df.sort_values('count', ascending=False).head(10)
        
        sns.barplot(x='count', y='model', hue='make', data=top_models, palette='viridis')
        plt.title('Top 10 Vehicle Models', fontsize=16)
        plt.xlabel('Count', fontsize=14)
        plt.ylabel('Model', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(title='Make')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'visualizations', 'top_models_bar_chart.png'), dpi=300)
        plt.close()
        
        # Average speed by make
        if 'avg_speed' in filtered_df.columns and not filtered_df['avg_speed'].isna().all():
            plt.figure(figsize=(12, 8))
            
            # Get makes with at least 3 vehicles
            valid_makes = filtered_df.groupby('make').filter(lambda x: x['count'].sum() >= 3)
            
            if len(valid_makes) > 0:
                make_speeds = valid_makes.groupby('make')['avg_speed'].mean().sort_values(ascending=False)
                
                sns.barplot(x=make_speeds.values, y=make_speeds.index, palette='coolwarm')
                plt.title('Average Speed by Vehicle Make', fontsize=16)
                plt.xlabel('Average Speed (km/h)', fontsize=14)
                plt.ylabel('Make', fontsize=14)
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(os.path.join(self.output_dir, 'visualizations', 'make_speed_bar_chart.png'), dpi=300)
                plt.close()
    
    def _create_summary_visualization(self):
        """Create summary visualization with key findings"""
        plt.figure(figsize=(14, 10))
        
        # Title and styling
        plt.suptitle('Traffic Analysis Summary', fontsize=20, fontweight='bold', y=0.98)
        plt.subplots_adjust(top=0.85)
        
        # Section for correlation findings
        if self.correlation_results:
            corr = self.correlation_results.get('pearson_correlation', 0)
            p_val = self.correlation_results.get('pearson_p_value', 1)
            relationship = self.correlation_results.get('relationship', 'unknown')
            significance = "statistically significant" if p_val < 0.05 else "not statistically significant"
            
            plt.figtext(
                0.5, 0.8,
                "Correlation Between Vehicle Count and Speed",
                fontsize=16, fontweight='bold', ha='center'
            )
            
            plt.figtext(
                0.5, 0.75,
                f"There is a {relationship} correlation ({corr:.3f}) between\n"
                f"the number of vehicles and their average speed, which is {significance}.",
                fontsize=14, ha='center'
            )
        
        # Section for regression findings
        if self.regression_results:
            better_model = self.regression_results.get('better_model', 'unknown')
            formula = self.regression_results.get('formula', 'unknown')
            r_squared = self.regression_results.get(better_model + '_model', {}).get('r_squared', 0)
            
            plt.figtext(
                0.5, 0.65,
                "Regression Analysis",
                fontsize=16, fontweight='bold', ha='center'
            )
            
            plt.figtext(
                0.5, 0.6,
                f"Best model: {better_model.capitalize()} (R² = {r_squared:.3f})\n"
                f"Formula: {formula}",
                fontsize=14, ha='center'
            )
        
        # Section for vehicle statistics
        if hasattr(self, 'vehicle_df') and self.vehicle_df is not None and len(self.vehicle_df) > 0:
            total_vehicles = len(self.vehicle_df)
            
            # Check if we have valid speed data
            vehicles_with_speed = len(self.vehicle_df[self.vehicle_df['valid_speed_count'] > 0])
            
            if vehicles_with_speed > 0:
                avg_speed = self.vehicle_df[self.vehicle_df['valid_speed_count'] > 0]['avg_speed'].mean()
                max_speed = self.vehicle_df[self.vehicle_df['valid_speed_count'] > 0]['max_speed'].max()
                speed_text = f"Average speed: {avg_speed:.1f} km/h\nMaximum speed: {max_speed:.1f} km/h"
            else:
                speed_text = "No valid speed data available"
            
            plt.figtext(
                0.5, 0.5,
                "Vehicle Statistics",
                fontsize=16, fontweight='bold', ha='center'
            )
            
            plt.figtext(
                0.5, 0.45,
                f"Total vehicles detected: {total_vehicles}\n"
                f"Vehicles with speed data: {vehicles_with_speed}/{total_vehicles} ({vehicles_with_speed/total_vehicles*100:.1f}%)\n"
                f"{speed_text}",
                fontsize=14, ha='center'
            )
        
        # Section for time patterns
        if hasattr(self, 'time_patterns') and self.time_patterns:
            vehicle_trend = self.time_patterns.get('vehicle_trend', 'unknown')
            speed_trend = self.time_patterns.get('speed_trend', 'unknown')
            
            plt.figtext(
                0.5, 0.35,
                "Time Patterns",
                fontsize=16, fontweight='bold', ha='center'
            )
            
            plt.figtext(
                0.5, 0.3,
                f"Vehicle count trend: {vehicle_trend}\n"
                f"Speed trend: {speed_trend}",
                fontsize=14, ha='center'
            )
        
        # Section for make/model statistics
        if hasattr(self, 'make_model_df') and self.make_model_df is not None:
            # Filter out Unknown makes and models
            filtered_df = self.make_model_df[
                (self.make_model_df['make'] != 'Unknown') & 
                (self.make_model_df['model'] != 'Unknown')
            ]
            
            if len(filtered_df) > 0:
                # Get top make
                top_make = filtered_df.groupby('make')['count'].sum().idxmax()
                top_make_count = filtered_df.groupby('make')['count'].sum().max()
                
                # Get top model
                top_model_row = filtered_df.sort_values('count', ascending=False).iloc[0]
                top_model = f"{top_model_row['make']} {top_model_row['model']}"
                top_model_count = top_model_row['count']
                
                plt.figtext(
                    0.5, 0.2,
                    "Vehicle Make/Model Statistics",
                    fontsize=16, fontweight='bold', ha='center'
                )
                
                plt.figtext(
                    0.5, 0.15,
                    f"Most common make: {top_make} ({top_make_count} vehicles)\n"
                    f"Most common model: {top_model} ({top_model_count} vehicles)",
                    fontsize=14, ha='center'
                )
        
        # Footer with date
        plt.figtext(
            0.5, 0.05,
            f"Analysis generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            fontsize=10, ha='center'
        )
        
        # Border
        plt.box(on=True)
        
        plt.savefig(os.path.join(self.output_dir, 'visualizations', 'analysis_summary.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_report(self):
        """Generate comprehensive analysis report"""
        print("Generating analysis report...")
        report_path = os.path.join(self.output_dir, "traffic_analysis_report.md")
        
        with open(report_path, 'w') as f:
            f.write("# Traffic Analysis Report\n\n")
            f.write(f"**Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Video information
            f.write("## Video Information\n\n")
            if 'video_info' in self.raw_data:
                video_info = self.raw_data['video_info']
                f.write(f"- **Filename:** {video_info.get('filename', 'Unknown')}\n")
                f.write(f"- **Resolution:** {video_info.get('resolution', 'Unknown')}\n")
                f.write(f"- **FPS:** {video_info.get('fps', 'Unknown')}\n")
                f.write(f"- **Processing Date:** {video_info.get('processed_date', 'Unknown')}\n\n")
            
            # Summary statistics
            f.write("## Summary Statistics\n\n")
            
            if hasattr(self, 'frame_df') and self.frame_df is not None:
                total_frames = len(self.frame_df)
                frames_with_vehicles = len(self.frame_df[self.frame_df['vehicle_count'] > 0])
                max_vehicles = self.frame_df['vehicle_count'].max()
                avg_vehicles = self.frame_df['vehicle_count'].mean()
                
                f.write(f"- **Total Frames Analyzed:** {total_frames}\n")
                f.write(f"- **Frames with Vehicles:** {frames_with_vehicles} ({frames_with_vehicles/total_frames*100:.1f}%)\n")
                f.write(f"- **Maximum Vehicles in a Frame:** {max_vehicles}\n")
                f.write(f"- **Average Vehicles per Frame:** {avg_vehicles:.2f}\n\n")
            
            if hasattr(self, 'vehicle_df') and self.vehicle_df is not None:
                total_vehicles = len(self.vehicle_df)
                
                # Check if we have valid speed data
                vehicles_with_speed = len(self.vehicle_df[self.vehicle_df['valid_speed_count'] > 0])
                
                f.write(f"- **Total Unique Vehicles:** {total_vehicles}\n")
                f.write(f"- **Vehicles with Speed Data:** {vehicles_with_speed} ({vehicles_with_speed/total_vehicles*100:.1f}%)\n")
                
                if vehicles_with_speed > 0:
                    speed_df = self.vehicle_df[self.vehicle_df['valid_speed_count'] > 0]
                    avg_speed = speed_df['avg_speed'].mean()
                    max_speed = speed_df['max_speed'].max()
                    min_speed = speed_df['min_speed'].min()
                    
                    f.write(f"- **Average Vehicle Speed:** {avg_speed:.2f} km/h\n")
                    f.write(f"- **Maximum Vehicle Speed:** {max_speed:.2f} km/h\n")
                    f.write(f"- **Minimum Vehicle Speed:** {min_speed:.2f} km/h\n\n")
                else:
                    f.write("- **No valid speed data available**\n\n")
            
            # Speed analysis section
            if hasattr(self, 'speed_analysis_results') and self.speed_analysis_results:
                f.write("## Speed Analysis\n\n")
                
                # Basic statistics
                if 'basic_stats' in self.speed_analysis_results:
                    stats = self.speed_analysis_results['basic_stats']
                    f.write("### Basic Speed Statistics\n\n")
                    f.write(f"- **Number of Speed Measurements:** {stats.get('count', 0)}\n")
                    f.write(f"- **Mean Speed:** {stats.get('mean', 0):.2f} km/h\n")
                    f.write(f"- **Median Speed:** {stats.get('median', 0):.2f} km/h\n")
                    f.write(f"- **Standard Deviation:** {stats.get('std', 0):.2f} km/h\n")
                    f.write(f"- **Range:** {stats.get('min', 0):.2f} - {stats.get('max', 0):.2f} km/h\n\n")
                    
                    if 'percentiles' in stats:
                        f.write("#### Speed Percentiles\n\n")
                        f.write("| Percentile | Speed (km/h) |\n")
                        f.write("|------------|-------------|\n")
                        for percentile, value in stats['percentiles'].items():
                            f.write(f"| {percentile} | {value:.2f} |\n")
                        f.write("\n")
                
                # Speed by vehicle class
                if 'class_speed' in self.speed_analysis_results and self.speed_analysis_results['class_speed']:
                    f.write("### Speed by Vehicle Class\n\n")
                    f.write("| Vehicle Class | Mean Speed (km/h) | Count | Min Speed (km/h) | Max Speed (km/h) |\n")
                    f.write("|--------------|-------------------|-------|------------------|------------------|\n")
                    
                    for row in self.speed_analysis_results['class_speed']:
                        f.write(f"| {row.get('vehicle_class', 'Unknown')} | {row.get('mean', 0):.2f} | {row.get('count', 0)} | {row.get('min', 0):.2f} | {row.get('max', 0):.2f} |\n")
                    
                    f.write("\n")
                
                # Include relevant visualizations
                f.write("### Speed Visualizations\n\n")
                f.write("![Speed Distribution](visualizations/speed_histogram.png)\n\n")
                if hasattr(self, 'speed_df') and 'vehicle_class' in self.speed_df.columns and self.speed_df['vehicle_class'].nunique() > 1:
                    f.write("![Speed by Vehicle Class](visualizations/speed_by_class.png)\n\n")
                f.write("![Vehicle Speeds Over Time](visualizations/speed_time_by_vehicle.png)\n\n")
            
            # Correlation analysis
            f.write("## Correlation Analysis\n\n")
            
            if self.correlation_results:
                corr = self.correlation_results.get('pearson_correlation', 0)
                p_val = self.correlation_results.get('pearson_p_value', 1)
                relationship = self.correlation_results.get('relationship', 'unknown')
                significance = "statistically significant" if p_val < 0.05 else "not statistically significant"
                
                f.write(f"- **Pearson Correlation Coefficient:** {corr:.3f}\n")
                f.write(f"- **P-value:** {p_val:.5f}\n")
                f.write(f"- **Statistical Significance:** {significance}\n")
                f.write(f"- **Relationship:** {relationship}\n\n")
                
                f.write("### Interpretation\n\n")
                f.write(f"There is a {relationship} correlation ({corr:.3f}) between the number of vehicles ")
                f.write(f"and their average speed, which is {significance}. ")
                
                if corr < -0.3:
                    f.write("This suggests that as the number of vehicles increases, the average speed tends to decrease, ")
                    f.write("which aligns with typical traffic congestion patterns.\n\n")
                elif corr > 0.3:
                    f.write("This suggests that as the number of vehicles increases, the average speed also tends to increase, ")
                    f.write("which is contrary to typical traffic patterns and may indicate unique road conditions or behaviors.\n\n")
                else:
                    f.write("This suggests a weak relationship between vehicle count and speed in this specific video, ")
                    f.write("which might be due to consistent traffic flow, limited congestion, or other factors.\n\n")
                
                f.write("![Correlation Plot](visualizations/correlation_scatter_plot.png)\n\n")
            else:
                f.write("Insufficient data for correlation analysis.\n\n")
            
            # Regression analysis
            f.write("## Regression Analysis\n\n")
            
            if self.regression_results:
                better_model = self.regression_results.get('better_model', 'unknown')
                formula = self.regression_results.get('formula', 'unknown')
                r_squared = self.regression_results.get(better_model + '_model', {}).get('r_squared', 0)
                
                f.write(f"- **Best Model Type:** {better_model.capitalize()}\n")
                f.write(f"- **Formula:** {formula}\n")
                f.write(f"- **R-squared:** {r_squared:.3f}\n\n")
                
                f.write("### Interpretation\n\n")
                f.write(f"The {better_model} model provides the best fit for the relationship between vehicle count and speed. ")
                
                if r_squared < 0.3:
                    f.write(f"The low R-squared value ({r_squared:.3f}) indicates that vehicle count alone explains only a small ")
                    f.write("portion of the variation in vehicle speeds. Other factors likely have significant influence.\n\n")
                elif r_squared < 0.7:
                    f.write(f"The moderate R-squared value ({r_squared:.3f}) suggests that vehicle count explains a reasonable ")
                    f.write("amount of the variation in speeds, but other factors also play important roles.\n\n")
                else:
                    f.write(f"The high R-squared value ({r_squared:.3f}) indicates that vehicle count is a strong predictor ")
                    f.write("of average speed in this video.\n\n")
                
                f.write("![Regression Model](visualizations/regression_model_plot.png)\n\n")
            else:
                f.write("Insufficient data for regression analysis.\n\n")
            
            # Time patterns
            f.write("## Time Pattern Analysis\n\n")
            
            if hasattr(self, 'time_patterns') and self.time_patterns:
                vehicle_trend = self.time_patterns.get('vehicle_trend', 'unknown')
                speed_trend = self.time_patterns.get('speed_trend', 'unknown')
                peak_vehicle = self.time_patterns.get('peak_vehicle_segment', {})
                peak_speed = self.time_patterns.get('peak_speed_segment', {})
                
                f.write(f"- **Vehicle Count Trend:** {vehicle_trend}\n")
                f.write(f"- **Speed Trend:** {speed_trend}\n")
                
                if 'segment_label' in peak_vehicle:
                    f.write(f"- **Peak Vehicle Period:** {peak_vehicle.get('segment_label')} ")
                    f.write(f"(avg: {peak_vehicle.get('vehicle_count_mean', 0):.2f}, ")
                    f.write(f"max: {peak_vehicle.get('vehicle_count_max', 0)})\n")
                
                if 'segment_label' in peak_speed and not pd.isna(peak_speed.get('avg_speed_mean', np.nan)):
                    f.write(f"- **Peak Speed Period:** {peak_speed.get('segment_label')} ")
                    f.write(f"(avg: {peak_speed.get('avg_speed_mean', 0):.2f} km/h, ")
                    f.write(f"max: {peak_speed.get('avg_speed_max', 0):.2f} km/h)\n\n")
                
                f.write("### Interpretation\n\n")
                f.write("The time series analysis reveals how traffic patterns evolved during the video duration. ")
                
                if vehicle_trend in ["increasing", "slightly increasing"] and speed_trend in ["decreasing", "slightly decreasing"]:
                    f.write("The combination of increasing vehicle counts and decreasing speeds suggests growing congestion over time.\n\n")
                elif vehicle_trend in ["decreasing", "slightly decreasing"] and speed_trend in ["increasing", "slightly increasing"]:
                    f.write("The combination of decreasing vehicle counts and increasing speeds suggests improving traffic flow over time.\n\n")
                else:
                    f.write(f"Vehicle counts showed a {vehicle_trend} trend while speeds showed a {speed_trend} trend.\n\n")
                
                f.write("![Time Series](visualizations/combined_time_series.png)\n\n")
            else:
                f.write("Insufficient data for time pattern analysis.\n\n")
            
            # Vehicle make/model analysis
            f.write("## Vehicle Make and Model Analysis\n\n")
            
            if hasattr(self, 'make_model_df') and self.make_model_df is not None:
                # Filter out Unknown makes and models
                filtered_df = self.make_model_df[
                    (self.make_model_df['make'] != 'Unknown') & 
                    (self.make_model_df['model'] != 'Unknown')
                ]
                
                if len(filtered_df) > 0:
                    # Count unique makes and models
                    unique_makes = filtered_df['make'].nunique()
                    unique_models = filtered_df.groupby(['make', 'model']).size().reset_index().shape[0]
                    
                    f.write(f"- **Unique Vehicle Makes:** {unique_makes}\n")
                    f.write(f"- **Unique Vehicle Models:** {unique_models}\n\n")
                    
                    # Top makes
                    top_makes = filtered_df.groupby('make')['count'].sum().sort_values(ascending=False).head(5)
                    
                    f.write("### Top 5 Vehicle Makes\n\n")
                    f.write("| Make | Count | Percentage |\n")
                    f.write("|------|-------|------------|\n")
                    
                    total_vehicles = filtered_df['count'].sum()
                    for make, count in top_makes.items():
                        percentage = count / total_vehicles * 100
                        f.write(f"| {make} | {count} | {percentage:.1f}% |\n")
                    
                    f.write("\n")
                    
                    # Top models
                    top_models = filtered_df.sort_values('count', ascending=False).head(5)
                    
                    f.write("### Top 5 Vehicle Models\n\n")
                    f.write("| Make | Model | Count | Percentage |\n")
                    f.write("|------|-------|-------|------------|\n")
                    
                    for _, row in top_models.iterrows():
                        percentage = row['count'] / total_vehicles * 100
                        f.write(f"| {row['make']} | {row['model']} | {row['count']} | {percentage:.1f}% |\n")
                    
                    f.write("\n")
                    
                    # Add speed analysis by make/model if available
                    if 'avg_speed' in filtered_df.columns and not filtered_df['avg_speed'].isna().all():
                        f.write("### Average Speed by Make\n\n")
                        f.write("| Make | Average Speed (km/h) | Number of Vehicles |\n")
                        f.write("|------|---------------------|-------------------|\n")
                        
                        make_speeds = filtered_df.groupby('make').agg({
                            'avg_speed': 'mean',
                            'count': 'sum'
                        }).sort_values('avg_speed', ascending=False).head(10)
                        
                        for make, row in make_speeds.iterrows():
                            if not pd.isna(row['avg_speed']):
                                f.write(f"| {make} | {row['avg_speed']:.2f} | {row['count']} |\n")
                        
                        f.write("\n")
                    
                    f.write("![Top Makes](visualizations/top_makes_bar_chart.png)\n\n")
                else:
                    f.write("No valid make/model data available for analysis.\n\n")
            else:
                f.write("No vehicle make/model data available for analysis.\n\n")
            
            # Conclusions
            f.write("## Conclusions\n\n")
            
            # Add conclusions about speed data
            if hasattr(self, 'vehicle_df') and self.vehicle_df is not None:
                vehicles_with_speed = len(self.vehicle_df[self.vehicle_df['valid_speed_count'] > 0])
                total_vehicles = len(self.vehicle_df)
                
                if vehicles_with_speed > 0:
                    speed_df = self.vehicle_df[self.vehicle_df['valid_speed_count'] > 0]
                    avg_speed = speed_df['avg_speed'].mean()
                    
                    f.write(f"1. Speed data was available for {vehicles_with_speed} out of {total_vehicles} vehicles ({vehicles_with_speed/total_vehicles*100:.1f}%).\n")
                    f.write(f"2. The average vehicle speed was {avg_speed:.2f} km/h.\n")
                else:
                    f.write("1. No valid speed data was collected for vehicles in this video.\n")
                    f.write("2. This may be due to calibration issues, short tracking durations, or other technical factors.\n")
            
            # Add conclusions about correlation
            if self.correlation_results and 'pearson_correlation' in self.correlation_results:
                corr = self.correlation_results['pearson_correlation']
                relationship = self.correlation_results['relationship']
                
                f.write(f"3. This analysis reveals a {relationship} correlation ({corr:.3f}) between vehicle count and speed.\n")
                
                if corr < -0.3:
                    f.write("4. As traffic density increases, vehicle speeds tend to decrease, indicating potential congestion patterns.\n")
                elif corr > 0.3:
                    f.write("4. As traffic density increases, vehicle speeds also tend to increase, which is unusual and may warrant further investigation.\n")
                else:
                    f.write("4. Vehicle count has a limited effect on speed in this particular video.\n")
            
            if hasattr(self, 'time_patterns') and self.time_patterns:
                vehicle_trend = self.time_patterns.get('vehicle_trend', 'unknown')
                speed_trend = self.time_patterns.get('speed_trend', 'unknown')
                
                if speed_trend != "no data":
                    f.write(f"5. Traffic patterns over time show a {vehicle_trend} trend in vehicle count and a {speed_trend} trend in speed.\n")
                else:
                    f.write(f"5. Traffic patterns over time show a {vehicle_trend} trend in vehicle count, but insufficient speed data to determine speed trends.\n")
            
            if hasattr(self, 'make_model_df') and self.make_model_df is not None and len(self.make_model_df) > 0:
                # Filter out Unknown makes and models
                filtered_df = self.make_model_df[
                    (self.make_model_df['make'] != 'Unknown') & 
                    (self.make_model_df['model'] != 'Unknown')
                ]
                
                if len(filtered_df) > 0:
                    top_make = filtered_df.groupby('make')['count'].sum().idxmax()
                    
                    f.write(f"6. The most common vehicle make detected was {top_make}, suggesting a prevalence of this brand in the local traffic.\n")
            
            f.write("\n7. This analysis provides valuable insights into traffic patterns that could inform traffic management strategies and urban planning decisions.\n")
        
        print(f"Report generated and saved to {report_path}")
        return report_path
    
    def save_data(self):
        """Save processed data to CSV files"""
        # Save frame data
        if self.frame_df is not None and len(self.frame_df) > 0:
            frame_csv = os.path.join(self.output_dir, "frame_data.csv")
            self.frame_df.to_csv(frame_csv, index=False)
            print(f"Frame data saved to {frame_csv}")
        
        # Save vehicle data
        if self.vehicle_df is not None and len(self.vehicle_df) > 0:
            vehicle_csv = os.path.join(self.output_dir, "vehicle_data.csv")
            self.vehicle_df.to_csv(vehicle_csv, index=False)
            print(f"Vehicle data saved to {vehicle_csv}")
        
        # Save make/model data
        if self.make_model_df is not None and len(self.make_model_df) > 0:
            make_model_csv = os.path.join(self.output_dir, "make_model_data.csv")
            self.make_model_df.to_csv(make_model_csv, index=False)
            print(f"Make/model data saved to {make_model_csv}")
        
        # Save speed data
        if self.speed_df is not None and len(self.speed_df) > 0:
            speed_csv = os.path.join(self.output_dir, "speed_data.csv")
            self.speed_df.to_csv(speed_csv, index=False)
            print(f"Speed data saved to {speed_csv}")
        
        # Save analysis results as JSON
        results = {
            'correlation': self.correlation_results,
            'regression': self.regression_results,
            'time_patterns': self.time_patterns if hasattr(self, 'time_patterns') else None,
            'make_model_results': self.make_model_results if hasattr(self, 'make_model_results') else None,
            'speed_analysis': self.speed_analysis_results if hasattr(self, 'speed_analysis_results') else None
        }
        
        results_json = os.path.join(self.output_dir, "analysis_results.json")
        with open(results_json, 'w') as f:
            json.dump(results, f, indent=4)
        
        print(f"Analysis results saved to {results_json}")
        
        return True
    
    def run_full_analysis(self):
        """Run the complete analysis pipeline"""
        print("Starting traffic data analysis...")
        
        # Step 1: Process the data
        if not self.process_data():
            print("Data processing failed. Analysis aborted.")
            return False
        
        # Step 2: Analyze correlation
        self.analyze_correlation()
        
        # Step 3: Run regression analysis
        self.run_regression_analysis()
        
        # Step 4: Analyze speed patterns (new function)
        self.analyze_speed_patterns()
        
        # Step 5: Analyze make/model trends
        self.analyze_make_model_trends()
        
        # Step 6: Analyze time patterns
        self.analyze_time_patterns()
        
        # Step 7: Generate visualizations
        self.generate_visualizations()
        
        # Step 8: Generate report
        self.generate_report()
        
        # Step 9: Save processed data
        self.save_data()
        
        print("Analysis complete!")
        return True


def main():
    """Main function to run the script"""
    parser = argparse.ArgumentParser(description='Advanced Traffic Data Analysis')
    parser.add_argument('--data', required=True, type=str, help='Path to JSON data file from vehicle recognition system')
    parser.add_argument('--output', default='analysis_output', type=str, help='Directory to save analysis outputs')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')
    
    args = parser.parse_args()
    
    print(f"Analyzing data from: {args.data}")
    print(f"Output directory: {args.output}")
    print(f"Debug mode: {'Enabled' if args.debug else 'Disabled'}")
    
    analyzer = TrafficDataAnalyzer(args.data, args.output, args.debug)
    analyzer.run_full_analysis()


if __name__ == "__main__":
    main()