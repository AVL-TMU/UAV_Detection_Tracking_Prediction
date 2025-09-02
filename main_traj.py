import os
import time
import cv2
import numpy as np
import torch
import pandas as pd
import datetime
import argparse
import matplotlib.pyplot as plt
import shutil
import pycuda.autoinit  # Initialize CUDA

# Import custom modules
from detection import Detector, DETECTION_THRESHOLD, CUSTOM_RESOLUTION
from tracking import ObjectTracker
from trajectory import TrajectoryPredictor, DEBUG_DRAW_HEADINGS, SUCCESS_THRESHOLD

# --- Default Configuration ---
DEFAULT_VIDEO_PATH = "/home/dasjetson/das_ws/UAV-Detection/data/videos/DJI_4.mp4"
FRAME_LIMIT = 5000       # Increased to give more frames for evaluation
FUTURE_FRAMES = 10       # How many frames ahead to predict
WARMUP_FRAMES = 20       # Frames to wait before starting metrics collection

# --- TensorRT Engine Path ---
ENGINE_PATH = "/home/dasjetson/das_ws/Dev_UAV/onnx-models_orig/inference_model.sim.engine"

def create_evaluation_folder(video_path):
    """Create a timestamped folder for the evaluation results"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    folder_name = f"evaluation_{video_name}_{timestamp}"
    
    # Create main folder
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        
    # Create subfolders
    os.makedirs(os.path.join(folder_name, "csv"), exist_ok=True)
    os.makedirs(os.path.join(folder_name, "plots"), exist_ok=True)
    
    return folder_name

def run_prediction_pipeline(video_path, output_folder, args):
    """Run the full detection, tracking, and prediction pipeline"""
    print(f"Starting video processing: {video_path}")
    print(f"Saving results to: {output_folder}")
    
    # Initialize components
    detector = Detector(ENGINE_PATH)
    tracker = ObjectTracker(track_thresh=args.threshold, frame_rate=30)
    predictor = TrajectoryPredictor(
        future_frames=args.future_frames,
        warmup_frames=WARMUP_FRAMES
    )
    
    # Output video path in the evaluation folder
    video_out_path = os.path.join(output_folder, "tracked_video.mp4")
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
        
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video info: {orig_w}x{orig_h}, FPS: {fps}, Total frames: {total_frames}")
    
    # Make sure we don't try to process more frames than exist in the video
    frame_limit = min(args.frame_limit, total_frames)
    print(f"Processing {frame_limit} frames")
    
    out = cv2.VideoWriter(video_out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (orig_w, orig_h))

    # Debug counters
    predictions_made = 0
    predictions_evaluated = 0
    tracks_seen = set()
    
    # Performance monitoring
    times, fps_list = [], []
    
    # Frame-by-frame metrics
    frame_metrics = []
    
    # Process video frames
    frame_id = 0
    
    while frame_id < frame_limit:
        ret, frame = cap.read()
        if not ret: 
            print(f"Failed to read frame at {frame_id}/{frame_limit}")
            break
            
        frame_id += 1
        if frame_id % 20 == 0:
            print(f"Processing frame {frame_id}/{frame_limit}")

        # 1. Detect objects
        det_arr, inf_t = detector.detect(frame, threshold=args.threshold)
        times.append(inf_t)
        fps_list.append(1.0/inf_t if inf_t>0 else 0)

        # Frame metrics data
        frame_data = {
            "frame_id": frame_id,
            "inference_time": inf_t,
            "fps": fps_list[-1],
            "detections": len(det_arr),
            "tracks": 0,
            "evaluated_predictions": 0,
            "avg_displacement_error": 0,
            "success_rate": 0
        }

        # 2. Track objects
        h0, w0 = frame.shape[:2]
        online_targets = tracker.update(det_arr, (h0, w0))
        
        frame_data["tracks"] = len(online_targets)
        
        if frame_id % 20 == 0:
            print(f"Frame {frame_id}: Found {len(online_targets)} tracked objects")
        
        # Frame evaluation metrics
        frame_displacement_errors = []
        frame_successful_predictions = 0
        frame_total_predictions = 0

        # Process each tracked object
        for target in online_targets:
            # Extract track info
            track_info = tracker.get_track_info(target)
            tid = track_info['id']
            x1, y1, x2, y2 = track_info['bbox']
            current_position = track_info['center']
            
            # Add to seen tracks
            tracks_seen.add(tid)

            # 3. Trajectory prediction
            pred_result = predictor.update(tid, current_position, frame_id)
            future_position = pred_result['predicted']
            predictions_made += 1

            # Draw current box & ID
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(frame, f"ID {tid}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

            # Draw predicted future (current frame's future diamond)
            cv2.drawMarker(
                frame, 
                (int(future_position[0]), int(future_position[1])),
                (0,0,255),
                markerType=cv2.MARKER_DIAMOND, 
                markerSize=12, 
                thickness=2
            )
            
            # 4. Evaluate predictions
            eval_result = predictor.evaluate(tid, current_position, frame_id)
            
            if eval_result:
                predictions_evaluated += 1
                frame_total_predictions += 1
                
                # Update frame metrics
                displacement_error = eval_result['displacement_error']
                frame_displacement_errors.append(displacement_error)
                
                if eval_result['success']:
                    frame_successful_predictions += 1
                
                # Draw evaluation visualizations
                frame = predictor.draw_predictions(frame, eval_result, debug_draw_headings=DEBUG_DRAW_HEADINGS)

        # Update frame metrics
        if frame_displacement_errors:
            frame_data["evaluated_predictions"] = len(frame_displacement_errors)
            frame_data["avg_displacement_error"] = np.mean(frame_displacement_errors)
            frame_data["success_rate"] = (
                frame_successful_predictions / frame_total_predictions if frame_total_predictions > 0 else 0
            )
        
        frame_metrics.append(frame_data)

        # Overlay FPS
        cv2.putText(frame, f"FPS:{fps_list[-1]:.2f}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
        out.write(frame)

    cap.release()
    out.release()
    
    # Debug info
    print(f"\nProcessing complete:")
    print(f"- Total frames processed: {frame_id}")
    print(f"- Unique tracks detected: {len(tracks_seen)}")
    print(f"- Total predictions made: {predictions_made}")
    print(f"- Predictions evaluated: {predictions_evaluated}")
    print(f"- Tracks with metrics: {len(predictor.track_metrics)}")
    
    # Calculate final metrics
    overall_metrics = predictor.compute_overall_metrics()
    
    # Save metrics to CSV
    print("Saving metrics to CSV files...")
    
    # 1. Overall metrics
    pd.DataFrame([overall_metrics]).to_csv(os.path.join(output_folder, "csv", "overall_metrics.csv"), index=False)
    
    # 2. Per-frame metrics
    pd.DataFrame(frame_metrics).to_csv(os.path.join(output_folder, "csv", "frame_metrics.csv"), index=False)
    
    # 3. Per-track metrics
    track_data = predictor.get_track_data()
    pd.DataFrame(track_data).to_csv(os.path.join(output_folder, "csv", "track_metrics.csv"), index=False)
    
    # 4. Raw errors
    pd.DataFrame({'displacement_errors': overall_metrics['displacement_errors']}).to_csv(
        os.path.join(output_folder, "csv", "raw_errors.csv"), index=False)
    
    # Generate summary text file
    generate_summary_file(output_folder, video_path, frame_id, overall_metrics, 
                         tracks_seen, track_data, args)
    
    # Return data for plotting
    has_metrics = len(overall_metrics['displacement_errors']) > 0
    return times, fps_list, overall_metrics, has_metrics

def generate_summary_file(output_folder, video_path, frame_count, metrics, tracks, track_data, args):
    """Generate a text summary of results"""
    summary_path = os.path.join(output_folder, "summary.txt")
    with open(summary_path, 'w') as f:
        f.write(f"Kalman Filter Prediction Evaluation Summary\n")
        f.write(f"=====================================\n\n")
        f.write(f"Video: {video_path}\n")
        f.write(f"Processed frames: {frame_count}\n")
        f.write(f"Unique tracks: {len(tracks)}\n")
        f.write(f"Detection threshold: {args.threshold}\n")
        f.write(f"Future prediction window: {args.future_frames} frames\n\n")
        
        f.write(f"Overall Metrics\n")
        f.write(f"-------------\n")
        f.write(f"ADE (Average Displacement Error): {metrics['ade']:.2f} pixels\n")
        f.write(f"FDE (Final Displacement Error): {metrics['fde']:.2f} pixels\n")
        f.write(f"Success Rate: {metrics['success_rate']*100:.2f}%\n")
        f.write(f"Average Heading Error: {metrics['heading_error']:.4f} radians\n")
        f.write(f"Average Speed Error: {metrics['speed_error']:.2f} pixels/frame\n")
        f.write(f"Standard Deviation of Errors: {metrics['std_dev']:.2f} pixels\n\n")
        
        f.write(f"Top 5 Tracks by Prediction Count\n")
        f.write(f"--------------------------\n")
        top_tracks = sorted(track_data, key=lambda x: x['predictions'], reverse=True)[:5]
        for i, track in enumerate(top_tracks, 1):
            f.write(f"{i}. Track ID {track['track_id']}: {track['predictions']} predictions, " 
                    f"Success rate: {track['success_rate']*100:.2f}%, "
                    f"Avg Error: {track['avg_displacement_error']:.2f} px\n")

def generate_plots(output_folder, times, fps_list, prediction_metrics, has_metrics, frame_metrics_df, track_metrics_df):
    """Generate visualizations and plots"""
    print("Generating plots...")
    
    # 1. Create inference time plot
    plt.figure(figsize=(10, 6))
    idxs = np.arange(len(times))
    plt.plot(idxs, times, label='TRT Time (s)')
    plt.legend()
    plt.title('Inference Time')
    plt.xlabel('Frame')
    plt.ylabel('Time (s)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "plots", "inference_time.png"))
    plt.close()
    
    # 2. Create FPS plot
    plt.figure(figsize=(10, 6))
    plt.plot(idxs, fps_list, label='FPS')
    plt.legend()
    plt.title('Frames Per Second')
    plt.xlabel('Frame')
    plt.ylabel('FPS')
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "plots", "fps.png"))
    plt.close()
    
    if has_metrics:
        # 3. Error distribution histogram
        plt.figure(figsize=(10, 6))
        if len(prediction_metrics['displacement_errors']) > 0:
            plt.hist(prediction_metrics['displacement_errors'], bins=20, alpha=0.7)
            plt.axvline(prediction_metrics['ade'], linestyle='--', label=f"ADE: {prediction_metrics['ade']:.2f}px")
            plt.axvline(prediction_metrics['fde'], linestyle='--', label=f"FDE: {prediction_metrics['fde']:.2f}px")
        plt.title('Displacement Error Distribution')
        plt.xlabel('Error (pixels)')
        plt.ylabel('Count')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, "plots", "error_distribution.png"))
        plt.close()
        
        # 4. Summary metrics bar chart
        plt.figure(figsize=(12, 6))
        metrics_names = ['ADE (px)', 'FDE (px)', 'Success Rate', 'Heading Error (rad)', 'Speed Error (px/frame)', 'Std Dev (px)']
        metrics_values = [
            prediction_metrics['ade'],
            prediction_metrics['fde'],
            prediction_metrics['success_rate'],
            prediction_metrics['heading_error'],
            prediction_metrics['speed_error'],
            prediction_metrics['std_dev']
        ]
        
        plt.bar(metrics_names, metrics_values)
        plt.title('Prediction Performance Metrics')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, "plots", "summary_metrics.png"))
        plt.close()
        
        # 5. Frame-by-frame metrics
        plt.figure(figsize=(12, 8))
        
        # Only plot frames that have evaluated predictions
        eval_frames = frame_metrics_df[frame_metrics_df['evaluated_predictions'] > 0]
        
        if not eval_frames.empty:
            plt.subplot(2, 1, 1)
            plt.plot(eval_frames['frame_id'], eval_frames['avg_displacement_error'], 'b-', label='Displacement Error')
            plt.title('Per-Frame Displacement Error')
            plt.xlabel('Frame')
            plt.ylabel('Pixels')
            plt.legend()
            
            plt.subplot(2, 1, 2)
            plt.plot(eval_frames['frame_id'], eval_frames['success_rate'], 'g-', label='Success Rate')
            plt.title('Per-Frame Success Rate')
            plt.xlabel('Frame')
            plt.ylabel('Success Rate')
            plt.legend()
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_folder, "plots", "frame_metrics.png"))
            plt.close()
        
        # 6. Track metrics
        if not track_metrics_df.empty and len(track_metrics_df) > 1:
            plt.figure(figsize=(12, 8))
            
            plt.subplot(2, 1, 1)
            track_ids = track_metrics_df['track_id'].astype(str).tolist()
            plt.bar(track_ids, track_metrics_df['avg_displacement_error'], alpha=0.7)
            plt.title('Average Displacement Error by Track')
            plt.xlabel('Track ID')
            plt.ylabel('Pixels')
            plt.xticks(rotation=90)
            
            plt.subplot(2, 1, 2)
            plt.bar(track_ids, track_metrics_df['success_rate'], alpha=0.7)
            plt.title('Success Rate by Track')
            plt.xlabel('Track ID')
            plt.ylabel('Success Rate')
            plt.xticks(rotation=90)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_folder, "plots", "track_metrics.png"))
            plt.close()
        
        # 7. Combined Overview Plot (main display)
        plt.figure(figsize=(18, 12))
        
        plt.subplot(2, 2, 1)
        idxs = np.arange(len(times))
        plt.plot(idxs, times, label='TRT Time (s)')
        plt.legend()
        plt.title('Inference Time')
        
        plt.subplot(2, 2, 2)
        if len(prediction_metrics['displacement_errors']) > 0:
            plt.hist(prediction_metrics['displacement_errors'], bins=20, alpha=0.7)
            plt.axvline(prediction_metrics['ade'], linestyle='--', label=f"ADE: {prediction_metrics['ade']:.2f}px")
            plt.axvline(prediction_metrics['fde'], linestyle='--', label=f"FDE: {prediction_metrics['fde']:.2f}px")
        plt.title('Displacement Error Distribution')
        plt.xlabel('Error (pixels)')
        plt.ylabel('Count')
        plt.legend()
        
        plt.subplot(2, 2, 3)
        metrics_names = ['ADE (px)', 'FDE (px)', 'Success Rate', 'Heading Error (rad)', 'Speed Error (px/frame)', 'Std Dev (px)']
        metrics_values = [
            prediction_metrics['ade'],
            prediction_metrics['fde'],
            prediction_metrics['success_rate'],
            prediction_metrics['heading_error'],
            prediction_metrics['speed_error'],
            prediction_metrics['std_dev']
        ]
        
        plt.bar(metrics_names, metrics_values)
        plt.title('Prediction Performance Metrics')
        plt.xticks(rotation=45)
        
        # Track count per frame
        plt.subplot(2, 2, 4)
        plt.plot(frame_metrics_df['frame_id'], frame_metrics_df['tracks'], 'r-')
        plt.title('Tracked Objects per Frame')
        plt.xlabel('Frame')
        plt.ylabel('Count')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, "overview.png"))
        plt.close()
        
    else:
        plt.figure(figsize=(10, 6))
        plt.text(0.5, 0.5, "No metrics collected\nTry increasing FRAME_LIMIT\nor check detection quality", 
                 ha='center', va='center', fontsize=14)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, "plots", "no_metrics.png"))
        plt.close()
    
    print(f"Plots saved to {os.path.join(output_folder, 'plots')}")

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Kalman filter prediction evaluation')
    parser.add_argument('--video', type=str, default=DEFAULT_VIDEO_PATH,
                      help='Path to the video file for processing')
    parser.add_argument('--frame-limit', type=int, default=FRAME_LIMIT,
                      help='Maximum number of frames to process')
    parser.add_argument('--future-frames', type=int, default=FUTURE_FRAMES,
                      help='Number of frames ahead to predict')
    parser.add_argument('--threshold', type=float, default=DETECTION_THRESHOLD,
                      help='Detection confidence threshold')
    return parser.parse_args()

if __name__ == '__main__':
    # Parse command-line arguments
    args = parse_args()
    
    print(f"Using video: {args.video}")
    print(f"Frame limit: {args.frame_limit}")
    print(f"Future frames: {args.future_frames}")
    print(f"Detection threshold: {args.threshold}")
    
    # Create output folder
    output_folder = create_evaluation_folder(args.video)
    
    # Run evaluation
    times_trt, fps_trt, prediction_metrics, has_metrics = run_prediction_pipeline(
        args.video, output_folder, args
    )
    
    # Load saved CSVs for plotting
    frame_metrics_df = pd.read_csv(os.path.join(output_folder, "csv", "frame_metrics.csv"))
    track_metrics_df = pd.read_csv(os.path.join(output_folder, "csv", "track_metrics.csv"))
    
    # Generate plots
    generate_plots(output_folder, times_trt, fps_trt, prediction_metrics, has_metrics, 
                  frame_metrics_df, track_metrics_df)
    
    # Print summary
    print(f"\nEvaluation complete! All results saved to: {output_folder}")
    print(f"- Video: {os.path.join(output_folder, 'tracked_video.mp4')}")
    print(f"- CSVs: {os.path.join(output_folder, 'csv')}")
    print(f"- Plots: {os.path.join(output_folder, 'plots')}")
    print(f"- Overview: {os.path.join(output_folder, 'overview.png')}")
    print(f"- Summary: {os.path.join(output_folder, 'summary.txt')}")
    
    if has_metrics:
        print("\nPrediction Performance Metrics Summary:")
        print(f"ADE (Average Displacement Error): {prediction_metrics['ade']:.2f} pixels")
        print(f"FDE (Final Displacement Error): {prediction_metrics['fde']:.2f} pixels")
        print(f"Success Rate: {prediction_metrics['success_rate']*100:.2f}%")
        print(f"Average Heading Error: {prediction_metrics['heading_error']:.4f} radians")
        print(f"Average Speed Error: {prediction_metrics['speed_error']:.2f} pixels/frame")
        print(f"Standard Deviation of Errors: {prediction_metrics['std_dev']:.2f} pixels") 