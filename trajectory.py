import cv2
import numpy as np
import math
from collections import deque, defaultdict

# Smoothing window for raw measurements
SMOOTH_WINDOW = 10

# Success threshold in pixels (prediction is successful if distance < SUCCESS_THRESHOLD)
SUCCESS_THRESHOLD = 50

# Debug drawing toggle for heading vectors
DEBUG_DRAW_HEADINGS = False

# Calculate Euclidean distance between two points
def calculate_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

# Calculate angle between two vectors (heading)
def calculate_heading(p1, p2):
    return math.atan2(p2[1] - p1[1], p2[0] - p1[0])

# Wrap angle to [-pi, pi]
def wrap_pi(a):
    return (a + math.pi) % (2 * math.pi) - math.pi

# Calculate speed (distance per frame)
def calculate_speed(p1, p2):
    return calculate_distance(p1, p2)

class TrajectoryPredictor:
    def __init__(self, future_frames=10, smooth_window=SMOOTH_WINDOW, warmup_frames=20):
        """Initialize trajectory predictor
        
        Args:
            future_frames: Number of frames ahead to predict
            smooth_window: Window size for position smoothing
            warmup_frames: Number of frames to wait before evaluation
        """
        self.future_frames = future_frames
        self.smooth_window = smooth_window
        self.warmup_frames = warmup_frames
        
        # Kalman filters and buffers per track
        self.kalman_dict = {}
        self.pos_buffer = {}
        
        # Store predictions for each track
        self.predictions = {}
        
        # Position history for each track
        self.position_history = defaultdict(list)
        
        # Metrics for evaluation
        self.metrics = {
            'displacement_errors': [],  # All displacement errors
            'heading_errors': [],       # All heading errors
            'speed_errors': [],         # All speed errors
        }
        
        # Per-track metrics
        self.track_metrics = defaultdict(lambda: {
            'predictions': 0,
            'successful_predictions': 0,
            'displacement_errors': [],
            'heading_errors': [],
            'speed_errors': []
        })
        
    def init_kalman_filter(self, track_id, position):
        """Initialize Kalman filter for a new track"""
        cx, cy = position
        
        # KF state: [x, y, vx, vy]
        kf = cv2.KalmanFilter(4, 2)
        kf.transitionMatrix = np.array([[1,0,1,0], [0,1,0,1], [0,0,1,0], [0,0,0,1]], np.float32)
        kf.measurementMatrix = np.array([[1,0,0,0], [0,1,0,0]], np.float32)
        kf.processNoiseCov = np.eye(4, dtype=np.float32)*1e-2
        kf.measurementNoiseCov = np.eye(2, dtype=np.float32)*1e-1
        kf.errorCovPost = np.eye(4, dtype=np.float32)
        kf.statePost = np.array([[cx],[cy],[0],[0]], np.float32)
        
        self.kalman_dict[track_id] = kf
        self.pos_buffer[track_id] = deque(maxlen=self.smooth_window)
        
    def update(self, track_id, position, frame_id):
        """Update trajectory for a track and make prediction
        
        Args:
            track_id: Unique ID of the track
            position: (x, y) center position of the track
            frame_id: Current frame ID
            
        Returns:
            Dictionary with predicted position and other info
        """
        cx, cy = position
        
        # Update position history
        self.position_history[track_id].append((frame_id, cx, cy))
        if len(self.position_history[track_id]) > self.future_frames + 1:
            self.position_history[track_id].pop(0)
        
        # Initialize new track if needed
        if track_id not in self.kalman_dict:
            self.init_kalman_filter(track_id, position)
        
        kf = self.kalman_dict[track_id]
        buf = self.pos_buffer[track_id]
        buf.append((cx, cy))
        
        # Average measurement over buffer
        avg_x = sum(p[0] for p in buf) / len(buf)
        avg_y = sum(p[1] for p in buf) / len(buf)
        
        # Predict / correct
        kf.predict()
        measurement = np.array([[np.float32(avg_x)], [np.float32(avg_y)]])
        kf.correct(measurement)
        
        # Future prediction
        st = kf.statePost.flatten()
        future_x = st[0] + st[2] * self.future_frames
        future_y = st[1] + st[3] * self.future_frames
        
        # Store prediction with frame ID for evaluation later
        if track_id not in self.predictions:
            self.predictions[track_id] = {}
        self.predictions[track_id][frame_id] = (future_x, future_y)
        
        return {
            'current': (cx, cy),
            'predicted': (future_x, future_y),
            'velocity': (st[2], st[3])
        }
        
    def evaluate(self, track_id, current_position, frame_id):
        """Evaluate prediction made future_frames ago
        
        Args:
            track_id: Track ID to evaluate
            current_position: Current (x, y) position to compare with prediction
            frame_id: Current frame ID
            
        Returns:
            Dictionary with evaluation metrics or None if no prediction to evaluate
        """
        # Evaluate predictions made FUTURE_FRAMES ago
        target_frame = frame_id - self.future_frames
        
        if target_frame <= self.warmup_frames:
            return None
            
        if track_id not in self.predictions or target_frame not in self.predictions[track_id]:
            return None
            
        pred_pos = self.predictions[track_id][target_frame]
        actual_pos = current_position
        
        # Calculate displacement error
        displacement_error = calculate_distance(pred_pos, actual_pos)
        self.metrics['displacement_errors'].append(displacement_error)
        self.track_metrics[track_id]['displacement_errors'].append(displacement_error)
        
        # Calculate if prediction was successful
        success = displacement_error < SUCCESS_THRESHOLD
        self.track_metrics[track_id]['predictions'] += 1
        if success:
            self.track_metrics[track_id]['successful_predictions'] += 1
        
        # Calculate heading and speed errors if possible
        heading_error = None
        speed_error = None
        
        # Calculate heading error (if we have enough history)
        if len(self.position_history[track_id]) > 1:
            prev_pos = self.position_history[track_id][-2][1:]  # Get previous position (x,y)
            actual_heading = calculate_heading(prev_pos, actual_pos)
            pred_heading = calculate_heading(prev_pos, pred_pos)
            heading_error_signed = wrap_pi(pred_heading - actual_heading)
            # store magnitude for metrics
            heading_error = abs(heading_error_signed)
            self.track_metrics[track_id]['heading_errors'].append(heading_error)
            self.metrics['heading_errors'].append(heading_error)
        
        # Calculate speed error (if we have enough history)
        if len(self.position_history[track_id]) > 1:
            prev_pos = self.position_history[track_id][-2][1:]  # Get previous position (x,y)
            actual_speed = calculate_speed(prev_pos, actual_pos)
            pred_speed = calculate_speed(prev_pos, pred_pos)
            speed_error = abs(actual_speed - pred_speed)
            self.track_metrics[track_id]['speed_errors'].append(speed_error)
            self.metrics['speed_errors'].append(speed_error)
        
        return {
            'pred_pos': pred_pos,
            'actual_pos': actual_pos,
            'displacement_error': displacement_error,
            'success': success,
            'heading_error': heading_error,
            'speed_error': speed_error,
            'prev_pos': prev_pos if len(self.position_history[track_id]) > 1 else None
        }
        
    def draw_predictions(self, frame, evaluation_result, debug_draw_headings=False):
        """Draw prediction results on frame
        
        Args:
            frame: CV2 image to draw on
            evaluation_result: Result from evaluate()
            debug_draw_headings: Whether to draw heading vectors
            
        Returns:
            Updated frame with visualizations
        """
        if evaluation_result is None:
            return frame
            
        pred_pos = evaluation_result['pred_pos']
        actual_pos = evaluation_result['actual_pos']
        displacement_error = evaluation_result['displacement_error']
        
        # Visualize evaluation (actual -> predicted)
        cv2.arrowedLine(
            frame,
            (int(actual_pos[0]), int(actual_pos[1])),
            (int(pred_pos[0]), int(pred_pos[1])),
            (255, 0, 255), 2, tipLength=0.3
        )
        
        cv2.putText(frame, f"Err: {displacement_error:.1f}px",
                    (int(actual_pos[0])+10, int(actual_pos[1])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
        
        # Optional debug: draw headings from prev_pos
        if debug_draw_headings and evaluation_result['prev_pos'] is not None:
            prev_pos = evaluation_result['prev_pos']
            L = 30  # Length of heading vectors
            
            # Calculate actual and predicted heading angles
            actual_heading = calculate_heading(prev_pos, actual_pos)
            pred_heading = calculate_heading(prev_pos, pred_pos)
            
            # Draw actual heading (green)
            ax_end = (int(prev_pos[0] + L * math.cos(actual_heading)),
                      int(prev_pos[1] + L * math.sin(actual_heading)))
            cv2.arrowedLine(frame, (int(prev_pos[0]), int(prev_pos[1])), ax_end, (0,255,0), 2, tipLength=0.3)
            
            # Draw predicted heading (red)
            px_end = (int(prev_pos[0] + L * math.cos(pred_heading)),
                      int(prev_pos[1] + L * math.sin(pred_heading)))
            cv2.arrowedLine(frame, (int(prev_pos[0]), int(prev_pos[1])), px_end, (0,0,255), 2, tipLength=0.3)
            
        return frame
        
    def compute_overall_metrics(self):
        """Compute aggregate metrics from all evaluations
        
        Returns:
            Dictionary with overall metrics
        """
        overall_metrics = {}
        
        if self.metrics['displacement_errors']:
            # Average Displacement Error
            overall_metrics['ade'] = np.mean(self.metrics['displacement_errors'])
            
            # For FDE, get the last error for each track if available
            fde_values = []
            for tid in self.track_metrics:
                if self.track_metrics[tid]['displacement_errors']:
                    fde_values.append(self.track_metrics[tid]['displacement_errors'][-1])
            
            # Final Displacement Error
            overall_metrics['fde'] = np.mean(fde_values) if fde_values else 0
            
            # Standard deviation of errors
            overall_metrics['std_dev'] = np.std(self.metrics['displacement_errors'])
            
            # Overall success rate
            total_predictions = sum(self.track_metrics[tid]['predictions'] for tid in self.track_metrics)
            total_successful = sum(self.track_metrics[tid]['successful_predictions'] for tid in self.track_metrics)
            overall_metrics['success_rate'] = total_successful / total_predictions if total_predictions > 0 else 0
            
            # Average heading error
            if self.metrics['heading_errors']:
                overall_metrics['heading_error'] = np.mean(self.metrics['heading_errors'])
            else:
                overall_metrics['heading_error'] = 0
            
            # Average speed error
            if self.metrics['speed_errors']:
                overall_metrics['speed_error'] = np.mean(self.metrics['speed_errors'])
            else:
                overall_metrics['speed_error'] = 0
            
            # Include raw errors for plotting
            overall_metrics['displacement_errors'] = self.metrics['displacement_errors']
        else:
            # Initialize with default values if no metrics collected
            overall_metrics['ade'] = 0
            overall_metrics['fde'] = 0
            overall_metrics['success_rate'] = 0
            overall_metrics['heading_error'] = 0
            overall_metrics['speed_error'] = 0
            overall_metrics['std_dev'] = 0
            overall_metrics['displacement_errors'] = []
            
        return overall_metrics
        
    def get_track_data(self):
        """Get per-track evaluation data for reporting
        
        Returns:
            List of dictionaries with track metrics
        """
        track_data = []
        for tid in self.track_metrics:
            tm = self.track_metrics[tid]
            avg_displacement = np.mean(tm['displacement_errors']) if tm['displacement_errors'] else 0
            success_rate = tm['successful_predictions'] / tm['predictions'] if tm['predictions'] > 0 else 0
            avg_heading_error = np.mean(tm['heading_errors']) if tm['heading_errors'] else 0
            avg_speed_error = np.mean(tm['speed_errors']) if tm['speed_errors'] else 0
            
            track_data.append({
                'track_id': tid,
                'predictions': tm['predictions'],
                'successful_predictions': tm['successful_predictions'],
                'success_rate': success_rate,
                'avg_displacement_error': avg_displacement,
                'avg_heading_error': avg_heading_error,
                'avg_speed_error': avg_speed_error,
                'track_lifespan': len(self.position_history[tid])
            })
            
        return track_data 