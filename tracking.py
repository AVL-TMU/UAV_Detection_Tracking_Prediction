import numpy as np
from types import SimpleNamespace
from ByteTrack.yolox.tracker.byte_tracker import BYTETracker

class ObjectTracker:
    def __init__(self, track_thresh=0.5, match_thresh=0.8, track_buffer=30, frame_rate=30, mot20=False):
        """Initialize the ByteTrack tracker with parameters"""
        # Tracker hyper-params
        self.tracker_args = {
            'track_thresh': track_thresh,
            'match_thresh': match_thresh,
            'track_buffer': track_buffer,
        }
        self.args = SimpleNamespace(**self.tracker_args)
        self.args.mot20 = mot20
        self.tracker = BYTETracker(self.args, frame_rate=frame_rate)
        
    def update(self, detections, frame_shape):
        """Update tracker with new detections
        
        Args:
            detections: numpy array of shape (N, 5) with [x1, y1, x2, y2, score]
            frame_shape: tuple of (height, width) of the frame
            
        Returns:
            List of online targets (tracks)
        """
        h0, w0 = frame_shape
        online_targets = []
        
        if len(detections) > 0:
            online_targets = self.tracker.update(detections, [h0, w0], [h0, w0])
            
        return online_targets
        
    def get_track_info(self, target):
        """Extract track information from a target
        
        Args:
            target: A track object from the ByteTracker
            
        Returns:
            Dictionary with track information
        """
        x1, y1, x2, y2 = map(int, target.tlbr)
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        tid = target.track_id
        
        return {
            'id': tid,
            'bbox': (x1, y1, x2, y2),
            'center': (cx, cy),
            'score': target.score
        } 