import cv2
import numpy as np

class VideoProcessor:
    """
    Handles video loading, frame extraction, and processing optimization.
    
    This class manages the OpenCV video capture object and provides an efficient
    method to iterate through frames with optional downsampling and skipping.

    Attributes:
        video_path (str): Path to the video file.
        resize_factor (float): Scale factor for resizing frames (e.g., 0.5 for half size).
        process_every_n_frames (int): Interval for processing frames (e.g., 10 to process every 10th frame).
        cap (cv2.VideoCapture): The OpenCV video capture object.
        fps (float): Frames per second of the video.
        total_frames (int): Total number of frames in the video.
        duration (float): Duration of the video in seconds.
    """
    def __init__(self, video_path, resize_factor=0.5, process_every_n_frames=10):
        self.video_path = video_path
        self.resize_factor = resize_factor
        self.process_every_n_frames = process_every_n_frames
        self.cap = cv2.VideoCapture(video_path)
        
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
            
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration = self.total_frames / self.fps if self.fps > 0 else 0

    def process_video(self, identifier_callback, progress_callback=None):
        """
        Iterates through the video, applying the identifier callback to selected frames.

        This method implements the main processing loop, handling frame skipping,
        resizing, and coordinate scaling.

        Args:
            identifier_callback (callable): A function that takes a frame (np.ndarray) and returns 
                                          a list of bounding boxes for detected faces.
            progress_callback (callable, optional): A function that accepts a float (0.0 to 1.0) 
                                                  representing the processing progress.

        Returns:
            list: A list of dictionaries, where each dictionary represents a frame with matches:
                  {
                      'timestamp': float,
                      'frame_index': int,
                      'matches': list of (top, right, bottom, left) tuples
                  }
        """
        results = []
        frame_count = 0
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
                
            frame_count += 1
            
            # Skip frames for performance
            if frame_count % self.process_every_n_frames != 0:
                continue

            # Resize for faster processing
            small_frame = cv2.resize(frame, (0, 0), fx=self.resize_factor, fy=self.resize_factor)
            
            # Convert BGR (OpenCV) to RGB (face_recognition)
            rgb_small_frame = small_frame[:, :, ::-1]
            
            # Run identification
            matches = identifier_callback(rgb_small_frame)
            
            if matches:
                # Scale match locations back up to original size
                scaled_matches = []
                scale = 1 / self.resize_factor
                for (top, right, bottom, left) in matches:
                    scaled_matches.append((
                        int(top * scale),
                        int(right * scale),
                        int(bottom * scale),
                        int(left * scale)
                    ))
                
                timestamp = frame_count / self.fps
                results.append({
                    'timestamp': timestamp,
                    'frame_index': frame_count,
                    'matches': scaled_matches
                })
            
            if progress_callback:
                progress_callback(frame_count / self.total_frames)

        self.cap.release()
        return results

    def get_frame_at_timestamp(self, timestamp):
        """
        Helper to retrieve a specific frame for display in the UI.
        """
        self.cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
        ret, frame = self.cap.read()
        if ret:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return None
