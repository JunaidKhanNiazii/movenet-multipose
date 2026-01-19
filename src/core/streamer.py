"""
streamer.py - Threaded Video Streamer
=====================================
Reads frames from video source in a separate thread to prevent I/O blocking.
"""

import cv2
import threading
import time
from queue import Queue, Empty

class VideoStreamer:
    """
    Reads frames from a video source or camera using a separate thread.
    Uses a thread-safe queue to store the latest frames.
    """
    
    def __init__(self, source, queue_size=128):
        self.source = source
        self.cap = cv2.VideoCapture(source)
        
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open video source: {source}")
            
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        
        self.queue = Queue(maxsize=queue_size)
        self.stopped = False
        self.thread = None

    def start(self):
        """Start the background frame reading thread."""
        if self.thread is not None:
            return self
            
        self.stopped = False
        self.thread = threading.Thread(target=self._update, name="StreamerThread", daemon=True)
        self.thread.start()
        return self

    def _update(self):
        """Internal loop to read frames from source."""
        while not self.stopped:
            if not self.queue.full():
                ret, frame = self.cap.read()
                if not ret:
                    self.stopped = True
                    break
                self.queue.put(frame)
            else:
                # Wait a bit if queue is full to prevent CPU pegging
                time.sleep(0.001)
                
        self.cap.release()

    def read(self):
        """Read the next frame from the queue."""
        try:
            return self.queue.get(timeout=1.0)
        except Empty:
            return None

    def more(self):
        """Check if there are more frames in the queue or if still running."""
        return not self.queue.empty() or not self.stopped

    def stop(self):
        """Stop the background thread and release resources."""
        self.stopped = True
        if self.thread:
            self.thread.join(timeout=2.0)
        if self.cap.isOpened():
            self.cap.release()
