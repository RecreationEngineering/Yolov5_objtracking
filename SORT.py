import torch
import cv2

import cv2
import numpy as np

# Modified KalmanBoxTracker class
class KalmanBoxTracker:
    def __init__(self, bbox):
        self.kalman_filter = cv2.KalmanFilter(6, 4)  # Kalman filter with state_size=6 and measurement_size=4
        self.kalman_filter.transitionMatrix = np.array([[1, 0, 1, 0, 0, 0],
                                                        [0, 1, 0, 1, 0, 0],
                                                        [0, 0, 1, 0, 1, 0],
                                                        [0, 0, 0, 1, 0, 1],
                                                        [0, 0, 0, 0, 1, 0],
                                                        [0, 0, 0, 0, 0, 1]], dtype=np.float32)
        self.kalman_filter.measurementMatrix = np.array([[1, 0, 0, 0, 0, 0],
                                                         [0, 1, 0, 0, 0, 0],
                                                         [0, 0, 0, 0, 1, 0],
                                                         [0, 0, 0, 0, 0, 1]], dtype=np.float32)
        self.kalman_filter.processNoiseCov = 1e-3 * np.eye(6, dtype=np.float32)
        self.kalman_filter.measurementNoiseCov = 1e-1 * np.eye(4, dtype=np.float32)
        self.kalman_filter.errorCovPost = 1 * np.eye(6, dtype=np.float32)
        
        self.kalman_filter.statePost = np.array([[bbox[0]],
                                                 [bbox[1]],
                                                 [bbox[2]],
                                                 [bbox[3]],
                                                 [0],
                                                 [0]], dtype=np.float32)  # Initialize width and height states to 0
        
        self.id = None
        self.skipped_frames = 0
        self.trace = []

    def update(self, bbox):
        self.kalman_filter.correct(np.array([[bbox[0]], [bbox[1]], [bbox[2]], [bbox[3]]], dtype=np.float32))
        prediction = self.kalman_filter.predict()
        self.trace.append((prediction[0, 0], prediction[1, 0]))
        
    def get_state(self):
        return self.kalman_filter.statePost


# Modified SORT class
class SORT:
    def __init__(self):
        self.trackers = []
        self.frame_count = 0
        
    def update(self, detections):
        detections = detections.iloc[:, :4].values.tolist()  # Selecting x, y, w, h from detections
        self.frame_count += 1
        
        for tracker in self.trackers:
            tracker.skipped_frames += 1
        
        matched_indices = []
        unmatched_detections = []
        unmatched_trackers = []
        
        for i, detection in enumerate(detections):
            best_match = None
            min_distance = float('inf')
            
            for j, tracker in enumerate(self.trackers):
                state = tracker.get_state()
                distance = np.linalg.norm(np.array(detection) - state[:4])
                
                if distance < min_distance:
                    min_distance = distance
                    best_match = j
            
            if best_match is not None and min_distance < 50:
                matched_indices.append((best_match, i))
            else:
                unmatched_detections.append(i)
        
        unmatched_trackers = [i for i in range(len(self.trackers)) if i not in [idx for idx, _ in matched_indices]]
        
        for idx_tracker, idx_detection in matched_indices:
            self.trackers[idx_tracker].update(detections[idx_detection])
            self.trackers[idx_tracker].skipped_frames = 0
        
        for idx in unmatched_trackers:
            self.trackers[idx].skipped_frames += 1
        
        new_trackers = [KalmanBoxTracker(detections[i]) for i in unmatched_detections]
        self.trackers.extend(new_trackers)
        
        active_trackers = [tracker for tracker in self.trackers if tracker.skipped_frames < 5]
        self.trackers = active_trackers
        
        return [tracker.get_state() for tracker in self.trackers if tracker.skipped_frames < 5]

if __name__ == '__main__':
    # Load YOLOv5 model
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', 'yolov5s.pt')  # Provide the path to your trained model
    # Set the model in evaluation mode
    model.eval()
    
    # Open video using OpenCV
    video_path = 'cars_on_highway.mp4'
    cap = cv2.VideoCapture(video_path)
    
    sort = SORT()
    
    while cap.isOpened():
        ret, frame = cap.read()
        
        if not ret:
            break

        # Perform object detection on the frame
        results = model(frame)
        
        # Perform object detection on the frame and get detections as [[x, y, w, h], ...]
        detections = results.pandas().xywh[0]  # Replace with your object detection results
        
        tracked_states = sort.update(detections)
        
        for state in tracked_states:
            x, y, w, h = int(state[0]), int(state[1]), int(state[2]), int(state[3])
            cv2.rectangle(frame, (x - w // 2, y - h // 2), (x + w // 2, y + h // 2), (0, 255, 0), 2)
        
        cv2.imshow('SORT Object Tracking', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

