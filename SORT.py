import torch
import cv2

if __name__ == '__main__':
    # Load YOLOv5 model
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', 'yolov5s.pt')  # Provide the path to your trained model

    # Set the model in evaluation mode
    model.eval()

    # Open video using OpenCV
    video_path = 'cars_on_highway.mp4'
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Perform object detection on the frame
        results = model(frame)
        
        # Process the detection results (results.raw.pred is a list of detected objects)
        for det in results.pred[0]:
            # Extract relevant information from detection
            class_id, confidence, bbox = det[5], det[4], det[:4]
            
            # Filter out detections below a certain confidence threshold
            if confidence > 0.5:
                # Draw bounding box on the frame
                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                class_name = model.names[int(class_id)]
                cv2.putText(frame, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Display the frame with detections
        cv2.imshow('YOLOv5 Object Detection', frame)
        
        # Exit when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()