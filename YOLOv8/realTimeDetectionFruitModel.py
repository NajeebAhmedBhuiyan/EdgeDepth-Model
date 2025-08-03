import cv2
from ultralytics import YOLO
import os

# Use absolute path to your model file
model_path = r'/home/nabq/object-detection/YOLOv8/best_fruit_detection.pt'

# Load trained model
model = YOLO(model_path)

# Class names (must match your training classes)
class_names = ["apple", "banana", "orange"]

# Rest of your webcam code remains the same
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Run detection
    results = model.track(frame, persist=True)
    
    # Process results manually
    for result in results:
        # Extract boxes, scores, and class IDs
        boxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy().astype(int)
        
        # Draw only high-confidence detections
        for box, score, class_id in zip(boxes, scores, class_ids):
            if score > 0.78:
                # Draw bounding box
                x1, y1, x2, y2 = map(int, box[:4])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Create label text
                label = f"{class_names[class_id]}: {score:.2f}"
                
                # Draw label background
                (text_width, text_height), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
                )
                cv2.rectangle(
                    frame,
                    (x1, y1 - text_height - 10),
                    (x1 + text_width, y1),
                    (0, 255, 0),
                    -1
                )
                
                # Draw label text
                cv2.putText(
                    frame,
                    label,
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 0),
                    2
                )
    
    # Display the frame
    cv2.imshow('Fruit Detection (Confidence > 0.78)', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()