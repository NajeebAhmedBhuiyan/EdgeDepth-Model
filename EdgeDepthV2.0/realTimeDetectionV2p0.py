import cv2
import numpy as np
import time

# Configuration
MODEL_PATH = "/home/nabq/object-detection/EdgeDepthV2.0/tiny_detector.onnx"
CONF_THRESHOLD = 0.5
INPUT_SIZE = (128, 128)  # Model expects 128x128 input
CLASSES = ["apple", "banana", "orange"]

# Load ONNX model
print(f"Loading model from {MODEL_PATH}...")
net = cv2.dnn.readNetFromONNX(MODEL_PATH)
print("Model loaded successfully")

# Initialize video capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera")
    exit()

# Frame counter for FPS calculation
frame_count = 0
start_time = time.time()

print("Starting real-time detection. Press 'q' to quit...")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame")
        break

    frame_count += 1
    orig_h, orig_w = frame.shape[:2]
    
    # Preprocess frame
    blob = cv2.dnn.blobFromImage(
        frame, 
        1/255.0, 
        INPUT_SIZE, 
        swapRB=True, 
        crop=False
    )
    
    # Run inference
    net.setInput(blob)
    output = net.forward()
    
    # Print output shape for debugging
    print(f"Output shape: {output.shape}")
    
    # If output is (1, 3), it's only class logits
    if output.shape == (1, 3):
        logits = output[0]
        
        # Convert logits to probabilities using softmax
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / exp_logits.sum()
        
        # Get class with highest probability
        class_id = np.argmax(probs)
        confidence = probs[class_id]
        
        # Display class prediction without bounding box
        if confidence > CONF_THRESHOLD:
            label = f"{CLASSES[class_id]}: {confidence:.2f}"
        else:
            label = "No detection"
            
        # Put text in the top-left corner
        cv2.putText(
            frame, 
            label, 
            (20, 40), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            1, 
            (0, 255, 0), 
            2
        )
    
    # Calculate and display FPS
    elapsed_time = time.time() - start_time
    fps = frame_count / elapsed_time
    cv2.putText(
        frame, 
        f"FPS: {fps:.1f}", 
        (20, 80), 
        cv2.FONT_HERSHEY_SIMPLEX, 
        0.7, 
        (0, 0, 255), 
        2
    )

    # Display frame
    cv2.imshow("Real-time Object Detection", frame)
    
    # Exit on 'q' press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
print("Detection stopped")