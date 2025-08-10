import cv2
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image

# Define the model architecture (double-check against training)
class TinyDetector(nn.Module):
    def __init__(self):
        super(TinyDetector, self).__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 48, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(48, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(output_size=1)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 128),
            nn.ReLU()
        )
        self.bbox_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 4),
            nn.Sigmoid()
        )
        self.cls_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3)  # 3 classes: apple, banana, orange
        )

    def forward(self, x):
        features = self.backbone(x)
        features = self.fc(features)
        bbox = self.bbox_head(features)
        logits = self.cls_head(features)
        return bbox, logits

# Configuration
MODEL_PATH = "/home/nabq/object-detection/EdgeDepthV2.0/best_tiny_detector.pth"
CONF_THRESHOLD = 0.7  # Increased threshold to reduce false positives
INPUT_SIZE = (128, 128)
CLASSES = ["apple", "banana", "orange"]

# Initialize model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = TinyDetector().to(device)

# Load model weights
print(f"Loading weights from {MODEL_PATH}...")
try:
    state_dict = torch.load(MODEL_PATH, map_location=device)
    
    # Handle possible state_dict wrapping
    if 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']
    elif 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    
    model.load_state_dict(state_dict)
    model.eval()
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Preprocessing transform - ensure it matches training
transform = transforms.Compose([
    transforms.Resize(INPUT_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Initialize video capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera")
    exit()

# Frame counter for FPS calculation
frame_count = 0
start_time = time.time()

# Debugging variables
last_outputs = []
debug_mode = False  # Set to True to see raw outputs

print("Starting real-time detection. Press 'q' to quit...")
print("Press 'd' to toggle debug mode")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame")
        break

    frame_count += 1
    orig_h, orig_w = frame.shape[:2]
    
    # Convert to PIL Image for preprocessing
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(frame_rgb)
    
    # Preprocess image
    input_tensor = transform(pil_img).unsqueeze(0).to(device)
    
    # Run inference
    with torch.no_grad():
        bbox, logits = model(input_tensor)
    
    # Convert outputs to numpy arrays
    bbox = bbox.cpu().numpy()[0]
    logits = logits.cpu().numpy()[0]
    
    # Store outputs for debugging
    last_outputs = (bbox, logits)
    
    # Convert logits to probabilities
    exp_logits = np.exp(logits - np.max(logits))
    probs = exp_logits / exp_logits.sum()
    
    # Get class with highest probability
    class_id = np.argmax(probs)
    confidence = probs[class_id]
    
    # Display information
    info_y = 30
    detection_info = ""
    
    # Calculate FPS
    elapsed_time = time.time() - start_time
    fps = frame_count / elapsed_time
    
    if confidence > CONF_THRESHOLD:
        # Convert normalized bbox to pixel coordinates
        x1, y1, x2, y2 = bbox
        x1 = int(x1 * orig_w)
        y1 = int(y1 * orig_h)
        x2 = int(x2 * orig_w)
        y2 = int(y2 * orig_h)
        
        # Calculate bounding box dimensions
        width_px = abs(x2 - x1)
        height_px = abs(y2 - y1)
        
        # Create detection info
        detection_info = f"Detected: {CLASSES[class_id]} ({confidence:.2f}) | Size: {width_px}×{height_px} px"
        text_color = (0, 255, 0)  # Green
    else:
        detection_info = "No object detected"
        text_color = (0, 0, 255)  # Red
    
    # Display detection info
    cv2.putText(
        frame, 
        detection_info, 
        (20, info_y), 
        cv2.FONT_HERSHEY_SIMPLEX, 
        0.7, 
        text_color, 
        2
    )
    
    # Display FPS
    fps_text = f"FPS: {fps:.1f}"
    cv2.putText(
        frame, 
        fps_text, 
        (20, info_y + 30), 
        cv2.FONT_HERSHEY_SIMPLEX, 
        0.7, 
        (0, 0, 255), 
        2
    )
    
    # Display debug info if enabled
    if debug_mode:
        debug_text = f"Raw outputs: bbox={bbox}, logits={logits}"
        cv2.putText(
            frame, 
            debug_text, 
            (20, info_y + 60), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.5, 
            (255, 255, 255), 
            1
        )
        debug_text2 = f"Probs: apple={probs[0]:.2f}, banana={probs[1]:.2f}, orange={probs[2]:.2f}"
        cv2.putText(
            frame, 
            debug_text2, 
            (20, info_y + 90), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.5, 
            (255, 255, 255), 
            1
        )

    # Display frame
    cv2.imshow("Real-time Object Detection", frame)
    
    # Handle key presses
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break
    elif key & 0xFF == ord('d'):
        debug_mode = not debug_mode
        print(f"Debug mode {'enabled' if debug_mode else 'disabled'}")

# Cleanup
cap.release()
cv2.destroyAllWindows()
print("Detection stopped")

# Print final outputs for debugging
print("\nLast outputs before exit:")
print("Bounding box:", last_outputs[0])
print("Logits:", last_outputs[1])