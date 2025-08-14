# realtime_edgedepth.py  — run locally (VS Code / terminal)
import cv2, time, argparse
import torch, numpy as np
from PIL import Image
import torchvision.transforms as T
from pathlib import Path

# ---- Edit these defaults or pass via args ----
MODEL_PTH = "/home/nabq/object-detection/EdgeDepthV2.1.1/micro_best_edgedepth.pth"
IMG_SIZE = 64
CLASSES = ["apple","banana","orange"]  # keep consistent
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- Model class must match your trained model ----
import torch.nn as nn
class MicroTinyDetector(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 12, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(12, 16, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(16, 32), nn.ReLU())
        self.bbox_head = nn.Sequential(nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 4), nn.Sigmoid())
        self.cls_head  = nn.Sequential(nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, num_classes))

    def forward(self, x):
        x = self.backbone(x)
        x = self.fc(x)
        return self.bbox_head(x), self.cls_head(x)

# load model
model = MicroTinyDetector(num_classes=len(CLASSES)).to(DEVICE)
if Path(MODEL_PTH).exists():
    model.load_state_dict(torch.load(MODEL_PTH, map_location=DEVICE))
    print("Loaded", MODEL_PTH)
else:
    raise SystemExit(f"Model file not found: {MODEL_PTH}")
model.eval()

# preprocessing
resize = T.Resize((IMG_SIZE, IMG_SIZE))
to_tensor = T.ToTensor()

def norm_to_pixels(box_norm, w, h):
    cx,cy,w_rel,h_rel = box_norm
    cx_p = cx * w; cy_p = cy * h
    w_p = w_rel * w; h_p = h_rel * h
    xmin = int(round(cx_p - w_p/2)); ymin = int(round(cy_p - h_p/2))
    xmax = int(round(cx_p + w_p/2)); ymax = int(round(cy_p + h_p/2))
    xmin = max(0, xmin); ymin = max(0, ymin)
    xmax = min(w-1, xmax); ymax = min(h-1, ymax)
    return xmin, ymin, xmax, ymax, int(round(w_p)), int(round(h_p))

# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise SystemExit("Cannot open webcam")

fps_time = time.time()
while True:
    ret, frame = cap.read()
    if not ret:
        break
    h, w = frame.shape[:2]
    # prepare input: convert BGR->RGB, PIL->transform
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    inp = resize(img_pil)
    inp_t = to_tensor(inp).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        bbox_t, logits = model(inp_t)
    bbox = bbox_t[0].cpu().numpy()  # normalized
    probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
    cls_idx = int(probs.argmax()); conf = float(probs.max())

    xmin,ymin,xmax,ymax, w_px, h_px = norm_to_pixels(bbox, w, h)
    # draw boxes and text
    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0,0,255), 2)  # predicted
    label = f"{CLASSES[cls_idx]} {conf:.2f}  {w_px}x{h_px}px"
    cv2.putText(frame, label, (xmin, max(20, ymin-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
    # FPS
    fps = 1.0 / max(1e-6, time.time() - fps_time); fps_time = time.time()
    cv2.putText(frame, f"FPS: {fps:.1f}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 1)

    cv2.imshow("EdgeDepth - live", frame)
    k = cv2.waitKey(1) & 0xFF
    if k == 27 or k == ord('q'):  # Esc or q to quit
        break

cap.release()
cv2.destroyAllWindows()
