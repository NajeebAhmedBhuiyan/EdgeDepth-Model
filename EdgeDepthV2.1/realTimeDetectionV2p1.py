import argparse
import time
import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

# try imports; helpful error messages
try:
    import torch
    import torch.nn as nn
    import torchvision.transforms as T
except Exception:
    torch = None
try:
    import onnxruntime as ort
except Exception:
    ort = None

# ---------------------------
# Micro model class (same as training)
# ---------------------------
class MicroTinyDetector(nn.Module):
    def __init__(self, num_classes=3):
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
        bbox = self.bbox_head(x)
        logits = self.cls_head(x)
        return bbox, logits

# ---------------------------
# Helpers
# ---------------------------
def preprocess_pil(img_pil, img_size=64):
    transform = T.Compose([T.Resize((img_size, img_size)), T.ToTensor()])
    t = transform(img_pil)  # tensor C,H,W, float32 0..1
    return t  # torch.Tensor

def postprocess_bbox_norm_to_pixels(bbox_norm, orig_w, orig_h):
    cx, cy, w, h = bbox_norm
    cx_p = cx * orig_w
    cy_p = cy * orig_h
    w_p = w * orig_w
    h_p = h * orig_h
    xmin = int(round(max(0, cx_p - w_p/2)))
    ymin = int(round(max(0, cy_p - h_p/2)))
    xmax = int(round(min(orig_w-1, cx_p + w_p/2)))
    ymax = int(round(min(orig_h-1, cy_p + h_p/2)))
    return [xmin, ymin, xmax, ymax], (int(round(w_p)), int(round(h_p))), (float(cx), float(cy), float(w), float(h))

# ---------------------------
# Main realtime loop
# ---------------------------
def realtime_loop(
    backend="pytorch", weights=None, traced=None, onnx_path=None,
    cam_index=0, img_size=64, class_names=None, smooth=0.6
):
    if class_names is None:
        class_names = ["apple", "banana", "orange"]

    # Prepare model/backend
    model = None
    sess = None
    device = None
    if backend == "pytorch":
        if torch is None:
            raise RuntimeError("PyTorch not installed. `pip install torch torchvision`")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # prefer traced if provided
        if traced and Path(traced).exists():
            # traced module: no class needed
            model = torch.jit.load(traced, map_location=device)
            model.eval()
            use_traced = True
            print("Loaded traced TorchScript model:", traced)
        else:
            # load class + weights
            model = MicroTinyDetector(num_classes=len(class_names)).to(device)
            assert weights and Path(weights).exists(), "weights path missing"
            state = torch.load(weights, map_location=device)
            # if state is a dict with "model_state" then adapt — but assume plain state_dict
            if isinstance(state, dict) and not any(k.startswith('backbone') for k in state.keys()):
                # might be full checkpoint, try common keys
                try:
                    model.load_state_dict(state)
                except Exception:
                    # try if state contains 'model'
                    if 'model' in state:
                        model.load_state_dict(state['model'])
                    else:
                        model.load_state_dict(state)
            else:
                model.load_state_dict(state)
            model.eval()
            use_traced = False
            print("Loaded PyTorch weights:", weights)
    elif backend == "onnx":
        if ort is None:
            raise RuntimeError("onnxruntime not installed. `pip install onnxruntime`")
        assert onnx_path and Path(onnx_path).exists(), "onnx_path must point to a valid ONNX file"
        sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
        print("Loaded ONNX model:", onnx_path)
    else:
        raise ValueError("backend must be 'pytorch' or 'onnx'")

    # Open camera
    cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW if os.name == 'nt' else 0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera (index {}).".format(cam_index))

    prev_bbox = None
    prev_time = time.time()
    fps = 0.0

    print("Starting realtime. Press 'q' to quit. Backend:", backend)
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Frame read error; exiting.")
            break
        orig_h, orig_w = frame.shape[:2]
        pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        # Preprocess
        if backend == "pytorch":
            inp_t = preprocess_pil(pil, img_size=img_size).unsqueeze(0).to(device)  # 1,C,H,W (torch tensor)
            with torch.no_grad():
                out = model(inp_t)
                # output may be tuple of tensors or TorchScript; normalize handling:
                if isinstance(out, tuple) or isinstance(out, list):
                    bbox_t, logits_t = out[0], out[1]
                else:
                    # some traced models return a single tuple-like object; try indexing
                    try:
                        bbox_t, logits_t = out[0], out[1]
                    except Exception:
                        raise RuntimeError("Unexpected model output shape/type for PyTorch backend.")
                bbox = bbox_t[0].cpu().numpy()
                logits = logits_t[0].cpu().numpy()
                # softmax
                exp = np.exp(logits - logits.max()); probs = exp / exp.sum()
        else:  # onnx
            inp_np = np.asarray(preprocess_pil(pil, img_size=img_size)).astype(np.float32)  # C,H,W
            inp_n = inp_np[np.newaxis, :]  # 1,C,H,W
            # ONNX session run; input name often 'input' as exported
            input_name = sess.get_inputs()[0].name
            out = sess.run(None, {input_name: inp_n})
            # expecting [bbox, logits]
            bbox = np.array(out[0][0], dtype=np.float32)
            logits = np.array(out[1][0], dtype=np.float32)
            exp = np.exp(logits - logits.max()); probs = exp / exp.sum()

        # smoothing (EMA)
        if prev_bbox is None:
            smooth_bbox = bbox
        else:
            smooth_bbox = smooth * np.array(prev_bbox) + (1.0 - smooth) * np.array(bbox)
        prev_bbox = smooth_bbox.tolist()

        # postprocess to pixel coords
        box_px, (bw_px, bh_px), bbox_norm = postprocess_bbox_norm_to_pixels(smooth_bbox, orig_w, orig_h)
        cls_idx = int(np.argmax(probs))
        conf = float(np.max(probs))

        # Draw on frame
        xmin, ymin, xmax, ymax = box_px
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
        label = f"{class_names[cls_idx]} {conf:.2f}"
        cv2.putText(frame, label, (xmin, max(12, ymin-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)
        size_text = f"w={bw_px}px h={bh_px}px"
        cv2.putText(frame, size_text, (xmin, min(orig_h-6, ymin + 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220,220,220), 1, cv2.LINE_AA)
        norm_text = f"nw={bbox_norm[2]:.2f} nh={bbox_norm[3]:.2f}"
        cv2.putText(frame, norm_text, (xmin, min(orig_h-6, ymin + 30)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (190,190,190), 1, cv2.LINE_AA)

        # FPS
        now = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / (now - prev_time + 1e-8))
        prev_time = now
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 1, cv2.LINE_AA)

        cv2.imshow("TinyDetector - realtime", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# ---------------------------
# CLI
# ---------------------------
if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Realtime test for micro_tiny_detector (PyTorch/ONNX)")
    parser.add_argument("--backend", choices=["pytorch","onnx"], default=None,
                        help="Which backend to use. If not set, auto-detected from available files.")
    parser.add_argument("--weights", default=None, help="PyTorch .pth file (optional)")
    parser.add_argument("--traced", default=None, help="TorchScript traced .pt (optional)")
    parser.add_argument("--onnx", default=None, help="ONNX model (optional)")
    parser.add_argument("--cam", type=int, default=0, help="camera index")
    parser.add_argument("--img-size", type=int, default=64, help="input size (default 64)")
    parser.add_argument("--smooth", type=float, default=0.6, help="EMA smoothing (0..1)")
    parser.add_argument("--class-names", default=None, help="comma-separated class names (default apple,banana,orange)")
    args = parser.parse_args()

    # class names
    if args.class_names:
        classes = args.class_names.split(",")
    else:
        classes = ["apple","banana","orange"]

    # Candidate filenames (in order of preference)
    candidate_weights_names = [
        args.weights, "micro_tiny_detector.pth", "micro_best_tiny_detector.pth",
        "micro_best_tiny_detector_final.pth", "micro_best_tiny_detector_latest.pth"
    ]
    candidate_traced_names = [
        args.traced, "micro_tiny_detector_traced.pt", "micro_tiny_detector.pt"
    ]
    candidate_onnx_names = [
        args.onnx, "micro_tiny_detector.onnx"
    ]

    # Search paths: script dir first (recommended for VSCode Run), then cwd, then home
    script_dir = Path(__file__).parent.resolve()
    cwd = Path.cwd().resolve()
    home = Path.home().resolve()
    search_dirs = [script_dir, cwd, home]

    def find_first(candidate_names):
        for name in candidate_names:
            if not name:
                continue
            p = Path(name)
            # if absolute path provided, check it directly
            if p.is_absolute():
                if p.exists():
                    return str(p)
                continue
            # otherwise search the list of directories
            for d in search_dirs:
                cand = (d / p).resolve()
                if cand.exists():
                    return str(cand)
        return None

    chosen_traced = find_first(candidate_traced_names)
    chosen_weights = find_first(candidate_weights_names)
    chosen_onnx = find_first(candidate_onnx_names)

    # If user forced backend via CLI, respect it; else auto-choose priority: traced > weights > onnx
    chosen_backend = args.backend
    if chosen_backend is None:
        if chosen_traced:
            chosen_backend = "pytorch"
        elif chosen_weights:
            chosen_backend = "pytorch"
        elif chosen_onnx:
            chosen_backend = "onnx"
        else:
            chosen_backend = None

    # Print what we will try (helpful debugging)
    print("Search directories (in order):")
    for d in search_dirs:
        print("  -", d)
    print("Auto-detected files (if any):")
    print("  traced:", chosen_traced)
    print("  weights:", chosen_weights)
    print("  onnx:", chosen_onnx)
    print("Chosen backend:", chosen_backend)

    # FINAL RUN
    if chosen_backend == "pytorch":
        # prefer traced .pt if available
        if chosen_traced:
            print(f"[AUTO] Using traced TorchScript: {chosen_traced}")
            realtime_loop(backend="pytorch", traced=chosen_traced, cam_index=args.cam,
                          img_size=args.img_size, class_names=classes, smooth=args.smooth)
        elif chosen_weights:
            print(f"[AUTO] Using PyTorch weights: {chosen_weights}")
            realtime_loop(backend="pytorch", weights=chosen_weights, cam_index=args.cam,
                          img_size=args.img_size, class_names=classes, smooth=args.smooth)
        else:
            print("\nERROR: No PyTorch model found to run with PyTorch backend. Tried these filenames in the search dirs:")
            for name in candidate_weights_names:
                if name:
                    print("  -", name)
            raise SystemExit(1)

    elif chosen_backend == "onnx":
        if chosen_onnx:
            print(f"[AUTO] Using ONNX model: {chosen_onnx}")
            realtime_loop(backend="onnx", onnx_path=chosen_onnx, cam_index=args.cam,
                          img_size=args.img_size, class_names=classes, smooth=args.smooth)
        else:
            print("\nERROR: No ONNX model found. Tried these filenames in the search dirs:")
            for name in candidate_onnx_names:
                if name:
                    print("  -", name)
            raise SystemExit(1)
    else:
        print("\nERROR: No suitable backend or model file found. To be sure, specify one explicitly:")
        print("  --backend pytorch --weights micro_best_tiny_detector.pth")
        print("  --backend pytorch --traced micro_tiny_detector_traced.pt")
        print("  --backend onnx --onnx micro_tiny_detector.onnx")
        raise SystemExit(1)
