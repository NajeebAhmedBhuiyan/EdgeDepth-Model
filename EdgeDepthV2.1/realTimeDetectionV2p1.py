#!/usr/bin/env python3
"""
realTimeDetection_debug.py

Realtime tester for micro_tiny_detector with debug features:
 - show the 64x64 model input window
 - print raw model outputs (cx,cy,w,h) and pixel size every N frames
 - optional synthetic scale test on an image to check sensitivity
 - supports PyTorch (.pth or traced .pt) and ONNX backends
 - auto-detects model files in script directory, current cwd, and home

Usage examples (from VS Code Run Python File you can just save & Run):
    python realTimeDetection_debug.py
    python realTimeDetection_debug.py --backend onnx --onnx path/to/micro_tiny_detector.onnx
    python realTimeDetection_debug.py --scale-test some_image.jpg

Press 'q' in the preview window to quit.
"""

import argparse
import time
import os
from pathlib import Path
from collections import deque

try:
    import cv2
except Exception as e:
    raise SystemExit("opencv-python is required. Install with: pip install opencv-python") from e

try:
    import numpy as np
    from PIL import Image
except Exception:
    raise SystemExit("numpy and pillow are required. Install with: pip install numpy pillow")

# optional imports
try:
    import torch
    import torch.nn as nn
    import torchvision.transforms as T
except Exception:
    torch = None
    nn = None
    T = None

try:
    import onnxruntime as ort
except Exception:
    ort = None

# ---------------------------
# Model class (must match training)
# ---------------------------
if torch is not None:
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
    """Return torch tensor C,H,W in range 0..1 (if torch available) or numpy"""
    if T is None:
        # fallback numpy resize and scale
        img_pil = img_pil.resize((img_size, img_size), Image.BILINEAR)
        arr = np.asarray(img_pil).astype(np.float32) / 255.0
        # convert HWC -> CHW
        return arr.transpose(2,0,1)
    transform = T.Compose([T.Resize((img_size, img_size)), T.ToTensor()])
    return transform(img_pil)

def postprocess_bbox_norm_to_pixels(bbox_norm, orig_w, orig_h):
    cx, cy, w, h = float(bbox_norm[0]), float(bbox_norm[1]), float(bbox_norm[2]), float(bbox_norm[3])
    cx_p = cx * orig_w
    cy_p = cy * orig_h
    w_p = w * orig_w
    h_p = h * orig_h
    xmin = int(round(max(0, cx_p - w_p/2)))
    ymin = int(round(max(0, cy_p - h_p/2)))
    xmax = int(round(min(orig_w-1, cx_p + w_p/2)))
    ymax = int(round(min(orig_h-1, cy_p + h_p/2)))
    return [xmin, ymin, xmax, ymax], (int(round(w_p)), int(round(h_p))), (cx, cy, w, h)

def load_pytorch_model(weights_path, device, num_classes=3):
    if torch is None:
        raise RuntimeError("PyTorch not installed.")
    model = MicroTinyDetector(num_classes=num_classes).to(device)
    st = torch.load(weights_path, map_location=device)
    # handle direct state_dict or full checkpoint
    if isinstance(st, dict) and any(k.startswith('backbone') for k in st.keys()):
        model.load_state_dict(st)
    elif isinstance(st, dict) and 'model' in st:
        model.load_state_dict(st['model'])
    else:
        try:
            model.load_state_dict(st)
        except Exception:
            # try to load as-is (maybe entire module saved)
            try:
                model = st
            except Exception as e:
                raise RuntimeError("Failed to load PyTorch weights.") from e
    model.eval()
    return model

def run_onnx_session(onnx_path):
    if ort is None:
        raise RuntimeError("onnxruntime not installed. Install: pip install onnxruntime")
    sess = ort.InferenceSession(str(onnx_path), providers=['CPUExecutionProvider'])
    return sess

# ---------------------------
# Synthetic scale test
# ---------------------------
def synthetic_scale_test(image_path, backend, model_obj=None, sess=None, device=None, img_size=64, scale_factors=None, classes=None):
    if scale_factors is None:
        scale_factors = [0.5, 0.75, 1.0, 1.25, 1.5]
    pil = Image.open(image_path).convert('RGB')
    w0, h0 = pil.size
    print("Synthetic scale test on:", image_path, "orig size:", (w0,h0))
    for s in scale_factors:
        new_w, new_h = int(w0*s), int(h0*s)
        img_s = pil.resize((new_w, new_h), Image.BILINEAR)
        if s > 1.0:
            left = (new_w - w0)//2; top = (new_h - h0)//2
            img_crop = img_s.crop((left, top, left+w0, top+h0))
            img_to_run = img_crop
        else:
            canvas = Image.new('RGB', (w0,h0), (0,0,0))
            left = (w0 - new_w)//2; top = (h0 - new_h)//2
            canvas.paste(img_s, (left, top))
            img_to_run = canvas

        inp_t = preprocess_pil(img_to_run, img_size=img_size)
        if backend == 'pytorch':
            t = inp_t.unsqueeze(0).to(device) if hasattr(inp_t, 'unsqueeze') else torch.from_numpy(inp_t).unsqueeze(0).to(device)
            with torch.no_grad():
                out = model_obj(t)
                bbox = out[0][0].cpu().numpy()
                logits = out[1][0].cpu().numpy()
                probs = np.exp(logits - logits.max()); probs /= probs.sum()
        else:
            inp_n = inp_t.astype(np.float32)[np.newaxis, :]  # N,C,H,W
            input_name = sess.get_inputs()[0].name
            out = sess.run(None, {input_name: inp_n})
            bbox = np.array(out[0][0], dtype=np.float32)
            logits = np.array(out[1][0], dtype=np.float32)
            probs = np.exp(logits - logits.max()); probs /= probs.sum()

        _, (bw_px, bh_px), _ = postprocess_bbox_norm_to_pixels(bbox, w0, h0)
        cls_idx = int(np.argmax(probs))
        cls_name = classes[cls_idx] if classes else str(cls_idx)
        print(f" scale={s:4.2f} -> bbox_norm={np.round(bbox,3)} -> size: {bw_px}px x {bh_px}px  class={cls_name} conf={probs.max():.3f}")

# ---------------------------
# Realtime loop with debugging
# ---------------------------
def realtime_loop(backend="pytorch", weights=None, traced=None, onnx_path=None,
                  cam_index=0, img_size=64, class_names=None, smooth=0.0,
                  debug=False, log_interval=10, device=None):
    if class_names is None:
        class_names = ["apple", "banana", "orange"]
    # prepare backend
    model = None
    sess = None
    if backend == "pytorch":
        if torch is None:
            raise RuntimeError("PyTorch not available.")
        if traced and Path(traced).exists():
            print("Loading traced TorchScript:", traced)
            model = torch.jit.load(traced, map_location=device)
            model.eval()
            using_traced = True
        else:
            assert weights and Path(weights).exists(), f"weights not found: {weights}"
            print("Loading PyTorch weights:", weights)
            model = load_pytorch_model(weights, device=device, num_classes=len(class_names))
            using_traced = False
    elif backend == "onnx":
        assert onnx_path and Path(onnx_path).exists(), f"ONNX not found: {onnx_path}"
        print("Loading ONNX:", onnx_path)
        sess = run_onnx_session(onnx_path)
    else:
        raise ValueError("Unknown backend")

    cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW if os.name == 'nt' else 0)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {cam_index}")

    prev_bbox = None
    frame_idx = 0
    fps = 0.0
    prev_time = time.time()
    history_w = deque(maxlen=500); history_h = deque(maxlen=500)

    print("Starting realtime loop. Press 'q' in window to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Frame read failed, exiting.")
            break
        orig_h, orig_w = frame.shape[:2]
        pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # preprocess
        inp_t = preprocess_pil(pil, img_size=img_size)  # torch tensor if torch available else numpy CHW
        # run model
        if backend == "pytorch":
            # ensure tensor
            if not isinstance(inp_t, (np.ndarray,)) and hasattr(inp_t, 'unsqueeze'):
                inp_tensor = inp_t.unsqueeze(0).to(device)
            else:
                inp_tensor = torch.from_numpy(inp_t).unsqueeze(0).to(device)
            with torch.no_grad():
                out = model(inp_tensor)
                # handle various output forms
                if isinstance(out, tuple) or isinstance(out, list):
                    bbox_raw = out[0][0].cpu().numpy()
                    logits = out[1][0].cpu().numpy()
                else:
                    # TorchScript may return a single object; attempt indexing
                    try:
                        bbox_raw = out[0][0].cpu().numpy()
                        logits = out[1][0].cpu().numpy()
                    except Exception:
                        # fallback: try converting to tuple
                        tup = tuple(out)
                        bbox_raw = tup[0][0].cpu().numpy()
                        logits = tup[1][0].cpu().numpy()
                probs = np.exp(logits - logits.max()); probs /= probs.sum()
            # keep a copy for debug display
            model_input_disp = inp_tensor[0].cpu().permute(1,2,0).numpy()
        else:
            # ONNX expects N,C,H,W as exported; use numpy
            inp_n = inp_t.astype(np.float32)[np.newaxis, :]
            input_name = sess.get_inputs()[0].name
            out = sess.run(None, {input_name: inp_n})
            bbox_raw = np.array(out[0][0], dtype=np.float32)
            logits = np.array(out[1][0], dtype=np.float32)
            probs = np.exp(logits - logits.max()); probs /= probs.sum()
            model_input_disp = inp_n[0].transpose(1,2,0)  # HWC for display (still 0..1)

        # smoothing (optional)
        if smooth and smooth > 0.0:
            if prev_bbox is None:
                smooth_bbox = bbox_raw
            else:
                smooth_bbox = smooth * np.array(prev_bbox) + (1.0 - smooth) * np.array(bbox_raw)
        else:
            smooth_bbox = bbox_raw
        prev_bbox = smooth_bbox.tolist()

        # postprocess to pixels
        box_px, (bw_px, bh_px), bbox_norm = postprocess_bbox_norm_to_pixels(smooth_bbox, orig_w, orig_h)
        cls_idx = int(np.argmax(probs))
        cls_name = class_names[cls_idx] if cls_idx < len(class_names) else str(cls_idx)
        conf = float(probs.max())

        # draw boxes and text
        xmin, ymin, xmax, ymax = box_px
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
        cv2.putText(frame, f"{cls_name} {conf:.2f}", (xmin, max(12, ymin-8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)
        cv2.putText(frame, f"w={bw_px}px h={bh_px}px", (xmin, min(orig_h-6, ymin + 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220,220,220), 1, cv2.LINE_AA)
        cv2.putText(frame, f"nw={bbox_norm[2]:.3f} nh={bbox_norm[3]:.3f}", (xmin, min(orig_h-6, ymin + 30)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (190,190,190), 1, cv2.LINE_AA)

        # FPS
        now = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / (now - prev_time + 1e-8))
        prev_time = now
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 1, cv2.LINE_AA)

        # debug: show 64x64 network input
        if debug:
            try:
                disp = np.clip(model_input_disp, 0.0, 1.0)
                disp_img = (disp * 255).astype(np.uint8)
                # if CHW -> HWC
                if disp_img.shape[0] != frame.shape[0] and disp_img.shape[2] == 3:
                    pass  # likely already HWC
                if disp_img.shape[2] == 3:
                    # already HWC
                    small = cv2.cvtColor(disp_img, cv2.COLOR_RGB2BGR)
                else:
                    # CHW -> HWC
                    small = cv2.cvtColor(disp_img, cv2.COLOR_RGB2BGR)
                cv2.imshow("Model input 64x64", small)
            except Exception as e:
                # ignore display problems
                pass

        cv2.imshow("TinyDetector - realtime (press q to quit)", frame)

        # logging raw values periodically
        frame_idx += 1
        history_w.append(bw_px); history_h.append(bh_px)
        if frame_idx % log_interval == 0:
            print(f"[FRAME {frame_idx}] raw_bbox_norm={np.round(bbox_raw,4)}  smoothed_norm={np.round(smooth_bbox,4)}")
            print(f"             pixel size: w={bw_px}px h={bh_px}px  class={cls_name} conf={conf:.3f}")
            # show last few width/h heights stats
            print(f"             recent w px: min {min(history_w):.0f}  max {max(history_w):.0f}  mean {np.mean(history_w):.1f}")

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# ---------------------------
# CLI & auto-detect
# ---------------------------
def find_model_files(args):
    # candidate names
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
    script_dir = Path(__file__).parent.resolve()
    cwd = Path.cwd().resolve()
    home = Path.home().resolve()
    search_dirs = [script_dir, cwd, home]

    def find_first(candidate_names):
        for name in candidate_names:
            if not name:
                continue
            p = Path(name)
            if p.is_absolute():
                if p.exists():
                    return str(p)
                continue
            for d in search_dirs:
                cand = (d / p).resolve()
                if cand.exists():
                    return str(cand)
        return None

    chosen_traced = find_first(candidate_traced_names)
    chosen_weights = find_first(candidate_weights_names)
    chosen_onnx = find_first(candidate_onnx_names)
    return chosen_traced, chosen_weights, chosen_onnx, search_dirs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["pytorch","onnx"], default=None)
    parser.add_argument("--weights", default=None)
    parser.add_argument("--traced", default=None)
    parser.add_argument("--onnx", default=None)
    parser.add_argument("--cam", type=int, default=0)
    parser.add_argument("--img-size", type=int, default=64)
    parser.add_argument("--smooth", type=float, default=0.0)
    parser.add_argument("--debug", action="store_true", help="Show model input and print debug info")
    parser.add_argument("--log-interval", type=int, default=10, help="Print raw bbox every N frames")
    parser.add_argument("--scale-test", default=None, help="Run synthetic scale test on given image and exit")
    parser.add_argument("--class-names", default=None, help="comma-separated class names")
    args = parser.parse_args()

    classes = args.class_names.split(",") if args.class_names else ["apple","banana","orange"]

    traced, weights, onnx_f, search_dirs = find_model_files(args)
    print("Search dirs:", [str(p) for p in search_dirs])
    print("Auto-detected: traced:", traced, "weights:", weights, "onnx:", onnx_f)

    # choose backend
    chosen_backend = args.backend
    if chosen_backend is None:
        if traced or weights:
            chosen_backend = "pytorch"
        elif onnx_f:
            chosen_backend = "onnx"
        else:
            chosen_backend = None

    # allow scale test without loading camera
    if args.scale_test:
        print("Running synthetic scale test...")
        if chosen_backend == "pytorch":
            # pick weights or traced (pref traced)
            if traced:
                device = torch.device('cpu')
                model_obj = torch.jit.load(traced, map_location=device)
                synthetic_scale_test(args.scale_test, 'pytorch', model_obj=model_obj, sess=None, device=device, img_size=args.img_size, classes=classes)
            elif weights:
                device = torch.device('cpu')
                model_obj = load_pytorch_model(weights, device=device, num_classes=len(classes))
                synthetic_scale_test(args.scale_test, 'pytorch', model_obj=model_obj, sess=None, device=device, img_size=args.img_size, classes=classes)
            else:
                print("No PyTorch model found for scale test.")
        elif chosen_backend == "onnx":
            if onnx_f:
                sess = run_onnx_session(onnx_f)
                synthetic_scale_test(args.scale_test, 'onnx', model_obj=None, sess=sess, device=None, img_size=args.img_size, classes=classes)
            else:
                print("No ONNX model found for scale test.")
        else:
            print("No backend/model available for scale test.")
        return

    # run realtime
    if chosen_backend == "pytorch":
        # prefer traced
        if traced:
            realtime_loop(backend="pytorch", traced=traced, cam_index=args.cam, img_size=args.img_size,
                          class_names=classes, smooth=args.smooth, debug=args.debug, log_interval=args.log_interval,
                          device=torch.device('cuda' if torch and torch.cuda.is_available() else 'cpu'))
        elif weights:
            realtime_loop(backend="pytorch", weights=weights, cam_index=args.cam, img_size=args.img_size,
                          class_names=classes, smooth=args.smooth, debug=args.debug, log_interval=args.log_interval,
                          device=torch.device('cuda' if torch and torch.cuda.is_available() else 'cpu'))
        else:
            print("No PyTorch model file found. Please put micro_best_tiny_detector.pth or traced .pt in script folder, or pass --weights / --traced")
            return
    elif chosen_backend == "onnx":
        if onnx_f:
            realtime_loop(backend="onnx", onnx_path=onnx_f, cam_index=args.cam, img_size=args.img_size,
                          class_names=classes, smooth=args.smooth, debug=args.debug, log_interval=args.log_interval)
        else:
            print("No ONNX file found. Please place micro_tiny_detector.onnx in script folder or pass --onnx")
            return
    else:
        print("No backend or model file detected. You can pass explicit flags:")
        print("  --backend pytorch --weights micro_best_tiny_detector.pth")
        print("  --backend pytorch --traced micro_tiny_detector_traced.pt")
        print("  --backend onnx --onnx micro_tiny_detector.onnx")
        return

if __name__ == "__main__":
    main()
