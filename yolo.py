import cv2
from ultralytics import YOLO
import torch
import numpy as np

device = 0 if torch.cuda.is_available() else "cpu"
model = YOLO("./best.pt")

# (optional) lock to a class name; set to None to allow any class
TARGET_CLASS_NAME = "rubber_duck"  # <-- change to your class name, or None

# simple centroid smoothing (EMA). set ALPHA=1.0 to disable smoothing
ALPHA = 0.25
prev_c = None

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Cannot open webcam. Try a different index (1, 2, ...)")

while True:
    ok, frame = cap.read()
    if not ok:
        break

    res = model.predict(
        source=frame, conf=0.40, imgsz=640, device=device, verbose=False
    )[0]
    boxes = res.boxes

    annotated = frame.copy()
    h, w = frame.shape[:2]
    cv2.drawMarker(
        annotated, (w // 2, h // 2), (255, 255, 255), cv2.MARKER_CROSS, 16, 2
    )  # image center

    if boxes is not None and len(boxes) > 0:
        xyxy = boxes.xyxy.cpu().numpy()  # (N,4)
        cls = boxes.cls.cpu().numpy().astype(int)  # (N,)
        conf = boxes.conf.cpu().numpy()  # (N,)

        # --- keep only the desired class (if set) ---
        if TARGET_CLASS_NAME is not None:
            # map name -> id once
            name2id = {v: k for k, v in model.names.items()}
            target_id = name2id.get(TARGET_CLASS_NAME, None)
            mask = (
                (cls == target_id)
                if target_id is not None
                else np.ones_like(cls, dtype=bool)
            )
        else:
            mask = np.ones_like(cls, dtype=bool)

        # pick ONE detection (highest confidence) among the filtered set
        if mask.any():
            idx_local = np.argmax(conf[mask])
            idx = np.where(mask)[0][idx_local]
        else:
            # fallback: take the overall most confident detection
            idx = int(np.argmax(conf))

        x1, y1, x2, y2 = xyxy[idx]
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

        # smooth centroid
        if prev_c is None:
            fx, fy = cx, cy
        else:
            fx = int(ALPHA * cx + (1 - ALPHA) * prev_c[0])
            fy = int(ALPHA * cy + (1 - ALPHA) * prev_c[1])
        prev_c = (fx, fy)

        # draw the one selected box + centroid
        cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.drawMarker(annotated, (fx, fy), (0, 255, 255), cv2.MARKER_CROSS, 14, 2)

        # offset from image center (useful for control)
        dx, dy = fx - w // 2, fy - h // 2
        cv2.putText(
            annotated,
            f"C=({fx},{fy}) d=({dx},{dy}) conf={conf[idx]:.2f}",
            (int(x1), max(0, int(y1) - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2,
        )
        # print to console if you need it for control logic
        # print(f"centroid=({fx},{fy}) offset=({dx},{dy})")
    else:
        # no detection: optionally keep showing last centroid
        if prev_c is not None:
            cv2.drawMarker(
                annotated, prev_c, (0, 165, 255), cv2.MARKER_TILTED_CROSS, 14, 2
            )
            cv2.putText(
                annotated,
                "no det (holding last)",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 165, 255),
                2,
            )

    # show HSV view if you still want it (optional)
    # cv2.imshow("frame (HSV)", cv2.cvtColor(frame, cv2.COLOR_BGR2HSV))

    cv2.imshow("YOLO one-target (q to quit)", annotated)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
