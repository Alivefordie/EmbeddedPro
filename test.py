# import the necessary packages
import numpy as np
import cv2
import cv2.aruco as aruco

clicks = []
FONT = cv2.FONT_HERSHEY_SIMPLEX
REAL_W = 2.3  # กว้างแกน X (เมตร)
REAL_H = 2.2  # สูงแกน Y (เมตร)  (ตามความหมายที่คุณนิยาม)
PX_PER_M = 300  # 300 px/m -> BEV ~ 690x660 px
STEP_M = 0.1
STEP_PX = int(round(STEP_M * PX_PER_M))
BEV_W = int(round(REAL_W * PX_PER_M))
BEV_H = int(round(REAL_H * PX_PER_M))
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
parameters = aruco.DetectorParameters()


def ArucoDetector(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect markers
    detector = aruco.ArucoDetector(aruco_dict, parameters)
    corners, ids, rejected = detector.detectMarkers(gray)

    # Draw detected markers
    if ids is not None:
        frame = aruco.drawDetectedMarkers(frame, corners)
        for i, corner in enumerate(corners):
            c = corner[0].mean(axis=0).astype(int)  # center point (x,y)
            marker_id = str(ids[i][0])

            cv2.putText(
                frame,
                f"({c[0]},{c[1]})",
                (c[0] + 10, c[1] - 10),  # move right & above
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
    return corners, ids, frame


def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype="float32")
    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    # return the ordered coordinates
    return rect


def warp_to_bev(image):
    bev = cv2.warpPerspective(image, H, (BEV_W, BEV_H))
    grid = cv2.addWeighted(bev, 1.0, grid_overlay, 0.35, 0)
    return grid


def on_mouse(event, x, y, flags, param):
    global clicks
    if event == cv2.EVENT_LBUTTONDOWN:
        clicks.append((x, y))
        print(f"Point {len(clicks)}: ({x}, {y})")
    elif event == cv2.EVENT_RBUTTONDOWN and clicks:
        clicks.pop()  # undo จุดล่าสุด


def draw_overlay(frame, pts):
    """วาด overlay: จุด, label, และเส้น polygon เชื่อมจุด"""
    canvas = frame.copy()

    # วาดจุด + label
    for i, (px, py) in enumerate(pts, start=1):
        cv2.circle(canvas, (px, py), 6, (0, 0, 255), -1)
        cv2.putText(
            canvas, f"P{i}", (px + 8, py - 8), FONT, 0.6, (0, 0, 255), 2, cv2.LINE_AA
        )

    # วาดโพลิกอนเชื่อมจุด (ตามลำดับที่คลิก)
    if len(pts) >= 2:
        cv2.polylines(
            canvas,
            [np.array(pts, dtype=np.int32)],
            isClosed=False,
            color=(0, 255, 255),
            thickness=2,
            lineType=cv2.LINE_AA,
        )

    # บาร์ช่วยเหลือ
    help_text = "L-click: add | R-click: undo | q: quit"
    cv2.putText(canvas, help_text, (10, 30), FONT, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    return canvas


def build_bev_grid_overlay(
    BEV_W,
    BEV_H,
    STEP_PX,
    PX_PER_M,
    label_every=5,  # ใส่ป้ายทุก 5 เส้น (เช่น 0.5 m ถ้า STEP_M=0.1)
    font=cv2.FONT_HERSHEY_SIMPLEX,
):
    overlay = np.zeros((BEV_H, BEV_W, 3), dtype=np.uint8)

    # เส้นแนวตั้ง
    for i, x in enumerate(range(0, BEV_W, STEP_PX)):
        cv2.line(overlay, (x, 0), (x, BEV_H - 1), (200, 200, 200), 1)
        if i % label_every == 0:
            dist_m = x / PX_PER_M
            txt = f"{dist_m:.1f}m"
            # outline (ดำ) + ตัวหนังสือ (แดง) ให้อ่านง่าย
            cv2.putText(
                overlay,
                txt,
                (min(x + 2, BEV_W - 60), 16),
                font,
                0.4,
                (0, 0, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                overlay,
                txt,
                (min(x + 2, BEV_W - 60), 16),
                font,
                0.3,
                (0, 0, 255),
                1,
                cv2.LINE_AA,
            )

    # เส้นแนวนอน
    for j, y in enumerate(range(0, BEV_H, STEP_PX)):
        cv2.line(overlay, (0, y), (BEV_W - 1, y), (200, 200, 200), 1)
        if j % label_every == 0:
            dist_m = y / PX_PER_M
            txt = f"{dist_m:.1f}m"
            y_lbl = max(12, y - 4)  # กันตัวหนังสือหลุดขอบบน
            cv2.putText(overlay, txt, (2, y_lbl), font, 0.4, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(
                overlay, txt, (2, y_lbl), font, 0.4, (0, 0, 255), 1, cv2.LINE_AA
            )

    # เน้นแกน (0 m) ให้หนาขึ้น
    cv2.line(overlay, (0, 0), (BEV_W - 1, 0), (120, 120, 120), 2)
    cv2.line(overlay, (0, 0), (0, BEV_H - 1), (120, 120, 120), 2)

    return overlay


def draw_line(image):
    canvas = image.copy()

    # วาดจุด + label
    for i, (px, py) in enumerate(clicks, start=1):
        cv2.circle(canvas, (px, py), 6, (0, 0, 255), -1)
        cv2.putText(
            canvas, f"P{i}", (px + 8, py - 8), FONT, 0.6, (0, 0, 255), 2, cv2.LINE_AA
        )

    # วาดโพลิกอนเชื่อมจุด (ตามลำดับที่คลิก)
    if len(clicks) >= 2:
        cv2.polylines(
            canvas,
            [np.array(clicks, dtype=np.int32)],
            isClosed=False,
            color=(0, 255, 255),
            thickness=2,
            lineType=cv2.LINE_AA,
        )
    return canvas


# cap = cv2.VideoCapture("./footage.mp4")
cap = cv2.VideoCapture(0)

cv2.namedWindow("Original", cv2.WINDOW_NORMAL)
# cv2.namedWindow("Original")
cv2.setMouseCallback("Original", on_mouse)
while True:
    ret, image = cap.read()
    scale = 0.8  # ย่อ 50%
    if not ret:
        # ถ้าอ่านเฟรมไม่สำเร็จ (วิดีโอจบแล้ว)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # ย้อนกลับไปเฟรมแรก
        continue

    # image = cv2.resize(image, (0, 0), fx=scale, fy=scale)
    out = draw_overlay(image, clicks)
    cv2.imshow("Original", out)
    k = cv2.waitKey(1) & 0xFF
    if k == ord("q") and len(clicks) == 4:
        break

cv2.destroyAllWindows()
pts = np.array(clicks, dtype="float32")
rect = order_points(pts)
dst_bev = np.array(
    [[0, 0], [BEV_W - 1, 0], [BEV_W - 1, BEV_H - 1], [0, BEV_H - 1]],
    dtype="float32",
)
grid_overlay = build_bev_grid_overlay(BEV_W, BEV_H, STEP_PX, PX_PER_M, label_every=1)
H = cv2.getPerspectiveTransform(rect, dst_bev)
cv2.namedWindow("Original Cap", cv2.WINDOW_NORMAL)
# cv2.namedWindow("Warped", cv2.WINDOW_NORMAL)

while True:
    ret, image = cap.read()
    if not ret:
        # ถ้าอ่านเฟรมไม่สำเร็จ (วิดีโอจบแล้ว)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # ย้อนกลับไปเฟรมแรก
        continue
    # image = cv2.resize(image, (0, 0), fx=scale, fy=scale)
    warped = warp_to_bev(image)
    _, _, warped = ArucoDetector(warped)
    # scale_warped = 0.7
    # warped = cv2.resize(warped, (0, 0), fx=scale_warped, fy=scale_warped)

    cv2.imshow("Warped", warped)
    canvas = draw_line(image)
    cv2.imshow("Original Cap", canvas)
    k = cv2.waitKey(1) & 0xFF
    if k == ord("q"):
        break

# show the original and warped images
cv2.destroyAllWindows()
cap.release()
