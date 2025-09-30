# import the necessary packages
import numpy as np
import cv2
import math
from ultralytics import YOLO
import torch
import time
import serial

# ==== Serial to boat MCU (Bluetooth) ====
PORT = "COM11"  # ปรับตามเครื่องคุณ (Linux/Mac ใช้ /dev/tty.*)
BAUD = 115200
TIMEOUT_S = 0.05

ser = serial.Serial(PORT, BAUD, timeout=TIMEOUT_S)
last_send = 0.0
SEND_HZ = 10.0  # จำกัดความถี่ส่งคำสั่ง ~10 Hz
MAX_PWM = 255  # ต้องไม่เกิน MAX_PWM ที่ฝั่ง MCU
FWD_PWM_BASE = 130  # ความแรงเดินหน้าเริ่มต้น
ROT_PWM_BASE = 120  # ความแรงหมุนเริ่มต้น
K_ROT = 1.0  # สเกล PWM ตามขนาด error องศา
ANGLE_TOL_DEG = 8.0  # ยอมให้เอียงได้เท่านี้แล้วเดินหน้า
STOP_DIST_M = 0.2  # ระยะหยุด (เมตร) เมื่อถึงเป้าหมาย


# 1) Pick device automatically: CUDA if available (your RTX 1650), else CPU
device = 0 if torch.cuda.is_available() else "cpu"

# 2) Load a small pretrained model (auto-downloads on first run)
model = YOLO("./best.pt")  # 'n' = nano, fastest
clicks = [(20, 454), (224, 129), (768, 119), (632, 485)]
boat_cx, boat_cy = -1, -1  # เริ่มต้นยังไม่รู้ตำแหน่งเรือ
H = None  # matrix แปลงมุมมอง
FONT = cv2.FONT_HERSHEY_SIMPLEX
REAL_W = 2.3  # กว้างแกน X (เมตร)
REAL_H = 2.2  # สูงแกน Y (เมตร)  (ตามความหมายที่คุณนิยาม)
PX_PER_M = 300  # 300 px/m -> BEV ~ 690x660 px
STEP_M = 0.1
STEP_PX = int(round(STEP_M * PX_PER_M))
BEV_W = int(round(REAL_W * PX_PER_M))
BEV_H = int(round(REAL_H * PX_PER_M))

# --- state สำหรับเก็บ "จุดหมาย" ---
# target_bev_px = None  # (xw, yw) บน (px)
# target_bev_m = None  # (Xm, Ym) บน BEV (m)BEV
target_bev_px = (662.9, 509.9)  # (xw, yw) บน BEV (px)
target_bev_m = (2.21, 1.70)  # (Xm, Ym) บน BEV (m)

TURN_INVERTED = False  # ถ้าหมุนกลับด้านอยู่ ให้ตั้ง True


def clamp_pwm(v):
    return max(0, min(int(round(v)), MAX_PWM))


def rate_limited_send(cmd):
    global last_send
    t = time.time()
    if t - last_send >= (1.0 / SEND_HZ):
        ser.write((cmd + "\n").encode("utf-8"))
        last_send = t


def send_stop():
    print("STOP")
    rate_limited_send("S-0")


def send_forward(p):
    print(f"FWD {p}")
    rate_limited_send(f"F-{clamp_pwm(p)}")


def send_left(p):
    print(f"LEFT {p}")
    rate_limited_send(f"L-{clamp_pwm(p)}")  # หมุนซ้ายอยู่กับที่


def send_right(p):
    print(f"RIGHT {p}")
    rate_limited_send(f"R-{clamp_pwm(p)}")  # หมุนขวาอยู่กับที่


def ang_wrap_deg(a):
    # ให้อยู่ช่วง (-180, 180]
    while a <= -180.0:
        a += 360.0
    while a > 180.0:
        a -= 360.0
    return a


def bearing_deg(src_px, dst_px):
    # ทั้ง src,dst เป็นพิกัด BEV (px) แกน y ลง, ใช้ atan2(dy,dx) แบบเดียวกับ yaw ที่คุณคำนวณ
    dx = dst_px[0] - src_px[0]
    dy = dst_px[1] - src_px[1]
    return math.degrees(math.atan2(dy, dx))


def dist_m_px(a_px, b_px, px_per_m=PX_PER_M):
    dx = (a_px[0] - b_px[0]) / px_per_m
    dy = (a_px[1] - b_px[1]) / px_per_m
    return (dx * dx + dy * dy) ** 0.5


def autopilot_step(head_xy_bev, tail_xy_bev, yaw_deg, target_bev_px):
    """
    head_xy_bev, tail_xy_bev: tuple(int,int) พิกัด BEV (px)
    yaw_deg: องศาหัวเรือ (จาก tail -> head) เช่น yaw_ema ของคุณ
    target_bev_px: จุดหมาย sticky target บน BEV (px) หรือ None
    """
    # ถ้าไม่มีข้อมูลครบนิ่ง → หยุด
    if (
        target_bev_px is None
        or head_xy_bev is None
        or tail_xy_bev is None
        or yaw_deg is None
    ):
        send_stop()

        return

    # ระยะถึงเป้าหมาย
    dist = dist_m_px(head_xy_bev, target_bev_px, PX_PER_M)
    if dist < STOP_DIST_M:
        print(f"At target (dist={dist:.2f}m) → STOP")
        send_stop()
        time.sleep(0.1)
        rate_limited_send("SOUND-0")
        time.sleep(0.1)
        # rate_limited_send("DROP-0")
        return

    # มุมที่ควรเผชิญหน้า (bearing) และ error
    bearing = bearing_deg(tail_xy_bev, target_bev_px)
    err = ang_wrap_deg(bearing - yaw_deg)

    # ถ้าเอียงมาก → หมุนอยู่กับที่
    if abs(err) > ANGLE_TOL_DEG:
        pwm = ROT_PWM_BASE + K_ROT * abs(err)  # เพิ่มแรงตาม error
        turn_left = err > 0  # ค่านี้คือทิศทางตามคณิตศาสตร์ปัจจุบัน

        if TURN_INVERTED:
            turn_left = not turn_left

        if turn_left:
            send_left(pwm)
        else:
            send_right(pwm)
        return

    # ถ้าเผชิญหน้าพอแล้ว → เดินหน้า
    send_forward(FWD_PWM_BASE)


def duck_detector(frame, H, BEV_W, BEV_H, px_per_m=PX_PER_M, cls_whitelist=None):
    """
    ตรวจ 'เป็ด' บน ORIGINAL → เลือก 1 กล่องที่ conf สูงสุด
    - ถ้าเจอ: คำนวณ centroid → project เป็น BEV → อัปเดต target_* (sticky)
    - ถ้าไม่เจอ: คง target_* เดิมไว้ (ไม่อัปเดต)
    คืนค่า: annotated_original, duck_xy_img, duck_xy_bev(หรือ None), duck_xy_m(หรือ None)
    """
    global target_bev_px, target_bev_m

    results = model.predict(
        source=frame, conf=0.40, imgsz=640, device=device, verbose=False
    )
    r = results[0]
    annotated = r.plot()

    # ไม่มี detection → ไม่รีเซ็ต เป้าหมายเดิม
    if r.boxes is None or r.boxes.shape[0] == 0:
        return annotated, None, None, None

    # กรองด้วย whitelist ถ้ามี
    keep = list(range(len(r.boxes)))
    if cls_whitelist is not None:
        cls = r.boxes.cls.int().cpu().tolist()
        keep = [i for i, c in enumerate(cls) if c in cls_whitelist]
        if not keep:
            return annotated, None, None, None

    # เลือกกล่อง conf สูงสุด
    conf = r.boxes.conf.cpu().numpy()
    best = max(keep, key=lambda i: conf[i])

    # centroid บน ORIGINAL
    x1, y1, x2, y2 = r.boxes.xyxy[best].cpu().numpy().tolist()
    cx = float((x1 + x2) * 0.5)
    cy = float((y1 + y2) * 0.5)
    cv2.circle(annotated, (int(round(cx)), int(round(cy))), 6, (0, 255, 255), -1)
    cv2.putText(
        annotated,
        "DUCK",
        (int(x1), max(0, int(y1) - 6)),
        FONT,
        0.6,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )

    # project → BEV
    if H is None:
        return annotated, (cx, cy), None, None

    pt = np.array([[[cx, cy]]], dtype=np.float32)
    xw, yw = cv2.perspectiveTransform(pt, H)[0, 0]
    # ในกรอบ BEV ไหม
    if not (0.0 <= xw < BEV_W and 0.0 <= yw < BEV_H):
        # ถ้ากล่องหลุดขอบ BEV ให้ “ไม่อัปเดต” เป้าหมาย (ยังใช้ค่าก่อนหน้า)
        return annotated, (cx, cy), None, None

    Xm, Ym = xw / px_per_m, yw / px_per_m

    # --- อัปเดต Sticky Target ---
    target_bev_px = (xw, yw)
    target_bev_m = (Xm, Ym)

    return annotated, (cx, cy), (xw, yw), (Xm, Ym)


def draw_sticky_target_on_bev(bev_img):
    """วาดจุดหมายล่าสุด (ถ้ามี) ลงบนภาพ BEV โดยไม่วาด trail"""
    if target_bev_px is None or target_bev_m is None:
        return bev_img
    xw, yw = target_bev_px
    Xm, Ym = target_bev_m
    out = bev_img.copy()
    cv2.drawMarker(
        out,
        (int(round(xw)), int(round(yw))),
        (0, 0, 255),
        markerType=cv2.MARKER_TILTED_CROSS,
        markerSize=20,
        thickness=3,
    )
    cv2.putText(
        out,
        f"Tgt BEV({xw:.1f},{yw:.1f})  W({Xm:.2f},{Ym:.2f})m",
        (int(round(xw)) + 10, max(18, int(round(yw)) - 10)),
        FONT,
        0.6,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )
    return out


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
    bev_image = image.copy()
    bev = cv2.warpPerspective(bev_image, H, (BEV_W, BEV_H))
    if grid_overlay is not None:
        return cv2.addWeighted(bev, 1.0, grid_overlay, 0.35, 0)
    else:
        return bev


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
    Hinv = np.linalg.inv(H)
    bev_box = np.float32(
        [[0, 0], [BEV_W - 1, 0], [BEV_W - 1, BEV_H - 1], [0, BEV_H - 1]]
    ).reshape(-1, 1, 2)
    box_on_img = cv2.perspectiveTransform(bev_box, Hinv).astype(int)
    cv2.polylines(canvas, [box_on_img.reshape(-1, 2)], True, (0, 200, 0), 2)
    # วาดจุด + label
    # for i, (px, py) in enumerate(clicks, start=1):
    #     cv2.circle(canvas, (px, py), 6, (0, 0, 255), -1)
    #     cv2.putText(
    #         canvas, f"P{i}", (px + 8, py - 8), FONT, 0.6, (0, 0, 255), 2, cv2.LINE_AA
    #     )

    # # วาดโพลิกอนเชื่อมจุด (ตามลำดับที่คลิก)
    # if len(clicks) >= 2:
    #     cv2.polylines(
    #         canvas,
    #         [np.array(clicks, dtype=np.int32)],
    #         isClosed=False,
    #         color=(0, 255, 255),
    #         thickness=2,
    #         lineType=cv2.LINE_AA,
    #     )
    return canvas


def bgr_to_hsv_range(bgr_color, tol=(10, 60, 60)):
    """
    แปลง BGR -> HSV แล้วคืนค่า (lower, upper) เป็น HSV สำหรับ inRange(hsv, lower, upper)
    ป้องกันปัญหา uint8 wrap ด้วยการแปลงเป็น int ก่อนคำนวณ
    """
    color_bgr = np.array([[bgr_color]], dtype=np.uint8)  # (1,1,3)
    hsv_color = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2HSV)[0, 0]

    # cast เป็น int ก่อนบวก/ลบ
    h = int(hsv_color[0])
    s = int(hsv_color[1])
    v = int(hsv_color[2])

    th, ts, tv = map(int, tol)

    lo = np.array([max(h - th, 0), max(s - ts, 0), max(v - tv, 0)], dtype=np.uint8)
    hi = np.array(
        [min(h + th, 179), min(s + ts, 255), min(v + tv, 255)], dtype=np.uint8
    )
    return lo, hi


def inrange_hsv_with_wrap(hsv_img, lower, upper):
    """
    ทำ inRange สำหรับ HSV โดยรองรับกรณี H ข้ามศูนย์ (เช่นสีแดง)
    ถ้า H lower <= H upper ใช้ช่วงเดียว
    ถ้า H lower > H upper แยกเป็น 2 ช่วงแล้ว OR กัน
    """
    h_lo, h_hi = int(lower[0]), int(upper[0])
    if h_lo <= h_hi:
        return cv2.inRange(hsv_img, lower, upper)
    # wrap-around: [0..h_hi] U [h_lo..179]
    lo1 = np.array([0, lower[1], lower[2]], dtype=np.uint8)
    hi1 = np.array([h_hi, upper[1], upper[2]], dtype=np.uint8)
    lo2 = np.array([h_lo, lower[1], lower[2]], dtype=np.uint8)
    hi2 = np.array([179, upper[1], upper[2]], dtype=np.uint8)
    return cv2.bitwise_or(
        cv2.inRange(hsv_img, lo1, hi1), cv2.inRange(hsv_img, lo2, hi2)
    )


def on_mouse_colour(event, x, y, flags, param):
    global head_colour, clicked_hsv, target_bev_px, target_bev_m

    # ซ้ายคลิก: เลือกสีสำหรับ marker (ของเดิม)
    if event == cv2.EVENT_LBUTTONDOWN and param["frame"] is not None:
        bgr = param["frame"][y, x, :].astype(np.uint8)
        head_colour = bgr
        hsv_pix = cv2.cvtColor(np.array([[bgr]], dtype=np.uint8), cv2.COLOR_BGR2HSV)[
            0, 0
        ]
        clicked_hsv = hsv_pix
        print(
            f"Picked BGR={tuple(int(v) for v in bgr)}, HSV={tuple(int(v) for v in hsv_pix)}"
        )

    # ขวาคลิก: กำหนด "จุดหมาย" จากจุดที่คลิก (Original -> BEV)
    elif event == cv2.EVENT_RBUTTONDOWN:
        if H is None:
            print("WARN: Homography H ยังไม่พร้อม ไม่สามารถตั้งจุดหมายได้")
            return

        # ใช้ helper ที่คุณมีอยู่แล้วจะสะดวก/ปลอดภัยกว่า
        xw, yw, inside = img_to_bev_point(x, y, H, BEV_W, BEV_H)
        if not inside:
            print(f"WARN: จุดที่คลิก ({x},{y}) project แล้วอยู่นอก BEV: ({xw:.1f},{yw:.1f})")
            return

        Xm, Ym = xw / PX_PER_M, yw / PX_PER_M
        target_bev_px = (xw, yw)
        target_bev_m = (Xm, Ym)

        print(f"Set TARGET from click: BEV({xw:.1f},{yw:.1f})  W({Xm:.2f},{Ym:.2f}) m")


def largest_centroid_from_mask(mask, min_area=400):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    bestA = 0
    for c in cnts:
        a = cv2.contourArea(c)
        if a < min_area:
            continue
        if a > bestA:
            M = cv2.moments(c)
            if M["m00"] > 1e-6:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                best = (cx, cy, c)
                bestA = a
    return best  # (cx,cy, contour) หรือ None


def compute_yaw_deg(head, tail):
    # head, tail เป็น (x,y) ใน BEV พิกเซล (แกน x ขวา+, y ลง+)
    dx = head[0] - tail[0]
    dy = head[1] - tail[1]
    return math.degrees(math.atan2(dy, dx))  # -180..180


def dist_m(a, b, px_per_m=PX_PER_M):
    dx = (a[0] - b[0]) / px_per_m
    dy = (a[1] - b[1]) / px_per_m
    return (dx * dx + dy * dy) ** 0.5


def process_frame(image, BEV, BEV_W, BEV_H):
    global yaw_ema

    # 1) warp to BEV (ถ้า H ยังไม่พร้อม ให้คืน image เดิม)
    # warped = image
    if BEV is None:
        warped = image
    else:
        warped = BEV

    lower_head, upper_head = bgr_to_hsv_range(head_colour, tol)
    lower_tail, upper_tail = bgr_to_hsv_range(tail_colour, tol)

    # 2) detect markers บน BEV
    mask_head = mask_hsv(warped, lower_head, upper_head)  # หัวเรือ (แดง)
    mask_tail = mask_hsv(warped, lower_tail, upper_tail)  # ท้ายเรือ (เขียว)

    head = largest_centroid_from_mask(mask_head, min_area=600)
    tail = largest_centroid_from_mask(mask_tail, min_area=600)

    # วาดช่วยบน BEV
    ColourframeBEV = warped.copy()
    if head:
        cx, cy, c = head
        cv2.circle(ColourframeBEV, (cx, cy), 7, (0, 0, 255), -1)
        cv2.drawContours(ColourframeBEV, [c], -1, (0, 0, 255), 2)
        cv2.putText(
            ColourframeBEV,
            "HEAD",
            (cx + 8, cy - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
        )
    if tail:
        cx, cy, c = tail
        cv2.circle(ColourframeBEV, (cx, cy), 7, (0, 255, 0), -1)
        cv2.drawContours(ColourframeBEV, [c], -1, (0, 255, 0), 2)
        cv2.putText(
            ColourframeBEV,
            "TAIL",
            (cx + 8, cy - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

    yaw_txt = "yaw: --"
    dist_txt = "len: -- m"

    if head and tail:
        head_xy = (head[0], head[1])
        tail_xy = (tail[0], tail[1])

        # 3) คำนวณ yaw และความยาวหัว-ท้าย (เมตร)
        yaw = compute_yaw_deg(head_xy, tail_xy)
        if yaw_ema is None:
            yaw_ema = yaw
        else:
            yaw_ema = (1 - ALPHA) * yaw_ema + ALPHA * yaw  # smoothing

        Lm = dist_m(head_xy, tail_xy, PX_PER_M)
        yaw_txt = f"yaw: {yaw_ema:6.1f} deg"
        dist_txt = f"len: {Lm:.2f} m"

        # 4) วาดเวกเตอร์ทิศจาก tail -> head
        cv2.arrowedLine(
            ColourframeBEV, tail_xy, head_xy, (255, 255, 0), 3, tipLength=0.25
        )

    # HUD
    cv2.putText(
        ColourframeBEV,
        yaw_txt,
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 0),
        2,
    )
    cv2.putText(
        ColourframeBEV,
        dist_txt,
        (10, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 0),
        2,
    )

    # 5) ถ้าต้องการวาดบน Original ด้วย: project ย้อน (ถ้าจำเป็น)
    # หรือจะ detect บน Original แยกก็ได้
    ColourframeORI = image.copy()
    ctx["frame"] = ColourframeORI
    head_xy_out = (head[0], head[1]) if head else None
    tail_xy_out = (tail[0], tail[1]) if tail else None
    return (
        ColourframeORI,
        ColourframeBEV,
        mask_head,
        mask_tail,
        head_xy_out,
        tail_xy_out,
    )

    return ColourframeORI, ColourframeBEV, mask_head, mask_tail


def mask_hsv(bgr, lo, hi):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lo = np.array(lo, np.uint8)
    hi = np.array(hi, np.uint8)
    m = cv2.inRange(hsv, lo, hi)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, K3, iterations=2)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, K5, iterations=2)
    return m


def draw_colour_mask(frame, frame_bev):
    global ctx
    Colourframe = frame.copy()
    ColourframeBEV = frame_bev.copy()
    # cv2.imshow("Colour Original", Colourframe)
    ctx["frame"] = Colourframe
    hsvFrame = cv2.cvtColor(Colourframe, cv2.COLOR_BGR2HSV)

    lower, upper = bgr_to_hsv_range(target_bgr, tol)
    # mask = inrange_hsv_with_wrap(hsvFrame, lower, upper)
    # # ขยาย (dilation) เพื่อลด noise
    # kernal = np.ones((5, 5), "uint8")
    # mask = cv2.dilate(mask, kernal)

    s_floor, v_floor = 60, 60
    lower[1] = max(int(lower[1]), s_floor)
    lower[2] = max(int(lower[2]), v_floor)
    mask = inrange_hsv_with_wrap(hsvFrame, lower, upper)

    # 2) Morphology: เปิดแล้วปิด (Open -> Close)
    kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel3, iterations=1)  # ตัดจุดเล็ก ๆ
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, kernel5, iterations=1
    )  # อุดรูใน object

    # 3) ลบคอมโพเนนต์เล็ก ๆ ด้วย connected components
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    min_area = 20  # ปรับตามสเกลภาพ/ขนาดวัตถุ
    mask_clean = np.zeros_like(mask)
    for i in range(1, num):  # ข้ามฉลาก 0 = พื้นหลัง
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            mask_clean[labels == i] = 255

    mask = mask_clean

    # 4) (ทางเลือก) ทำ median blur เล็กน้อยให้ขอบเรียบ
    mask = cv2.medianBlur(mask, 5)

    # 5) ต่อไปเหมือนเดิม แต่ใช้ RETR_EXTERNAL เพื่อตัดรู/เส้นใน
    res = cv2.bitwise_and(Colourframe, Colourframe, mask=mask)
    # print(rect)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 300:
            x, y, w, h = cv2.boundingRect(contour)
            # --- centroid ด้วย moments ---
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:  # fallback: ศูนย์กลางกรอบสี่เหลี่ยม
                cx = x + w // 2
                cy = y + h // 2
            global boat_cx, boat_cy
            pt = np.array([[[cx, cy]]], dtype=np.float32)  # centroid ใน Original
            if not inside_ref_quad(cx, cy, rect):  # อยู่นอกพื้นที่ที่ H เชื่อถือได้ → ข้าม/เตือน
                cv2.putText(
                    Colourframe,
                    "out of ref",
                    (x, y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    2,
                )
                continue
            boat_cx, boat_cy = cx, cy  # วาดกรอบ + centroid

            pt_bev = cv2.perspectiveTransform(pt, H)  # → พิกัดใน BEV
            xw, yw = pt_bev[0, 0]  # จุดใน BEV (หน่วยพิกเซล)
            Xm, Ym = xw / PX_PER_M, yw / PX_PER_M  # แปลงเป็นเมตร
            # print("centroid:", cx, cy)
            # # print("bev:", xw, yw, " inside:", 0 <= xw < BEV_W and 0 <= yw < BEV_H)
            # วาดผลบนภาพ Original
            cv2.rectangle(Colourframe, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.drawMarker(
                Colourframe,
                (cx, cy),
                (0, 255, 255),
                markerType=cv2.MARKER_CROSS,
                markerSize=14,
                thickness=2,
            )

            # แสดงพิกัด BEV (px) + World (m) แบบย่อ
            cv2.putText(
                Colourframe,
                f"BEV({xw:.1f},{yw:.1f})  W({Xm:.2f}m,{Ym:.2f}m)",
                (cx + 8, cy - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                2,
            )

            cv2.putText(
                Colourframe,
                "Target Colour",
                (x, y - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 0, 0),
                2,
            )

            cv2.drawMarker(
                ColourframeBEV,
                (int(round(xw)), int(round(yw))),
                (0, 255, 255),
                markerType=cv2.MARKER_CROSS,
                markerSize=14,
                thickness=2,
            )
            lbl_pos = (int(round(xw)) + 8, int(round(yw)) - 8)
            print(f"BEV({xw:.1f},{yw:.1f})  W({Xm:.2f}m,{Ym:.2f}m)")
            cv2.putText(
                ColourframeBEV,
                f"BEV({xw:.1f},{yw:.1f})  W({Xm:.2f}m,{Ym:.2f}m)",
                lbl_pos,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                2,
            )
    return Colourframe, mask, res, ColourframeBEV


# cap = cv2.VideoCapture("./footage.mp4")
# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture(1)

cv2.namedWindow("Original", cv2.WINDOW_NORMAL)
cv2.namedWindow("Original")
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
    if k == ord("c"):
        clicks = []
    if k == ord("q") and len(clicks) == 4:
        break

cv2.destroyAllWindows()
# print(clicks)
pts = np.array(clicks, dtype="float32")
rect = order_points(pts)
dst_bev = np.array(
    [[0, 0], [BEV_W - 1, 0], [BEV_W - 1, BEV_H - 1], [0, BEV_H - 1]],
    dtype="float32",
)
grid_overlay = build_bev_grid_overlay(BEV_W, BEV_H, STEP_PX, PX_PER_M, label_every=1)
H = cv2.getPerspectiveTransform(rect, dst_bev)

target_bgr = np.array([171, 100, 32], dtype=np.uint8)  # (B,G,R)
tol = (10, 60, 60)  # ปรับได้ตามต้องการ
# cv2.namedWindow("Original Cap", cv2.WINDOW_NORMAL)
cv2.namedWindow("Original Cap")
# cv2.namedWindow("Warped", cv2.WINDOW_NORMAL)

ctx = {"frame": None}
# cv2.namedWindow("Colour Original")
cv2.setMouseCallback("Original Cap", on_mouse_colour, ctx)
clicked_hsv = None


head_colour = np.array([79, 184, 165], dtype=np.uint8)
tail_colour = np.array([61, 54, 123], dtype=np.uint8)
K3 = np.ones((3, 3), np.uint8)
K5 = np.ones((5, 5), np.uint8)

yaw_ema = None
ALPHA = 0.2  # smoothing


def inside_ref_quad(cx, cy, rect4):
    poly = rect4.reshape(-1, 1, 2).astype(np.float32)  # rect4 = (tl,tr,br,bl)
    return cv2.pointPolygonTest(poly, (float(cx), float(cy)), False) >= 0


def img_to_bev_point(cx, cy, H, bev_w, bev_h):
    pt = np.array([[[float(cx), float(cy)]]], dtype=np.float32)
    out = cv2.perspectiveTransform(pt, H)[0, 0]
    xw, yw = float(out[0]), float(out[1])
    inside = (0.0 <= xw < bev_w) and (0.0 <= yw < bev_h)
    return xw, yw, inside


while True:
    ret, image = cap.read()
    if not ret:
        # ถ้าอ่านเฟรมไม่สำเร็จ (วิดีโอจบแล้ว)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # ย้อนกลับไปเฟรมแรก
        continue
    # image, mask, res = draw_colour_mask(image)
    # image = cv2.resize(image, (0, 0), fx=scale, fy=scale)
    # cv2.imshow("debug", image)
    warped = warp_to_bev(image)
    # overlay_duck_on_bev(warped, dets, H, BEV_H, BEV_H)
    # annot, duck_xy_img, duck_xy_bev, duck_xy_m = duck_detector(
    #     image,
    #     H,
    #     BEV_W,
    #     BEV_H,
    #     PX_PER_M,
    #     cls_whitelist=None,  # ระบุ class id ของ "เป็ด" ได้ถ้าต้อง
    # )
    warped = draw_sticky_target_on_bev(warped)
    # cv2.imshow("bev_out", bev_out)
    # _, _, warped = ArucoDetector(warped)
    # scale_warped = 0.7
    # warped = cv2.resize(warped, (0, 0), fx=scale_warped, fy=scale_warped)

    # color, mask, res, warped = draw_colour_mask(image, warped)
    # color, warped, mask_head, mask_tail = process_frame(image, warped, BEV_W, BEV_H)

    color, warped, mask_head, mask_tail, head_xy_bev, tail_xy_bev = process_frame(
        image, warped, BEV_W, BEV_H
    )
    # ใช้ yaw_ema ที่คุณคำนวณอยู่แล้วใน process_frame
    curr_yaw = yaw_ema  # ถ้าอยากใช้แบบดิบ ให้ใช้ 'yaw' ที่คำนวณก่อน EMA

    # สั่งควบคุมอัตโนมัติ: หมุนอยู่กับที่ให้หัวหันไปหาเป้าหมาย แล้วเดินหน้า
    autopilot_step(head_xy_bev, tail_xy_bev, curr_yaw, target_bev_px)

    cv2.imshow("Warped", warped)
    # canvas = draw_line(color)
    cv2.polylines(color, [rect.astype(np.int32)], True, (0, 200, 0), 2, cv2.LINE_AA)
    cv2.imshow("Original Cap", color)
    # cv2.imshow("Target Mask", mask)
    # cv2.imshow("Target Detection", res)
    k = cv2.waitKey(1) & 0xFF
    if k == ord("q"):
        break
    elif k == ord("t"):
        tail_colour = head_colour.copy()
        print(f"Set tail colour to {tail_colour}")

# show the original and warped images
cv2.destroyAllWindows()
cap.release()
