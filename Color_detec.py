import cv2
import numpy as np


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
    global target_bgr, clicked_hsv
    if event == cv2.EVENT_LBUTTONDOWN and param["frame"] is not None:
        # ดึงสี BGR จากจุดคลิก แล้วตั้งเป็นเป้าหมายใหม่
        bgr = param["frame"][y, x, :].astype(np.uint8)
        target_bgr = bgr
        hsv_pix = cv2.cvtColor(np.array([[bgr]], dtype=np.uint8), cv2.COLOR_BGR2HSV)[
            0, 0
        ]
        clicked_hsv = hsv_pix
        print(
            f"Picked BGR={tuple(int(v) for v in bgr)}, HSV={tuple(int(v) for v in hsv_pix)}"
        )


cap = cv2.VideoCapture(1)
target_bgr = np.array([61, 68, 158], dtype=np.uint8)  # (B,G,R)
tol = (10, 60, 60)  # ปรับได้ตามต้องการ
ctx = {"frame": None}
cv2.namedWindow("Colour Original")
cv2.setMouseCallback("Colour Original", on_mouse_colour, ctx)
clicked_hsv = None

while True:
    ret, imageFrame = cap.read()
    if not ret:
        break
    ctx["frame"] = imageFrame
    hsvFrame = cv2.cvtColor(imageFrame, cv2.COLOR_BGR2HSV)

    lower, upper = bgr_to_hsv_range(target_bgr, tol)
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
    min_area = 500  # ปรับตามสเกลภาพ/ขนาดวัตถุ
    mask_clean = np.zeros_like(mask)
    for i in range(1, num):  # ข้ามฉลาก 0 = พื้นหลัง
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            mask_clean[labels == i] = 255

    mask = mask_clean

    # 4) (ทางเลือก) ทำ median blur เล็กน้อยให้ขอบเรียบ
    # mask = cv2.medianBlur(mask, 5)

    # 5) ต่อไปเหมือนเดิม แต่ใช้ RETR_EXTERNAL เพื่อตัดรู/เส้นใน
    res = cv2.bitwise_and(imageFrame, imageFrame, mask=mask)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 300:
            x, y, w, h = cv2.boundingRect(contour)

            # (ทางเลือก) กรองความเพี้ยนรูปร่างเพิ่มเติม
            # aspect = w / float(h)
            # if aspect < 0.3 or aspect > 3.5: continue  # ตัดวัตถุยาว/แคบผิดปกติ
            # solidity = area / cv2.contourArea(cv2.convexHull(contour))
            # if solidity < 0.8: continue  # ตัดรูปร่างเว้าหนัก ๆ

            cv2.rectangle(imageFrame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(
                imageFrame,
                "Target Colour",
                (x, y - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 0, 0),
                2,
            )

    # แสดงผล
    cv2.imshow("Colour Original", imageFrame)
    cv2.imshow("Target Mask", mask)
    cv2.imshow("Target Detection", res)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("=") or key == ord("+"):
        # เพิ่ม tolerance
        th, ts, tv = tol
        tol = (min(th + 1, 90), min(ts + 5, 127), min(tv + 5, 127))
    elif key == ord("-") or key == ord("_"):
        # ลด tolerance
        th, ts, tv = tol
        tol = (max(th - 1, 0), max(ts - 5, 0), max(tv - 5, 0))

cap.release()
cv2.destroyAllWindows()
