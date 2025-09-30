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


# ---------- Demo ----------
def main():
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    # เริ่มด้วยสีฟ้า BGR
    target_bgr = np.array([61, 68, 158], dtype=np.uint8)  # (B,G,R)
    tol = (10, 60, 60)  # ปรับได้ตามต้องการ

    clicked_hsv = None

    def on_mouse(event, x, y, flags, param):
        nonlocal target_bgr, clicked_hsv
        if event == cv2.EVENT_LBUTTONDOWN and param["frame"] is not None:
            # ดึงสี BGR จากจุดคลิก แล้วตั้งเป็นเป้าหมายใหม่
            bgr = param["frame"][y, x, :].astype(np.uint8)
            target_bgr = bgr
            hsv_pix = cv2.cvtColor(
                np.array([[bgr]], dtype=np.uint8), cv2.COLOR_BGR2HSV
            )[0, 0]
            clicked_hsv = hsv_pix
            print(
                f"Picked BGR={tuple(int(v) for v in bgr)}, HSV={tuple(int(v) for v in hsv_pix)}"
            )

    ctx = {"frame": None}
    cv2.namedWindow("Frame")
    cv2.setMouseCallback("Frame", on_mouse, ctx)

    print("Instructions:")
    print(" - คลิกซ้ายบนภาพเพื่อเลือกสีจากพิกเซลนั้น")
    print(" - ปุ่ม +/- : ลด/เพิ่ม tolerance (ΔH,ΔS,ΔV) เป็น (10,60,60) +/- (1,5,5) ตามปุ่ม")
    print(" - ปุ่ม q   : ออกจากโปรแกรม")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        ctx["frame"] = frame
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lower, upper = bgr_to_hsv_range(target_bgr, tol=tol)
        mask = inrange_hsv_with_wrap(hsv, lower, upper)
        # mask = cv2.inRange(hsv, lower, upper)
        result = cv2.bitwise_and(frame, frame, mask=mask)

        # แสดงข้อมูลบนภาพ
        info = f"BGR target={tuple(int(v) for v in target_bgr)} | HSV lo={tuple(int(v) for v in lower)} hi={tuple(int(v) for v in upper)} | tol={tol}"
        if clicked_hsv is not None:
            info += f" | lastHSV={tuple(int(v) for v in clicked_hsv)}"
        disp = frame.copy()
        cv2.putText(
            disp,
            info,
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            disp,
            info,
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        cv2.imshow("Frame", disp)
        cv2.imshow("Mask", mask)
        cv2.imshow("Result", result)

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


if __name__ == "__main__":
    main()
