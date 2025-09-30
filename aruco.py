import cv2
import cv2.aruco as aruco

# --- Load camera (0 = default webcam) ---
cap = cv2.VideoCapture(0)

# --- Define the dictionary of markers ---
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
parameters = aruco.DetectorParameters()

# --- Main loop ---
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale
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
                marker_id,
                (c[0], c[1]),  # position
                cv2.FONT_HERSHEY_SIMPLEX,  # font
                2.0,  # font scale (bigger = larger text)
                (0, 0, 255),  # color (red)
                3,  # thickness
                cv2.LINE_AA,
            )
    cv2.imshow("ArUco Detection", frame)

    # Exit with 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
