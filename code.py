import cv2
import pytesseract
import numpy as np
import imutils

# 👇 CHANGE THIS PATH TO YOUR TESSERACT LOCATION

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

VIDEO_PATH = "video.mp4"

cap = cv2.VideoCapture(VIDEO_PATH)
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # --- Resize for better processing ---
    frame = imutils.resize(frame, width=800)

    # --- Convert to grayscale ---
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # --- Reduce noise ---
    gray = cv2.bilateralFilter(gray, 11, 17, 17)

    # --- Increase contrast (VERY IMPORTANT) ---
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)

    # --- Edge detection ---
    edged = cv2.Canny(gray, 30, 200)

    # --- Find contours ---
    cnts, _ = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:10]

    plate = None

    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.018 * peri, True)

        if len(approx) == 4:  # Plate is rectangular
            x, y, w, h = cv2.boundingRect(approx)
            plate = gray[y:y+h, x:x+w]
            break

    if plate is not None:
        # --- Threshold plate ---
        _, plate = cv2.threshold(plate, 0, 255,
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # --- OCR ---
        text = pytesseract.image_to_string(
            plate,
            config="--psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        )

        text = text.strip()

        if len(text) >= 5:
            print(f"[FRAME {frame_count}] Possible Plate: {text}")
            cv2.imshow("Plate", plate)
            cv2.waitKey(0)

    # Show processing (optional)
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
