import torch
import cv2

# Load YOLOv5s (versi kecil, cepat)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Buka webcam (0 = default webcam)
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Gagal membaca frame dari webcam.")
        break

    # Deteksi objek dengan YOLOv5
    results = model(frame)

    # Render hasil (bounding box, label, confidence)
    annotated_frame = results.render()[0]

    # Tampilkan hasil
    cv2.imshow("YOLOv5 Real-Time Detection", annotated_frame)

    # Tekan 'q' untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
