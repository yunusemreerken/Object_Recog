import cv2
from ultralytics import YOLO

# YOLOv8 modelini yükle (Önceden eğitilmiş model)
model = YOLO("yolov8n.pt")

# Kamerayı başlat
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Model ile tahmin yap
    results = model(frame)

    # Sonuçları ekrana çiz
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Koordinatları al
            label = model.names[int(box.cls[0])]  # Etiket al
            score = box.conf[0].item()  # Güven skoru

            # Eğer etiket "cell phone", "clock" veya "key" ise çiz
            if label in ["cell phone", "clock", "key"]:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} {score:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("YOLOv8 Nesne Tanıma", frame)

    # Çıkış için 'q' tuşuna bas
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
