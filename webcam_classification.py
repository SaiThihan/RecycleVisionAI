#webcam_classification.py
import cv2
import torch
import numpy as np
from torchvision import transforms
from model import load_model

import sys
sys.path.insert(0, './yolov5')
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.torch_utils import select_device
from yolov5.utils.general import non_max_suppression, scale_boxes

model = load_model('best_model.pth')
print("Model loaded successfully!")

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

device = select_device('')
yolo_model = DetectMultiBackend('./yolov5/yolov5s.pt', device=device)
yolo_model.eval()

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

labels = ['Non-Recyclable','Recyclable']

print("Starting webcam feed...")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame from webcam.")
        break

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    img = torch.from_numpy(img).to(device).float() / 255.0  # Convert to tensor and normalize
    img = img.permute(2, 0, 1).unsqueeze(0)  # Rearrange dimensions to [1, 3, H, W]

    with torch.no_grad():
        pred = yolo_model(img)
        pred = non_max_suppression(pred, 0.4, 0.5)

    for det in pred:
        if len(det):
            print(f"Detections: {det}")
            print(f"Shape of det: {det.shape}")

            if len(det.shape) > 1 and det.shape[1] >= 4:
                det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], frame.shape).round()

                for *xyxy, conf, cls in det:
                    x1, y1, x2, y2 = map(int, xyxy)

                    cropped_img = frame[y1:y2, x1:x2]

                    image = transform(cropped_img).unsqueeze(0)

                    with torch.no_grad():
                        outputs = model(image)
                        _, predicted = torch.max(outputs.data, 1)
                        label = labels[predicted.item()]

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow('Webcam Classification', frame)

    key = cv2.waitKey(1) 
    if key == ord('q') or cv2.getWindowProperty('Webcam Classification', cv2.WND_PROP_VISIBLE) < 1: 
        break

print("Releasing resources...")

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
print("Done.")
