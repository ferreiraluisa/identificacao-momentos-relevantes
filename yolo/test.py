import cv2
from ultralytics import YOLO
import json

# check torch using gpu, device
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

model = YOLO('runs/detect/train/weights/best.pt')

image = cv2.imread("../image/img3.png")
results = model(image, verbose=False)

for i, result in enumerate(results):
    print(result['box'])

# cap = cv2.VideoCapture("../videos/C5_XkThyaWk.mp4")  

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     results = model(frame, verbose=False)

#     for i, result in enumerate(results):
#         print(result)
#     break

    # for image_result in results:
    #     for xmin, ymin, xmax, ymax, conf, cls in image_result.xyxy:
    #         cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)
    #         cv2.putText(frame, f"{cls} ({conf:.2f})", (int(xmin), int(ymin) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # cv2.imshow("Video", frame)

    # if cv2.waitKey(1) & 0xFF == ord("q"):
    #     break

# Release resources
# cap.release()
# cv2.destroyAllWindows()
