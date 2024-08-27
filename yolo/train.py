from ultralytics import YOLOv10 as YOLO

#check torch using gpu, device
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

model = YOLO.from_pretrained('jameslahm/yolov10m')

# model = YOLO('yolov10/runs/detect/yolos_mgd_30_puro/weights/best.pt')

model.train(data='mgd/data.yaml', epochs=30, batch=8, device=device)