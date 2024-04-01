from ultralytics import YOLO

#check torch using gpu, device
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

model = YOLO('weights/yolov8n.pt')

#https://github.com/orgs/ultralytics/discussions/3276
freeze_layers = range(24)  # Layers to freeze (0-23 for YOLOv8)
for name, param in model.named_parameters():
    if any(f'model.{layer}.' in name for layer in freeze_layers):
        param.requires_grad = False

model.train(data='dataset/weapon/data.yaml', epochs=100, imgsz=640, device=device)