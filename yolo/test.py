from ultralytics import YOLOv10 as YOLO
import cv2
import json
import os

# # Carregar modelo
model = YOLO('/home/luisa/runs/detect/train4/weights/best.pt')

# files = os.listdir('images/')
# results = model([f"images/{file}" for file in files], verbose=False)

# for i, result in enumerate(results):
#     result.save(filename=f"result{i}.jpg")

# Carregar vídeo
cap = cv2.VideoCapture('../videos/ilBvt9EWC8g.mp4')

# Definir codec e criar objeto VideoWriter
codec = cv2.VideoWriter_fourcc(*'XVID')
output_video = cv2.VideoWriter('ilBvt9EWC8g2.avi', codec, 30, (int(cap.get(3)), int(cap.get(4))))

while True:
    # Ler frame
    ret, frame = cap.read()

    if not ret:
        break

    # Fazer inferência
    results = model(frame, verbose=False)
    # res_json = json.loads(results[0].tojson())

    # # Visualizar bounding boxes
    # for i, result in enumerate(res_json):
    #     higher_conf = result['confidence']
    #     if(higher_conf <= 0.5):
    #         continue
    #     boxes = result['box']
    #     label = result['class']

    #     x1 = int(boxes['x1'])
    #     y1 = int(boxes['y1'])
    #     x2 = int(boxes['x2'])
    #     y2 = int(boxes['y2'])
    #     if label == 0:
    #         write = "Person"
    #         color = (0, 255, 0)
    #     else:
    #         write = "Gun"
    #         color = (0, 0, 255)
    #     cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    #     cv2.putText(frame, str(higher_conf), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    for result in results:
        result.save(filename='result.jpg')
    img = cv2.imread('result.jpg')

    # Escrever frame no arquivo de vídeo de saída
    output_video.write(img)

# Liberar recursos
cap.release()
output_video.release()
cv2.destroyAllWindows()
