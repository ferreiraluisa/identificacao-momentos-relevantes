from ultralytics import YOLO
import cv2
from ultralytics import YOLO
import sys
import numpy as np

# file = sys.argv[1]
model = YOLO('yolov8n-pose.pt')

# YOLO Keypoints
# 0 : Nose
# 1 : Left Eye
# 2 : Right Eye
# 3 : Left Ear
# 4 : Right Ear
# 5 : Left Sholder
# 6 : Right Sholder
# 7 : Left Elbow
# 8 : Right Elbow
# 9 : Left Wrist
# 10 : Right Wrist
# 11 : Left Hip
# 12 : Right Hip
# 13 : Left Knee
# 14 : Right Knee
# 15 : Left Ankle
# 16 : Right Ankle


filename = sys.argv[1]

model = YOLO('yolov8m-pose.pt')

cap = cv2.VideoCapture('../videos/' + filename)


def are_sholders_between_hands_and_hips(kps):
    """Check if the shoulders are between the hands and hips.

    Args:
        kps: A list of lists of the coordinates of the skeleton points.

    Returns:
        True if the shoulders are between the hands and hips, False otherwise.
    """

    # Get the coordinates of the key points of the hands, shoulders, and hips.
    left_hand_x, left_hand_y = kps[9]
    left_shoulder_x, left_shoulder_y = kps[5]
    left_hip_x, left_hip_y = kps[11]
    right_hand_x, right_hand_y = kps[10]
    right_shoulder_x, right_shoulder_y = kps[6]
    right_hip_x, right_hip_y = kps[12]

    if (right_hand_x == 0 and right_hand_y == 0) and (left_hand_x == 0 and left_hand_y == 0):
        return False
    if(right_hip_x == 0 and right_hip_y == 0) and (left_hip_x == 0 and left_hip_y == 0):
        return False
    if(right_shoulder_x == 0 and right_shoulder_y == 0) and (left_shoulder_x == 0 and left_shoulder_y == 0):
        return False

    shoulder_hand = (left_hand_x - left_shoulder_x, left_hand_y - left_shoulder_y)
    shoulder_hip = (left_hip_x - left_shoulder_x, left_hip_y - left_shoulder_y)

    dot = shoulder_hand[0] * shoulder_hip[0] + shoulder_hand[1] * shoulder_hip[1]

    norm_shoulder_hand = np.sqrt(shoulder_hand[0] ** 2 + shoulder_hand[1] ** 2)
    norm_shoulder_hip = np.sqrt(shoulder_hip[0] ** 2 + shoulder_hip[1] ** 2)

    cos_left = dot / (norm_shoulder_hand * norm_shoulder_hip)

    shoulder_hand = (right_hand_x - right_shoulder_x, right_hand_y - right_shoulder_y)
    shoulder_hip = (right_hip_x - right_shoulder_x, right_hip_y - right_shoulder_y)
    dot = shoulder_hand[0] * shoulder_hip[0] + shoulder_hand[1] * shoulder_hip[1]

    norm_shoulder_hand = np.sqrt(shoulder_hand[0] ** 2 + shoulder_hand[1] ** 2)
    norm_shoulder_hip = np.sqrt(shoulder_hip[0] ** 2 + shoulder_hip[1] ** 2)

    cos_right = dot / (norm_shoulder_hand * norm_shoulder_hip)

    return cos_left < 0 and cos_right < 0

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
codec = cv2.VideoWriter_fourcc(*'mp4v')
output_video = cv2.VideoWriter('resultados_yolo/' + filename, codec, fps, (width, height))
print(cap.isOpened())
hist_pessoa = []
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, verbose=False)

    annotated_frame = results[0].plot()
    height, width, _ = annotated_frame.shape

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    kps = results[0].keypoints.xy.cpu().numpy()
    pessoa_empe = False
    if len(kps[0]) != 0 :
        j = 50
        for i, kp in enumerate(kps):
            if are_sholders_between_hands_and_hips(kp):
                # print(j)
                bbox = results[0].boxes.xyxy[i].cpu().numpy()
                confidence = results[0].boxes.conf.cpu().numpy()[i]
                if(confidence < 0.5):
                    continue
                confidence_rounded = np.around(confidence, decimals=2)
                
                # Desenha o bounding box
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
                cv2.putText(frame, f"Hands above shoulders", (int(bbox[0]), int(bbox[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                j += 50
                pessoa_empe = True
        
    output_video.write(frame)
    if pessoa_empe:
        hist_pessoa.append(1)
    else:
        hist_pessoa.append(0)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
output_video.release()
cv2.destroyAllWindows()
import matplotlib.pyplot as plt
import json

hist_guns = [float(value) for value in hist_pessoa]

data = {
    "people": hist_pessoa
}
with open(f"{filename.split('.')[0]}.json", 'w') as arquivo_json:
    json.dump(data, arquivo_json, indent=4)



tempo_segundos = [i / fps for i in range(len(hist_pessoa))]

plt.figure(figsize=(10, 6))
plt.hist(tempo_segundos, bins=len(hist_pessoa), weights=hist_pessoa, edgecolor='black')

plt.xlabel('Tempo (segundos)')
plt.ylabel('Frequência')
plt.title('Aparência de Multidões')

plt.savefig(f'{filename}_people.png')
plt.clf()





