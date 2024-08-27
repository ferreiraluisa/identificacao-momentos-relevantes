from ultralytics import YOLO
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math


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
    print(math.degrees(math.acos(cos_left))> 90)
    print(math.degrees(math.acos(cos_right)) > 90)

    return math.degrees(math.acos(cos_left)) > 90 or math.degrees(math.acos(cos_right)) > 90
    


def are_both_hands_above_shoulders_head(kps):
    """Check if both hands are above the shoulders/head.

    Args:
        kps: A list of lists of the coordinates of the skeleton points.

    Returns:
        True if both hands are above the shoulders/head, False otherwise.
    """

    # Pega as coordenadas dos pontos chave das mãos e ombros.
    left_hand_x, left_hand_y = kps[9]
    left_shoulder_x, left_shoulder_y = kps[5]
    right_hand_x, right_hand_y = kps[10]
    right_shoulder_x, right_shoulder_y = kps[6]

    # Verifica se as coordenadas das mãos são válidas
    if (left_hand_x == 0 and left_hand_y == 0) or (right_hand_x == 0 and right_hand_y == 0):
        return False

    # Se a altura das mãos é maior que a altura dos ombros, então as mãos estão acima dos ombros/cabeça.
    if left_hand_y < left_shoulder_y and right_hand_y < right_shoulder_y:
        return True
    elif left_hand_x > left_shoulder_x and right_hand_x > right_shoulder_x:
        return True
    else:
        return False

# Carregar a imagem
# Carregar o modelo YOLO
model = YOLO('yolov8m-pose.pt')
# model = YOLO('yolov8x-pose-p6.pt')
import sys
path = sys.argv[1]
frame = cv2.imread(path)

# Obter os resultados do modelo
results = model(frame, verbose=False)

annotated_frame = results[0].plot()
frame2 = model.predict(path, conf=0.85, hide_labels=True, hide_conf=True, show=False, boxes=False, save=True)
# cv2.imwrite(f"{path.split('/')[0]}/pontos.jpg", frame2)
# Processar as detecções
height, width, _ = frame.shape

def desenha(kps, frame):
    blank_image = np.zeros_like(frame)
    if kps is not None:
        # keypoints = kps.xy
        for keypoint in kps:
            for point in keypoint:
                cv2.circle(blank_image, (int(point[0]), int(point[1])), 10, (0, 255, 0), -1)

    return blank_image

kps = results[0].keypoints.xy.cpu().numpy()
annotated_frame = results[0].plot()
if len(kps[0]) != 0:
    for i, kp in enumerate(kps):
        r = desenha(kps, frame)
        cv2.imwrite("pontos.jpg", r)
        if are_sholders_between_hands_and_hips(kp):
            print('que??')
            # Obtém a caixa delimitadora
            bbox = results[0].boxes.xyxy[i].cpu().numpy()
            confidence = results[0].boxes.conf.cpu().numpy()[i]
            if(confidence < 0.5):
                continue
            confidence_rounded = np.around(confidence, decimals=2)
            
            # Desenha o bounding box
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
            cv2.putText(frame, f"Hands above shoulders", (int(bbox[0]), int(bbox[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

# Salvar a imagem anotada
# cv2.imwrite(f"{path.split('/')[0]}/result.jpg", annotated_frame)
# cv2.imwrite(f"result.jpg", frame)