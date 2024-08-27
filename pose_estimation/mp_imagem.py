import cv2
import mediapipe as mp

# Inicializar os módulos de pose e mãos do MediaPipe
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands

pose_tracker = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
hands_tracker = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Carregar a imagem
frame = cv2.imread('TtNUbUWg4xY.png')

# Detectar a pose e as mãos na imagem
pose_results = pose_tracker.process(frame)
hands_results = hands_tracker.process(frame)

# Verificar se as mãos estão acima dos ombros ou da cabeça
if pose_results.pose_landmarks and hands_results.multi_hand_landmarks:
    # Obter as coordenadas dos ombros
    landmarks = pose_results.pose_landmarks.landmark
    shoulder_left = (landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * frame.shape[1], 
                     landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * frame.shape[0])
    shoulder_right = (landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * frame.shape[1], 
                      landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * frame.shape[0])

    # Verificar a posição de cada mão em relação aos ombros
    for hand_landmarks in hands_results.multi_hand_landmarks:
        hand_above_shoulder = False
        for landmark in hand_landmarks.landmark:
            x, y = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
            if y < shoulder_left[1] and y < shoulder_right[1]:
                hand_above_shoulder = True
                break
        if hand_above_shoulder:
            cv2.putText(frame, "Person with hands above shoulders", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            break

# Desenhar o esqueleto da pose e os pontos das mãos
if pose_results.pose_landmarks:
    for connection in mp_pose.POSE_CONNECTIONS:
        idx1, idx2 = connection
        x1, y1 = int(pose_results.pose_landmarks.landmark[idx1].x * frame.shape[1]), int(pose_results.pose_landmarks.landmark[idx1].y * frame.shape[0])
        x2, y2 = int(pose_results.pose_landmarks.landmark[idx2].x * frame.shape[1]), int(pose_results.pose_landmarks.landmark[idx2].y * frame.shape[0])
        cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

if hands_results.multi_hand_landmarks:
    for hand_landmarks in hands_results.multi_hand_landmarks:
        for landmark in hand_landmarks.landmark:
            x, y = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

# Salvar a imagem anotada
cv2.imwrite("result.jpg", frame)

