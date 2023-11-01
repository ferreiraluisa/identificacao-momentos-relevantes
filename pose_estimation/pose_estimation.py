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

model = YOLO('yolov8n-pose.pt')

cap = cv2.VideoCapture('videos/' + filename)

def are_both_hands_above_shoulders_head(kps):
    """Check if both hands are above the shoulders/head.

    Args:
        kps: A list of lists of the coordinates of the skeleton points.

    Returns:
        True if both hands are above the shoulders/head, False otherwise.
    """

    # Get the coordinates of both hand and shoulder keypoints.
    left_hand_x, left_hand_y = kps[9]
    left_shoulder_x, left_shoulder_y = kps[5]
    right_hand_x, right_hand_y = kps[10]
    right_shoulder_x, right_shoulder_y = kps[6]
    if (left_hand_x == 0 and left_hand_y == 0) or (right_hand_x == 0 and right_hand_y == 0):
        return False

    # If the height of either hand keypoint is greater than the height of the corresponding shoulder keypoint, then the hand is above the shoulders/head.
    if left_hand_y < left_shoulder_y and right_hand_y < right_shoulder_y:
        return True
    elif left_hand_x > left_shoulder_x and right_hand_x > right_shoulder_x:
        return True
    else:
        return False

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
codec = cv2.VideoWriter_fourcc(*'mp4v')
output_video = cv2.VideoWriter('resultados_yolo/' + filename, codec, fps, (width, height))
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
    if len(kps[0]) != 0 :
        j = 50
        for i, kp in enumerate(kps):
            if are_both_hands_above_shoulders_head(kp):
                # print(j)
                confidence = results[0].boxes.conf.cpu().numpy()[i]
                confidence_rounded = np.around(confidence, decimals=2)
                cv2.putText(annotated_frame, "Person "+str(confidence_rounded)+ " with Hands above shoulders", (50, j ), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                j += 50
    output_video.write(annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break





