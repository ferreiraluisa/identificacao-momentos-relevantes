import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands

pose_tracker = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
hands_tracker = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# open the input video file
cap = cv2.VideoCapture("videos/O1p47ykn3N0.mp4")

# get the video dimensions and frame rate
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# create a video writer object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('resultados_mediapipe/O1p47ykn3N0.mp4', fourcc, fps, (width, height))

while True:
    # read a frame from the input video
    ret, frame = cap.read()

    # if there are no more frames, exit the loop
    if not ret:
        break

    # Convert the frame to RGB format
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect the pose and hands in the frame
    pose_results = pose_tracker.process(frame)
    hands_results = hands_tracker.process(frame)

    # Check if the hands are above the shoulders or head
    if pose_results.pose_landmarks and hands_results.multi_hand_landmarks:
        # Get the coordinates of the shoulders and head
        landmarks = pose_results.pose_landmarks.landmark
        shoulder_left = (landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * frame.shape[1], landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * frame.shape[0])
        shoulder_right = (landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * frame.shape[1], landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * frame.shape[0])
        head = (landmarks[mp_pose.PoseLandmark.NOSE.value].x * frame.shape[1], landmarks[mp_pose.PoseLandmark.NOSE.value].y * frame.shape[0])

        # Check each hand for position relative to the shoulders/head
        for hand_landmarks in hands_results.multi_hand_landmarks:
            hand_above_shoulder = False
            hand_above_head = False
            for landmark in hand_landmarks.landmark:
                x, y = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
                if x < shoulder_left[0] and x < shoulder_right[0]:
                    hand_above_shoulder = True
                elif  y < shoulder_left[1] and y < shoulder_right[1]:
                    hand_above_shoulder = True
            if hand_above_shoulder:
                cv2.putText(frame, "Hand above shoulder", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Draw the pose skeleton and hand landmarks
    if pose_results.pose_landmarks:
        # Define the connections between the keypoints
        connections = mp_pose.POSE_CONNECTIONS
        for connection in connections:
            # Get the indices of the keypoints for the connection
            idx1, idx2 = connection
            # Get the coordinates of the keypoints
            x1, y1 = int(pose_results.pose_landmarks.landmark[idx1].x * frame.shape[1]), int(pose_results.pose_landmarks.landmark[idx1].y * frame.shape[0])
            x2, y2 = int(pose_results.pose_landmarks.landmark[idx2].x * frame.shape[1]), int(pose_results.pose_landmarks.landmark[idx2].y * frame.shape[0])
            # Draw a line between the keypoints
            cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
    if hands_results.multi_hand_landmarks:
        for hand_landmarks in hands_results.multi_hand_landmarks:
            for landmark in hand_landmarks.landmark:
                x, y = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

    # Write the frame to the output video
    out.write(frame)

cap.release()
out.release()
cv2.destroyAllWindows()