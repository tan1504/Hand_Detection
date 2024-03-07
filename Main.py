import cv2 as cv
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
# Đọc webcam
cap = cv.VideoCapture(0)
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
  while cap.isOpened():
    success, img = cap.read()
    if not success:
      break

    img.flags.writeable = False
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    results = hands.process(img)

    img.flags.writeable = True
    img = cv.cvtColor(img, cv.COLOR_RGB2BGR)

    counter = 0

    if results.multi_hand_landmarks:

      for hand_landmarks in results.multi_hand_landmarks:
        # Check left and right
        handIndex = results.multi_hand_landmarks.index(hand_landmarks)
        handLabel = results.multi_handedness[handIndex].classification[0].label

        handLandmarks = []
        
        for landmarks in hand_landmarks.landmark: #Check từng điểm ngón tay
          handLandmarks.append([landmarks.x, landmarks.y])

        if handLabel == "Left" and handLandmarks[4][0] > handLandmarks[3][0]:
          counter = counter+1
        elif handLabel == "Right" and handLandmarks[4][0] < handLandmarks[3][0]:
          counter = counter+1

        if handLandmarks[8][1] < handLandmarks[6][1]:       #Ngón trỏ
          counter = counter+1
        if handLandmarks[12][1] < handLandmarks[10][1]:     #Ngón giữa
          counter = counter+1
        if handLandmarks[16][1] < handLandmarks[14][1]:     #Ngón áp út
          counter = counter+1
        if handLandmarks[20][1] < handLandmarks[18][1]:     #Ngón út
          counter = counter+1

        # Draw hand
        mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Hiển thị số ngón tay
    cv.putText(img, str(counter), (50, 100), cv.FONT_HERSHEY_SIMPLEX, 3, (255,0,255), 10)
    cv.imshow('Hand Detection', img)
    if cv.waitKey(5) & 0xFF == 27:
      break
cap.release()