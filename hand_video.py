import cv2
import sys
import mediapipe as mp
import math
import numpy as np
from tensorflow.keras.models import load_model

def grade(file_name):

    actions = ['rock_paper', 'shaking_hands', 'moving_fingers', 'opp_rock_paper', 'finger_clap', 'rock_clap', 'count_number']
    seq_length = 30

    model = load_model('models/model.h5')

    # MediaPipe hands model
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5)

    cap = cv2.VideoCapture(file_name)

    if not cap.isOpened():
        print("Error: Could not open video.")
        sys.exit()
    
    end = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  #동영상 파일의 전체 프레임수를 가져와 저장
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  #카메라로 촬영하는 영상의 가로 픽셀 수

    _, frame = cap.read()

    seq = []
    action_seq = []
    this_action = '?'

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        hand_landmarks = result.multi_hand_landmarks

        if hand_landmarks:
            for res in result.multi_hand_landmarks:
                joint = np.zeros((21, 4))
                for j, lm in enumerate(res.landmark):
                    joint[j] = [lm.x, lm.y, lm.z, lm.visibility]

                # Compute angles between joints
                v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19], :3] # Parent joint
                v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], :3] # Child joint
                v = v2 - v1 # [20, 3]
                # Normalize v
                v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

                # Get angle using arcos of dot product
                angle = np.arccos(np.einsum('nt,nt->n',
                    v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:], 
                    v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:])) # [15,]

                angle = np.degrees(angle) # Convert radian to degree

                d = np.concatenate([joint.flatten(), angle])

                seq.append(d)

                mp_drawing.draw_landmarks(frame, res, mp_hands.HAND_CONNECTIONS)

                if len(seq) < seq_length:
                    continue

                input_data = np.expand_dims(np.array(seq[-seq_length:], dtype=np.float32), axis=0)

                y_pred = model.predict(input_data).squeeze()

                i_pred = int(np.argmax(y_pred))
                conf = y_pred[i_pred]

                if conf < 0.9:  #90퍼센트 이하의 confidence
                    continue

                print("i_pred = ", i_pred)

                '''
                action = actions[i_pred]
                action_seq.append(action)
                '''

                if i_pred < len(actions):  # actions 리스트의 길이를 초과하지 않도록 확인
                    action = actions[i_pred]
                    action_seq.append(action)


                    if len(action_seq) < 3:
                        continue

                    # this_action = '?'
                    if action_seq[-1] == action_seq[-2] == action_seq[-3]:  # 마지막 3개의 액션이 모두 같은 액션일 떄 
                        this_action = action

                    # cv2.putText(img, f'{this_action.upper()}', org=(int(res.landmark[0].x * img.shape[1]), int(res.landmark[0].y * img.shape[0] + 20)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)
                    
                    print(this_action)
                else:
                    print(f"Warning: i_pred ({i_pred}) is out of range for actions list.")          
        
        if cap.get(cv2.CAP_PROP_POS_FRAMES) < end - 1:  #현재 프레임이 동영상의 마지막 프레임일 경우에는 재생하지 않도록
            ret, frame = cap.read()
        else:
            break

        # cv2.imshow("Video", frame)
    
    cv2.destroyAllWindows()
    cap.release()
    return this_action