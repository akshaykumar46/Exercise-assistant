import cv2
import mediapipe as mp
import numpy as np
import time
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


def calculate_angle(a,b,c):
    a = np.array(a) 
    b = np.array(b) 
    c = np.array(c) 
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle >180.0:
        angle = 360-angle
        
    return angle 

rep_count = 0

cap = cv2.VideoCapture("/Users/akshaykumar/Downloads/KneeBendVideo.mp4")

timer = 0
start_time = 0
flag = True

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        try:
            ret, frame = cap.read()

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
        
            results = pose.process(image)

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        
            landmarks = results.pose_landmarks.landmark
            left_hip=landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
            left_knee=landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
            left_ankle=landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]

            right_hip=landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
            right_knee=landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]
            right_ankle=landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]

            left_legVisi=left_hip.visibility+left_ankle.visibility+left_knee.visibility
            right_legVisi=right_hip.visibility+right_ankle.visibility+right_knee.visibility

            if left_legVisi>right_legVisi:
                hip=[left_hip.x,left_hip.y]
                knee=[left_knee.x,left_knee.y]
                ankle=[left_ankle.x,left_ankle.y]
                
            else:
                hip=[right_hip.x,right_hip.y]
                knee=[right_knee.x,right_knee.y]
                ankle=[right_ankle.x,right_ankle.y]
            
            angle=calculate_angle(hip,knee,ankle)

            print(angle)
            
            
            if angle < 140:
                if timer == 0:
                    start_time = time.time()
                timer = 1
            else:
                timer = 0

            if timer == 1:
                if time.time() - start_time > 8:
                    if flag:
                        rep_count += 1
                        flag=False
                    timer = 0
                else:
                    if flag:
                        remaining_time = 8 - (time.time() - start_time)
                        cv2.putText(image, "Remaining time: {:.2f} seconds".format(remaining_time), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    else:
                        cv2.putText(image, "You may relax your leg now for next rep", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0, 0), 2)

            else:
                cv2.putText(image, "Keep your knee bent", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                flag = True

            cv2.putText(image, "Rep Count: " + str(rep_count), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        except:
            cv2.putText(image, "Error in reading frame", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)

        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                 ) 
        
        
        cv2.imshow('Knee Bend Exercise', image)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break


    cap.release()
    cv2.destroyAllWindows()

