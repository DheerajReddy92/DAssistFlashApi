import cv2
import numpy as np
import dlib
import pygame

# Initialize pygame mixer
pygame.mixer.init()

# Load dlib's pre-trained shape predictor model for facial landmarks
predictor = dlib.shape_predictor('shape_predictor.dat')

# Desired EAR threshold for drowsiness
EAR_THRESHOLD = 0.25
ALARM_FILE = 'alarm.wav'

# Load the alarm sound
pygame.mixer.music.load(ALARM_FILE)

def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

detector = dlib.get_frontal_face_detector()
video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_frame = gray_frame.astype('uint8')

    faces = detector(gray_frame)

    for face in faces:
        landmarks = predictor(gray_frame, face)
        
        left_eye = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)], dtype=np.float32)
        right_eye = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)], dtype=np.float32)

        ear_left = eye_aspect_ratio(left_eye)
        ear_right = eye_aspect_ratio(right_eye)
        ear = (ear_left + ear_right) / 2.0
        
        if ear < EAR_THRESHOLD:
            print("Drowsiness detected! EAR:", ear)
            if not pygame.mixer.music.get_busy():
                pygame.mixer.music.play()
        else:
            pygame.mixer.music.stop()
            
        cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (255, 0, 0), 2)

    cv2.imshow('Drowsiness Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
pygame.mixer.quit()