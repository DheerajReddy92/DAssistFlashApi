# from flask import Flask, request, jsonify
# import cv2
import numpy as np
# import dlib

app = Flask(__name__)

# Initialize dlib's face detector and shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    return (A + B) / (2.0 * C)

EAR_THRESHOLD = 0.25

@app.route('/api/detect_drowsiness', methods=['POST'])
def detect_drowsiness():
    file = request.files['frame']
    npimg = np.fromstring(file.read(), np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    
    for face in faces:
        landmarks = predictor(gray, face)
        
        left_eye = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)])
        right_eye = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)])
        
        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)
        ear = (left_ear + right_ear) / 2.0
        
        if ear < EAR_THRESHOLD:
            return jsonify({'alert': 'Drowsiness detected!'})
    
    return jsonify({'alert': ''})

if __name__ == '__main__':
    app.run(debug=True)