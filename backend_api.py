from flask import Flask, request, jsonify
import cv2
import numpy as np
import pickle
import face_recognition
import os
from werkzeug.utils import secure_filename
from flask_cors import CORS
        
app = Flask(__name__)
CORS(app)

# Load the saved encodings
with open(os.path.join(os.path.dirname(__file__), '../EncodeFile.p'), 'rb') as file:
    encodeListKnownWithIds = pickle.load(file)

encodeListKnown = encodeListKnownWithIds[0]  # Encodings
studentIds = encodeListKnownWithIds[1]       # Student IDs

@app.route('/recognize', methods=['POST'])
def recognize():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    file = request.files['image']
    filename = secure_filename(file.filename)
    file_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({'error': 'Invalid image'}), 400

    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
    faceCurFrame = face_recognition.face_locations(imgS)
    encodeCurFrame = face_recognition.face_encodings(imgS, faceCurFrame)

    results = []
    for encodeFace, faceLoc in zip(encodeCurFrame, faceCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        if len(faceDis) > 0:
            best_match_index = np.argmin(faceDis)
            if matches[best_match_index]:
                matched_id = studentIds[best_match_index]
                results.append({'id': matched_id, 'distance': float(faceDis[best_match_index]), 'location': [int(x) for x in faceLoc]})
            else:
                results.append({'id': 'unknown', 'distance': None, 'location': [int(x) for x in faceLoc]})
        else:
            results.append({'id': 'unknown', 'distance': None, 'location': [int(x) for x in faceLoc]})

    return jsonify({'results': results})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True) 
