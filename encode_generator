import cv2
print(cv2.__version__)

import os
import numpy as np
import pickle
import face_recognition

cap = cv2.VideoCapture(0)  # Change 1 to 0 if you only have one camera
cap.set(3, 640)  # Set frame width
cap.set(4, 720)  # Set frame height

if not cap.isOpened():
    print("Camera could not be opened")
    exit()

# Correct the filename if necessarynn
imgBackground = cv2.imread('Resources/background.png')

# Importing mode images into a list
folderModePath = 'Resources/Modes'
modePathList = os.listdir(folderModePath)
imgModeList = []

for path in modePathList:
    full_path = os.path.join(folderModePath, path)
    img = cv2.imread(full_path)  # Load the full image path
    if img is not None:          # Only append valid images
        imgModeList.append(img)
    else:
        print(f"Failed to load image: {full_path}")

print(f"Loaded {len(imgModeList)} images into imgModeList")

if imgBackground is None:
    print("Failed to load background image")
    exit()

print("loading encode file")
import pickle

# Load the saved encodings
with open('EncodeFile.p', 'rb') as file:
    encodeListKnownWithIds = pickle.load(file)

encodeListKnown = encodeListKnownWithIds[0]  # Encodings
studentIds = encodeListKnownWithIds[1]       # Student IDs

# Now you can use `encodeListKnown` and `studentIds` for face recognition tasks
#print(f"Loaded {len(encodeListKnown)} encodings.")
#print(f"Student IDs: {studentIds}")
print("Encode file loaded")


while True:
    success, img = cap.read()
    if not success:
        print("Failed to grab frame")
        break

    # resize our image, S represents Small
    imgS=cv2.resize(img,(0,0), None,0.25,0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB) 

    # saving the imagS location, so that it can be encode whenever called inside encoding function
    faceCurFrame = face_recognition.face_locations(imgS)
    encodeCurFrame = face_recognition.face_encodings(imgS,faceCurFrame)

    if img.shape[0] <= 480 and img.shape[1] <= 640:
        imgBackground[162:162 + 480, 55:55 + 640] = img
  # Connect webcam with background image

    if len(imgModeList) > 0:
        imgBackground[44:44 + 633, 808:808 + 414] = imgModeList[0]  # Display the first mode image
    

    for encodeFace, faceLoc in zip(encodeCurFrame, faceCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)

        print("matches", matches)
        print("faceDis", faceDis)
        print(studentIds)

        if len(faceDis) > 0:
            best_match_index = np.argmin(faceDis)

            if matches[best_match_index]:
                matched_id = studentIds[best_match_index]
                print(f"✅ Match found: {matched_id} with distance {faceDis[best_match_index]:.2f}")

            # Optional: draw a box and label on the face in webcam image
                y1, x2, y2, x1 = faceLoc
                y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4  # Scale back up

                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, matched_id, (x1, y2 + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            else:
                print("❌ No valid match found.")
        else:
            print("⚠️ No faces compared.")


    # cv2.putText(imgBackground, studentIds[best_match_index], (860, 445), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)


        

    cv2.imshow("Webcam", img)  # Display webcam
    cv2.imshow("Face Attendance", imgBackground)  # Display background image on screen
    
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
        break

cap.release()
cv2.destroyAllWindows()
