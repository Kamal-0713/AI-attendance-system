import cv2
import numpy as np
import os
import csv
from datetime import datetime

dataset_path = 'dataset'

# Must use opencv-contrib-python for cv2.face
recognizer = cv2.face.LBPHFaceRecognizer_create()

faces = []
labels = []
label_map = {}
current_label = 0

# Load dataset
for person in os.listdir(dataset_path):
    person_path = os.path.join(dataset_path, person)
    if not os.path.isdir(person_path):
        continue

    label_map[current_label] = person

    for img_name in os.listdir(person_path):
        img_path = os.path.join(person_path, img_name)

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Skipping invalid image: {img_path}")
            continue

        # Resize all images to same shape
        img = cv2.resize(img, (128, 128))

        faces.append(img)
        labels.append(current_label)

    current_label += 1

# Convert lists to arrays
faces = np.array(faces)
labels = np.array(labels)

# Train recognizer
recognizer.train(faces, labels)

# Haarcascade for face detection
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

cam = cv2.VideoCapture(0)

while True:
    ret, frame = cam.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces_detected = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces_detected:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (128, 128))

        label, confidence = recognizer.predict(face)

        # If high confidence, find name else Unknown
        if confidence < 80:
            name = label_map.get(label, "Unknown")
        else:
            name = "Unknown"

        now = datetime.now()
        date = now.strftime('%Y-%m-%d')
        time = now.strftime('%H:%M:%S')

        # ⭐ Avoid duplicates: check if this person has attendance for today
        already_logged = False
        if os.path.exists('attendance.csv'):
            with open('attendance.csv', 'r') as f:
                for row in csv.reader(f):
                    if len(row) >= 2 and row[0] == name and row[1] == date:
                        already_logged = True
                        break

        if not already_logged:
            # ⭐ Write to CSV with separate columns (Name, Date, Time)
            with open('attendance.csv', 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([name, date, time])

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, name, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow("Attendance System", frame)

    # Press 'q' to stop webcam
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
