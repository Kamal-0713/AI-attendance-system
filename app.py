from flask import Flask, render_template, request
import cv2
import os

app = Flask(__name__)

@app.route('/', methods=['GET','POST'])
def index():
    if request.method == 'POST':
        name = request.form['username']

        path = os.path.join('dataset', name)
        os.makedirs(path, exist_ok=True)

        cam = cv2.VideoCapture(0)
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

        count = 0
        while count < 30:
            ret, frame = cam.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray,1.3,5)

            for (x,y,w,h) in faces:
                count += 1
                face = gray[y:y+h, x:x+w]
                face = cv2.resize(face,(128,128))
                cv2.imwrite(f"{path}/{count}.jpg", face)
                cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

            cv2.imshow("Capturing Dataset", frame)
            if cv2.waitKey(1) == 27:
                break

        cam.release()
        cv2.destroyAllWindows()
        return f"Dataset created for {name}"

    return render_template('index.html')

app.run(debug=True)