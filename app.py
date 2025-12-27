from flask import Flask, render_template, request
import os, base64

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    data = request.get_json()
    name = data["username"]
    img = data["img"].split(",")[1]

    folder = f"dataset/{name}"
    os.makedirs(folder, exist_ok=True)

    img_count = len(os.listdir(folder)) + 1
    with open(f"{folder}/{img_count}.jpg", "wb") as f:
        f.write(base64.b64decode(img))

    return {"status":"saved"}

app.run(host="0.0.0.0", port=5000, debug=True)
