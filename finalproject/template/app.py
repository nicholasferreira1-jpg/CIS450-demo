# final project example template
#
# to build:   docker build -t app .      
# to run:     docker run -p 80:80 app
# in browser: http://localhost

from flask import Flask, request, send_file, render_template_string, send_from_directory
import os
import cv2
import socket

app = Flask(__name__)

hostname = socket.gethostname()

UPLOAD_FOLDER = "static"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
LAST_T = 100

HTML = """
<h2>Image Processor</h2>
<p><i>host={{ hostname }}</i></p>

<form method="POST" action="/edges" enctype="multipart/form-data">
    <label>Image:</label>
    <input type="file" name="file">

    <label>Threshold:</label>
    <input type="number" name="T" value="{{ threshold }}">

    <input type="submit" value="Run">
</form>

<hr>

{% if original and processed %}

<h3>Result (Threshold: {{ threshold }})</h3>

<div style="display:flex; gap:20px; align-items:flex-start;">

    <div>
        <p><b>Original</b></p>
        <img src="{{ original }}" style="max-width:350px;">
    </div>

    <div>
        <p><b>Processed</b></p>
        <img src="{{ processed }}" style="max-width:350px;">
    </div>

</div>

{% endif %}
"""

def detect_edges(input_path, output_path, T=100):
    T1 = int(T * 0.5)
    T2 = int(T)
    img = cv2.imread(input_path)
    edges = cv2.Canny(img, T1, T2)
    cv2.imwrite(output_path, edges)
    return output_path


@app.route("/")
def home():
    return render_template_string(
        HTML,
        threshold=LAST_T,
        hostname=hostname
    )


@app.route("/output.jpg")
def output_image():
    return send_file("output.jpg", mimetype="image/jpeg")


@app.route("/edges", methods=["POST"])
def edges_route():
    T = request.form.get("T", default=100, type=int)
    LAST_T = T

    file = request.files.get("file", None)

    original_path = os.path.join(UPLOAD_FOLDER, "original.jpg")
    output_path = os.path.join(UPLOAD_FOLDER, "output.jpg")

    if file and file.filename != "":
        file.save(original_path)

    if not os.path.exists(original_path):
        return "No image uploaded yet."

    detect_edges(original_path, output_path, T)

    return render_template_string(
        HTML,
        original="/static/original.jpg",
        processed="/static/output.jpg",
        threshold=LAST_T,
        hostname=hostname
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 80))
    app.run(host="0.0.0.0", port=port)