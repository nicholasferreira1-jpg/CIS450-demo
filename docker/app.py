# docker example
# build:   docker build -t edge-app
# run:     docker run -p 5000:5000 edge-app
# postamn: http://localhost:5000/edges
#          POST file, T
#          returns image

from flask import Flask, request, send_file
import os
import cv2
 
app = Flask(__name__)
 
def detect_edges(input_path, output_path, T=100):
    T1 = int(T * 0.5)
    T2 = int(T)
    img = cv2.imread(input_path)
    edges = cv2.Canny(img, T1, T2)
    cv2.imwrite(output_path, edges)
    return output_path

@app.route("/edges", methods=["POST"])
def edges_route():
    file = request.files["file"]
    T = request.form.get("T", default=100, type=int)
 
    input_path = "input.jpg"
    output_path = "output.jpg"
 
    file.save(input_path)
 
    detect_edges(input_path, output_path, T)
 
    return send_file(output_path, mimetype="image/jpeg")
 
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)