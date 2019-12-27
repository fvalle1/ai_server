"""
Copyright (C) 2019 Filippo Valle

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""


from classify import classifier
import cv2
from flask import Flask, render_template, make_response
app = Flask(__name__)

model = classifier()

@app.route("/")
def server():
    message='Hello raspberry'
    return render_template("index.html", message=message)

@app.route("/trigger")
def click():
    classes = model.classify(model.shot())
    return render_template("index.html", message="classified",class0=classes[0], class1=classes[1],class2=classes[2],class3=classes[3],class4=classes[4])

@app.route("/shot")
def shot():
    model.shot()
    return render_template("index.html", message='shot')

@app.route("/image")
def image():
    retval, buffer = cv2.imencode('.jpg', cv2.imread("input.jpg"))
    response = make_response(buffer.tobytes())
    response.headers['Content-Type'] = 'image/png'
    return response

if __name__ == "__main__":
    app.run(host="0.0.0.0")
