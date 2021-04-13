from classify import classifier
from face_recogniser import face_recogniser
from age_recogniser import age_recogniser
from yolo_model import yolo_model
#from face_recogniser_status import face_recogniser_status
import logging
import sys,os
import cv2
from flask import Flask, render_template, make_response, Response

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'common'))

logging.basicConfig(format="[ %(levelname)s ] %(message)s", level=logging.INFO, stream=sys.stdout)
log = logging.getLogger()

app = Flask(__name__)
log.info("Loading server..")

#model = classifier()
#model = face_recogniser()
#model = age_recogniser()
model = yolo_model(model = "yolo-v3-tf.xml", raw_output_message=True, labels = "object_detection_classes_yolov3.txt")
#model = face_recogniser_status()

@app.route("/")
def server():
    log.info("Main page")
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
    retval, buffer = cv2.imencode('.jpg', cv2.imread("/home/pi/inception/server/input.jpg"))
    response = make_response(buffer.tobytes())
    response.headers['Content-Type'] = 'image/png'
    return response



def generator():
    import io
    global model
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        frame = cv2.resize(frame, (256,256))
        try:
            frame = model.process(frame)
        except:
            log.error(*sys.exc_info())
            frame = None
        if frame is None:
            continue
        encode_return_code, image_buffer = cv2.imencode('.jpg', frame)
        io_buf = io.BytesIO(image_buffer)
        yield (b'--frame\r\n'+
               b'Content-Type: image/jpeg\r\n\r\n' + io_buf.read() + b'\r\n')

@app.route("/video")
def video():
    return Response(generator(),mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/live")
def live():
	return render_template("live.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0")
