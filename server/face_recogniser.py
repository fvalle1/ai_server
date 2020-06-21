import cv2 as cv
from model import model

class face_recogniser(model):
    def __init__(self):
        super()
        self.net = cv.dnn.readNet('/home/pi/inception/face-detection-adas-0001.xml','/home/pi/inception/face-detection-adas-0001.bin')
        self.net.setPreferableTarget(cv.dnn.DNN_TARGET_MYRIAD)

    def add_face_rectangle(self, frame):
        # Prepare input blob and perform an inference.
        blob = cv.dnn.blobFromImage(frame, size=(672, 384), ddepth=cv.CV_8U)
        self.net.setInput(blob)
        out = self.net.forward()
        # Draw detected faces on the frame.
        for detection in out.reshape(-1, 7):
            confidence = float(detection[2])
            xmin = int(detection[3] * frame.shape[1])
            ymin = int(detection[4] * frame.shape[0])
            xmax = int(detection[5] * frame.shape[1])
            ymax = int(detection[6] * frame.shape[0])
            if confidence > 0.5:
                cv.rectangle(frame, (xmin, ymin), (xmax, ymax), color=(0, 255, 0))
        return frame

    def process(self, frame):
        return self.add_face_rectangle(frame)
