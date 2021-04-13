import cv2 as cv
import logging as log
import sys
from openvino.inference_engine import IECore
from model import model

class face_recogniser(model):
    def __init__(self):
        super().__init__()

        self.device="MYRIAD"

        log.basicConfig(format="[ %(levelname)s ] %(message)s",
                        level=log.INFO, stream=sys.stdout)
        model_xml = 'face-detection-adas-0001.xml'
        model_bin = 'face-detection-adas-0001.bin'

        # Plugin initialization for specified device and load extensions library if specified
        log.info("Creating Inference Engine")
        ie = IECore()
        # Read IR
        log.info("Loading network files:\n\t{}\n\t{}".format(
            model_xml, model_bin))
        self.net = ie.read_network(model=model_xml, weights=model_bin)

        assert self.net.input_info["data"].input_data.shape[0] == 1, "Sample supports only single input topologies"
        assert len(
            self.net.outputs) == 1, "Sample supports only single output topologies"

        log.info("Preparing input blobs")
        self.input_blob = next(iter(self.net.input_info))
        self.out_blob = next(iter(self.net.outputs))
        self.net.batch_size = 1

        # Loading model to the plugin
        log.info("Loading model to the plugin")
        self.exec_net = ie.load_network(
            network=self.net, device_name=self.device)

        self.nFaces = 0

    def add_face_rectangle(self, frame):
        # Prepare input blob and perform an inference.
        blob = cv.dnn.blobFromImage(frame, size=(672, 384), ddepth=cv.CV_8U)
        res = self.exec_net.infer(inputs={self.input_blob: blob})
        out = res[self.out_blob]

        # Draw detected faces on the frame.
        self.nFaces = 0
        for detection in out.reshape(-1, 7):
            confidence = float(detection[2])
            xmin = int(detection[3] * frame.shape[1])
            ymin = int(detection[4] * frame.shape[0])
            xmax = int(detection[5] * frame.shape[1])
            ymax = int(detection[6] * frame.shape[0])
            if confidence > 0.5:
                cv.rectangle(frame, (xmin, ymin), (xmax, ymax), color=(0, 255, 0))
                self.nFaces+=1
        return frame

    def process(self, frame):
        return self.add_face_rectangle(frame)
