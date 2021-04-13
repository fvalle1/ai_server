import cv2 as cv
import logging as log
import sys
from openvino.inference_engine import IECore

from model import model

class age_recogniser(model):
    def __init__(self):
        super().__init__()
        log.basicConfig(format="[ %(levelname)s ] %(message)s",
                        level=log.INFO, stream=sys.stdout)

        self.device = "MYRIAD"
        model_xml = 'age-gender-recognition-retail-0013.xml'
        model_bin = 'age-gender-recognition-retail-0013.bin'

        # Plugin initialization for specified device and load extensions library if specified
        log.info("Creating Inference Engine")
        ie = IECore()
        # Read IR
        log.info("Loading network files:\n\t{}\n\t{}".format(
            model_xml, model_bin))
        self.net = ie.read_network(model=model_xml, weights=model_bin)

        assert self.net.input_info["data"].input_data.shape[0] == 1, "Sample supports only single input topologies"
        assert len(self.net.outputs) == 2, "Sample supports two output topologies"

        log.info("Preparing input blobs")
        self.input_blob = next(iter(self.net.input_info))
        self.out_blob = next(iter(self.net.outputs))
        self.net.batch_size = 1

        # Loading model to the plugin
        log.info("Loading model to the plugin")
        self.exec_net = ie.load_network(
            network=self.net, device_name=self.device)


    def add_age_info(self, frame):
        # Prepare input blob and perform an inference.
        blob = cv.dnn.blobFromImage(frame, 1.0, (62,62))
        res = self.exec_net.infer(inputs={self.input_blob: blob})

        # Draw detected faces on the frame.
        gender = res["prob"].ravel()[1]
        age = res["age_conv3"].ravel()[0]
        gender = "male" if gender > 0.5 else "female"
        age = age * 100
        #age, gender = (10,"a")
        font = cv.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (10,250)
        fontScale  = 1
        if gender == "male":
            fontColor = (50,50,255)
        else:
            fontColor = (50,255,55)
        lineType = 2
        cv.putText(frame,"%s %.0f"%(gender,age),
          bottomLeftCornerOfText,
          font,
          fontScale,
          fontColor,
          lineType)
        return frame

    def process(self, frame):
        return self.add_age_info(frame)
