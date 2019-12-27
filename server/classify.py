#!/usr/bin/env python
"""
Copyright (C) 2018-2019 Intel Corporation

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

"""
The original work was modified by Filippo Valle
"""

from __future__ import print_function
import sys
import os
from argparse import ArgumentParser, SUPPRESS
import cv2
import cv2 as cv
import numpy as np
import logging as log
from time import time
from openvino.inference_engine import IENetwork, IECore
import json
from telepyth import TelepythClient

class classifier:
    def __init__(self):
        self.model = 'inception-v4.xml'
        self.device = 'MYRIAD'
        self.number_top = 10
        self.input=['input.jpg']
        self.load_model()

    def load_model(self):
        log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)
        model_xml = self.model
        model_bin = os.path.splitext(model_xml)[0] + ".bin"

        # Plugin initialization for specified device and load extensions library if specified
        log.info("Creating Inference Engine")
        ie = IECore()
        # Read IR
        log.info("Loading network files:\n\t{}\n\t{}".format(model_xml, model_bin))
        self.net = IENetwork(model=model_xml, weights=model_bin)

        assert len(self.net.inputs.keys()) == 1, "Sample supports only single input topologies"
        assert len(self.net.outputs) == 1, "Sample supports only single output topologies"

        log.info("Preparing input blobs")
        self.input_blob = next(iter(self.net.inputs))
        self.out_blob = next(iter(self.net.outputs))
        self.net.batch_size = len(self.input)

        # Loading model to the plugin
        log.info("Loading model to the plugin")
        self.exec_net = ie.load_network(network=self.net, device_name=self.device)

        #load classes
        with open("/home/pi/inception/imagenet_class_index.json",'r') as file:
            self.labels_map=json.load(file)

    def shot(self):
        log.info('Shoting')
        # capture camera
        cap = cv2.VideoCapture(0)

        # Read an image.
        #frame = cv.imread('input.jpg')
        #if frame is None:
        #    raise Exception('Image not found!')

        ret, frame = cap.read()
        # Display the resulting frame
        #Save the frame to an image file.
        cv.imwrite('input.jpg', frame)

        # When everything done, release the capture
        cap.release()
        cv2.destroyAllWindows()
        return frame

    def telegram_send(self, fig=None, text=None, key='KEY'):
        global tp
        tp = TelepythClient(key)
        if fig is not None:
            tp.send_figure(fig)
            if text is not None:
                tp.send_text(text)


    def classify(self, image):
        # Read and pre-process input images
        n, c, h, w = self.net.inputs[self.input_blob].shape
        images = np.ndarray(shape=(n, c, h, w))
        #image = cv2.imread(self.input[0])
        if image.shape[:-1] != (h, w):
            log.warning("Image {} is resized from {} to {}".format(self.input[0], image.shape[:-1], (h, w)))
            image = cv2.resize(image, (w, h))
            image = image.transpose((2, 0, 1))  # Change data layout from HWC to CHW
        images[0] = image
        log.info("Batch size is {}".format(n))

        # Start sync inference
        log.info("Starting inference in synchronous mode")
        res = self.exec_net.infer(inputs={self.input_blob: images})

        # Processing output blob
        log.info("Processing output blob")
        res = res[self.out_blob]
        log.info("Top {} results: ".format(self.number_top))

        classid_str = "classid"
        probability_str = "probability"
        for i, probs in enumerate(res):
            probs = np.squeeze(probs)
            top_ind = np.argsort(probs)[-self.number_top:][::-1]
        print("Image {}\n".format(self.input[i]))
        print(classid_str, probability_str)
        print("{} {}".format('-' * len(classid_str), '-' * len(probability_str)))
        for id in top_ind:
            det_label = self.labels_map[str(id)][1] if self.labels_map else "{}".format(id)
            label_length = len(det_label)
            space_num_before = (len(classid_str) - label_length) // 2
            space_num_after = len(classid_str) - (space_num_before + label_length) + 2
            space_num_before_prob = (len(probability_str) - len(str(probs[id]))) // 2
            print("{}{}\t{}{}{:.7f}".format(' ' * space_num_before, det_label,
            ' ' * space_num_after, ' ' * space_num_before_prob,
            probs[id]))

        print("\n")
        #telegram_send(text="%s with p: %f"%(self.labels_map[str(0)][1], probs[0]))
        return ["{}{}\t{}{}{:.7f}".format(' ' * space_num_before, self.labels_map[str(top_ind[0])][1] if self.labels_map else "{}".format(top_ind[0]),
        ' ' * space_num_after, ' ' * space_num_before_prob,
        probs[top_ind[0]]),
        "{}{}\t{}{}{:.7f}".format(' ' * space_num_before, self.labels_map[str(top_ind[1])][1] if self.labels_map else "{}".format(top_ind[0]),
        ' ' * space_num_after, ' ' * space_num_before_prob,
        probs[top_ind[1]]),
        "{}{}\t{}{}{:.7f}".format(' ' * space_num_before, self.labels_map[str(top_ind[2])][1] if self.labels_map else "{}".format(top_ind[0]),
        ' ' * space_num_after, ' ' * space_num_before_prob,
        probs[top_ind[2]]),
        "{}{}\t{}{}{:.7f}".format(' ' * space_num_before, self.labels_map[str(top_ind[3])][1] if self.labels_map else "{}".format(top_ind[0]),
        ' ' * space_num_after, ' ' * space_num_before_prob,
        probs[top_ind[3]]),
        "{}{}\t{}{}{:.7f}".format(' ' * space_num_before, self.labels_map[str(top_ind[4])][1] if self.labels_map else "{}".format(top_ind[0]),
        ' ' * space_num_after, ' ' * space_num_before_prob,
        probs[top_ind[4]])
        ]



if __name__ == '__main__':
    sys.exit(classify() or 0)
