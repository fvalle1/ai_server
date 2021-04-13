#!/usr/bin/env python3
"""
 Copyright (C) 2018-2020 Intel Corporation

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

import monitors
import logging
import threading
import os
import sys
from collections import deque
from argparse import ArgumentParser, SUPPRESS
from math import exp as exp
from time import perf_counter
from enum import Enum

#import ngraph as ng #please check get_object when this will be installed

import cv2
import numpy as np
from openvino.inference_engine import IECore

sys.path.append(os.path.join(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))), 'common'))


logging.basicConfig(format="[ %(levelname)s ] %(message)s",
                    level=logging.INFO, stream=sys.stdout)
log = logging.getLogger()

class YoloParams:
    # ------------------------------------------- Extracting layer parameters ------------------------------------------
    # Magic numbers are copied from yolo samples
    def __init__(self, param, side):
        self.num = 3 if 'num' not in param else int(param['num'])
        self.coords = 4 if 'coords' not in param else int(param['coords'])
        self.classes = 80 if 'classes' not in param else int(param['classes'])
        self.side = side
        self.anchors = [10.0, 13.0, 16.0, 30.0, 33.0, 23.0, 30.0, 61.0, 62.0, 45.0, 59.0, 119.0, 116.0, 90.0, 156.0,
                        198.0,
                        373.0, 326.0] if 'anchors' not in param else [float(a) for a in param['anchors']]

        self.isYoloV3 = False

        if param.get('mask'):
            mask = [int(idx) for idx in param['mask']]
            self.num = len(mask)

            maskedAnchors = []
            for idx in mask:
                maskedAnchors += [self.anchors[idx * 2],
                                  self.anchors[idx * 2 + 1]]
            self.anchors = maskedAnchors

            self.isYoloV3 = True  # Weak way to determine but the only one.


class Modes(Enum):
    USER_SPECIFIED = 0
    MIN_LATENCY = 1


class Mode():
    def __init__(self, value):
        self.current = value

    def next(self):
        if self.current.value + 1 < len(Modes):
            self.current = Modes(self.current.value + 1)
        else:
            self.current = Modes(0)


class ModeInfo():
    def __init__(self):
        self.last_start_time = perf_counter()
        self.last_end_time = None
        self.frames_count = 0
        self.latency_sum = 0


def scale_bbox(x, y, h, w, class_id, confidence, h_scale, w_scale):
    xmin = int((x - w / 2) * w_scale)
    ymin = int((y - h / 2) * h_scale)
    xmax = int(xmin + w * w_scale)
    ymax = int(ymin + h * h_scale)
    return dict(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, class_id=class_id, confidence=confidence)

def entry_index(side, coord, classes, location, entry):
    side_power_2 = side ** 2
    n = location // side_power_2
    loc = location % side_power_2
    return int(side_power_2 * (n * (coord + classes + 1) + entry) + loc)


def parse_yolo_region(blob, resized_image_shape, original_im_shape, params, threshold, is_proportional=True):
    # ------------------------------------------ Validating output parameters ------------------------------------------
    _, _, out_blob_h, out_blob_w = blob.shape
    assert out_blob_w == out_blob_h, "Invalid size of output blob. It sould be in NCHW layout and height should " \
                                     "be equal to width. Current height = {}, current width = {}" \
                                     "".format(out_blob_h, out_blob_w)

    # ------------------------------------------ Extracting layer parameters -------------------------------------------
    orig_im_h, orig_im_w = original_im_shape
    resized_image_h, resized_image_w = resized_image_shape
    objects = list()
    predictions = blob.flatten()
    side_square = params.side * params.side

    # ------------------------------------------- Parsing YOLO Region output -------------------------------------------
    for i in range(side_square):
        row = i // params.side
        col = i % params.side
        for n in range(params.num):
            obj_index = entry_index(
                params.side, params.coords, params.classes, n * side_square + i, params.coords)
            scale = predictions[obj_index]
            if scale < threshold:
                continue
            box_index = entry_index(
                params.side, params.coords, params.classes, n * side_square + i, 0)
            # Network produces location predictions in absolute coordinates of feature maps.
            # Scale it to relative coordinates.
            x = (col + predictions[box_index + 0 * side_square]) / params.side
            y = (row + predictions[box_index + 1 * side_square]) / params.side
            # Value for exp is very big number in some cases so following construction is using here
            try:
                w_exp = exp(predictions[box_index + 2 * side_square])
                h_exp = exp(predictions[box_index + 3 * side_square])
            except OverflowError:
                continue
            # Depends on topology we need to normalize sizes by feature maps (up to YOLOv3) or by input shape (YOLOv3)
            w = w_exp * \
                params.anchors[2 * n] / \
                (resized_image_w if params.isYoloV3 else params.side)
            h = h_exp * params.anchors[2 * n + 1] / \
                (resized_image_h if params.isYoloV3 else params.side)
            for j in range(params.classes):
                class_index = entry_index(params.side, params.coords, params.classes, n * side_square + i,
                                          params.coords + 1 + j)
                confidence = scale * predictions[class_index]
                if confidence < threshold:
                    continue
                objects.append(scale_bbox(x=x, y=y, h=h, w=w, class_id=j, confidence=confidence,
                                          h_scale=orig_im_h, w_scale=orig_im_w))
    return objects


def intersection_over_union(box_1, box_2):
    width_of_overlap_area = min(
        box_1['xmax'], box_2['xmax']) - max(box_1['xmin'], box_2['xmin'])
    height_of_overlap_area = min(
        box_1['ymax'], box_2['ymax']) - max(box_1['ymin'], box_2['ymin'])
    if width_of_overlap_area < 0 or height_of_overlap_area < 0:
        area_of_overlap = 0
    else:
        area_of_overlap = width_of_overlap_area * height_of_overlap_area
    box_1_area = (box_1['ymax'] - box_1['ymin']) * \
        (box_1['xmax'] - box_1['xmin'])
    box_2_area = (box_2['ymax'] - box_2['ymin']) * \
        (box_2['xmax'] - box_2['xmin'])
    area_of_union = box_1_area + box_2_area - area_of_overlap
    if area_of_union == 0:
        return 0
    return area_of_overlap / area_of_union


def resize(image, size, keep_aspect_ratio, interpolation=cv2.INTER_LINEAR):
    if not keep_aspect_ratio:
        return cv2.resize(image, size, interpolation=interpolation)

    iw, ih = image.shape[0:2][::-1]
    w, h = size
    scale = min(w/iw, h/ih)
    nw = int(iw*scale)
    nh = int(ih*scale)
    image = cv2.resize(image, (nw, nh), interpolation=interpolation)
    new_image = np.full((size[1], size[0], 3), 128, dtype=np.uint8)
    dx = (w-nw)//2
    dy = (h-nh)//2
    new_image[dy:dy+nh, dx:dx+nw, :] = image
    return new_image


def preprocess_frame(frame, input_height, input_width, nchw_shape, keep_aspect_ratio):
    in_frame = resize(frame, (input_width, input_height), keep_aspect_ratio)
    if nchw_shape:
        # Change data layout from HWC to CHW
        in_frame = in_frame.transpose((2, 0, 1))
    in_frame = np.expand_dims(in_frame, axis=0)
    return in_frame

###temporary and weak
from params import default_params

def get_objects(output, net, new_frame_height_width, source_height_width, prob_threshold, is_proportional):
    objects = list()
    #function = ng.function_from_cnn(net)
    i=0
    for layer_name, out_blob in output.items():
        out_blob = out_blob.reshape(net.outputs[layer_name].shape)
        #params = [x._get_attributes() for x in net.get_ordered_ops()
        #          if x.get_friendly_name() == layer_name][0]
        #layer_params = YoloParams(params, out_blob.shape[2])
        ###temporary until ngraph is installed, on Raspbian
        params = default_params[i]
        i+=1

        layer_params = YoloParams(params, out_blob.shape[2])
        objects += parse_yolo_region(out_blob, net.input_info["input_1"].input_data.shape[2:], new_frame_height_width, layer_params,
                                     prob_threshold, is_proportional)

    return objects


def filter_objects(objects, iou_threshold, prob_threshold):
    # Filtering overlapping boxes with respect to the --iou_threshold CLI parameter
    objects = sorted(objects, key=lambda obj: obj['confidence'], reverse=True)
    for i in range(len(objects)):
        if objects[i]['confidence'] == 0:
            continue
        for j in range(i + 1, len(objects)):
            if intersection_over_union(objects[i], objects[j]) > iou_threshold:
                objects[j]['confidence'] = 0

    return tuple(obj for obj in objects if obj['confidence'] >= prob_threshold)


def async_callback(status, callback_args):
    request, frame_id, frame_mode, frame, start_time, completed_request_results, empty_requests, \
        mode, event, callback_exceptions = callback_args

    try:
        if status != 0:
            raise RuntimeError(
                'Infer Request has returned status code {}'.format(status))

        completed_request_results[frame_id] = (
            frame, request.output_blobs, start_time, frame_mode == mode.current)

        if mode.current == frame_mode:
            empty_requests.append(request)
    except Exception as e:
        callback_exceptions.append(e)

    event.set()


def put_highlighted_text(frame, message, position, font_face, font_scale, color, thickness, labels=None):
    cv2.putText(frame, message, position, font_face, font_scale,
                (255, 255, 255), thickness + 1)  # white border
    cv2.putText(frame, message, position, font_face,
                font_scale, color, thickness)


def await_requests_completion(requests):
    for request in requests:
        request.wait()
