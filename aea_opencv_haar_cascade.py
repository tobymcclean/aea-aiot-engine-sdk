import argparse
import json
import logging as log
import sys
from typing import Tuple, Union

import cv2

from adl_edge_iot.datacls import PyDetectionBox
from aea_aicv_sdk import ObjectDetector, frame_data_2_np_array


def argument_parser():
    log.info('Creating the argument parser...')
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('-pc', '--primary', type=str, required=True,
                        help='The primary cascade file.')
    parser.add_argument('-pl', '--primary_label', type=str, required=True,
                        help='The primary label.')
    parser.add_argument('-sc', '--secondary', type=str, required=False,
                        help='The secondary cascade file.', default=None)
    parser.add_argument('-sl', '--secondary_label', type=str, required=False,
                        help='The secondary label.', default='')
    parser.add_argument('-p', '--properties', type=str, required=False,
                        help='The URI (without file://) to the properties file.',
                        default='./config/Viewer.json')


def build_engine(primary_cascade_file: str, primary_label: str, secondary_cascade_file: Union[None, str] = None,
                 secondary_label: str = ''):
    cascade = cv2.CascadeClassifier()

    if not cascade.load(primary_cascade_file):
        log.error(f'Error loading cascade: {primary_cascade_file}')
        return None

    if secondary_cascade_file is not None:
        secondary_cascade = cv2.CascadeClassifier()
        if not secondary_cascade.load(secondary_cascade_file):
            log.error(f'Error loading cascade: {secondary_cascade_file}')
            return None
    else:
        secondary_cascade = None

    def score(flow_id: str, frame: object) -> Tuple[str, PyDetectionBox]:
        img = frame_data_2_np_array(frame)
        frame_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        frame_gray = cv2.equalizeHist(frame_gray)

        detection_box = PyDetectionBox(frame_id=frame.frame_id, engine_id='opencv-haar-cascades')

        # detection
        detections = cascade.detectMultiScale(frame_gray)

        for (x, y, w, h) in detections:
            detection_box.add_box(category_label=primary_label, x1=x, y1=y, x2=x + w, y2=y + h)
            if secondary_cascade is not None:
                roi = frame_gray[y:y + h, x:x + w]
                secondary_detections = secondary_cascade.detectMultiScale(roi)
                for (x_s, y_s, w_s, h_s) in secondary_detections:
                    detection_box.add_box(category_label=secondary_label, x1=x_s, y1=y_s, x2=x_s + w_s, y2=y_s + h_s)

        return flow_id, detection_box

    return score


def main():
    args = vars(argument_parser())
    with open(args['properties']) as f:
        properties_str = json.load(f)
        properties_str = json.dumps(properties_str) if properties_str is not None else None

    engine = ObjectDetector(properties_str=properties_str,
                            inference=build_engine(args['primary'], args['primary_label'], args['secondary'],
                                                   args['secondary_label']))

    engine.run()
    return 0


if __name__ == '__main__':
    sys.exit(main())
