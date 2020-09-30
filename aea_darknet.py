import logging as log
from io import BytesIO
import json
import sys

import darknet
import argparse

from adl_edge_iot.datacls import PyDetectionBox
from aea_aicv_sdk import frame_data_2_image, ObjectDetector


def argument_parser():
    log.info('Creating the argument parser...')
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('-c', '--config', type=str, required=True,
                        help='The config file')
    parser.add_argument('-d', '--data', type=str, required=True,
                        help='The data file')
    parser.add_argument('-w', '--weights', type=str, required=True,
                        help='The weights file')
    parser.add_argument('-t', '--threshold', type=float, required=False, default=0.5,
                        help='')
    parser.add_argument('-p', '--properties', type=str, required=False,
                        help='The URI (without file://) to the properties file.',
                        default='./config/Viewer.json')

    return parser.parse_args()


def frame_data_2_bytes(frame : object, input_width : int, input_height : int) -> bytes:
    image = frame_data_2_image(frame, input_width, input_height)
    with BytesIO() as output:
        image.save(output, 'BMP')
    return output.getvalue()


def build_detection_engine(config_file : str, data_file : str, weights : str, thresh : float):
    network, class_names, class_colors = darknet.load_network(
        config_file,
        data_file,
        weights,
        batch_size=1
    )
    labels_rev = {}
    for i, label in enumerate(class_names):
        labels_rev[label] = i

    width = darknet.network_width(network)
    height = darknet.network_height(network)
    darknet_image = darknet.make_image(width, height, 3)

    def inference(flow_id:str, frame:object):
        image = frame_data_2_bytes(frame, width, height)
        darknet.copy_image_from_bytes(darknet_image, image)
        detections = darknet.detect_image(network, class_names, darknet_image, thresh=thresh)
        result = PyDetectionBox(stream_id=frame.stream_id, frame_id=frame.frame_id)
        for label, confidence, bbox in detections:
            left, top, right, bottom = darknet.bbox2points(bbox)
            result.add_box(0, '', labels_rev.get(label, ''), label, left, top, right, left, float(confidence), '')
        darknet.free_image(darknet_image)

        return flow_id, result

    return inference

def main():
    args = vars(argument_parser())
    with open(args['properties']) as f:
        properties_str = json.load(f)
        properties_str = json.dumps(properties_str) if properties_str is not None else None


    engine = ObjectDetector(properties_str=properties_str,
                            inference=build_detection_engine(args['config'], args['data'], args['weights']))

    engine.run()
    return 0


if __name__ == '__main__':
    sys.exit(main())