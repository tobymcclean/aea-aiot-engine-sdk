import json
import logging as log
import argparse
import sys

from gstgva import VideoFrame, util
from gstreamer import GstContext
from adl_edge_iot.datacls import PyDetectionBox
from aea_aicv_gst_sdk import GstEngine


def argument_parser():
    log.info('Creating the argument parser...')
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('-c', '--command', type=str, required=True,
                        help='The gstreamer command, and it must start with appsrc and end with appsink')
    parser.add_argument('-f', '--flow_id', type=str, required=True,
                        help='The flow_id to write the results to')
    parser.add_argument('-p', '--properties', type=str, required=False,
                        help='The URI (without file://) to the properties file.',
                        default='./config/Inference.json')

    return parser.parse_args()


def build_handler():
    class shared:
        pass

    shared.frame_id = 0

    def frame_handler(buffer, caps) -> PyDetectionBox:
        shared.frame_id += 1
        frame = VideoFrame(buffer, caps=caps)
        result = PyDetectionBox(frame_id=shared.frame_id, engine_id='deep-learning-streamer')
        for roi in frame.regions():
            rect = roi.normalized_rect()
            x1, y1, x2, y2 = rect.x, rect.y, rect.x + rect.w, rect.y + rect.h
            for tensor in roi.tensors():
                object_id = tensor.object_id()
                object_id = 0 if object_id is None else object_id
                if tensor.is_detection():
                    confidence = roi.confidence()
                    confidence = 0.0 if confidence is None else confidence
                    result.add_box(category_id=roi.label_id(), category_label=roi.label(), x1=x1, y1=y1, x2=x2, y2=y2,
                                   probability=float(confidence))
                else:
                    confidence = tensor.confidence()
                    confidence = 0.0 if confidence is None else confidence
                    label_id = tensor.label_id()
                    label_id = 0 if label_id is None else 0
                    result.add_box(tracker_obj_id=object_id, category_id=label_id, category_label=tensor.label(), x1=x1,
                                   y1=y1, x2=x2, y2=y2,
                                   probability=float(confidence),
                                   metadata=f'name={tensor.name()},layer={tensor.layer_name()},model={tensor.model_name()}')
        return result

    return frame_handler


def main():
    args = vars(argument_parser())
    with open(args['properties']) as f:
        properties_str = json.load(f)
        properties_str = json.dumps(properties_str) if properties_str is not None else None

    with GstContext():
        engine = GstEngine(properties_str=properties_str, flow_id=args['flow_id'], command=args['command'],
                           sink_handler=build_handler())

    engine.run()
    return 0


if __name__ == '__main__':
    sys.exit(main())
