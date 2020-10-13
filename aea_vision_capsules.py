import argparse
import json
import logging as log
import sys
from pathlib import Path
from typing import Tuple

from vcap import NodeDescription, DetectionNode
from vcap.loading.capsule_loading import load_capsule

from adl_edge_iot.datacls import PyDetectionBox
from aea_aicv_sdk import frame_data_2_np_array, ObjectDetector

log.basicConfig(format='[ %(levelname)s ] %(message)s', level=log.INFO, stream=sys.stdout)



def argument_parser():
    log.info('Creating the argument parser...')
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('-c', '--capsule', type=str, required=True,
                        help='The capsule file')
    parser.add_argument('-p', '--properties', type=str, required=False,
                        help='The URI (without file://) to the properties file.',
                        default='./config/Viewer.json')


def add_detection_node_2_detection_box(node: DetectionNode, container: PyDetectionBox) -> None:
    container.add_box(category_label=node.name, x1=node.bbox.x1, y1=node.bbox.y1, x2=node.bbox.x2, y2=node.bbox.y2)


def build_engine(capsule_file: Path):
    capsule = load_capsule(capsule_file)

    def score(flow_id: str, frame: object) -> Tuple[str, PyDetectionBox]:
        img = frame_data_2_np_array(frame)
        if capsule.input_type.size is NodeDescription.Size.NONE:
            input_node = None
        else:
            input_node = DetectionNode(name='', coords=[[0, 0], [frame.width, 0], [frame.width, frame.height],
                                                        [0, frame.height]])

        if capsule.input_type.size is NodeDescription.Size.ALL:
            input_node = [input_node]

        result = capsule.process_frame(frame=img, detection_node=input_node, options=capsule.default_options,
                                       state=capsule.stream_state())
        detection_box = PyDetectionBox(frame_id=frame.frame_id, engine_id='vision_capsules')
        if isinstance(result, list):
            for node in result:
                add_detection_node_2_detection_box(node, detection_box)
        elif isinstance(result, DetectionNode):
            add_detection_node_2_detection_box(result, detection_box)

        return flow_id, detection_box

    return score


def main():
    args = vars(argument_parser())
    with open(args['properties']) as f:
        properties_str = json.load(f)
        properties_str = json.dumps(properties_str) if properties_str is not None else None

    engine = ObjectDetector(properties_str=properties_str,
                             inference=build_engine(args['capsule']))

    engine.run()
    return 0


if __name__ == '__main__':
    sys.exit(main())
