import argparse
import sys
import logging as log
import json
import time

import cv2

from adl_edge_iot.datacls import PyFrameData
from adl_edge_iot.datariver.things.edge_thing import EdgeThing
from adl_edge_iot.datariver.utils import write_tag, SIGTERMHandler

PROPERTIES_FILE = './etc/config/properties.json'

log.basicConfig(format='[ %(levelname)s ] %(message)s', level=log.INFO, stream=sys.stdout)

def argument_parser():
    log.info('Creating the argument parser...')
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('-i', '--video_file', type=str, required=True,
                        help='Mandatory file name of the video file to load and stream.')
    parser.add_argument('-f', '--fps', type=int, required=False,
                        help='The framerate to read the video file',
                        default=30)
    parser.add_argument('-p', '--properties', type=str, required=False,
                        help='The URI (without file://) to the properties file.',
                        default='etc/config/frame_streamer.json')
    return parser.parse_args()


def init_edge_thing(properties_file) -> EdgeThing:
    with open(properties_file) as f:
        properties_str = json.load(f)
        properties_str = json.dumps(properties_str) if properties_str is not None else None

    return EdgeThing(properties_str=properties_str,
                     tag_groups=['com.adlinktech.vision.capture/2.000/VideoFrameTagGroup'],
                     thing_cls=['com.adlinktech.vision/FrameStreamer'])

def build_exit_handler(thing):
    def exit_handler(signum, frame):
        log.info('Exiting the frame streamer.')
        thing.__exit__()

    return exit_handler

def main() -> int:
    """
    The main loop of the Thing, when it exits the lifecycle of the Thing is done.
    It loads the video file using OpenCV and then loops reading each frame individually. Within
    """
    args = vars(argument_parser())
    video_file = args['video_file']
    fps = args['fps']
    properties_file = args['properties']

    thing = init_edge_thing(properties_file)
    sigterm_handler = SIGTERMHandler(build_exit_handler(thing))

    frame_counter = 0
    frame_data = PyFrameData()
    delay = 1 / fps
    cap = cv2.VideoCapture(video_file)
    ret, frame = cap.read()

    if ret:
        frame_data.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_data.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_data.channels = 1 if frame.ndim == 2 else frame.shape[2]
        frame_data.framerate = fps

    while ret:
        frame_counter += 1
        frame_bytes = frame.tobytes()
        frame_data.frame_id = frame_counter
        frame_data.timestamp = int(time.time())
        frame_data.size = len(frame_bytes)
        frame_data.video_data = frame_bytes
        write_tag(thing.thing, 'VideoFrameData', frame_data.dr_data)
        time.sleep(delay)
        ret, frame = cap.read()

    cap.release()
    thing.__exit__()

    return 0

if __name__ == '__main__':
    sys.exit(main())