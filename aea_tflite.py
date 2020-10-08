import logging as log
import sys
import argparse
import json

import numpy as np
import tflite_runtime.interpreter as tflite
from PIL import Image

from adl_edge_iot.datacls import PyDetectionBox, PyClassification
from aea_aicv_sdk import ObjectDetector, FrameClassifier, load_labels, frame_data_2_image


def argument_parser():
    log.info('Creating the argument parser...')
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('-m', '--model', type=str, required=True,
                        help='The model file')
    parser.add_argument('-l', '--label', type=str, required=True,
                        help='The label file')
    parser.add_argument('-d', '--is_detector', type=bool, required=False, default=True,
                        help='Indicate whether this is a detection model or a classification model.')
    parser.add_argument('-p', '--properties', type=str, required=False,
                        help='The URI (without file://) to the properties file.',
                        default='./config/Viewer.json')

    return parser.parse_args()


def get_output_tensor(interpreter: tflite.Interpreter, index: int) -> np.ndarray:
    """Returns the output tensor at the given index."""
    output_details = interpreter.get_output_details()[index]
    return np.squeeze(interpreter.get_tensor(output_details['index']))


def set_input_tensor(interpreter: tflite.Interpreter, frame: Image) -> None:
    """Sets the input tensor of the model to the current frame"""
    tensor_index = interpreter.get_input_details()[0]['index']
    input_tensor = interpreter.tensor(tensor_index)()[0]
    input_tensor[:, :] = frame


def build_classification_engine(model_file: str, labels_file: str, top_k=1):
    labels = load_labels(labels_file)
    interpreter = tflite.Interpreter(model_path=model_file)
    interpreter.allocate_tensors()
    _, input_height, input_width, _ = interpreter.get_input_details()[0]['shape']

    def inference(flow_id: str, frame: object) -> PyClassification:
        image = frame_data_2_image(frame, input_width, input_height)
        set_input_tensor(interpreter, image)
        interpreter.invoke()

        output_details = interpreter.get_output_details()[0]
        output = np.squeeze(interpreter.get_tensor(output_details['index']))

        # if the model is quantized (uint8 data), then dequantize the results
        if output_details['dtype'] == np.uint8:
            scale, zero_point = output_details['quantization']
            output = scale * (output - zero_point)

        ordered = np.argpartition(-output, top_k)

        result = PyClassification(frame_id=frame.frame_id, engine_id='tflite')
        for label_id in ordered[:top_k]:
            result.add_classification(category_id=label_id, category_label=labels[label_id], probability=output[label_id])

        return flow_id, result

    return inference


def build_detection_engine(model_file: str, labels_file: str):
    labels = load_labels(labels_file)
    interpreter = tflite.Interpreter(model_path=model_file)
    interpreter.allocate_tensors()
    _, input_height, input_width, _ = interpreter.get_input_details()[0]['shape']

    def inference(flow_id: str, frame: object) -> PyDetectionBox:
        """
        Returns a list of detection results, each a dictionary of object info
        """
        image = frame_data_2_image(frame, input_width, input_height)
        set_input_tensor(interpreter, image)
        interpreter.invoke()

        # Get all output details
        boxes = get_output_tensor(interpreter, 0)
        classes = get_output_tensor(interpreter, 1)
        scores = get_output_tensor(interpreter, 2)
        count = int(get_output_tensor(interpreter, 3))

        result = PyDetectionBox(frame_id=frame.frame_id, engine_id='tflite')
        for i in range(count):
            result.add_box(category_id=classes[i],category_label=labels.get(classes[i], ''),
                x1=boxes[i][1], y1=boxes[i][0], x2=boxes[i][3], y2=boxes[i][2], probability=float(scores[i]))

        return flow_id, result

    return inference


def main():
    args = vars(argument_parser())
    with open(args['properties']) as f:
        properties_str = json.load(f)
        properties_str = json.dumps(properties_str) if properties_str is not None else None

    if args['is_detector']:
        engine = ObjectDetector(properties_str=properties_str,
                                inference=build_detection_engine(args['model'], args['label']))
    else:
        engine = FrameClassifier(properties_str=properties_str,
                                 inference=build_classification_engine(args['model'], args['label']))

    engine.run()
    return 0


if __name__ == '__main__':
    sys.exit(main())
