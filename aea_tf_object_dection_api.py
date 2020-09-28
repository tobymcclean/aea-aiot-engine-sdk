import logging as log
import os
import sys
import argparse
import json

import numpy as np
from PIL import Image

from adl_edge_iot.datacls import PyDetectionBox
from aea_aicv_sdk import ObjectDetector, load_labels, frame_data_2_image, frame_data_2_np_array

import tensorflow as tf
from object_detection.utils import label_map_util, config_util
from object_detection.builders import model_builder


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

def get_input_tensor(frame: np.ndarray) -> tf.Tensor:
    """Gets the input tensor of the model to the current frame"""
    return tf.convert_to_tensor(
        np.expand_dims(frame, 0), dtype=tf.float32)


def get_interpreter(pipeline_config : str, model_dir: str):
    configs = config_util.get_configs_from_pipeline_file(pipeline_config)
    model_config = configs['model']
    model = model_builder.build(
        model_config=model_config, is_training=False)
    ckpt = tf.compat.v2.train.Checkpoint(model=model)
    ckpt.restore(os.path.join(model_dir, 'ckpt-0')).expect_partial()


    @tf.function
    def inference(frame : object):
        image, shapes = model.preprocess(frame)
        prediction_dict = model.predict(image, shapes)
        detections = model.postprocess(prediction_dict, shapes)
        return detections, prediction_dict, tf.reshape(shapes, [-1])

    return inference


def build_detection_engine(pipeline_config : str, model_dir: str, labels_file: str):
    labels = label_map_util.create_category_index_from_labelmap(labels_file, use_display_name=True)
    interpreter = get_interpreter(pipeline_config, model_dir)

    def inference(flow_id: str, frame: object) -> PyDetectionBox:
        """
        Returns a list of detection results, each a dictionary of object info
        """
        image = frame_data_2_np_array(frame)
        input_tensor = get_input_tensor(image)
        detections, predictions_dict, shapes = interpreter(input_tensor)
        boxes = detections['detection_boxes'][0].numpy()
        classes = detections['detection_classes'][0].numpy()
        scores = detections['dection_scores'][0].numpy()

        result = PyDetectionBox(frame_id=frame.frame_id, stream_id=flow_id)

        for i, box in enumerate(boxes):
            ymin, xmin, ymax, xmax = box.tolist()
            probability = float(scores[i])
            class_id = classes[i]
            class_label = labels.get(class_id, 'N/A')
            result.add_box(0, '', class_id, class_label, xmin, ymin, xmax, ymax, probability, '')

        return flow_id, result

    return inference

def main():
    args = vars(argument_parser())
    with open(args['properties']) as f:
        properties_str = json.load(f)
        properties_str = json.dumps(properties_str) if properties_str is not None else None

    engine = ObjectDetector(properties_str=properties_str,
                                inference=build_detection_engine(args['model'], args['label']))

    engine.run()
    return 0


if __name__ == '__main__':
    sys.exit(main())