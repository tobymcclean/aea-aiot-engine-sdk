import argparse
import json
import logging as log
import sys
from collections import namedtuple
from typing import Callable

import torch
from torchvision import models, transforms
from torchvision.transforms import Compose

from adl_edge_iot.datacls import PyDetectionBox, PyClassification
from aea_aicv_sdk import ObjectDetector, FrameClassifier, load_labels, frame_data_2_image

TVModel = namedtuple('TVModel', ['model', 'type'])

tv_models = {
    'resnet18': TVModel(models.resnet18, 'classification'),
    'alexnet': TVModel(models.alexnet, 'classification'),
    'squeezenet': TVModel(models.squeezenet1_1, 'classification'),
    'vgg16': TVModel(models.vgg16, 'classification'),
    'densenet': TVModel(models.densenet161, 'classification'),
    'inception': TVModel(models.inception_v3, 'classification'),
    'googlenet': TVModel(models.googlenet, 'classification'),
    'shufflenet': TVModel(models.shufflenet_v2_x1_0, 'classification'),
    'mobilenet': TVModel(models.mobilenet_v2, 'classification'),
    'resnext50': TVModel(models.resnext50_32x4d, 'classification'),
    'wide_resnet50': TVModel(models.wide_resnet50_2, 'classification'),
    'mnasnet': TVModel(models.mnasnet1_0, 'classification'),
    'fasterrcnn_resnet50': TVModel(models.detection.fasterrcnn_resnet50_fpn, 'detection')
}


def argument_parser():
    log.info('Creating the argument parser...')
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('-m', '--model', type=str, required=True,
                        help='The model', choices=['resnet18', 'alexnet', 'squeezenet', 'vgg16',
                                                   'densenet', 'inception', 'googlenet', 'shufflenet',
                                                   'mobilenet', 'resnext50', 'wide_resnet50', 'mnasnet',
                                                   'fasterrcnn_resnet50'])
    parser.add_argument('-l', '--label', type=str, required=True,
                        help='The label file')
    parser.add_argument('-p', '--properties', type=str, required=False,
                        help='The URI (without file://) to the properties file.',
                        default='./config/Viewer.json')

    return parser.parse_args()


def get_transform(input_size: int = 256, center_crop: int = 224) -> Compose:
    return transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(center_crop),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def build_classification_engine(model: Callable, labels_file: str, top_k=1):
    labels = load_labels(labels_file)
    engine = model(pretrained=True)
    engine.eval()
    transform = get_transform()

    def inference(flow_id: str, frame: object) -> PyClassification:
        image = frame_data_2_image(frame, frame.width, frame.height)
        image_t = transform(image)
        image_t = torch.unsqueeze(image_t, 0)

        output = engine(image_t)
        percentage = torch.nn.functional.softmax(output, dim=1)[0]
        _, indices = torch.sort(output, descending=True)
        result = PyClassification(frame_id=frame.frame_id, engine_id='torchvision')
        for idx in indices[0][:top_k]:
            result.add_classification(category_id=idx, category_label=labels[idx],
                                      probability=percentage[idx].item())

        return flow_id, result

    return inference


def build_detection_engine(model: Callable, labels_file: str):
    labels = load_labels(labels_file)
    engine = model(pretrained=True)
    engine.eval()
    transform = get_transform()

    def inference(flow_id: str, frame: object) -> PyDetectionBox:
        """
        Returns a list of detection results, each a dictionary of object info
        """
        image = frame_data_2_image(frame, frame.width, frame.height)
        image_t = transform(image)
        image_t = torch.unsqueeze(image_t, 0)

        output = engine(image_t)

        # Get all output details
        boxes = output[0]['boxes']
        classes = output[0]['labels']
        scores = output[0]['scores']

        result = PyDetectionBox(frame_id=frame.frame_id, engine_id='tflite')
        for i, box in enumerate(boxes):
            result.add_box(category_id=classes[i], category_label=labels.get(classes[i], ''),
                           x1=box[0], y1=box[1], x2=box[2], y2=box[3], probability=float(scores[i]))

        return flow_id, result

    return inference


def main():
    args = vars(argument_parser())
    with open(args['properties']) as f:
        properties_str = json.load(f)
        properties_str = json.dumps(properties_str) if properties_str is not None else None

    tv_model = tv_models[args['model']]

    if tv_model.type == 'detection':
        engine = ObjectDetector(properties_str=properties_str,
                                inference=build_detection_engine(tv_model.model, args['label']))
    elif tv_model.type == 'classification':
        engine = FrameClassifier(properties_str=properties_str,
                                 inference=build_classification_engine(tv_model.model, args['label']))
    else:
        log.error('Unrecognized model type.')
        return 1

    engine.run()
    return 0


if __name__ == '__main__':
    sys.exit(main())
