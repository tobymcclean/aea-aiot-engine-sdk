import json
import logging as log
import sys
import argparse

import numpy as np
import onnxruntime

from adl_edge_iot.datacls import PyClassification
from aea_aicv_sdk import frame_data_2_image, load_labels, FrameClassifier

log.basicConfig(format='[ %(levelname)s ] %(message)s', level=log.INFO, stream=sys.stdout)

def argument_parser():
    log.info('Creating the argument parser...')
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('-m', '--model', type=str, required=True,
                        help='The model file')
    parser.add_argument('-l', '--label', type=str, required=True,
                        help='The label file')
    parser.add_argument('-p', '--properties', type=str, required=False,
                        help='The URI (without file://) to the properties file.',
                        default='./config/Viewer.json')

def softmax(x):
    x = x.reshape(-1)
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def preprocess(input_data):
    # convert the input data into float32
    img_data = input_data.astype('float32')

    # normalize
    mean_vec = np.array([0.485, 0.456,0.406])
    stddev_vec = np.array([0.229, 0.224, 0.225])
    norm_img_data = np.zeros(img_data.shape).astype('float32')
    for i in range(img_data.shape[0]):
        norm_img_data[i,:,:] = (img_data[i,:,:]/255 - mean_vec[i]) / stddev_vec[i]

    # add batch channel
    norm_img_data = norm_img_data.reshape(1, 3, 224, 224).astype('float32')

    return norm_img_data

def postprocess(result):
    return softmax(np.array(result)).tolist()


def build_classification_engine(model_file:str, labels_file:str, top_k:int = 5):
    labels = load_labels(labels_file)
    session = onnxruntime.InferenceSession(model_file, None)

    # get the name of the first input of the model
    input_name = session.get_inputs()[0].name
    _, input_height, input_width, _ = session.get_inputs()[0].shape
    input_type = session.get_inputs()[0].shape

    log.info(f'Model input name: {input_name}')
    log.info(f'Model input shape: {input_height} x {input_width}')
    log.info(f'Model input type: {input_type}')


    def score(flow_id:str, frame:object) -> PyClassification:
        image = frame_data_2_image(frame, input_width, input_height)
        image_data = np.array(image).transpose(2, 0, 1)
        image_data = preprocess(image_data)
        raw_result = session.run([], {input_name:image_data})
        res = postprocess(raw_result)
        res_idx = np.squeeze(np.argsort(res))[-top_k:]
        result = PyClassification(engine_id='onnx_runtime', frame_id=frame.frame_id)
        for idx in res_idx:
            result.add_classification(category_id=idx, category_label= labels.get(idx, ''), probability=float(res[idx]))


        return flow_id, result

    return score


def main():
    args = vars(argument_parser())
    with open(args['properties']) as f:
        properties_str = json.load(f)
        properties_str = json.dumps(properties_str) if properties_str is not None else None

    engine = FrameClassifier(properties_str=properties_str,
                             inference=build_classification_engine(args['model'], args['label']))

    engine.run()
    return 0


if __name__ == '__main__':
    sys.exit(main())