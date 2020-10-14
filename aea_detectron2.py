import json
import logging as log
import argparse
import sys
from typing import Tuple

import cv2
import torch

from detectron2.utils.logger import setup_logger

setup_logger()

import numpy as np
import pycocotools.mask as mask_util
from detectron2.structures import Boxes, RotatedBoxes, PolygonMasks, BitMasks
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog

from adl_edge_iot.datacls import PyDetectionBox
from aea_aicv_sdk import frame_data_2_np_array, ObjectDetector

detectron2_models = {
    # COCO Detection with Faster R-CNN
    "faster_rcnn_R_50_C4_1x": "COCO-Detection/faster_rcnn_R_50_C4_1x.yaml",
    "faster_rcnn_R_50_DC5_1x": "COCO-Detection/faster_rcnn_R_50_DC5_1x.yaml",
    "faster_rcnn_R_50_FPN_1x": "COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml",
    "faster_rcnn_R_50_C4_3x": "COCO-Detection/faster_rcnn_R_50_C4_3x.yaml",
    "faster_rcnn_R_50_FPN_3x": "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml",
    "faster_rcnn_R_101_C4_3x": "COCO-Detection/faster_rcnn_R_101_C4_3x.yaml",
    "faster_rcnn_R_101_DC5_3x": "COCO-Detection/faster_rcnn_R_101_DC5_3x.yaml",
    "faster_rcnn_R_101_FPN_3x": "COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml",
    "faster_rcnn_X_101_32x8d_FPN_3x": "COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml",
    # COCO Detection with RetinaNet
    "retinanet_R_50_FPN_1x": "COCO-Detection/retinanet_R_50_FPN_1x.yaml",
    "retinanet_R_50_FPN_3x": "COCO-Detection/retinanet_R_50_FPN_3x.yaml",
    "retinanet_R_101_FPN_3x": "COCO-Detection/retinanet_R_101_FPN_3x.yaml",
    # COCO Detection with RPN and Fast R-CNN
    "rpn_R_50_C4_1x": "COCO-Detection/rpn_R_50_C4_1x.yaml",
    "rpn_R_50_FPN_1x": "COCO-Detection/rpn_R_50_FPN_1x.yaml",
    "fast_rcnn_R_50_FPN_1x": "COCO-Detection/fast_rcnn_R_50_FPN_1x.yaml",
    # COCO Instance Segmentation Baselines with Mask R-CNN
    "mask_rcnn_R_50_C4_1x": "COCO-InstanceSegmentation/mask_rcnn_R_50_C4_1x.yaml",
    "mask_rcnn_R_50_DC5_1x": "COCO-InstanceSegmentation/mask_rcnn_R_50_DC5_1x.yaml",
    "mask_rcnn_R_50_FPN_1x": "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml",
    "mask_rcnn_R_50_C4_3x": "COCO-InstanceSegmentation/mask_rcnn_R_50_C4_3x.yaml",
    "mask_rcnn_R_50_DC5_3x": "COCO-InstanceSegmentation/mask_rcnn_R_50_DC5_3x.yaml",
    "mask_rcnn_R_50_FPN_3x": "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml",
    "mask_rcnn_R_101_C4_3x": "COCO-InstanceSegmentation/mask_rcnn_R_101_C4_3x.yaml",
    "mask_rcnn_R_101_DC5_3x": "COCO-InstanceSegmentation/mask_rcnn_R_101_DC5_3x.yaml",
    "mask_rcnn_R_101_FPN_3x": "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml",
    "mask_rcnn_X_101_32x8d_FPN_3x": "COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"  # noqa
}

class GenericMask:
    """
    From Detectron2 visualizer.py
    Attribute:
        polygons (list[ndarray]): list[ndarray]: polygons for this mask.
            Each ndarray has format [x, y, x, y, ...]
        mask (ndarray): a binary mask
    """

    def __init__(self, mask_or_polygons, height, width):
        self._mask = self._polygons = self._has_holes = None
        self.height = height
        self.width = width

        m = mask_or_polygons
        if isinstance(m, dict):
            # RLEs
            assert "counts" in m and "size" in m
            if isinstance(m["counts"], list):  # uncompressed RLEs
                h, w = m["size"]
                assert h == height and w == width
                m = mask_util.frPyObjects(m, h, w)
            self._mask = mask_util.decode(m)[:, :]
            return

        if isinstance(m, list):  # list[ndarray]
            self._polygons = [np.asarray(x).reshape(-1) for x in m]
            return

        if isinstance(m, np.ndarray):  # assumed to be a binary mask
            assert m.shape[1] != 2, m.shape
            assert m.shape == (height, width), m.shape
            self._mask = m.astype("uint8")
            return

        raise ValueError("GenericMask cannot handle object {} of type '{}'".format(m, type(m)))

    @property
    def polygons(self):
        if self._polygons is None:
            self._polygons, self._has_holes = self.mask_to_polygons(self._mask)
        return self._polygons

    def mask_to_polygons(self, mask):
        # cv2.RETR_CCOMP flag retrieves all the contours and arranges them to a 2-level
        # hierarchy. External contours (boundary) of the object are placed in hierarchy-1.
        # Internal contours (holes) are placed in hierarchy-2.
        # cv2.CHAIN_APPROX_NONE flag gets vertices of polygons from contours.
        mask = np.ascontiguousarray(mask)  # some versions of cv2 does not support incontiguous arr
        res = cv2.findContours(mask.astype("uint8"), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        hierarchy = res[-1]
        if hierarchy is None:  # empty mask
            return [], False
        has_holes = (hierarchy.reshape(-1, 4)[:, 3] >= 0).sum() > 0
        res = res[-2]
        res = [x.flatten() for x in res]
        res = [x for x in res if len(x) >= 6]
        return res, has_holes

    def bbox(self):
        p = mask_util.frPyObjects(self.polygons, self.height, self.width)
        p = mask_util.merge(p)
        bbox = mask_util.toBbox(p)
        bbox[2] += bbox[0]
        bbox[3] += bbox[1]
        return bbox


def argument_parser():
    log.info('Creating the argument parser...')
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('-m', '--model', type=str, required=True,
                        help='The name of the model')
    parser.add_argument('-p', '--properties', type=str, required=False,
                        help='The URI (without file://) to the properties file.',
                        default='./config/Viewer.json')

    return parser.parse_args()


def get_labels(classes, scores, class_names):
    labels = None
    if classes is not None and class_names is not None and len(class_names) > 0:
        labels = [class_names[i] for i in classes]
    if scores is not None:
        if labels is None:
            labels = ["{:.0f}%".format(s * 100) for s in scores]
        else:
            labels = ["{} {:.0f}%".format(l, s * 100) for l, s in zip(labels, scores)]
    return labels


def get_boxes(predictions):
    boxes = predictions.pred_boxes if predictions.has("pred_boxes") else None
    if boxes is not None:
        if isinstance(boxes, Boxes) or isinstance(boxes, RotatedBoxes):
            boxes = boxes.tensor.numpy()
        else:
            boxes = np.asarray(boxes)

    return boxes


def get_masks(predictions, height, width):
    if predictions.has("pred_masks"):
        masks = np.asarray(predictions.pred_masks)
        masks = [GenericMask(x, height, width) for x in masks]
    else:
        masks = None

    if masks is not None:
        if isinstance(masks, PolygonMasks):
            masks = masks.polygons
        if isinstance(masks, BitMasks):
            masks = masks.tensor.numpy()
        if isinstance(masks, torch.Tensor):
            masks = masks.numpy()

        ret = []
        for x in masks:
            if isinstance(x, GenericMask):
                ret.append(x)
            else:
                ret.append(GenericMask(x, height, width))
        masks = ret
    return masks


def build_engine(config_file: str):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(config_file))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(config_file)
    predictor = DefaultPredictor(cfg)
    metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
    labels = metadata.get('thing_classes', None)

    def inference(flow_id: str, frame: object) -> Tuple[str, PyDetectionBox]:
        image = frame_data_2_np_array(frame)
        outputs = predictor(image)
        predictions = outputs["instances"].to("cpu")
        boxes = get_boxes(predictions)
        masks = get_masks(predictions, frame.height, frame.width)
        scores = predictions.scores if predictions.has("scores") else None
        classes = predictions.pred_classes if predictions.has("pred_classes") else None
        box_labels = get_labels(classes, scores, labels)

        result = PyDetectionBox(frame_id=frame.frame_id, engine_id='detectron2')
        if boxes is not None or masks is not None:
            num_predictions = len(boxes) if boxes is not None else len(masks)
            for idx in range(num_predictions):
                x1, y1, x2, y2 = boxes[idx] if boxes is not None else masks[idx].bbox()
                result.add_box(category_id=classes[idx], category_label=box_labels[idx],
                               x1=x1, y1=y1, x2=x2, y2=x2,
                               probability=float(scores[idx]))

        return flow_id, result

    return inference


def main():
    args = vars(argument_parser())
    with open(args['properties']) as f:
        properties_str = json.load(f)
        properties_str = json.dumps(properties_str) if properties_str is not None else None

    model = detectron2_models.get(args['model'], None)
    if model is None:
        log.error(f"Unrecognied model {args['model']}. Should be one of \n{detectron2_models.keys()}")
        return 1

    engine = ObjectDetector(properties_str=properties_str,
                            inference=build_engine(model))
    engine.run()
    return 0


if __name__ == '__main__':
    sys.exit(main())
