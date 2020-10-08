Awareness is created by augmenting raw data (stemming from sensors) with derived-values (created by inference-engines that exploit machine learning models.) In many cases it is necessary to create such awareness at the edge. The awareness is then used to generate actionable insights that are timely and accurate. The insights present value in solving a particular problem.

One of the challenges many solution developers face is bringing their **_models_** to the edge to generate the insights in a timely manner. The challenges are many including
* the variety of data sources; and
* easy access to the awareness to generate the **_actionable insights_**.

This repository provides a set of Python utility functions and classes for building ADLINK Edge SDK applications to generate awareness (in a normalized and consistent format) based on machine vision. The integrations focus on integrating inference engines (e.g. TinyML, Intel OpenVINO, and NVIDIA TensorRT) as a stream processor. This is in contrast to solutions like TensorFlow serving that expose a model through a request/reply end-point (most often REST and/or gRPC).

For those interested in more on ADLINK Edge applications and the ADLINK Edge SDK please refer to [goto50.ai](http://www.goto50.ai).

### The devil is in the details
#### Building a new object detection integration
With the `aea_aicv_sdk.ObjectDetector` class the integration is as easy as providing a function that processes a frame/image and produces a `PyObjectDetection` object that contains the bounding boxes for the objects that were detected. An example of the function signature
```python
def inference(flow_id: str, frame: object) -> Tuple[str, PyDetectionBox]:
```
The `flow_id` parameter identifies the source context of the frame and allows for the application to process more that one source of data. And the `frame` parameter is an object with all of the attributes of the `DetectionBox` type found in _definitions/TagGroup/com.vision.data/VideoFrameTagGroup.json_ file. The resulting PyDetectionBox objects will be made available to other ADLINK Edge applications through the `DetectionBox` tag group (think database table).

| Engine | Reference integration | Inference funtion |
| ------ | --------------------- | ----------------- |
|Tensorflow Lite | aea_tflite.py | build_detection_engine |
| [Tensorflow 2 Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection) | aea_tf_object_detection_api.py | build_detection_engine |
| [Darknet](https://github.com/AlexeyAB/darknet) | aea_darknet.py | build_detection_engine |
| [NVIDIA Tensor RT](https://developer.nvidia.com/tensorrt)| work in progress | |
| [Intel OpenVINO](https://software.intel.com/en-us/openvino-toolkit) | work in progress | |
| [ArmNN](https://github.com/ARM-software/armnn) | work in progress | |
| [Rockchip NPU]() | work in progress | |
| [Qualcomm Neural Processing SDK](https://developer.qualcomm.com/software/qualcomm-neural-processing-sdk) | work in progress | |


#### Building a new frame classifierer integration
With the `aea_aicv_sdk.FrameClassifier` class the integration is as easy as providing a function that processes a frame/image and produces a `PyClassification` object that contains the top-K classifications for the frame along with their confidence levels. An example of the function signature
```python
def inference(flow_id: str, frame: object) -> Tuple[str, PyClassification]:
```
The `flow_id` parameter identifies the source context of the frame and allows for the application to process more that one source of data. And the `frame` parameter is an object with all of the attributes of the `DetectionBox` type found in _definitions/TagGroup/com.vision.data/VideoFrameTagGroup.json_ file. The resulting PyClassification objects will be made available to other ADLINK Edge applications through the `Classification` tag group (think database table).

| Engine | Reference integration | Inference funtion |
| ------ | --------------------- | ----------------- |
|Tensorflow Lite | aea_tflite.py | build_classification_engine|
| [ONNX Runtime](https://github.com/microsoft/onnxruntime) | aea_onnx_runtime.py | build_classification_engine |

## Dependencies
| Software | Version |
| -------- | ------- |
| ADLINK Edge SDK | 1.6.0 |