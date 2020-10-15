This directory contains guidance on setting up an environment using [Anaconda](https://www.anaconda.com/) (_where possible_) for each of the reference integrations. For most of the environments a file with the list of packages is provided.

Installation instructions for Anaconda can be found [here](https://docs.anaconda.com/anaconda/install/) or altneratively the more compact Miniconda can be found [here](https://docs.conda.io/en/latest/miniconda.html).

- [Detectron2](#detectron2)
- [Darknet](#darknet)
- [ONNX Runtime](#onnxruntime)
- [OpenCV](#OpenCV)
- [OpenVINO Deep Learning Streamer](#ov_dls)
- [Tensorflow Lite](#tflite)
- [Torch Vision](#torch_vision)
- [Open Vision Capsules](#vision_capsules)


<a name="detectron2"/>

## Detectron2
```bash
$ conda create -n detectron2 --file detectron2_req.txt

$ conda activate detectron2

$ pip install ~/ADLINK/EdgeSDK/1.6.0/python/adlinktech_datariver-1.6.0-py2.py3-none-any.whl
```

<a name="onnxruntime"/>

## ONNX Runtime
```bash
$ conda create -n onnx_runtime --file onnx_runtime_req.txt

$ conda activate onnx_runtime

$ pip install -r onnx_runtime_pip.txt
$ pip install ~/ADLINK/EdgeSDK/1.6.0/python/adlinktech_datariver-1.6.0-py2.py3-none-any.whl
```

<a name="darknet"/>

## Darknet
```bash
$ conda create -n darknet --file darknet_req.txt

$ conda activate darknet
$ pip install ~/ADLINK/EdgeSDK/1.6.0/python/adlinktech_datariver-1.6.0-py2.py3-none-any.whl
```

<a name="OpenCV"/>

## OpenCV
```bash
$ conda create -n opencv --file opencv_req.txt

$ conda activate opencv
$ pip install ~/ADLINK/EdgeSDK/1.6.0/python/adlinktech_datariver-1.6.0-py2.py3-none-any.whl
```

<a name="ov_dls"/>

## OpenVINO Deep Learning Streamer
```bash
$ conda create -n ov_dls --file ov_dls_req.txt

$ conda activate ov_dls
$ pip install ~/ADLINK/EdgeSDK/1.6.0/python/adlinktech_datariver-1.6.0-py2.py3-none-any.whl
```

<a name="tflite"/>

## Tensorflow Lite
```bash
$ conda create -n tflite --file tflite_req.txt

$ conda activate tflite
$ pip install ~/ADLINK/EdgeSDK/1.6.0/python/adlinktech_datariver-1.6.0-py2.py3-none-any.whl
```

<a name="torch_vision"/>

## Torch Vision
```bash
$ conda create -n torch_vision --file torch_vision_req.txt

$ conda activate torch_vision
$ pip install ~/ADLINK/EdgeSDK/1.6.0/python/adlinktech_datariver-1.6.0-py2.py3-none-any.whl
```

<a name="vision_capsules"/>

## Open Vision Capsules
```bash
$ conda create -n vision_capsules --file vision_capsules_req.txt

$ conda activate vision_capsules
$ pip install ~/ADLINK/EdgeSDK/1.6.0/python/adlinktech_datariver-1.6.0-py2.py3-none-any.whl
```
