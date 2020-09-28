import logging as log
import sys
import traceback
from typing import Callable, List, Tuple

from rx.subject import Subject

from adlinktech.datariver import IotNvpDataAvailableListener, Dispatcher, FlowState
from adlinktech.datariver import class_from_thing_input

from adl_edge_iot.datacls import PyClassification
from adl_edge_iot.datariver.things.edge_thing import EdgeThing
from adl_edge_iot.datacls.PyDetectionBox import PyDetectionBox
from adl_edge_iot.datariver.utils import write_tag


class FrameListener(IotNvpDataAvailableListener):
    def __init__(self, subject, data_class):
        super().__init__()
        self.__subject = subject
        self.__data_class = data_class

    def notify_data_available(self, samples):
        for sample in samples:
            if sample.flow_state == FlowState.ALIVE and sample.data.size() > 0:
                try:
                    data = self.__data_class(sample.data)
                    self.__subject.on_next((sample.flow_id, data))
                except Exception as e:
                    log.error(f'FrameListener: notify_data_available - An unexpected error occured: {e}')
                    traceback.print_exc(file=sys.stdout)


class __InferenceEngine(EdgeThing):
    def __init__(self,
                 properties_str: str,
                 inference: Callable[[str, object], Tuple[str,object]],
                 tag_groups: List[str],
                 thing_cls: List[str]):
        super().__init__(properties_str=properties_str,
                         tag_groups=tag_groups,
                         thing_cls=thing_cls)
        self.__frame_data_class = class_from_thing_input(self.dr, self.thing, 'VideoFrameData')
        self.__frame_subject = Subject()
        self.__listener = FrameListener(self.__frame_subject, self.__frame_data_class)
        self.__inference_fn = inference
        self.__frame_subject.map(lambda s: self.__inference_fn(s[0], s[1])).subscribe(self._write_inference)

    def _write_inference(self, obj: object) -> None:
        pass

    def run(self) -> None:
        """
        The main loop of the Thing, when it exits the lifecycle of the Thing is done.
        A listener is attached to the VideoFrame input and as the frames are received they a passed through
        the TensorFlow Object Detection API to transform the frame into a set of Regions of Interest (or
        Detection Boxes) that have a classification of what is in them.
        The result is packaged into a DetectionBox and written to the Data River.
        It exists when the `terminate` flage is set on the Thing by calling __exit__
        """
        dispatcher = Dispatcher()
        self.thing.add_listener(self.__listener, 'VideoFrameData', dispatcher)

        while not self.terminate:
            try:
                dispatcher.process_events(1000)
            except:
                continue


class ObjectDetector(__InferenceEngine):
    def __init__(self,
                 properties_str: str,
                 inference: Callable[[str, object], Tuple[str, PyDetectionBox]]):
        super().__init__(properties_str=properties_str,
                         inference=inference,
                         tag_groups=['com.vision.data/DetectionBoxTagGroup', 'com.vision.data/VideoFrameTagGroup'],
                         thing_cls=['com.vision.data/ObjectDetector'])

    def _write_inference(self, obj: Tuple[str, PyDetectionBox]) -> None:
        write_tag(self.thing, 'DetectionBoxData', obj[1].dr_data, flow=obj[0])


class FrameClassifier(__InferenceEngine):
    def __init__(self,
                 properties_str: str,
                 inference: Callable[[str, object], Tuple[str, PyClassification]]):
        super().__init__(properties_str=properties_str,
                         inference=inference,
                         tag_groups=['com.vision.data/ClassificationBoxTagGroup', 'com.vision.data/VideoFrameTagGroup'],
                         thing_cls=['com.vision.data/FrameClassifier'])

    def _write_inference(self, obj: Tuple[str, PyClassification]) -> None:
        write_tag(self.thing, 'ClassificationData', obj[1].dr_data, flow=obj[0])
