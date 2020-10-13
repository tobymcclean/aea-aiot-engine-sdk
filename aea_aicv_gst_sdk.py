import logging as log
import time
from typing import Union, Any, Callable

from fractions import Fraction

from gstreamer import GstPipeline, GstApp, Gst, GstVideo, GLib
import gstreamer.utils as utils

from rx.subject import Subject

from adlinktech.datariver import Dispatcher
from adlinktech.datariver import class_from_thing_input

from adl_edge_iot.datariver.things.edge_thing import EdgeThing
from adl_edge_iot.datacls import PyDetectionBox
from aea_aicv_sdk import FrameListener, frame_data_2_np_array
from adl_edge_iot.datariver.utils import write_tag


def parse_caps(pipeline : str) -> Union[dict, None]:
    try:
        caps = [prop for prop in pipeline.split('!')[0].split(' ') if 'caps' in prop][0]
        return dict([p.split('') for p in caps.split(',') if '=' in p])
    except IndexError as err:
        log.error('There was a problem parsing the appsrc caps.')
        return None


def fraction_to_str(fraction:Fraction) -> str:
    return f'{fraction.numerator}/{fraction.denominator}'


class GstEngine(EdgeThing):
    def __init__(self, properties_str: str, flow_id: str, sink_handler : Callable, command : str, height_default:int=480, width_default:int=640,
                 framerate_default:Fraction=Fraction(30), video_format_default:str='RGB'):
        super().__init__(properties_str=properties_str,
                         tag_groups=['com.adlinktech.vision.inference/2.000/DetectionBoxTagGroup',
                                     'com.adlinktech.vision.capture/2.000/VideoFrameTagGroup'],
                         thing_cls=['com.adlinktech.vision/ObjectDetector'])
        self.__flow_id = flow_id
        self.__sink_handler = sink_handler
        self.__frame_data_class = class_from_thing_input(self.dr, self.thing, 'VideoFrameData')
        self.__frame_subject = Subject()
        self.__listener = FrameListener(self.__frame_subject, self.__frame_data_class)
        args_caps = parse_caps(command)
        self.command = command
        self.width = int(args_caps.get('width', width_default))
        self.height = int(args_caps.get('height', height_default))
        fps = Fraction(args_caps.get('framerate', framerate_default))
        self.video_format = args_caps.get('format', video_format_default)
        self.channels = utils.get_num_channels(self.video_format)
        self.dtype = utils.get_np_dtype(self.video_format)
        self.fps_str = fraction_to_str(fps)
        self.caps = f'video/x-raw,forma={self.video_format},width={self.width},height={self.height},framerate={self.fps_str}'
        self.duration = 10 ** 9 / (fps.numerator / fps.denominator)  # frame duration
        self.pipeline = None
        self.app_src = None
        self.app_sink = None
        self.terminated = False
        self.pts = self._pts()
        self.__frame_subject.map(lambda s: self.__emit(s[0], s[1]))


    def _pts(self):
        pts = 0
        while True:
            pts += self.duration
            yield pts


    def __on_pipeline_init(self):
        app_src = self.pipeline.get_by_cls(GstApp.AppSrc)[0]  # get AppSrc
        app_sink = self.pipeline.get_by_cls(GstApp.AppSink)[0]  # get AppSrc

        # instructs appsrc that we will be dealing with a timed buffer
        app_src.set_property('format', Gst.Format.TIME)

        # instructs appsrc to block pushing buffers until ones in queue are preprocessed
        # allows to avoid huge queue size in appsrc
        app_src.set_property('block', True)

        # set input format (caps)
        app_src.set_caps(Gst.Caps.from_string(self.caps))

        # instructs appsink to emit signals
        app_sink.set_property('emit-signals', True)

        app_sink.connect('new-sample', self.__on_buffer, None)



    def run(self):
        self.pipeline = GstPipeline(self.command)
        # override on_pipeline_init to se specific properties before launching pipeline
        self.pipeline._on_pipeline_init = self.__on_pipeline_init

        try:
            self.pipeline.startup()
            self.app_src = self.pipeline.get_by_cls(GstApp.AppSrc)[0]
            self.app_sink = self.pipeline.get_by_cls(GstApp.AppSink)[0]
        except Exception as e:
            log.error('Problem starting pipeline')
            self.terminate()

        dispatcher = Dispatcher()
        self.thing.add_listener(self.__listener, 'VideoFrameData', dispatcher)

        while not self.terminate:
            try:
                dispatcher.process_events(1000)
            except:
                continue

    def __emit(self, flow_id: str, frame: object):
        array = frame_data_2_np_array(frame)
        gst_buffer = utils.ndarray_to_gst_buffer(array)
        gst_buffer.pts = next(self.pts)
        gst_buffer.duration = self.duration

        self.app_src.__emit('push-buffer', gst_buffer)


    def __on_buffer(self, sink: GstApp.AppSink, data: Any) -> Gst.FlowReturn:
        sample = sink.__emit('pull-sample')
        buffer = sample.get_buffer()
        caps = sample.get_caps()
        result = self.__sink_handler(buffer, caps)
        self.__write_inference(result)
        return Gst.FlowReturn.OK


    def __write_inference(self, obj: PyDetectionBox) -> None:
        write_tag(self.thing, 'DetectionBoxData', obj.dr_data, flow=self.__flow_id)

    def terminate(self):
        if self.app_src is not None:
            self.terminated = True
            self.app_src.__emit('end-of-stream')
        if self.pipeline is not None:
            while not self.pipeline.is_done:
                time.sleep(.1)
            self.pipeline.shutdown()


