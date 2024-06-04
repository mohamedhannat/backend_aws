import multiprocessing
from multiprocessing import Process
from collections import deque
import traceback
import time as tm
from ultralytics import YOLO
import json
import os
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'TV', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush'
]


class StreamProcess:
    
    def __init__(self):
        self.videos_frame_ts = deque(maxlen=20)
        self.signals_to_send = multiprocessing.Manager().Queue()
        self.video_frames = multiprocessing.Manager().Queue()
        self.yolo_process = None

    def read_video_frame(self, frame, ts):
        """
        Read video frame received from webrtc server and save it inside models_input to be served by 
        ModelFrameGenerator to models that need video frames as input.
        """
        self.video_frames.put((frame,ts))
       
    def get_signals_to_send(self):
        return(self.signals_to_send)

    def get_state(self):
        """
        Check and Return if there is some FE signal to send to webrtc client.
       
        Returns:
            str: FE Signal to send to the webrtc client.
        """
        if(self.signals_to_send.qsize()>0):
            data= self.signals_to_send.get()
            return(data)
        else:
            return("")
    
    def yolo_processing(self, model_name):
        ov_model = YOLO("yolov8n_openvino_model/", task="detect")
        #load yolo model
        results = ov_model.predict(np.zeros((320, 320, 3)), imgsz=320)

        while(True):
            try: 
                frame, ts = self.video_frames.get()
                
                # Process the frame with YOLOv8
                results = ov_model.predict(frame, imgsz=320)

                # Extract person detections
                detections = []
                for result in results[0].boxes:
                    x1, y1, x2, y2 = result.xyxy[0]
                    confidence = result.conf[0]
                    cls = int(result.cls)
                    class_name = COCO_CLASSES[cls]
                    detections.append({
                        'x1': int(x1),
                        'y1': int(y1),
                        'x2': int(x2),
                        'y2': int(y2),
                        'confidence': float(confidence),
                        'class': class_name
                    })
                #print("process frame: ", ts, detections)
                self.signals_to_send.put(json.dumps(detections))
            except:
                pass

    def run(self):
        model_name = "yolov8n.pt"
        model_name = "yolov8n.onnx"
        self.yolo_process = Process(target= self.yolo_processing, args=(model_name,), daemon=True)
        self.yolo_process.start()

    
    def stop(self):
        """
        Terminate process.
        """
        try:
            if(self.yolo_process is not None):
                self.yolo_process.terminate()
        except:
            pass
    

   
