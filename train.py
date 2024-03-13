import utils.functions as f
from modelworker import YoloModelWorker
import os

if not os.path.exists('datasets'):
    f.load_datasets("realtime-drone-detected",
        "test-real-time-drone-detected",
        3,
        "yolov5"
    )

model = YoloModelWorker('yolov5s.yaml',"yolov5su.pt")


_ = model.train(
    data="./datasets/data.yaml",
)

_ = model.val(
    data="./datasets/data.yaml",
)

_ = model.export(
    format="onnx",
    imgsz=(640, 640),
    half=True,
    dynamic=False,
    simplify=True,
    opset=12
)