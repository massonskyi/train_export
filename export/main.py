import os
from export import Export

MODEL: str = "yolov5n_512x512_leakyleru_1e.pt"
TYPE_EXPORT: str= "tflite"
TYPE_CONVERT: str = "tflite"

if __name__ == '__main__':
    export = Export(MODEL, os.getcwd())
    export.export(TYPE_EXPORT, int8=True, imgsz=(512,512))
    export.convert2rknn(TYPE_CONVERT)