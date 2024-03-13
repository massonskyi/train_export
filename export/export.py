import os
import shutil
from typing import Union
from ultralytics import YOLO
from rknn.api import RKNN

__all__ = ['Export']

from utils import Tools
import utils.functions as f
QUANTIZE_ON = [True,
               True,
               False,
               False,
               False
]



class Export:
    def __init__(self, _model_path: str, _working_dir: str = os.getcwd()) -> None:
        self._model_path = _model_path
        self.model = YOLO(self._model_path)
        self.working_dir = _working_dir

    @Tools.timeit
    def export(self, export_type: str, opset: int = 12, imgsz: Union[int, tuple] = (640, 640),
        half: bool = False, int8:bool = False) \
            -> bool:
        if export_type == "onnx":
            return self._export_onnx(opset)
        elif export_type == "tflite":
            try:
                if self._export_tflite(opset, imgsz, half, int8):
                    self.process_files_after_export(self._find_dir())
                    return True
                else:
                    return False
            except:
                ...
            finally:
                self.process_files_after_export(self._find_dir())
                return True
        else:
            raise ValueError("Invalid export type")

    @Tools.timeit
    def _export_onnx(self, _opset: int = 12) -> bool:
        return self.model.export(format="onnx", opset=_opset)

    @Tools.timeit
    def _export_tflite(self, _opset: int = 12,
        imgsz: Union[int, tuple] = (640, 640), half: bool = False, int8:bool = False) \
            -> bool:
        return self.model.export(format="tflite", opset=_opset, imgsz=imgsz, half=half, int8=int8)

    @Tools.timeit
    def convert2rknn(self, model_type: str,  _target_platform='rk3588'):

        DATASET = './dataset.txt'

        f.check_dataset(DATASET)
        f.checking_dir([self.working_dir + "/models", self.working_dir + "/export"])

        models_path = self.working_dir + "/models"
        export_path = self.working_dir + "/export"

        models = [f"{models_path}/f16.tflite",
                  f"{models_path}/f32.tflite",
                  f"{models_path}/fiq.tflite",
                  f"{models_path}/i8.tflite",
                  f"{models_path}/iq.tflite"
        ]

        for i, model in enumerate(models):
            # Create RKNN object
            rknn = RKNN(verbose=True)

            # pre-process config
            print('--> Config model')
            rknn.config(mean_values=[[0, 0, 0]],
                std_values=[[255, 255, 255]],
                target_platform=_target_platform

            )
            print('done')

            # Load ONNX model
            if model_type == "tflite":
                print('--> Loading model')
                ret = rknn.load_tflite(model=model)
                if ret != 0:
                    print('Load model failed!')
                    exit(ret)
                print('done')
            elif model_type == "onnx":
                print('--> Loading model')
                ret = rknn.load_onnx(model=model)
                if ret != 0:
                    print('Load model failed!')
                    exit(ret)
                print('done')
            # Build model
            print('--> Building model')
            ret = rknn.build(do_quantization=QUANTIZE_ON[i], dataset=DATASET)
            if ret != 0:
                print('Build model failed!')
                exit(ret)
            print('done')

            # Export RKNN model
            print('--> Export rknn model')
            ret = rknn.export_rknn(f"{export_path}/{f.get_filename_without_extension(model)}.rknn")
            if ret != 0:
                print('Export rknn model failed!')
                exit(ret)
            print('done')
            print(f"Saved model to {self.working_dir}/export")

    @Tools.timeit
    def _find_dir(self):
        item_path = f"{os.path.join(self.working_dir, self._model_path[:-3])}_saved_model"

        if os.path.isdir(item_path):
           return item_path

        return None

    @Tools.timeit
    def process_files_after_export(self, dir):
        @Tools.timeit
        def preprocess(directory):
            for item in os.listdir(directory):
                item_path = os.path.join(directory, item)

                if os.path.isdir(item_path):
                    preprocess(item_path)
                    os.rmdir(item_path)
                    print(f"Removed directory: {item_path}")

                elif os.path.isfile(item_path) and not item.endswith(".tflite"):
                    os.remove(item_path)
                    print(f"Removed file: {item_path}")

        @Tools.timeit
        def process(directory, old_pattern, new_pattern):
            for filename in os.listdir(directory):
                for old, new in zip(old_pattern, new_pattern):
                    if old in filename:
                        # Create the new filename using the new pattern
                        new_filename = filename.replace(old, new)

                        # Create the full paths for the old and new filenames
                        old_file_path = os.path.join(directory, filename)
                        new_file_path = os.path.join(directory, new_filename)

                        # Rename the file
                        os.rename(old_file_path, new_file_path)

                        print(f"Renamed {old_file_path} to {new_file_path}")

        @Tools.timeit
        def postprocess(src_dir, dest_dir):
            if not os.path.exists(dest_dir):
                os.makedirs(dest_dir)

            # Iterate over all files in the source directory
            for filename in os.listdir(src_dir):
                # Create the full paths for the source and destination files
                src_file_path = os.path.join(src_dir, filename)
                dest_file_path = os.path.join(dest_dir, filename)

                # Copy the file to the destination directory
                shutil.copy2(src_file_path, dest_file_path)

                print(f"Copied {src_file_path} to {dest_file_path}")

            for item in os.listdir(src_dir):
                item_path = os.path.join(src_dir, item)

                if os.path.isdir(item_path):
                    preprocess(item_path)
                    os.rmdir(item_path)
                    print(f"Removed directory: {item_path}")

                elif os.path.isfile(item_path):
                    os.remove(item_path)
                    print(f"Removed file: {item_path}")

            os.rmdir(src_dir)


        preprocess(dir)

        model_pattern = [f"{self._model_path[:-3]}_float16.tflite", f"{self._model_path[:-3]}_float32.tflite",
                         f"{self._model_path[:-3]}_full_integer_quant.tflite", f"{self._model_path[:-3]}_int8.tflite",
                         f"{self._model_path[:-3]}_integer_quant.tflite"]
        new_pattern = [
            "f16.tflite", "f32.tflite", "fiq.tflite", "i8.tflite", "iq.tflite"
        ]

        process(dir, model_pattern, new_pattern)
        postprocess(dir, self.working_dir + "/models/")
        print(f"Saved model to {self.working_dir}/models")
