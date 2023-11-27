import math
import multiprocessing
import traceback
from pathlib import Path
import os
import numpy as np
import numpy.linalg as npla

import samplelib
from core import pathex
from core.cv2ex import *
from core.interact import interact as io
from core.joblib import MPClassFuncOnDemand, MPFunc
from core.leras import nn
from DFLIMG import DFLIMG
import facelib
from facelib import FaceEnhancer, FaceType, LandmarksProcessor, XSegNet
from merger import FrameInfo, InteractiveMergerSubprocessor, MergerConfig
from mainscripts.Extractor_custome import main_extract

model_class_name=None,
saved_models_path=None,
training_data_src_path=None,
force_model_name=None,
input_path=None,
output_path=None,
output_mask_path=None,
aligned_path=None,
force_gpu_idxs=None,
cpu_only=None
io.log_info ("Running merger.\r\n")
path_root = str(Path(__file__)).split("DeepFaceLab_Linux")[0]
model_class_name = 'SAEHD'
saved_models_path = "/home/ubuntu/quyennv/DeepFaceLab_Linux/workspace/model/Simon-model"
force_gpu_idxs = 0
force_model_name = None
# try:
# Initialize model
import models
# def initial_DFLmodel(saved_models_path):
# saved_models_path =Path(saved_models_path)
# if not os.path.exists( saved_models_path):
#     io.log_err('Model directory not found. Please ensure it exists.')
    # return
# model = models.import_model(model_class_name)(is_training=False,
#                                               saved_models_path=saved_models_path,
#                                               force_gpu_idxs=force_gpu_idxs,
#                                               force_model_name=force_model_name,
#                                               cpu_only=cpu_only)
# # print("OK")
# predictor_func, predictor_input_shape, cfg = model.get_MergerConfig()
# predictor_func = MPFunc(predictor_func)
# run_on_cpu = False
# run_on_cpu = len(nn.getCurrentDeviceConfig().devices) == 0
# face_enhancer_func =FaceEnhancer(place_model_on_cpu=True,run_on_cpu=run_on_cpu)
# if cpu_only:
#     device_config = nn.DeviceConfig.CPU()
#     place_model_on_cpu = True
# else:
#     device_config = nn.DeviceConfig.GPUIndexes ([0])
#     place_model_on_cpu = device_config.devices[0].total_mem_gb < 4
#
# nn.initialize (device_config)

# self.log_info (f"Running on {client_dict['device_name'] }")
# config_extract = main_extract(face_type=cfg.face_type,
#                                  max_faces_from_image=None,
#                                  image_size=None,
#                                  jpeg_quality=None,
#                                  )
# print(config_extract)
# rects_extractor = facelib.S3FDExtractor(place_model_on_cpu=place_model_on_cpu)
#
#     # for head type, extract "3D landmarks"
# landmarks_extractor = facelib.FANExtractor(landmarks_3D=cfg.face_type >= FaceType.HEAD,
#                                                         place_model_on_cpu=place_model_on_cpu)
    # print(type(model,predictor_func,predictor_input_shape,cfg,xseg_256_extract_func,face_enhancer_func))
    # return model,predictor_func,predictor_input_shape,cfg,xseg_256_extract_func,face_enhancer_func,rects_extractor,landmarks_extractor,config_extract,device_config
