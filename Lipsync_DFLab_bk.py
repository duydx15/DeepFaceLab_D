# import multiprocessing
# from re import L
# from xml.etree.ElementTree import TreeBuilder
# multiprocessing.set_start_method("spawn")
from pathlib import Path
from core.leras import nn
nn.initialize_main_env()
import os
import sys
import time
import argparse
import cv2
from core import pathex
from core import osex
from pathlib import Path
from core.interact import interact as io
import ffmpeg
import subprocess
from core.leras.device import Devices
# from mainscripts.Extractor_custome import *
from mainscripts.Extractor_custome import ExtractSubprocessor,main_extract
from merger.MergeMasked_custome import MergeMasked,MergeMaskedFace
from Retinaface_Mediapipe.pipeline_mobile_resnet import loadmodelface, detection_face
from tqdm import tqdm
from Load_model_func import *

from PIL import Image
from core.joblib import MPClassFuncOnDemand, MPFunc
import facelib
from facelib import FaceEnhancer, FaceType, LandmarksProcessor, XSegNet
# from XSeg_video import init_XSeg
# import tensorflow as tf
# gpu_options=tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.9)
# config=tf.compat.v1.ConfigProto(gpu_options=gpu_options)
# session = tf.compat.v1.Session(config=config)

LOGURU_FFMPEG_LOGLEVELS = {
    "trace": "trace",
    "debug": "debug",
    "info": "info",
    "success": "info",
    "warning": "warning",
    "error": "error",
    "critical": "fatal",
}

def ffmpeg_encoder(outfile, fps, width, height):
    frames = ffmpeg.input(
        "pipe:0",
        format="rawvideo",
        pix_fmt="rgb24",
        vsync="1",
        s='{}x{}'.format(width, height),
        r=fps,
    )

    encoder_ = subprocess.Popen(
        ffmpeg.compile(
            ffmpeg.output(
                frames,
                outfile,
                pix_fmt="yuv420p",
                vcodec="libx264",
                acodec="copy",
                r=fps,
                crf=17,
                vsync="1",
            )
            .global_args("-hide_banner")
            .global_args("-nostats")
            .global_args(
                "-loglevel",
                LOGURU_FFMPEG_LOGLEVELS.get(
                    os.environ.get("LOGURU_LEVEL", "INFO").lower()
                ),
            ),
            overwrite_output=True,
        ),
        stdin=subprocess.PIPE,
        # stdout=subprocess.DEVNULL,
        # stderr=subprocess.DEVNULL,
    )
    return encoder_


def retinaface_check(blocks):
    block_check = []
    for frame in blocks:
        h,w = frame.shape[:2]
        count_loss_detect = 0
        padding_ratio = 0.2
        detect_fail = False
        previous_model = 0
        crop_image = None

        detected_face,_ = detection_face(mobile_net,resnet_net, frame, device,padding_ratio)
        block_check.append(detected_face)
    return block_check

def reset_data():
    global block,detected_frame,frame_cop
    block = []
    detected_frame = []
    # block.append(frame)

def  extract_face(image,rects_extractor,landmarks_extractor):
    global config_extract,device_config
    data = {'rects':None,
            'rects_retation':None,
            'landmarks':None,
            'image':None,
            'face_image_landmarks':None,
            'faces_detected':None,
            'landmarks_accurate':False}
    h, w, c = image.shape

    data = ExtractSubprocessor.Cli.rects_stage (data=data,
                                                image=image,
                                                max_faces_from_image=config_extract['max_faces_from_image'],
                                                rects_extractor=rects_extractor,
                                                )

    data = ExtractSubprocessor.Cli.landmarks_stage (data=data,
                                                    image=image,
                                                    landmarks_extractor=landmarks_extractor,
                                                    rects_extractor=rects_extractor,
                                                    )

    data = ExtractSubprocessor.Cli.final_stage(data=data,
                                               image=image,
                                               face_type=config_extract['face_type'],
                                               image_size=config_extract['image_size'],
                                               jpeg_quality=config_extract['jpeg_quality'],
                                               output_debug_path=None,
                                               final_output_path=None,
                                               )
    return data

# def merge_face(predictor_func,
                 # predictor_input_shape,
                 # face_enhancer_func,
                 # xseg_256_extract_func,
                 # cfg,data):

def init_XSeg(model_path, device='cpu'):
    """
    from pathlib import Path
    ******DeepFaceLab*******
    from core.leras import nn
    from facelib import XSegNet
    from core.leras.device import Devices
    """
    if device=='cpu':
        Devices.initialize_main_env()
        device_config = nn.DeviceConfig.CPU()
        nn.initialize(device_config)
    else:
        Devices.initialize_main_env()
        device_config = nn.DeviceConfig.GPUIndexes([0]) # change GPU index here
        nn.initialize(device_config)
    # print(nn.data_format)
    model_path = Path(model_path)
    xseg = XSegNet(name='XSeg',
                    load_weights=True,
                    weights_file_root=model_path,
                    data_format=nn.data_format,
                    raise_on_no_model_files=True)
    return xseg

def write_frame(images,encoder_video):
    images = cv2.cvtColor(images,cv2.COLOR_RGB2BGR)
    imageout = Image.fromarray(np.uint8(images))
    encoder_video.stdin.write(imageout.tobytes())

if __name__ == "__main__":
    if sys.version_info[0] < 3 or (sys.version_info[0] == 3 and sys.version_info[1] < 6):
        raise Exception("This program requires at least Python 3.6")
    from config_merger_model import *
    FRAME_PATH = '/home/ubuntu/Duy_test_folder/Retinaface_Mediapipe/Dr_video/DrDisrespect-Falls-in-Love-with-Warzone-again-thanks-to-new-Game-Mode-30FPS.mp4'
    device = 'cuda'
    output_path = '/home/ubuntu/quyennv/DeepFaceLab_Linux/DeepFaceLab/test_video/DFLab_workflow_debug.mp4'
    model_path = '/home/ubuntu/quyennv/DeepFaceLab_Linux/workspace/model'

    mobile_net, resnet_net = loadmodelface()    #Face Occlusion

    xseg_256_extract_func = init_XSeg(model_path, device='cuda')

    config_merge_mask = cfg_merge
    config_merge_mask['face_type'] = cfg.face_type


    capFrame = cv2.VideoCapture(FRAME_PATH)
    fps = capFrame.get(cv2.CAP_PROP_FPS)
    width_  = int(capFrame.get(cv2.CAP_PROP_FRAME_WIDTH))
    height_ = int(capFrame.get(cv2.CAP_PROP_FRAME_HEIGHT))
    encoder_video = ffmpeg_encoder(output_path, fps, width_, height_)
    totalF = int(capFrame.get(cv2.CAP_PROP_FRAME_COUNT))

    pbar = tqdm(total=totalF)
    count_frame = 0
    fps_block = 10
    block = []
    block_merge = []

    minute_start = 0
    second_start = 0
    minute_stop =0
    second_stop =10
    frame_start = int(minute_start*60*fps+second_start*fps)
    frame_stop = int(minute_stop*60*fps+second_stop*fps)

    while capFrame.isOpened():
        count_frame +=1
        ret,frame = capFrame.read()
        frame_cop = frame.copy()

        if count_frame <= frame_start:
            continue
        elif count_frame > frame_stop:
            break
        elif count_frame > frame_start and count_frame <= frame_stop:

            if len(block) < fps_block and ret:
                block.append(frame)

            if len(block) == fps_block or not ret:
                detected_frame = retinaface_check(block)
                # print(count_frame,detected_frame)

                ##Process images in block
                for idx in range(fps_block):
                    pbar.update(1)
                    if not detected_frame[idx]:
                        block_merge.append(block[idx])
                        write_frame(block[idx],encoder_video)
                    else:
                        data = extract_face(block[idx],rects_extractor,landmarks_extractor)
                        # print(data)
                        merge_face_img = MergeMasked(predictor_func,
                                                     predictor_input_shape,
                                                     face_enhancer_func,
                                                     xseg_256_extract_func.extract,
                                                     config_merge_mask,data,block[idx])
                        # print(merge_face_img)
                        # cv2.imwrite("Final_merge.png",merge_face_img[...,0:3])
                        block_merge.append(merge_face_img[...,0:3])
                        write_frame(merge_face_img[...,0:3],encoder_video)

                    # break

                reset_data()

    pbar.close()
    encoder_video.stdin.flush()
    encoder_video.stdin.close()
