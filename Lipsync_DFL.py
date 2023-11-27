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
from Retinaface_Mediapipe.common import random_crop, normalize_channels
from tqdm import tqdm
from Load_model_func import *
import json
from PIL import Image
from core.joblib import MPClassFuncOnDemand, MPFunc
import facelib
from facelib import FaceEnhancer, FaceType, LandmarksProcessor, XSegNet
import imutils
import mediapipe as mp
from skimage import measure
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

with open("/home/ubuntu/Duy_test_folder/Retinaface_Mediapipe/config_lipsync.json",'r') as f_lips:
	list_keypoint = json.load(f_lips)
streamer = "Dr"
right_threshold = list_keypoint[streamer]["right_threshold"]
left_threshold = list_keypoint[streamer]["left_threshold"]
straight_threshold = list_keypoint[streamer]["straight_threshold"]
pitch_up_threshold = list_keypoint[streamer]["pitch_up_threshold"]
FACEMESH_lips_1 = list_keypoint[streamer]["FACEMESH_lips_1"]
FACEMESH_lips = list_keypoint[streamer]["FACEMESH_lips"]
FACEMESH_lips_2 = list_keypoint[streamer]["FACEMESH_lips_2"]
FACEMESH_lips_2_face_up = list_keypoint[streamer]["FACEMESH_lips_2_face_up"]
FACEMESH_lips_2_up = list_keypoint[streamer]["FACEMESH_lips_2_up"]
FACEMESH_lips_2_left = list_keypoint[streamer]["FACEMESH_lips_2_left"]
FACEMESH_lips_2_right = list_keypoint[streamer]["FACEMESH_lips_2_right"]
FACEMESH_lips_intermediate_left = list_keypoint[streamer]["FACEMESH_lips_2"]
FACEMESH_lips_intermediate_right = list_keypoint[streamer]["FACEMESH_lips_2"]
landmark_points_68 = list_keypoint[streamer]["landmark_points_68"]

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
        padding_ratio = 0.08
        crop_image = None

        detected_face,_ = detection_face(mobile_net,resnet_net, frame, device,padding_ratio)
        block_check.append(detected_face)
    return block_check

def reset_data():
    global block,detected_frame,frame_cop,block_ori
    block = []
    detected_frame = []
    block_ori = []
    # block.append(frame)

landmark_points_68 = [162,234,93,58,172,136,149,148,152,377,378,365,397,288,323,454,389,71,63,105,66,107,336,
                      296,334,293,301,168,197,5,4,75,97,2,326,305,33,160,158,133,153,144,362,385,387,263,373,
                      380,61,39,37,0,267,269,291,405,314,17,84,181,78,82,13,312,308,317,14,87 ]
landmark_mouth_point = [2,326, 423,425 ,411,416,430,431, 262, 428, 199, 208, 32, 211,210,192, 187, 205 , 203, 97]
# landmark_points_68 = [162,234,93,58,172,136,149,148,152,377,378,365,397,288,323,454,389,71,63,105,66,107,336,
#                       296,334,293,301,168,197,5,4,75,97,2,326,305,33,160,158,133,153,144,362,385,387,263,373,
#                       380,]

def extract_face_retina(image,face_mesh_wide):
    global config_extract,device_config
    data = {'rects':None,
            'rects_rotation':None,
            'landmarks':None,
            'image':None,
            'face_image_landmarks':None,
            'faces_detected':None,
            'image_landmarks':None,
            'landmarks_accurate':False,
            # cfg['masked_hist_match':False,
            'crops_coors':None}
    h, w, c = image.shape
    padding_ratio = 0.1
    crop_image = None
    detected_face,l_coordinate = detection_face(mobile_net,resnet_net, image, device,padding_ratio)
    rects = data['rects'] = l_coordinate
    face_landmarks = None
    crop_images_coors = None
    try:
        for i in range(len(l_coordinate)):
            topleft, bottomright = l_coordinate[i]

            crop_image = image[topleft[1]:bottomright[1], topleft[0]:bottomright[0],:]
            crop_images_coors = [topleft[1],bottomright[1], topleft[0],bottomright[0]]
            data['crops_coors'] = crop_images_coors
            results = face_mesh_wide.process(cv2.cvtColor(crop_image, cv2.COLOR_BGR2RGB))
            if not results.multi_face_landmarks:
                print("Mediapipe not detect")
                continue
            face_landmarks = results.multi_face_landmarks[0]
            curbox = []
            curbox.append(topleft[0])
            curbox.append(topleft[1])
            curbox.append(bottomright[0])
            curbox.append(bottomright[1])
            bbox = curbox
        # if not face_landmarks:

        bbox_w = bbox[2] - bbox[0]
        bbox_h = bbox[3] - bbox[1]
        listpoint = []

        for i in range(len(landmark_points_68)):
            idx = landmark_points_68[i]
            x = face_landmarks.landmark[idx].x
            y = face_landmarks.landmark[idx].y

            realx = x * bbox_w + bbox[0]
            realy = y * bbox_h + bbox[1]
            listpoint.append([realx, realy])


        srcpts = np.array(listpoint, np.int32)
        data['image_landmarks'] = srcpts

        listpoint2 = []
        for i in range(len(landmark_mouth_point)):
            idx = landmark_mouth_point[i]
            x = face_landmarks.landmark[idx].x
            y = face_landmarks.landmark[idx].y

            realx = x * bbox_w + bbox[0]
            realy = y * bbox_h + bbox[1]
            listpoint2.append([realx, realy])


        mouth_point = np.array(listpoint2, np.int32)
        # srcpts = srcpts.reshape(-1,1,2)
        data['mouth_point'] = mouth_point
    # print(srcpts,len(srcpts))
    except Exception:
        pass
    return data


def extract_face_S3FD(image,rects_extractor,landmarks_extractor,face_mesh_wide):
    global config_extract,device_config
    data = {'rects':None,
            'rects_rotation':None,
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
    if data['rects_rotation'] is None:
        print("Not detect rect_rotation")
        return None
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

def my_cv2_resize(img, w, h):
    if img.size > w*h:
        return cv2.resize(img, (w, h), cv2.INTER_AREA)
    else:
        return cv2.resize(img, (w, h), cv2.INTER_CUBIC)

def fillhole(input_image):
    # input_image = 255 - input_image
    # labels_mask = measure.label(input_image)
    # regions = measure.regionprops(labels_mask)
    # regions.sort(key=lambda x: x.area, reverse=True)
    # if len(regions) > 1:
    #     for rg in regions[1:]:
    #         labels_mask[rg.coords[:,0], rg.coords[:,1]] = 0
    # labels_mask[labels_mask!=0] = 1
    # input_image = labels_mask

    im_flood_fill = input_image.copy()
    h, w = input_image.shape[:2]
    masktmp = np.zeros((h + 2, w + 2), np.uint8)
    im_flood_fill = im_flood_fill.astype("uint8")
    cv2.floodFill(im_flood_fill, masktmp, (0, 0), 255)
    im_flood_fill_inv = cv2.bitwise_not(im_flood_fill)
    img_out = input_image | im_flood_fill_inv

    # bighole = bwareaopen(im_flood_fill_inv, int(w/3))
    # img_out = img_out - bighole
    return img_out

def find_ratio_intersection(videoimg,crops_coors,videopts_out,mask_mount=None):


    crop_image = videoimg[crops_coors[0]:crops_coors[1],crops_coors[2]:crops_coors[3],:]
    crop_image_occ = np.copy(crop_image)
    xseg_res = xseg_256_extract_func.get_resolution()
    h, w, c = crop_image.shape
    crop_image = my_cv2_resize(crop_image, xseg_res, xseg_res)
    Xseg_mask = xseg_256_extract_func.extract(crop_image)
    Xseg_mask = my_cv2_resize(Xseg_mask, w, h)
    Xseg_mask = np.array(255*Xseg_mask, dtype=np.uint8)
    Xseg_mask[Xseg_mask>0] = 255
    Xseg_mask = fillhole(Xseg_mask)

    video_h,video_w, = videoimg.shape[:2]

    img_face_occ = np.zeros((video_h,video_w), np.uint8)
    img_face_occ[crops_coors[0]:crops_coors[1],crops_coors[2]:crops_coors[3]] = Xseg_mask

    # img_face_occ = Xseg_mask

    var_middle_occ = np.copy(mask_mount[:,:])
    # var_middle_occ[var_middle_occ >0]=1
    img_mouth_mask_occ = cv2.bitwise_and(img_face_occ,var_middle_occ)
    # print("Max mouth Occ:", np.max(img_mouth_mask_occ)," ", img_mouth_mask_occ.shape)
    mount_mask_Occ = img_mouth_mask_occ.copy()

    #cal ratio between 2 mouth mask Mediapipe and FaceOcc
    mask_mount = np.atleast_3d(mask_mount).astype('float') / 255.
    mask_mount[mask_mount != 1] = 0

    img_mouth_mask_occ = np.atleast_3d(img_mouth_mask_occ).astype('float') / 255.
    img_mouth_mask_occ[img_mouth_mask_occ != 1] = 0
    newval = len(img_mouth_mask_occ[img_mouth_mask_occ > 0])/len(mask_mount[mask_mount >0])
    return newval,mount_mask_Occ,img_face_occ

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

def write_block(blocks,encoder):
    global block,detected_frame,frame_cop, block_ori,block_merge,remerge_up, count_remerge_up
    block = []
    detected_frame = []
    block_ori = []

    for images in blocks:
        # output_fr +=1
        write_frame(images,encoder)

    block = []
    block_merge = []
    detected_frame = []
    block_ori = []
    remerge_up = False
    count_remerge_up = 0

def backup_data(data):
    global data_previous,transfer_step
    # print('data',len(data),'data_previous',len(data_previous))
    if len(data_previous) < 1:
        for i in range(transfer_step):
            if len(data_previous) == transfer_step:
                break
            else:
                data_previous.append(data)
    else:
        data_previous.pop(0)
        data_previous.append(data)

    return data_previous

if __name__ == "__main__":
    if sys.version_info[0] < 3 or (sys.version_info[0] == 3 and sys.version_info[1] < 6):
        raise Exception("This program requires at least Python 3.6")
    from config_merger_model import *
    FACE_PATH = '/home/ubuntu/Duy_test_folder/Retinaface_Mediapipe/Wav2lip_14Sep_Dr_ES.mp4'
    FRAME_PATH = '/home/ubuntu/Duy_test_folder/Retinaface_Mediapipe/Dr_video/DrDisrespect-Falls-in-Love-with-Warzone-again-thanks-to-new-Game-Mode-30FPS.mp4'
    device = 'cuda'
    output_path = '/home/ubuntu/quyennv/DeepFaceLab_Linux/DeepFaceLab/test_video/DFLab_workflow_debug_switchmode.mp4'
    model_path = '/home/ubuntu/quyennv/DeepFaceLab_Linux/workspace/model'

    mobile_net, resnet_net = loadmodelface()    #Face Occlusion
    xseg_256_extract_func = init_XSeg(model_path, device='cuda')

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh_wide = mp_face_mesh.FaceMesh(min_detection_confidence=0.5)

    transfer_step = 5
    config_merge_mask = cfg_merge
    config_merge_mask['face_type'] = cfg.face_type
    scale_range = np.abs(config_merge_mask['output_face_scale'] - 2)/transfer_step

    capFrame = cv2.VideoCapture(FRAME_PATH)
    capFace = cv2.VideoCapture(FACE_PATH)
    fps = capFrame.get(cv2.CAP_PROP_FPS)
    width_  = int(capFrame.get(cv2.CAP_PROP_FRAME_WIDTH))
    height_ = int(capFrame.get(cv2.CAP_PROP_FRAME_HEIGHT))
    encoder_video = ffmpeg_encoder(output_path, fps, width_, height_)
    totalF = int(capFrame.get(cv2.CAP_PROP_FRAME_COUNT))

    pbar = tqdm(total=totalF)
    count_frame = 0
    fps_block = 20
    block = []
    block_ori = []
    block_merge = []
    blur_threshold = 10.69


    minute_start = 0
    second_start = 19
    minute_stop =0
    second_stop =40
    frame_start = int(minute_start*60*fps+second_start*fps)
    frame_stop = int(minute_stop*60*fps+second_stop*fps)
    # frame_start = 1280
    # frame_stop = 1630
    data_previous = []
    remerge_down = True
    remerge_up = False
    count_remerge_up = 0

    while capFrame.isOpened():
        count_frame +=1
        ret,frame = capFace.read()
        ret2,frame_ori = capFrame.read()
        frame_cop = frame.copy()
        cv2.putText(frame, text='Fr:'+str(count_frame), org=(100, 40), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=1.1, color=(0, 255, 0),thickness=2)
        cv2.putText(frame_ori, text='Fr: '+str(count_frame), org=(100, 40), fontFace=cv2.FONT_HERSHEY_TRIPLEX,fontScale=1.1, color=(0, 255, 0),thickness=2)
        if count_frame <= frame_start:
            continue
        elif count_frame > frame_stop:
            break
        elif count_frame > frame_start and count_frame <= frame_stop:

            if len(block) < fps_block and ret:
                block.append(frame)
                block_ori.append(frame_ori)
            if len(block) == fps_block or not ret:
                # detected_frame = retinaface_check(block)
                ##Process images in block
                for idx in range(fps_block):
                    pbar.update(1)
                    # print("Length block merge",count_frame,len(block_merge))
                    data = extract_face_retina(block[idx],face_mesh_wide)
                    # print(data)
                    if data['image_landmarks'] is None or data['crops_coors'] is None:
                        # print( data['image_landmarks'] is None,  data['crops_coors'] is None)
                        print("Not extract:",count_frame)
                        block_merge.append(block_ori[idx])
                        data_previous = backup_data(data)
                        remerge_up = True
                        continue
                    crops_coors = data['crops_coors']

                    #check size of cutted face
                    size_box_face = np.abs((crops_coors[0]-crops_coors[1])*(crops_coors[2]-crops_coors[3]))
                    # cv2.putText(videoimg, text='Size_box'+str(size_box_face/(width_*height_)), org=(100, 150), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=1.1, color=(0, 255, 0),thickness=2)
                    if size_box_face/(width_*height_) < 0.0115:
                        print("Small face box",count_frame,size_box_face/(width_*height_))
                        block_merge.append(block_ori[idx])
                        data_previous = backup_data(data)
                        remerge_up = True
                        continue

                    #check blur face
                    face = frame_ori[crops_coors[0]:crops_coors[1],crops_coors[2]:crops_coors[3],:]
                    face = imutils.resize(face, width=500)
                    focus_measure = cv2.Laplacian(face, cv2.CV_64F).var()
                    if focus_measure < blur_threshold:      ## Blur face condition ##
                        print("Blur face",count_frame,"|",focus_measure)
                        block_merge.append(block_ori[idx])
                        data_previous = backup_data(data)
                        remerge_up = True
                        continue

                    #### Calc and Check Xseg condition
                    img_mouth_mask = np.zeros((height_,width_), np.uint8)
                    cv2.fillPoly(img_mouth_mask, pts =[data['mouth_point']], color=(255,255,255))
                    newval,_,img_face_Xseg = find_ratio_intersection(block_ori[idx],crops_coors,data['mouth_point'],mask_mount = img_mouth_mask)
                    cv2.putText(block[idx], text='Intersec ratio: '+str(newval), org=(100, 100), fontFace=cv2.FONT_HERSHEY_TRIPLEX,fontScale=1.1, color=(0, 255, 0),thickness=2)
                    cv2.putText(block_ori[idx], text='Intersec ratio: '+str(newval), org=(100, 100), fontFace=cv2.FONT_HERSHEY_TRIPLEX,fontScale=1.1, color=(0, 255, 0),thickness=2)

                    if newval < 0.85:
                        print("Ocluded face",count_frame,"|",newval)
                        if remerge_down:
                            config_tmp_down = config_merge_mask.copy()
                            for i in range(transfer_step):
                                idx_bk = idx - transfer_step +i
                                try:
                                    config_tmp_down['output_face_scale'] = config_tmp_down['output_face_scale'] + scale_range
                                    merge_face_img = MergeMasked(predictor_func,
                                                                 predictor_input_shape,
                                                                 face_enhancer_func.enhance,
                                                                 xseg_256_extract_func.extract,
                                                                 config_tmp_down,data_previous[i],block[idx_bk],remerge = remerge_down)
                                    cv2.putText(merge_face_img, text="RE-MERGE", org=(100, 200), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=1.5, color=(0, 0, 255),thickness=2)
                                    block_merge[idx_bk] = merge_face_img[...,0:3]
                                except Exception:
                                    pass
                            print("Re-merge success")
                            remerge_down = False
                        remerge_up = True
                        block_merge.append(block_ori[idx])
                        data_previous = backup_data(data)
                        continue
                    #### Merge face by "hist-match" to make face trasform smoothy

                    if not remerge_up:
                        merge_face_img = MergeMasked(predictor_func,
                                                     predictor_input_shape,
                                                     face_enhancer_func.enhance,
                                                     xseg_256_extract_func.extract,
                                                     config_merge_mask,data,block[idx],remerge = False)
                        count_remerge_up = 0
                    elif remerge_up and count_remerge_up <= transfer_step:
                        print("remerge UP")
                        config_tmp_up = config_merge_mask.copy()
                        count_remerge_up +=1
                        if count_remerge_up == transfer_step:
                            remerge_up = False
                        try:
                            config_tmp_up['output_face_scale'] = config_tmp_up['output_face_scale'] + scale_range*(transfer_step-count_remerge_up)
                            merge_face_img = MergeMasked(predictor_func,
                                                         predictor_input_shape,
                                                         face_enhancer_func.enhance,
                                                         xseg_256_extract_func.extract,
                                                         config_tmp_up,data,block[idx],remerge= True)
                            cv2.putText(merge_face_img, text="RE-MERGE-UP", org=(100, 230), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=1.5, color=(0, 0, 255),thickness=2)
                        except Exception:
                            merge_face_img = block[idx]
                    # print("count_remerge_up",count_remerge_up)
                    remerge_down = True
                    # # print(merge_face_img)
                    # # cv2.imwrite("Final_merge.png",merge_face_img[...,0:3])
                    block_merge.append(merge_face_img[...,0:3])
                    data_previous = backup_data(data)
                    # write_frame(merge_face_img[...,0:3],encoder_video)
                write_block(block_merge,encoder_video)
                # reset_data()

    pbar.close()
    encoder_video.stdin.flush()
    encoder_video.stdin.close()
