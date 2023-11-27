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
import torch
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
from merger.MergeMasked_custome import MergeMasked,MergeMaskedFace,MergeMaskedFace_assume
from Retinaface_Mediapipe.pipeline_mobile_resnet import loadmodelface, detection_face, args
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
import time
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

class KalmanTracking(object):
    # init kalman filter object
    def __init__(self, point):
        deltatime = 1/30 # 30fps
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0],
         [0, 1, 0, 0]], np.float32)

        self.kalman.transitionMatrix = np.array([[1, 0, 1, 0],
        [0, 1, 0, 1],
        [0, 0, 1, 0],
        [0, 0, 0, 1]], np.float32)

        self.kalman.processNoiseCov = np.array([[1, 0, 0, 0],
         [0, 1, 0, 0],
         [0, 0, 1, 0],
         [0, 0, 0, 1]], np.float32) * deltatime # 0.03
        # self.kalman.measurementNoiseCov = np.array([[1, 0],
        #                                             [0, 1]], np.float32) *0.1
        self.measurement = np.array((point[0], point[1]), np.float32)

    def getpoint(self, kp):
        self.kalman.correct(kp-self.measurement)
        # get new kalman filter prediction
        prediction = self.kalman.predict()
        prediction[0][0] = prediction[0][0] +  self.measurement[0]
        prediction[1][0] = prediction[1][0] +  self.measurement[1]

        return prediction

def binaryMaskIOU_(mask1, mask2):
    mask1_area = torch.count_nonzero(mask1)
    mask2_area = torch.count_nonzero(mask2)
    # print("mask1_area ", mask1_area)
    # print("mask2_area ", mask2_area)
    intersection = torch.count_nonzero(torch.logical_and(mask1, mask2))
    iou = intersection / (mask1_area + mask2_area - intersection)
    return iou.numpy()

class KalmanArray(object):
    def __init__(self):
        self.kflist = []
        # self.oldmask = None
        # self.resetVal = 1
        self.w = 0
        self.h = 0

    def noneArray(self):
        return len(self.kflist) == 0

    def setpoints(self, points, w=1920, h=1080):
        for value in points:
            intpoint = np.array([np.float32(value[0]), np.float32(value[1])], np.float32)
            self.kflist.append(KalmanTracking(intpoint))

        self.w = w
        self.h = h
        # self.oldmask = np.zeros(image_shape[0:2]+(1,),dtype=np.float32)


        # print('setpoints ', self.w)


    def getpoints(self, kps):
        # print('old ', kps[:3])
        # print("KPS:",len(kps),'\n', kps)
        # print("Kflist",len(self.kflist))
        orginmask = np.zeros((self.h,self.w),dtype=np.float32)
        orginmask = cv2.fillConvexPoly(orginmask, np.array(kps[:], np.int32), 1) #kps[:-18]
        kps_o = kps.copy()
        # print("kps",len(kps))
        # print("kflist",len(self.kflist))
        if len(kps) <= len(self.kflist):
            kps_final = len(kps)
        else:
            kps_final = len(self.kflist)
        for i in range(kps_final):
            # print(i)
            # kps[i] = kflist[i]
            intpoint = np.array([np.float32(kps[i][0]), np.float32(kps[i][1])], np.float32)
            tmp = self.kflist[i].getpoint(intpoint)
            kps[i] = (tmp[0][0], tmp[1][0])

        newmask = np.zeros((self.h,self.w),dtype=np.float32)
        newmask = cv2.fillConvexPoly(newmask, np.array(kps[:], np.int32), 1) #kps[:-18]
        # cv2.imwrite('orginmask.jpg' , orginmask*255)
        val = binaryMaskIOU_(torch.from_numpy(orginmask), torch.from_numpy(newmask))
        # print('binaryMaskIOU_ ', val)

        # distance = spatial.distance.cosine(orgindata, newdata)
        # print(distance)
        if val < 0.9:
            del self.kflist[:]
            # self.oldmask = None
            self.setpoints(kps_o,self.w, self.h)
            return kps_o

        # self.olddata = newdata
        # print('new ', kps[:3])
        return kps

def cross_point(line1, line2):
    x1 = line1[0]
    y1 = line1[1]
    x2 = line1[2]
    y2 = line1[3]

    x3 = line2[0]
    y3 = line2[1]
    x4 = line2[2]
    y4 = line2[3]

    k1 = (y2 - y1) * 1.0 / (x2 - x1)
    b1 = y1 * 1.0 - x1 * k1 * 1.0
    if (x4 - x3) == 0:
        k2 = None
        b2 = 0
    else:
        k2 = (y4 - y3) * 1.0 / (x4 - x3)
        b2 = y3 * 1.0 - x3 * k2 * 1.0
    if k2 == None:
        x = x3
    else:
        x = (b2 - b1) * 1.0 / (k1 - k2)
    y = k1 * x * 1.0 + b1 * 1.0
    return [x, y]

def point_line(point,line):
    x1 = line[0]
    y1 = line[1]
    x2 = line[2]
    y2 = line[3]

    x3 = point[0]
    y3 = point[1]

    k1 = (y2 - y1)*1.0 /(x2 -x1)
    b1 = y1 *1.0 - x1 *k1 *1.0
    k2 = -1.0/k1
    b2 = y3 *1.0 -x3 * k2 *1.0
    x = (b2 - b1) * 1.0 /(k1 - k2)
    y = k1 * x *1.0 +b1 *1.0
    return [x,y]

def point_point(point_1,point_2):
    x1 = point_1[0]
    y1 = point_1[1]
    x2 = point_2[0]
    y2 = point_2[1]
    # distance = ((x1-x2)**2 +(y1-y2)**2)**0.5
    distance = math.sqrt((x1-x2) ** 2 + (y1-y2) ** 2)
    # distance = math.hypot(x2-x2, y1-y2)
    # if distance == 0:
    #     distance = distance + 0.1
    return distance

def facePose(point1, point31, point51, point60, point72):
    crossover51 = point_line(point51, [point1[0], point1[1], point31[0], point31[1]])
    yaw_mean = point_point(point1, point31) / 2
    yaw_right = point_point(point1, crossover51)
    yaw = (yaw_mean - yaw_right) / yaw_mean
    if math.isnan(yaw):
        return None, None, None
    yaw = int(yaw * 71.58 + 0.7037)

    #pitch
    pitch_dis = point_point(point51, crossover51)
    if point51[1] < crossover51[1]:
        pitch_dis = -pitch_dis
    if math.isnan(pitch_dis):
        return None, None, None
    pitch = int(1.497 * pitch_dis + 18.97)

    #roll
    roll_tan = abs(point60[1] - point72[1]) / abs(point60[0] - point72[0])
    roll = math.atan(roll_tan)
    roll = math.degrees(roll)
    if math.isnan(roll):
        return None, None, None
    if point60[1] >point72[1]:
        roll = -roll
    roll = int(roll)

    return yaw, pitch, roll

def is_on_line(x1, y1, x2, y2, x3):
    slope = (y2 - y1) / (x2 - x1)
    return slope * (x3 - x1) + y1

# def ffmpeg_encoder(outfile, fps, width, height):
#     frames = ffmpeg.input(
#         "pipe:0",
#         format="rawvideo",
#         pix_fmt="rgb24",
#         vsync="1",
#         s='{}x{}'.format(width, height),
#         r=fps,
#     )
#
#     encoder_ = subprocess.Popen(
#         ffmpeg.compile(
#             ffmpeg.output(
#                 frames,
#                 outfile,
#                 pix_fmt="yuv420p",
#                 vcodec="libx264",
#                 acodec="copy",
#                 r=fps,
#                 crf=17,
#                 vsync="1",
#             )
#             .global_args("-hide_banner")
#             .global_args("-nostats")
#             .global_args(
#                 "-loglevel",
#                 LOGURU_FFMPEG_LOGLEVELS.get(
#                     os.environ.get("LOGURU_LEVEL", "INFO").lower()
#                 ),
#             ),
#             overwrite_output=True,
#         ),
#         stdin=subprocess.PIPE,
#         # stdout=subprocess.DEVNULL,
#         # stderr=subprocess.DEVNULL,
#     )
#     return encoder_

def ffmpeg_encoder(outfile, fps, width, height):
    LOGURU_FFMPEG_LOGLEVELS = {
        "trace": "trace",
        "debug": "debug",
        "info": "info",
        "success": "info",
        "warning": "warning",
        "error": "error",
        "critical": "fatal",
        }


    if torch.cuda.is_available():
        codec = "h264_nvenc"
        frames = ffmpeg.input(
        "pipe:0",
        format="rawvideo",
        pix_fmt="rgb24",
        vsync="2",
        s='{}x{}'.format(width, height),
        r=fps,
        hwaccel="cuda",
        hwaccel_device="0",
        # hwaccel_output_format="cuda",
        thread_queue_size=2,
    )
    else:
        codec = "libx264"
        frames = ffmpeg.input(
        "pipe:0",
        format="rawvideo",
        pix_fmt="rgb24",
        vsync="2",
        s='{}x{}'.format(width, height),
        r=fps,
        # hwaccel="cuda",
        # hwaccel_device="0",
        # hwaccel_output_format="cuda",
        thread_queue_size=2,
    )
    # print("###########33", codec)
    encoder_ = subprocess.Popen(
        ffmpeg.compile(
            ffmpeg.output(
                frames,
                outfile,
                pix_fmt="yuv420p",
                # vcodec="libx264",
                vcodec=codec,
                acodec="copy",
                r=fps,
                # crf=1,
                vsync="2",
                # async=4,
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
landmark_mouth_point = [94,370,326, 423,425 ,411,416,430,431, 262, 428, 199, 208, 32, 211,210,192, 187, 205 , 203, 97,141]
#[2,326, 423,425 ,411,416,430,431, 262, 428, 199, 208, 32, 211,210,192, 187, 205 , 203, 97]
# landmark_points_68 = [162,234,93,58,172,136,149,148,152,377,378,365,397,288,323,454,389,71,63,105,66,107,336,
#                       296,334,293,301,168,197,5,4,75,97,2,326,305,33,160,158,133,153,144,362,385,387,263,373,
#                       380,]

def extract_face_retina(image,face_mesh_wide,kf_mouth,kf):
    global config_extract,device_config,count_frame
    data = {'rects':None,
            'rects_rotation':None,
            'landmarks':None,
            'image':None,
            'face_image_landmarks':None,
            'faces_detected':None,
            'image_landmarks':None,
            'landmarks_accurate':False,
            'crops_coors':None,
            'face_type':None}
    h, w, c = image.shape
    padding_ratio = [0.3,0.1,0]
    crop_image = None
    for r_idx in range(len(padding_ratio)):
        detected_face,l_coordinate = detection_face(mobile_net,resnet_net, image, device,padding_ratio[r_idx])
        if not detected_face and r_idx == len(padding_ratio)-1:
            # count_loss_detect = 1
            # detect_fail = True
            # print("New_model loss")
            return data
        elif not detected_face and r_idx < len(padding_ratio)-1:
          # previous_model =  1
          continue
        rects = data['rects'] = l_coordinate
        face_landmarks = None
        crop_images_coors = None
        # try:
        bbox = []
        for i in range(len(l_coordinate)):
            topleft, bottomright = l_coordinate[i]

            crop_image = image[topleft[1]:bottomright[1], topleft[0]:bottomright[0],:]

            crop_images_coors = [topleft[1],bottomright[1], topleft[0],bottomright[0]]
            data['crops_coors'] = crop_images_coors
            # crop_image = imutils.resize(crop_image, width = 400)
            results = face_mesh_wide.process(cv2.cvtColor(crop_image, cv2.COLOR_BGR2RGB))
            if not results.multi_face_landmarks:
                # print("Mediapipe not detect",padding_ratio[r_idx])
                # cv2.imwrite("crop_image.png",crop_image)
                continue
            face_landmarks = results.multi_face_landmarks[0]
            curbox = []
            curbox.append(topleft[0])
            curbox.append(topleft[1])
            curbox.append(bottomright[0])
            curbox.append(bottomright[1])
            bbox = curbox
        # if not face_landmarks:
        if not face_landmarks and r_idx == len(padding_ratio)-1:
            # print("Medipipe loss")
            return data
        elif not face_landmarks and r_idx < len(padding_ratio)-1 :
            # print("Medipipe skip")
            continue
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

        if kf is not None:
            if kf.noneArray():
                # kf.setpoints(listpointLocal, 1e-03)
                kf.setpoints(listpoint, w, h)

            else:
                listpoint = kf.getpoints(listpoint)

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

        if kf_mouth is not None:
            # print("Use ")
            if kf_mouth.noneArray():
                # kf.setpoints(listpointLocal, 1e-03)
                kf_mouth.setpoints(listpoint2, w, h)

            else:
                listpoint2 = kf_mouth.getpoints(listpoint2)
        mouth_point = np.array(listpoint2, np.int32)
        # srcpts = srcpts.reshape(-1,1,2)
        data['mouth_point'] = mouth_point
        return data
        # print(srcpts,len(srcpts))
        # except Exception:
        #     pass
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

def expand_box_face(image,box):
    H,W = image.shape[:2]
    bbox = [box[2],box[0],box[3],box[1]]#((xmin, ymin , xmax, ymax))
    topleft = (int(bbox[0]), int(bbox[1]))
    bottomright = (int(bbox[2]), int(bbox[3]))
    hw_ratio = (bottomright[1] - topleft[1])/(bottomright[0] - topleft[0])
    # print("HW_ratio:",hw_ratio)
    # center = (int((bbox[0]+bbox[2])/2), int((bbox[1]+bbox[3])/2))
    padding_X = int((bottomright[0] - topleft[0]) *hw_ratio*1.2)
    padding_Y = int((bottomright[1] - topleft[1])  /hw_ratio)
    padding_topleft = (max(0, topleft[0] - padding_X), max(0, topleft[1]- padding_Y))
    padding_bottomright = (min(W, bottomright[0] + padding_X), min(H, bottomright[1] + int(padding_Y/1.3)))
    # coordinate = (padding_topleft, padding_bottomright)
    return [padding_topleft[1],padding_bottomright[1],padding_topleft[0],padding_bottomright[0]]

def find_ratio_intersection_v2(videoimg,crops_coors,videopts_out,data,mask_mount=None):

    img_size = videoimg.shape[1], videoimg.shape[0]
    xseg_res = xseg_256_extract_func.get_resolution()
    img_face_landmarks = np.array(data['image_landmarks'])
    output_size = 512
    # print("Face_type",data['face_type'],type(data['face_type']))
    # data['face_type'] = 'wf'
    face_mask_output_mat = LandmarksProcessor.get_transform_mat (img_face_landmarks, output_size, face_type=data['face_type'], scale= 1.0)
    # img_face_mask_a = LandmarksProcessor.get_image_hull_mask (videoimg.shape, img_landmarks)
    xseg_mat            = LandmarksProcessor.get_transform_mat (img_face_landmarks, xseg_res, face_type=data['face_type'])
    dst_face_xseg_bgr   = cv2.warpAffine(videoimg, xseg_mat, (xseg_res,)*2, flags=cv2.INTER_CUBIC )
    # cv2.imwrite("Xseg_debug.png",dst_face_xseg_bgr)
    dst_face_xseg_mask  = xseg_256_extract_func.extract(dst_face_xseg_bgr)
    # dst_face_xseg_mask  = fillhole(dst_face_xseg_mask)
    X_dst_face_mask_a_0 = cv2.resize (dst_face_xseg_mask, (output_size,output_size), interpolation=cv2.INTER_CUBIC)
    wrk_face_mask_a_0 = X_dst_face_mask_a_0
    wrk_face_mask_a_0[ wrk_face_mask_a_0 < (1.0/255.0) ] = 0.0
    img_face_mask_a = cv2.warpAffine( wrk_face_mask_a_0, face_mask_output_mat, img_size, np.zeros(videoimg.shape[0:2], dtype=np.float32), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_CUBIC )[...,None]
    img_face_mask_a = np.clip (img_face_mask_a, 0.0, 1.0)
    img_face_mask_a [ img_face_mask_a < (1.0/255.0) ] = 0.0
    img_face_mask_a [ img_face_mask_a >0.0 ] = 1.0

    img_face_occ = np.array(255*img_face_mask_a, dtype=np.uint8)
    var_middle_occ = np.copy(mask_mount[:,:])
    img_mouth_mask_occ = cv2.bitwise_and(img_face_occ,var_middle_occ)
    mount_mask_Occ = img_mouth_mask_occ.copy()

    #cal ratio between 2 mouth mask Mediapipe and FaceOcc
    mask_mount = np.atleast_3d(mask_mount).astype('float') / 255.
    mask_mount[mask_mount != 1] = 0

    img_mouth_mask_occ = np.atleast_3d(img_mouth_mask_occ).astype('float') / 255.
    img_mouth_mask_occ[img_mouth_mask_occ != 1] = 0
    newval = len(img_mouth_mask_occ[img_mouth_mask_occ > 0])/len(mask_mount[mask_mount >0])
    return newval,mount_mask_Occ,img_face_occ

def find_ratio_intersection(videoimg,crops_coors,videopts_out,mask_mount=None):

    xseg_res = xseg_256_extract_func.get_resolution()


    crops_coors = expand_box_face(videoimg,crops_coors)
    crop_image = videoimg[crops_coors[0]:crops_coors[1],crops_coors[2]:crops_coors[3],:]
    crop_image_occ = np.copy(crop_image)
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

    # img_face_occ = np.array(255*img_face_mask_a, dtype=np.uint8)
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

def mask2box(mask2d):
    (y, x) = np.where(mask2d > 0)
    topy = np.min(y)
    topx = np.min(x)
    bottomy = np.max(y)
    bottomx = np.max(x)
    # (topy, topx) = (np.min(y), np.min(x))
    # (bottomy, bottomx) = (np.max(y), np.max(x))
    center_ = (int((topx+bottomx)/2),int((bottomy+topy)/2))

    return topy, topx, bottomy, bottomx, center_

def write_frame(images,encoder_video):
    global total_record
    total_record +=1
    images = cv2.cvtColor(images,cv2.COLOR_RGB2BGR)
    imageout = Image.fromarray(np.uint8(images))
    encoder_video.stdin.write(imageout.tobytes())


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description='Retinaface')
    # args = parser.parse_args()
    input_video = []
    output_video = []
    input_audio = []
    with open(args.list_input_video,'r') as file:
        # print(file.read()[0])
        for line in file.read().split("\n")[:-1]:
            input_video.append(line)
            output_video.append(line.split(".mp4")[0]+"_WithDFL_noGAN.mp4")
    with open(args.list_input_audio,'r') as file_audio:
        for line in file_audio.read().split("\n")[:-1]:
            input_audio.append(line)

    if sys.version_info[0] < 3 or (sys.version_info[0] == 3 and sys.version_info[1] < 6):
        raise Exception("This program requires at least Python 3.6")
    from config_merger_model import *
    device = 'cuda'
    savepath_nonsound = "./output_nonsound_1.mp4"
    # model_path = '/home/ubuntu/quyennv/DeepFaceLab_Linux/workspace/model/Kaja-model'
    mobile_net, resnet_net = loadmodelface()    #Face Occlusion
    xseg_256_extract_func = init_XSeg(args.dfl_model, device='cuda')

    model = models.import_model(model_class_name)(is_training=False,
                                                  saved_models_path=Path(args.dfl_model),
                                                  force_gpu_idxs=force_gpu_idxs,
                                                  force_model_name=force_model_name,
                                                  cpu_only=cpu_only)
    # print("OK")
    predictor_func, predictor_input_shape, cfg = model.get_MergerConfig()
    predictor_func = MPFunc(predictor_func)
    run_on_cpu = False
    run_on_cpu = len(nn.getCurrentDeviceConfig().devices) == 0
    face_enhancer_func =FaceEnhancer(place_model_on_cpu=True,run_on_cpu=run_on_cpu)
    if cpu_only:
        device_config = nn.DeviceConfig.CPU()
        place_model_on_cpu = True
    else:
        device_config = nn.DeviceConfig.GPUIndexes ([0])
        place_model_on_cpu = device_config.devices[0].total_mem_gb < 4

    nn.initialize (device_config)

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh_wide = mp_face_mesh.FaceMesh(min_detection_confidence=0.2)

    trackkpVideoFace = KalmanArray()
    trackkpVideoMount = KalmanArray()
    print(input_video)
    print(input_audio)
    for idx in range(len(input_video)):
        config_merge_mask = cfg_merge
        config_merge_mask['face_type'] = cfg.face_type
        FACE_PATH = input_video[idx]
        wavpath = input_audio[idx]
        output_path = output_video[idx]
        # FACE_PATH = '/home/ubuntu/Duy_test_folder/SadTalker_samples/videostoberunfromsadtalker_11/1/video_scale/simon_ref_audio1_gfp_WC.mp4'
        # FACE_PATH = '/home/ubuntu/quyennv/DeepFaceLab_Linux/DeepFaceLab/test_video/Test_codeformer_18Nov.mp4'
        # FRAME_PATH = '/home/ubuntu/Duy_test_folder/Retinaface_Mediapipe/Dr_video/DrDisrespect-Falls-in-Love-with-Warzone-again-thanks-to-new-Game-Mode.mp4'
        # wavpath = '/home/ubuntu/Duy_test_folder/SadTalker_samples/AvatarVideo_VoiceSamples/Simon/simonneuteng1_1.mp3'
        # output_path = '/home/ubuntu/Duy_test_folder/SadTalker_samples/videostoberunfromsadtalker_11/1/video_scale/simon_ref_audio1_gfp_WithDFL.mp4'
        FRAME_PATH = FACE_PATH#'/home/ubuntu/Duy_test_folder/Retinaface_Mediapipe/Dr_video/Doc_lipsync_Apr6/video/Doc_clip508.mp4'

        capFrame = cv2.VideoCapture(FRAME_PATH)
        capFace = cv2.VideoCapture(FACE_PATH)
        fps = capFrame.get(cv2.CAP_PROP_FPS)
        print("FPS: ",fps)
        width_  = int(capFrame.get(cv2.CAP_PROP_FRAME_WIDTH))
        height_ = int(capFrame.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(capFrame.get(cv2.CAP_PROP_FRAME_COUNT))
        encoder_video = ffmpeg_encoder(savepath_nonsound, fps, width_, height_)

        count_frame = 0
        fps_block = 10
        block = []
        block_ori = []
        block_merge = []
        blur_threshold = 10.69
        border_th = 13
        minute_start =0
        second_start = 0
        minute_stop =1
        second_stop =35
        frame_start = int(minute_start*60*fps+second_start*fps)
        frame_stop = int(minute_stop*60*fps+second_stop*fps)
        totalF = int(frame_stop-frame_start)
        # frame_skip = [6060,6720]
        pbar = tqdm(total=total_frames)

        total_record = 0
        try:
            while capFrame.isOpened():
                count_frame +=1
                ret2,frame_ori = capFrame.read()
                ret,frame = capFace.read()

                # if not ret:
                #     break
                # frame_cop = frame.copy()
                # cv2.putText(frame, text='Fr:'+str(count_frame), org=(100, 40), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=1.1, color=(0, 255, 0),thickness=2)
                # cv2.putText(frame_ori, text='Fr: '+str(count_frame), org=(100, 40), fontFace=cv2.FONT_HERSHEY_TRIPLEX,fontScale=1.1, color=(0, 255, 0),thickness=2)
                if count_frame <= frame_start:
                    continue
                elif count_frame > frame_stop:
                    break
                elif count_frame > frame_start and count_frame <= frame_stop:
                    # ret,frame = capFace.read()
                    # frame_cop = frame.copy()
                    # cv2.putText(frame, text='Fr:'+str(count_frame), org=(100, 40), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=1.1, color=(0, 255, 0),thickness=2)

                    # if count_frame >=frame_skip[0] and count_frame <= frame_skip[1]:
                    #     frame = frame_ori

                    if len(block) < fps_block and ret:
                        block.append(frame)
                        block_ori.append(frame_ori)

                    if len(block) == fps_block or not ret:
                        # detected_frame = retinaface_check(block)

                        ##Process images in block
                        for idx in range(fps_block):
                            pbar.update(1)
                            # print("Total frame recoreded" ,total_record ," - ", idx)
                            data = extract_face_retina(block[idx],face_mesh_wide,trackkpVideoMount,trackkpVideoFace)
                            # print(data)
                            if data['image_landmarks'] is None or data['crops_coors'] is None:
                                # print( data['image_landmarks'] is None,  data['crops_coors'] is None)
                                print("Not extract:",count_frame)
                                write_frame(block_ori[idx],encoder_video)
                                continue
                            crops_coors = data['crops_coors']
                            data['face_type'] = config_merge_mask['face_type']

                            #check size of cutted face
                            size_box_face = np.abs((crops_coors[0]-crops_coors[1])*(crops_coors[2]-crops_coors[3]))
                            # cv2.putText(videoimg, text='Size_box'+str(size_box_face/(width_*height_)), org=(100, 150), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=1.1, color=(0, 255, 0),thickness=2)
                            if size_box_face/(width_*height_) <= 0.01:
                                print("Small face box",count_frame,size_box_face/(width_*height_))
                                write_frame(block_ori[idx],encoder_video)
                                continue

                            # #check blur face
                            # face = block[idx][crops_coors[0]:crops_coors[1],crops_coors[2]:crops_coors[3],:]
                            # face = imutils.resize(face, width=500)
                            # focus_measure = cv2.Laplacian(face, cv2.CV_64F).var()
                            # if focus_measure < blur_threshold:      ## Blur face condition ##
                            #     print("Blur face",count_frame,"|",focus_measure)
                            #     write_frame(block_ori[idx],encoder_video)
                            #     continue

                            #### Calc and Check Xseg condition
                            img_mouth_mask = np.zeros((height_,width_), np.uint8)
                            cv2.fillPoly(img_mouth_mask, pts =[data['mouth_point']], color=(255,255,255))

                            config_merge_mask_cop = config_merge_mask
                            m_topy, m_topx, m_bottomy, m_bottomx,center_mount = mask2box(img_mouth_mask)
                            if m_topy <border_th or m_topx <border_th or m_bottomy > height_-border_th or m_bottomx > width_-border_th:
                                print("Box touched")
                                config_merge_mask_cop['mask_mode'] = 8
                                merge_face_img = MergeMasked(predictor_func,
                                                             predictor_input_shape,
                                                             face_enhancer_func.enhance,
                                                             xseg_256_extract_func.extract,
                                                             config_merge_mask_cop,data,block[idx],block_ori[idx],remerge=True,skip_merge=True)
                            else:
                                #     write_frame(block_ori[idx],encoder_video)
                                #     continue

                                newval,_,img_face_Xseg = find_ratio_intersection_v2(block_ori[idx],crops_coors,data['mouth_point'],data,mask_mount = img_mouth_mask)
                                # cv2.putText(block[idx], text='Intersec ratio: '+str(newval), org=(100, 100), fontFace=cv2.FONT_HERSHEY_TRIPLEX,fontScale=1.1, color=(0, 255, 0),thickness=2)
                                # cv2.putText(block_ori[idx], text='Intersec ratio: '+str(newval), org=(100, 100), fontFace=cv2.FONT_HERSHEY_TRIPLEX,fontScale=1.1, color=(0, 255, 0),thickness=2)

                                # if newval < 0.71:
                                #     print("Ocluded face",count_frame,"|",newval)
                                #     config_merge_mask_cop['mask_mode'] = 8

                                ### Merge face by "hist-match" to make face trasform smoothy
                                merge_face_img = MergeMasked(predictor_func,
                                                             predictor_input_shape,
                                                             face_enhancer_func.enhance,
                                                             xseg_256_extract_func.extract,
                                                             config_merge_mask_cop,data,block[idx],block_ori[idx],remerge=False,skip_merge=False)

                            # # print(merge_face_img)
                            # if count_frame ==550:
                            #     cv2.imwrite("Final_merge.png",merge_face_img[...,0:3])
                            #     print("Saved image")
                            # block_merge.append(merge_face_img[...,0:3])
                            out_img1 = merge_face_img[...,0:3]
                            # img_bgr_uint8_2 = normalize_channels(merge_face_img[...,3], 3)
                            # img_bgr_2 = img_bgr_uint8_2.astype(np.uint8)
                            # out_img1 = cv2.addWeighted(out_img1, 0.6,img_bgr_2 , 0.4, 0)
                            # cv2.polylines(out_img1,[data['mouth_point']], True,color=(0,0,255),thickness=2)
                            write_frame(out_img1,encoder_video)


                            # break

                        reset_data()
                    if not ret:
                        break
                # break

        except Exception:
            pass
        pbar.close()
        encoder_video.stdin.flush()
        encoder_video.stdin.close()
        time.sleep(2)
        ffmpeg_cmd = f"""ffmpeg -y  -hide_banner -loglevel quiet -i {savepath_nonsound} -i '{wavpath}' -c:a aac -c:v copy -crf 17 {output_path}"""
        print(ffmpeg_cmd)
        os.system(ffmpeg_cmd)
