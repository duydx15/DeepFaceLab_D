import os
import sys
import cv2

from pipeline_mobile_resnet import loadmodelface, detection_face

mobile_net, resnet_net = loadmodelface()

def retinaface_check(blocks):
    block_check = []
    for frame in blocks:
        h,w = frame.shape[:2]
        count_loss_detect = 0
        padding_ratio = 0.2
        detect_fail = False
        previous_model = 0
        crop_image = None

        detected_face = detection_face(mobile_net,resnet_net, frame, device,padding_ratio)
        block_check.append(detected_face)
    return block_check
