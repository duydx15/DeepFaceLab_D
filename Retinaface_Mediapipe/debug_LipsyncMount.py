import cv2
import mediapipe as mp
import os
from tqdm import tqdm
import numpy as np
import math
from PIL import Image
import ffmpeg
import subprocess
from pipeline_mobile_resnet import loadmodelface, detection_face
from color_transfer import color_transfer, color_transfer_mix, color_transfer_sot, color_transfer_mkl, color_transfer_idt, color_hist_match, reinhard_color_transfer, linear_color_transfer
from common import random_crop, normalize_channels, cut_odd_image, overlay_alpha_image

import torch
from scipy.spatial import Delaunay, ConvexHull
LOGURU_FFMPEG_LOGLEVELS = {
	"trace": "trace",
	"debug": "debug",
	"info": "info",
	"success": "info",
	"warning": "warning",
	"error": "error",
	"critical": "fatal",
}


import warnings
warnings.filterwarnings("ignore")

class KalmanTracking(object):
	# init kalman filter object
	def __init__(self, point):
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
		 [0, 0, 0, 1]], np.float32) * 0.03
		self.measurement = np.array((point[0], point[1]), np.float32)
		# print(self.measurement)

	def getpoint(self, kp):
		self.kalman.correct(kp-self.measurement)
		# get new kalman filter prediction
		prediction = self.kalman.predict()
		prediction[0][0] = prediction[0][0] +  self.measurement[0]
		prediction[1][0] = prediction[1][0] +  self.measurement[1]

		return prediction

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
	# 	distance = distance + 0.1
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

debuglist = [8, 298, 301, 368, 347, 329, 437, 168, 217, 100, 118, 139, 71, 68, #sunglass
	164, 391, 436, 432, 430, 273, 409, 11, 185, 210, 212, 216, 165,  #mustache
	13, 311, 308, 402, 14, 178, 78, 81, #lips Inner
	0, 269, 291, 375, 405, 17, 181, 146, 61, 39, #lips Outer
	4, 429, 327, 2, 98, 209, #noise
	151, 200, 175, 65, 69, 194, 140, 205, 214, 135, 215,
	177, 137, 34, 295, 299, 418, 369,
	425, 434, 364, 435, 401, 366, 264, 43]
# FACEMESH_bigmask = [151, 337, 299, 333, 298, 301, 368, 264, 447, 366, 401, 435, 367, 364, 394, 395, 369, 396, 175,
# 						171, 140, 170, 179, 135, 138, 215, 177,137, 227, 34, 139, 71, 68, 104, 69, 108]

FACEMESH_bigmask = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
					  397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
					  172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]

FACEMESH_lips_1 = [164, 393, 391, 322, 410, 432, 422, 424, 418, 421, 200, 201,
					   194, 204, 202, 212, 186, 92, 165, 167]
FACEMESH_lips_2 = [326, 426, 436, 434, 430, 431, 262, 428, 199, 208, 32, 211, 210,
					   214, 216, 206, 97]
# FACEMESH_bigmask = [197, 419, 399, 437, 353, 371, 266, 425, 427, 434, 430, 431, 262, 428, 199,
# 					208, 32, 211, 210, 214, 207, 205, 36, 142, 126, 217, 174, 196]

# FACEMESH_bigmask = [197, 419, 399, 437, 353, 371, 266, 425, 427, 434, 394, 395, 369, 396, 175, 171, 140, 170, 169, 214 , 207, 205, 36, 142, 126, 217, 174, 196]

FACEMESH_pose_estimation = [34,264,168,33, 263]

def pointInRect(point, rect):
	x1, y1, x2, y2 = rect
	wbox = abs(x2-x1)
	xo = (x1+x2)/2
	yo = (y1+y2)/2
	x, y = point
	dist1 = math.hypot(x-xo, y-yo)

	aaa = dist1/wbox if wbox>0 else 1
	# print(dist1, ' ', wbox, ' ',aaa)
	# print('cur: ', point, '\told: ',  (xo,yo))
	# print('oldbox: ', rect)
	if (x1 < x and x < x2):
		if (y1 < y and y < y2):
			if aaa <= 0.06:
				return True
	return False

def facemeshTrackByBox(multi_face_landmarks,  w, h):
	# print(bbox)
	for faceIdx, face_landmarks in enumerate(multi_face_landmarks):
		listpoint = []
		for i in range(len(FACEMESH_lips_1)):
			idx = FACEMESH_lips_1[i]
			x = face_landmarks.landmark[idx].x
			y = face_landmarks.landmark[idx].y

			realx = x * w
			realy = y * h
			listpoint.append((realx, realy))

		video_leftmost_x = min(x for x, y in listpoint)
		video_bottom_y = min(y for x, y in listpoint)
		video_rightmost_x = max(x for x, y in listpoint)
		video_top_y = max(y for x, y in listpoint)

		# x = (video_leftmost_x+video_rightmost_x)/2
		y = (video_bottom_y+video_top_y)/2
		# point = (x,y)
		# print(point, ' ', h, w)
		if y < h/2:
			continue
		# if pointInRect(point, bbox):
		return faceIdx
	return -1

def facemeshTrackByBoxCrop(multi_face_landmarks,  w, h):
	for faceIdx, face_landmarks in enumerate(multi_face_landmarks):
		listpoint = []
		for i in range(len(FACEMESH_lips_1)):
			idx = FACEMESH_lips_1[i]
			x = face_landmarks.landmark[idx].x
			y = face_landmarks.landmark[idx].y

			realx = x * w
			realy = y * h
			listpoint.append((realx, realy))

		video_leftmost_x = min(x for x, y in listpoint)
		video_bottom_y = min(y for x, y in listpoint)
		video_rightmost_x = max(x for x, y in listpoint)
		video_top_y = max(y for x, y in listpoint)

		y = (video_bottom_y+video_top_y)/2
		# point = (x,y)
		# print(point, ' ', h, w)
		if y < h/2:
			continue
		# if pointInRect(point, bbox):
		return faceIdx
	return -1

def get_face_by_RetinaFace(facea, inimg, mobile_net, resnet_net, device, kf=None):

	h,w = inimg.shape[:2]

	l_coordinate, detected_face = detection_face(mobile_net,resnet_net, inimg, device)
	if not detected_face:
		return None, None, None

	face_landmarks = None
	bbox = None
	# print('get_face_by_RetinaFace ', l_coordinate)
	for i in range(len(l_coordinate)):
		topleft, bottomright = l_coordinate[i]
		if bottomright[1] < h/3:
			continue

		crop_image = inimg[topleft[1]:bottomright[1], topleft[0]:bottomright[0],:]
		# result = insightface_facial_landmark(insight_app, crop_image)
		# print('l_coordinate[i] ', l_coordinate[i])
		# result, detected_keypoint = facial_landmark_detection(face_mesh, crop_image)

		results = facea.process(cv2.cvtColor(crop_image, cv2.COLOR_BGR2RGB))
		if not results.multi_face_landmarks:
			continue

		face_landmarks = results.multi_face_landmarks[0]
		curbox = []
		curbox.append(topleft[0])
		curbox.append(topleft[1])
		curbox.append(bottomright[0])
		curbox.append(bottomright[1])
		bbox = curbox

	if not face_landmarks:
		return None, None, None
	# print('bbox ', bbox)
	bbox_w = bbox[2] - bbox[0]
	bbox_h = bbox[3] - bbox[1]

	# if kf:
	# 	tmp = facemeshTrackByBox(results.multi_face_landmarks, w, h)
	# 	if tmp<0:
	# 		kf[:]
	# 		return None, None
	# 	else:
	# 		face_landmarks = results.multi_face_landmarks[tmp]
	# else:
	# 	face_landmarks = results.multi_face_landmarks[0]
	#
	# posePoint = []
	# for i in range(len(FACEMESH_pose_estimation)):
	# 	idx = FACEMESH_pose_estimation[i]
	# 	x = face_landmarks.landmark[idx].x
	# 	y = face_landmarks.landmark[idx].y
	#
	# 	realx = x * w
	# 	realy = y * h
	# 	posePoint.append((realx, realy))
	# yaw, pitch, roll = facePose(posePoint[0], posePoint[1], posePoint[2], posePoint[3], posePoint[4])
	# if yaw is None or abs(yaw) >= 45 or abs(pitch) >= 35:
	# 	return None, None

	listpoint = []

	for i in range(len(FACEMESH_lips_1)):
		idx = FACEMESH_lips_1[i]
		x = face_landmarks.landmark[idx].x
		y = face_landmarks.landmark[idx].y

		realx = x * bbox_w + bbox[0]
		realy = y * bbox_h + bbox[1]
		listpoint.append((realx, realy))

	# video_leftmost_x = min(x for x, y in listpoint)
	# video_bottom_y = min(y for x, y in listpoint)
	# video_rightmost_x = max(x for x, y in listpoint)
	# video_top_y = max(y for x, y in listpoint)


	# x = (video_leftmost_x+video_rightmost_x)/2
	# y = (video_bottom_y+video_top_y)/2
	# tmppoint = (x,y)
	# resetKF = False
	# if iswide:
	# 	global oldBboxtagget
	# 	if not pointInRect(tmppoint, oldBboxtagget):
	# 		resetKF = True
	# 	if resetKF:
	# 		del kf[:]
	# 	oldBboxtagget = (video_leftmost_x, video_bottom_y, video_rightmost_x, video_top_y)

	if kf is not None:
		if len(kf) == 0:
			for i in range(len(listpoint)):
				kf.append(KalmanTracking((listpoint[i][0],listpoint[i][1])))
		else:
			for i in range(len(listpoint)):
				intpoint = np.array([np.float32(listpoint[i][0]), np.float32(listpoint[i][1])], np.float32)
				tmp = kf[i].getpoint(intpoint)
				listpoint[i] = (tmp[0][0], tmp[1][0])

	srcpts = np.array(listpoint, np.int32)
	srcpts = srcpts.reshape(-1,1,2)

	listpoint2 = []
	for i in range(len(FACEMESH_lips_2)):
		idx = FACEMESH_lips_2[i]
		x = face_landmarks.landmark[idx].x
		y = face_landmarks.landmark[idx].y

		realx = x * bbox_w + bbox[0]
		realy = y * bbox_h + bbox[1]
		listpoint2.append((realx, realy))

	# if kf is not None:
	# 	if len(kf) == 0:
	# 		for i in range(len(listpoint2)):
	# 			kf.append(KalmanTracking((listpoint2[i][0],listpoint2[i][1])))
	# 	else:
	# 		for i in range(len(listpoint2)):
	# 			intpoint = np.array([np.float32(listpoint2[i][0]), np.float32(listpoint2[i][1])], np.float32)
	# 			tmp = kf[i].getpoint(intpoint)
	# 			listpoint2[i] = (tmp[0][0], tmp[1][0])

	srcpts2 = np.array(listpoint2, np.int32)
	srcpts2 = srcpts2.reshape(-1,1,2)

	listpoint3 = []
	for i in range(len(face_landmarks.landmark)):
		# idx = FACEMESH_bigmask[i]
		x = face_landmarks.landmark[i].x
		y = face_landmarks.landmark[i].y

		realx = x * bbox_w + bbox[0]
		realy = y * bbox_h + bbox[1]
		listpoint3.append((realx, realy))

	srcpts3 = np.array(listpoint3, np.int32)
	srcpts3 = srcpts3.reshape(-1,1,2)

	return srcpts, srcpts2, srcpts3


def get_face(facea, inimg, kf=None, iswide = False):
	listpoint = []
	h,w = inimg.shape[:2]
	# print(inimg.shape[:2])
	results = facea.process(cv2.cvtColor(inimg, cv2.COLOR_BGR2RGB))
	if not results.multi_face_landmarks:
		return None, None

	face_landmarks = None

	if kf:
		tmp = facemeshTrackByBox(results.multi_face_landmarks, w, h)
		if tmp<0:
			kf[:]
			return None, None
		else:
			face_landmarks = results.multi_face_landmarks[tmp]
	else:
		face_landmarks = results.multi_face_landmarks[0]

	posePoint = []
	for i in range(len(FACEMESH_pose_estimation)):
		idx = FACEMESH_pose_estimation[i]
		x = face_landmarks.landmark[idx].x
		y = face_landmarks.landmark[idx].y

		realx = x * w
		realy = y * h
		posePoint.append((realx, realy))
	# yaw, pitch, roll = facePose(posePoint[0], posePoint[1], posePoint[2], posePoint[3], posePoint[4])
	# if yaw is None or abs(yaw) >= 45 or abs(pitch) >= 35:
	# 	return None, None

	for i in range(len(FACEMESH_lips_1)):
		idx = FACEMESH_lips_1[i]
		x = face_landmarks.landmark[idx].x
		y = face_landmarks.landmark[idx].y

		realx = x * w
		realy = y * h
		listpoint.append((realx, realy))

	video_leftmost_x = min(x for x, y in listpoint)
	video_bottom_y = min(y for x, y in listpoint)
	video_rightmost_x = max(x for x, y in listpoint)
	video_top_y = max(y for x, y in listpoint)


	x = (video_leftmost_x+video_rightmost_x)/2
	y = (video_bottom_y+video_top_y)/2
	tmppoint = (x,y)
	resetKF = False
	# if iswide:
	# 	global oldBboxtagget
	# 	if not pointInRect(tmppoint, oldBboxtagget):
	# 		resetKF = True
	# 	if resetKF:
	# 		del kf[:]
	# 	oldBboxtagget = (video_leftmost_x, video_bottom_y, video_rightmost_x, video_top_y)

	if kf is not None:
		if len(kf) == 0:
			for i in range(len(listpoint)):
				kf.append(KalmanTracking((listpoint[i][0],listpoint[i][1])))
		else:
			for i in range(len(listpoint)):
				intpoint = np.array([np.float32(listpoint[i][0]), np.float32(listpoint[i][1])], np.float32)
				tmp = kf[i].getpoint(intpoint)
				listpoint[i] = (tmp[0][0], tmp[1][0])

	srcpts = np.array(listpoint, np.int32)
	srcpts = srcpts.reshape(-1,1,2)

	listpoint2 = []
	for i in range(len(FACEMESH_lips_2)):
		idx = FACEMESH_lips_2[i]
		x = face_landmarks.landmark[idx].x
		y = face_landmarks.landmark[idx].y

		realx = x * w
		realy = y * h
		listpoint2.append((realx, realy))

	# if kf is not None:
	# 	if len(kf) == 0:
	# 		for i in range(len(listpoint2)):
	# 			kf.append(KalmanTracking((listpoint2[i][0],listpoint2[i][1])))
	# 	else:
	# 		for i in range(len(listpoint2)):
	# 			intpoint = np.array([np.float32(listpoint2[i][0]), np.float32(listpoint2[i][1])], np.float32)
	# 			tmp = kf[i].getpoint(intpoint)
	# 			listpoint2[i] = (tmp[0][0], tmp[1][0])

	srcpts2 = np.array(listpoint2, np.int32)
	srcpts2 = srcpts2.reshape(-1,1,2)

	return srcpts, srcpts2

def getKeypointByMediapipe(face_mesh_wide, videoimg,face_mesh_256, faceimg, kf, kf_driving, mobile_net,resnet_net,device):
	# videopts, videopts_out = get_face(face_mesh_wide, videoimg, kf)
	# if videopts is None:
		# return None, None, None, None
	videopts, videopts_out, videopts_big = get_face_by_RetinaFace(face_mesh_wide, videoimg, mobile_net, resnet_net, device, kf)
	if videopts is None:
		return None, None, None, None, None
	facepts, facepts_out = get_face(face_mesh_256, faceimg, kf_driving)
	return videopts, videopts_out, videopts_big, facepts, facepts_out


# def is_on_line(x1, y1, x2, y2, x3):
# 	slope = (y2 - y1) / (x2 - x1)
# 	return slope * (x3 - x1) + y1

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

def get_concat_h(im1, im2):
	dst = Image.new('RGB', (im1.width + im2.width, im1.height))
	dst.paste(im1, (0, 0))
	dst.paste(im2, (im1.width, 0))
	# print(dst.size)
	return dst
def get_concat_v(im1, im2):
	dst = Image.new('RGB', (im1.width, im1.height + im2.height))
	dst.paste(im1, (0, 0))
	dst.paste(im2, (0, im1.height))
	return dst

def applyAffineTransform(src, srcTri, dstTri, size):
	# Given a pair of triangles, find the affine transform.
	warpMat = cv2.getAffineTransform(np.float32(srcTri), np.float32(dstTri))

	# Apply the Affine Transform just found to the src image
	dst = cv2.warpAffine(src, warpMat, (size[0], size[1]), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

	return dst

def warpTriangle(img1, img2, t1, t2):
	# Find bounding rectangle for each triangle
	r1 = cv2.boundingRect(np.float32(t1))
	r2 = cv2.boundingRect(np.float32(t2))

	# Offset points by left top corner of the respective rectangles
	t1Rect = []
	t2Rect = []
	t2RectInt = []

	for i in range(0, 3):
		# print(t1[i][0] - r1[0])
		# print(i, ' ' , t1[i][0][1] )
		# print(r1[1])
		t1Rect.append(((t1[i][0][0] - r1[0]), (t1[i][0][1] - r1[1])))
		t2Rect.append(((t2[i][0][0] - r2[0]), (t2[i][0][1] - r2[1])))
		t2RectInt.append(((t2[i][0][0] - r2[0]), (t2[i][0][1] - r2[1])))

	# Get mask by filling triangle
	mask = np.zeros((r2[3], r2[2], 3), dtype=np.float32)
	cv2.fillConvexPoly(mask, np.int32(t2RectInt), (1.0, 1.0, 1.0), 16, 0)

	# Apply warpImage to small rectangular patches
	img1Rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
	size = (r2[2], r2[3])
	img2Rect = applyAffineTransform(img1Rect, t1Rect, t2Rect, size)
	img2Rect = img2Rect * mask

	# Copy triangular region of the rectangular patch to the output image
	img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] * ((1., 1., 1.) - mask)
	img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] + img2Rect
font = cv2.FONT_HERSHEY_SIMPLEX


def image_stats(image):
  (l, a, b) = cv2.split(image)
  (lMean, lStd) = (l.mean(), l.std())
  (aMean, aStd) = (a.mean(), a.std())
  (bMean, bStd) = (b.mean(), b.std())

  return (lMean, lStd, aMean, aStd, bMean, bStd)


def adjust_avg_color(img_old,img_new):
	# print('adjust_avg_color ', img_new.shape)
	w,h,c = img_new.shape
	for i in range(img_new.shape[-1]):
		old_avg = img_old[:, :, i].mean()
		new_avg = img_new[:, :, i].mean()
		diff_int = (int)(old_avg - new_avg)
		for m in range(img_new.shape[0]):
			for n in range(img_new.shape[1]):
				temp = (img_new[m,n,i] + diff_int)
				if temp < 0:
					img_new[m,n,i] = 0
				elif temp > 255:
					img_new[m,n,i] = 255
				else:
					img_new[m,n,i] = temp

def transfer_avg_color(img_old,img_new):
  # assert(img_old.shape==img_new.shape)
  source = cv2.cvtColor(img_old, cv2.COLOR_BGR2LAB).astype("float32")
  target = cv2.cvtColor(img_new, cv2.COLOR_BGR2LAB).astype("float32")

  (lMeanSrc, lStdSrc, aMeanSrc, aStdSrc, bMeanSrc, bStdSrc) = image_stats(source)
  (lMeanTar, lStdTar, aMeanTar, aStdTar, bMeanTar, bStdTar) = image_stats(target)

  (l, a, b) = cv2.split(target)

  l -= lMeanTar
  a -= aMeanTar
  b -= bMeanTar

  l = (lStdTar / lStdSrc) * l
  a = (aStdTar / aStdSrc) * a
  b = (bStdTar / bStdSrc) * b

  l += lMeanSrc
  a += aMeanSrc
  b += bMeanSrc

  l = np.clip(l, 0, 255)
  a = np.clip(a, 0, 255)
  b = np.clip(b, 0, 255)

  transfer = cv2.merge([l, a, b])
  transfer = cv2.cvtColor(transfer.astype("uint8"), cv2.COLOR_LAB2BGR)

  return transfer

# def adjust_avg_color(img_old,img_new):
#     w,h,c = img_new.shape
#     for i in range(img_new.shape[-1]):
#         old_avg = img_old[:, :, i].mean()
#         new_avg = img_new[:, :, i].mean()
#         diff_int = (int)(old_avg - new_avg)
#         for m in range(img_new.shape[0]):
#             for n in range(img_new.shape[1]):
#                 temp = (img_new[m,n,i] + diff_int)
#                 if temp < 0:
#                     img_new[m,n,i] = 0
#                 elif temp > 255:
#                     img_new[m,n,i] = 255
#                 else:
#                     img_new[m,n,i] = temp

if __name__ == "__main__":

	FRAME_PATH = 'in.mp4'
	FACE_PATH = '3source.mp4'
	# FACE_PATH = '4m256.mp4'

	mobile_net, resnet_net = loadmodelface()
	device = torch.device("cuda")

	mp_face_mesh = mp.solutions.face_mesh
	# face_mesh_256 = mp_face_mesh.FaceMesh(min_detection_confidence=0.2, min_tracking_confidence=0.2,max_num_faces=1)
	face_mesh_256 = mp_face_mesh.FaceMesh(min_detection_confidence=0.2, max_num_faces=1,refine_landmarks=True)
	face_mesh_wide = mp_face_mesh.FaceMesh(min_detection_confidence=0.5)

	trackkpfacemesh = None #[]
	trackkpfacemesh_driving = None # []

	global oldBboxtagget
	oldBboxtagget = [0,0,0,0]
	global oldReset
	oldReset = False

	facecout = 0
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))

	capFrame = cv2.VideoCapture(FRAME_PATH)
	capFace = cv2.VideoCapture(FACE_PATH)

	fps = capFrame.get(cv2.CAP_PROP_FPS)
	width_  = capFrame.get(cv2.CAP_PROP_FRAME_WIDTH)
	height_ = capFrame.get(cv2.CAP_PROP_FRAME_HEIGHT)

	encoder = ffmpeg_encoder('3_source_Lipsync.mp4', fps, int(width_), int(height_))
	# encoder2 = ffmpeg_encoder('vidout.mp4', fps, 256, 256)

	totalF = int(capFace.get(cv2.CAP_PROP_FRAME_COUNT))
	pbar = tqdm(total=totalF)

	while capFrame.isOpened() or capFace.isOpened():
		okay1  , videoimg = capFrame.read()
		okay2 , faceimg = capFace.read()

		if not okay1 or not okay2:
			print('Cant read the video , Exit!')
			break
		pbar.update(1)
		video_h,video_w, = videoimg.shape[:2]

		videopts, videopts_out, videopts_big, facepts, facepts_out = getKeypointByMediapipe(face_mesh_wide, videoimg, face_mesh_256, faceimg, trackkpfacemesh, trackkpfacemesh_driving, mobile_net,resnet_net,device)

		if videopts is None or facepts is None:
			# use orgin frame
			image_draw = cv2.cvtColor(videoimg,cv2.COLOR_RGB2BGR)
			imageout = Image.fromarray(image_draw)
			encoder.stdin.write(imageout.tobytes())
			continue

		try:
			# debugimg = cv2.polylines(faceimg, [facepts_out], True, (0, 0, 255), 1)
			# debugimage_draw = cv2.cvtColor(debugimg,cv2.COLOR_RGB2BGR)
			# debugimageout = Image.fromarray(debugimage_draw)
			# encoder2.stdin.write(debugimageout.tobytes())


			M, _ = cv2.findHomography(facepts, videopts)
			faceimg = cv2.warpPerspective(faceimg,M,(video_w, video_h))

			# facepts_out = cv2.perspectiveTransform(facepts_out.astype(np.float32), M).astype(np.int32)

			# points1 = []
			# for value in videopts_out:
			# 	points1.append([value[0][0], value[0][1]])
			# points1 = np.array(points1)
			# tri = Delaunay(points1)
			# tris1 = []
			# tris2 = []
			# for i in range(0, len(tri.simplices)):
			# 	tri1 = []
			# 	tri2 = []
			# 	for j in range(0, 3):
			# 		tri1.append(videopts_out[tri.simplices[i][j]])
			# 		tri2.append(facepts_out[tri.simplices[i][j]])
			#
			# 	tris1.append(tri1)
			# 	tris2.append(tri2)
			# tris1 = np.array(tris1, np.float32)
			# tris2 = np.array(tris2, np.float32)
			#
			# videoimgWarped = videoimg.copy()
			# for i in range(0, len(tris1)):
			# 	warpTriangle(faceimg, videoimgWarped, tris1[i], tris2[i])
			# videopts_up = [videopts_out[0], videopts_out[1], videopts_out[2], videopts_out[3], videopts_out[17],
			# 				videopts_out[13], videopts_out[14], videopts_out[15], videopts_out[16]]
			# videopts_up = np.array(videopts_up, np.int32)
			# videopts_up = videopts_up.reshape(-1,1,2)
			#
			# videopts_down = [videopts_out[4], videopts_out[5], videopts_out[6], videopts_out[7],
			# 				videopts_out[8], videopts_out[9], videopts_out[10], videopts_out[11], videopts_out[12],
			# 				videopts_out[17]]
			# videopts_down = np.array(videopts_down, np.int32)
			# videopts_down = videopts_down.reshape(-1,1,2)
			#
			# mask_down = np.zeros((video_h,video_w), np.uint8)
			# cv2.fillPoly(mask_down, pts =[videopts_down], color=(255,255,255,0))
			# (y, x) = np.where(mask_down >0)
			# (topy_down, topx_down) = (np.min(y), np.min(x))
			# (bottomy_down, bottomx_down) = (np.max(y), np.max(x))
			# a, b = calculate_V_(videoimg[topy_down:bottomy_down, topx_down:bottomx_down])
			# image = cv2.putText(image, strtext, (150, 150), font,  1, (255, 0, 0), 2, cv2.LINE_AA)

			# center = (int((topx+bottomx)/2),int((bottomy+topy)/2))
			# output = cv2.seamlessClone(faceimg, videoimg, mask_blur, center, cv2.NORMAL_CLONE)

			# mask_blur = np.zeros((video_h,video_w), np.uint8)
			# cv2.fillPoly(mask_blur, pts =[videopts_down], color=(255,255,255,0))
			# (y, x) = np.where(mask_blur >0)
			# (topy, topx) = (np.min(y), np.min(x))
			# (bottomy, bottomx) = (np.max(y), np.max(x))
			# center = (int((topx+bottomx)/2),int((bottomy+topy)/2))
			# # cv2.imwrite("mask_down.jpg",mask_blur)
			#




			# cropface = faceimg[topy:bottomy, topx:bottomx]
			# cropvideo = videoimg[topy:bottomy, topx:bottomx]
			#
			# adjust_avg_color(cropvideo,cropface)
			#
			# faceimg[topy:bottomy, topx:bottomx] = cropface
			# videoimg[topy:bottomy, topx:bottomx] = cropvideo

			# center = (int((topx+bottomx)/2),int((bottomy+topy)/2))
			# outputface = cv2.seamlessClone(faceimg, videoimg, mask_big, center, cv2.NORMAL_CLONE)


			mask_blur = np.zeros((video_h,video_w), np.uint8)
			cv2.fillPoly(mask_blur, pts =[videopts_out], color=255)

			img2_mount_mask = cv2.bitwise_not(mask_blur)
			img2_nomount = cv2.bitwise_and(videoimg, videoimg, mask=img2_mount_mask)
			mountimg_only = cv2.bitwise_and(faceimg, faceimg, mask=mask_blur)
			result = cv2.add(img2_nomount, mountimg_only)

			# mask_diff = mask_big - mask_blur
			(y, x) = np.where(mask_blur >0)
			(topy, topx) = (np.min(y), np.min(x))
			(bottomy, bottomx) = (np.max(y), np.max(x))
			center = (int((topx+bottomx)/2),int((bottomy+topy)/2))
			result = cv2.seamlessClone(result, videoimg, mask_blur, center, cv2.NORMAL_CLONE)

			# videoimg = cv2.inpaint(videoimg,mask_blur,3,cv2.INPAINT_TELEA)


			# print(wrk_face_mask_a_0.shape)

			# mask_big = np.zeros((video_h,video_w), np.uint8)
			# cv2.fillPoly(mask_big, pts =[videopts_big], color=(255,255,255))
			# (y, x) = np.where(mask_big >0)
			# (topy, topx) = (np.min(y), np.min(x))
			# (bottomy, bottomx) = (np.max(y), np.max(x))

			img_bgr_uint8_1 = normalize_channels(result[topy:bottomy, topx:bottomx], 3)
			img_bgr_1 = img_bgr_uint8_1.astype(np.float32) / 255.0
			img_bgr_1 = np.clip(img_bgr_1, 0, 1)
			img_bgr_uint8_2 = normalize_channels(videoimg[topy:bottomy, topx:bottomx], 3)
			img_bgr_2 = img_bgr_uint8_2.astype(np.float32) / 255.0
			img_bgr_2 = np.clip(img_bgr_2, 0, 1)


			new_binary = mask_blur[topy:bottomy, topx:bottomx][..., np.newaxis]
			# new_binary = mask_big[..., np.newaxis]
			img_face_mask_a = np.clip(new_binary, 0.0, 1.0)
			img_face_mask_a[ img_face_mask_a < (1.0/255.0) ] = 0.0 # get rid of noise

			# print(img_face_mask_a.shape)

			# result_new = color_transfer_mkl(img_bgr_1*img_face_mask_a,img_bgr_2*img_face_mask_a)
			# result_new = reinhard_color_transfer (img_bgr_1, img_bgr_2, target_mask=img_face_mask_a, source_mask=img_face_mask_a)
			result_new = linear_color_transfer (img_bgr_1, img_bgr_2)


			final_img = color_hist_match(result_new, img_bgr_2, 238).astype(dtype=np.float32)

			# result[topy:bottomy, topx:bottomx]
			final_img = (final_img*255).astype(np.uint8)
			# resized = cv2.resize(final_img, (256,256), interpolation = cv2.INTER_AREA)
			# cv2.imshow('Tesster', final_img)
			# if cv2.waitKey(5) & 0xFF == 27:
			# 	break
			# final_img = final_img.astype(np.uint8)

			# img2_face_mask = cv2.bitwise_not(mask_big)
			# img2_noface = cv2.bitwise_and(videoimg, videoimg, mask=img2_face_mask)
			result[topy:bottomy, topx:bottomx] = final_img # cv2.add(img2_nomount[topy:bottomy, topx:bottomx], final_img)

			output = cv2.seamlessClone(result, videoimg, mask_blur, center, cv2.NORMAL_CLONE)
			# output = result

			# result[topy:bottomy, topx:bottomx] = transfer_avg_color(videoimg[topy:bottomy, topx:bottomx], result[topy:bottomy, topx:bottomx])



			# image_mask = cv2.merge((mask_blur,mask_blur,mask_blur))
			# image_mask = cv2.dilate(image_mask,kernel,iterations = 1)
			# image_mask = cv2.GaussianBlur(image_mask,(11,11),0)
			# foreground = cv2.multiply(image_mask.astype(np.float64), output.astype(np.float64))
			# background = cv2.multiply(1.0 - image_mask.astype(np.float64), videoimg.astype(np.float64))
			# output = numpy.add(background,foreground)

			# output = cv2.rectangle(output, (topx_down, topy_down), (bottomx_down, bottomy_down), (255, 0, 0), 1)
			# strtext = str(b/a)
			# output = cv2.putText(output, strtext, (150, 150), font,  1, (255, 0, 0), 2, cv2.LINE_AA)

			# dim = (256, 256)
			# videoimg = cv2.polylines(videoimg, [videopts_out], True, (0, 0, 255), 1)
			# videoimg = cv2.resize(videoimg[topy:bottomy, topx:bottomx,:], dim, interpolation = cv2.INTER_AREA)
			# image_draw1 = cv2.cvtColor(videoimg,cv2.COLOR_RGB2BGR)
			# imageout1 = Image.fromarray(image_draw1)
			# faceimg = cv2.polylines(faceimg, [videopts_out], True, (0, 0, 255), 1)
			# faceimg = cv2.resize(faceimg[topy:bottomy, topx:bottomx,:], dim, interpolation = cv2.INTER_AREA)
			# image_draw2 = cv2.cvtColor(faceimg,cv2.COLOR_RGB2BGR)
			# imageout2 = Image.fromarray(image_draw2)
			# outputface = cv2.polylines(outputface, [videopts_out], True, (0, 0, 255), 1)
			# outputface = cv2.polylines(outputface, [videopts_big], True, (0, 255, 0), 1)
			# output = cv2.circle(output, center, radius=0, color=(0, 0, 255), thickness=-1)
			# output = cv2.resize(output[topy:bottomy, topx:bottomx,:], dim, interpolation = cv2.INTER_AREA)
			# image_draw3 = cv2.cvtColor(output,cv2.COLOR_RGB2BGR)
			# imageout3 = Image.fromarray(image_draw3)
			#
			# outfinal1 = get_concat_h(imageout1, imageout2)
			# outfinal2 = get_concat_h(outfinal1, imageout3)


			image_draw = cv2.cvtColor(output,cv2.COLOR_RGB2BGR)
			imageout = Image.fromarray(image_draw)
			encoder.stdin.write(imageout.tobytes())

		except Exception:
			# use orgin frame
			image_draw = cv2.cvtColor(videoimg,cv2.COLOR_RGB2BGR)
			imageout = Image.fromarray(image_draw)
			encoder.stdin.write(imageout.tobytes())
			continue



	pbar.close()
	encoder.stdin.flush()
	encoder.stdin.close()

	# encoder2.stdin.flush()
	# encoder2.stdin.close()
