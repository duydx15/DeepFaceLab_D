import sys
import os
import dlib
import glob
import numpy as np
from skimage import io
import cv2
import math
from subprocess import Popen, PIPE
from PIL import Image
def makeDelaunay(theSize1,theSize0,theList):

    # Check if a point is inside a rectangle
    def rect_contains(rect, point) :
        if point[0] < rect[0] :
            return False
        elif point[1] < rect[1] :
            return False
        elif point[0] > rect[2] :
            return False
        elif point[1] > rect[3] :
            return False
        return True

    # Write the delaunay triangles into a file
    def draw_delaunay(subdiv,dictionary1) :

        list4=[]

        triangleList = subdiv.getTriangleList();
        r = (0, 0, theSize1,theSize0)

        for t in triangleList :
            pt1 = (int(t[0]), int(t[1]))
            pt2 = (int(t[2]), int(t[3]))
            pt3 = (int(t[4]), int(t[5]))
            if rect_contains(r, pt1) and rect_contains(r, pt2) and rect_contains(r, pt3) :
                list4.append((dictionary1[pt1],dictionary1[pt2],dictionary1[pt3]))
        dictionary1={}

        return list4

    # Make a rectangle.
    rect = (0, 0, theSize1,theSize0)

    # Create an instance of Subdiv2D.
    subdiv = cv2.Subdiv2D(rect);

    # Make a points list and a searchable dictionary.
    theList=theList.tolist()
    points=[(int(x[0]),int(x[1])) for x in theList]
    dictionary={x[0]:x[1] for x in list(zip(points,range(76)))}
    # Insert points into subdiv
    for p in points :
        subdiv.insert(p)

    # Make a delaunay triangulation list.
    list4=draw_delaunay(subdiv,dictionary);

    # Return the list.
    return list4

def makeCorrespondence(theImage1,theImage2):

    # # Detect the points of face.
    # predictor_path = thePredictor
    # detector = dlib.get_frontal_face_detector()
    # predictor = dlib.shape_predictor(predictor_path)

    # Setting up some initial values.
    array = np.zeros((68,2))
    size=(0,0)
    imgList=[theImage1,theImage2]
    list1=[]
    list2=[]
    j=1
    idx = 0
    for img in imgList:

        size=(img.shape[0],img.shape[1])
        if(j==1):
            currList=list1
        else:
            currList=list2

        # Ask the detector to find the bounding boxes of each face. The 1 in the
        # second argument indicates that we should upsample the image 1 time. This
        # will make everything bigger and allow us to detect more faces.
        # Also give error if face is not found.
        dets = list_point68[idx]
        if(len(dets)==0):
            if(isinstance(f,str)):
                return [[0,f],0,0,0,0,0]
            else:
                return [[0,"No. "+str(j)],0,0,0,0,0]
        j=j+1

        for k, d in enumerate(dets):

            # Get the landmarks/parts for the face in box d.
            shape = predictor(img, d)
            for i in range(0,68):
                currList.append((int(shape.part(i).x),int(shape.part(i).y)))
                array[i][0]+=shape.part(i).x
                array[i][1]+=shape.part(i).y
            currList.append((1,1))
            currList.append((size[1]-1,1))
            currList.append(((size[1]-1)//2,1))
            currList.append((1,size[0]-1))
            currList.append((1,(size[0]-1)//2))
            currList.append(((size[1]-1)//2,size[0]-1))
            currList.append((size[1]-1,size[0]-1))
            currList.append(((size[1]-1)//2,(size[0]-1)//2))
        idx +=1
        
    narray=array/2
    narray=np.append(narray,[[1,1]],axis=0)
    narray=np.append(narray,[[size[1]-1,1]],axis=0)
    narray=np.append(narray,[[(size[1]-1)//2,1]],axis=0)
    narray=np.append(narray,[[1,size[0]-1]],axis=0)
    narray=np.append(narray,[[1,(size[0]-1)//2]],axis=0)
    narray=np.append(narray,[[(size[1]-1)//2,size[0]-1]],axis=0)
    narray=np.append(narray,[[size[1]-1,size[0]-1]],axis=0)
    narray=np.append(narray,[[(size[1]-1)//2,(size[0]-1)//2]],axis=0)

    return [size,imgList[0],imgList[1],list1,list2,narray]

def applyAffineTransform(src, srcTri, dstTri, size) :

    # Given a pair of triangles, find the affine transform.
    warpMat = cv2.getAffineTransform( np.float32(srcTri), np.float32(dstTri) )

    # Apply the Affine Transform just found to the src image
    dst = cv2.warpAffine( src, warpMat, (size[0], size[1]), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101 )

    return dst


# Warps and alpha blends triangular regions from img1 and img2 to img
def morphTriangle(img1, img2, img, t1, t2, t, alpha) :

    # Find bounding rectangle for each triangle
    r1 = cv2.boundingRect(np.float32([t1]))
    r2 = cv2.boundingRect(np.float32([t2]))
    r = cv2.boundingRect(np.float32([t]))


    # Offset points by left top corner of the respective rectangles
    t1Rect = []
    t2Rect = []
    tRect = []


    for i in range(0, 3):
        tRect.append(((t[i][0] - r[0]),(t[i][1] - r[1])))
        t1Rect.append(((t1[i][0] - r1[0]),(t1[i][1] - r1[1])))
        t2Rect.append(((t2[i][0] - r2[0]),(t2[i][1] - r2[1])))


    # Get mask by filling triangle
    mask = np.zeros((r[3], r[2], 3), dtype = np.float32)
    cv2.fillConvexPoly(mask, np.int32(tRect), (1.0, 1.0, 1.0), 16, 0);

    # Apply warpImage to small rectangular patches
    img1Rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
    img2Rect = img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]]

    size = (r[2], r[3])
    warpImage1 = applyAffineTransform(img1Rect, t1Rect, tRect, size)
    warpImage2 = applyAffineTransform(img2Rect, t2Rect, tRect, size)

    # Alpha blend rectangular patches
    imgRect = (1.0 - alpha) * warpImage1 + alpha * warpImage2

    # Copy triangular region of the rectangular patch to the output image
    img[r[1]:r[1]+r[3], r[0]:r[0]+r[2]] = img[r[1]:r[1]+r[3], r[0]:r[0]+r[2]] * ( 1 - mask ) + imgRect * mask


def makeMorphs(theDuration,theFrameRate,theImage1,theImage2,theList1,theList2,theList4,size,theResult):

    totalImages=int(theDuration*theFrameRate)

    # p = Popen(['ffmpeg', '-y', '-f', 'image2pipe', '-r', str(theFrameRate),'-s',str(size[1])+'x'+str(size[0]), '-i', '-', '-c:v', 'libx264', '-crf', '25','-vf','scale=trunc(iw/2)*2:trunc(ih/2)*2','-pix_fmt','yuv420p', theResult], stdin=PIPE)
    for j in range(0,totalImages):

        # Read images
        img1 = theImage1
        img2 = theImage2

        # Convert Mat to float data type
        img1 = np.float32(img1)
        img2 = np.float32(img2)

        # Read array of corresponding points
        points1 = theList1
        points2 = theList2
        points = [];
        alpha=j/(totalImages-1)

        # Compute weighted average point coordinates
        for i in range(0, len(points1)):
            x = ( 1 - alpha ) * points1[i][0] + alpha * points2[i][0]
            y = ( 1 - alpha ) * points1[i][1] + alpha * points2[i][1]
            points.append((x,y))


        # Allocate space for final output
        imgMorph = np.zeros(img1.shape, dtype = img1.dtype)

        # Read triangles from delaunay_output.txt
        for i in range(len(theList4)):
            x = int(theList4[i][0])
            y = int(theList4[i][1])
            z = int(theList4[i][2])

            t1 = [points1[x], points1[y], points1[z]]
            t2 = [points2[x], points2[y], points2[z]]
            t = [ points[x], points[y], points[z] ]

            # Morph one triangle at a time.
            morphTriangle(img1, img2, imgMorph, t1, t2, t, alpha)

        temp_res=cv2.cvtColor(np.uint8(imgMorph),cv2.COLOR_BGR2RGB)
        res=Image.fromarray(temp_res)
        # res.save(p.stdin,'JPEG')

    # p.stdin.close()
    # p.wait()

# makeDelaunay(1080,1440)
