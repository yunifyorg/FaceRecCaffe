import math

import caffe
import cv2
import time
import numpy as np

from mtcnn import MTCNN

GPU_ID = 0
caffe.set_mode_gpu()
caffe.set_device(GPU_ID)

def find_new_locations(points, angle, center, adj, M):
    M_inv = cv2.getRotationMatrix2D(center, angle, 1.0)
    M_inv[0, 2] += adj[0]
    #M_inv[1, 2] += adjustments[1]
    ones = np.ones(shape=(len(points), 1))
    points_ones = np.hstack([points, ones])
    new_points = M_inv.dot(points_ones.T).T
    return new_points

def guard(x, N):
    x[x<0] = 0
    x[:2][x[:2]>N[1]-1] = N[1]-1
    x[2:][x[2:]>N[0]-1] = N[0]-1
    x = x.astype(int)
    return x

def imrotate(image, angle, points=None):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
 
    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
 
    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
 
    # adjust the rotation matrix to take into account translation
    adjustment = ((nW / 2) - cX, (nH / 2) - cY)
    M[0, 2] += adjustment[0]
    M[1, 2] += adjustment[1]
    
    new_points = find_new_locations(points, angle, (cX, cY), adjustment, M) if points is not None else None
    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH)), new_points


def align(img, f5pt, crop_size=144, ec_mc_y=48, ec_y=48):
    pi = math.pi
    ang_delta = math.atan2(f5pt[1, 0]-f5pt[0,0], f5pt[1,1]-f5pt[0,1])
    img_rot, f5pt_rot = imrotate(img, -ang_delta/pi*180, f5pt)
    eyec = (f5pt_rot[0, :] + f5pt_rot[1, :])/2
    mouthc = (f5pt_rot[3, :] + f5pt_rot[4, :])/2
    resize_scale = ec_mc_y/(mouthc[0]-eyec[0])
    img_resize = cv2.resize(img_rot, (0,0), None, fx=resize_scale, fy=resize_scale)
    
    res = img_resize
    rot_shape = np.array(img_rot.shape)[:-1]
    res_shape = np.array(img_resize.shape)[:-1]
    eyec2 = (eyec - rot_shape/2) * resize_scale + res_shape/2
    eyec2 = np.round(eyec2).astype(int)
    img_crop = np.zeros((crop_size, crop_size, img_rot.shape[2]))
    crop_y = int(eyec2[0] - ec_y)
    crop_y_end = int(crop_y + crop_size - 1)
    crop_x = int(eyec2[1]-math.floor(crop_size/2))
    crop_x_end = int(crop_x + crop_size - 1)
    box = guard(np.array([crop_x, crop_x_end, crop_y, crop_y_end]), img_resize.shape)
    img_crop[box[2]-crop_y+1:box[3]-crop_y+1, box[0]-crop_x+1:box[1]-crop_x+1,:] = img_resize[box[2]:box[3],box[0]:box[1],:]
    cropped = img_crop/255
    return res, eyec2, cropped, resize_scale



class FaceFeatureExtractor():
    def __init__(self):
        self.face_detector = MTCNN()
        self.net = caffe.Net('./mtcnn/LightenedCNN_B.prototxt', './mtcnn/LightenedCNN_B.caffemodel', caffe.TEST)

    def getFeatures(self, img):
        t0 = time.time()
        img_aligned, times = self.getAlignedImg(img)
        t1 = time.time()
        im_gray = cv2.cvtColor(img_aligned.astype(np.float32), cv2.COLOR_RGB2GRAY)
        t2 = time.time()
        self.net.blobs['data'].data[...] = cv2.resize(im_gray, (128, 128))
        result = self.net.forward()
        result = result[result.keys()[0]]
        t3 = time.time()

        times.update({'features': t3 - t2})
        return result, times

    def getAlignedImg(self, img):
        t0 = time.time()
        _, f5pt = self.face_detector.detect(img)
        t1 = time.time()
        _, _, img_aligned, _ = align(img, f5pt[0])
        t2 = time.time()
        times = {'detection': t1 - t0, 'alignment': t2 - t1}
        return img_aligned, times

if __name__ == '__main__':
    feature_extractor = FaceFeatureExtractor()
