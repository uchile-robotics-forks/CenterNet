from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os
import cv2
import numpy as np

import sys
sys.path.append("..")

from opts_pose import opts
from detectors.detector_factory import detector_factory

image_ext = ['jpg', 'jpeg', 'png', 'webp']
video_ext = ['mp4', 'mov', 'avi', 'mkv']
time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge']

def print_poses(image,bbox):

  img = image.copy()
  cv2.circle(img,(bbox[5],bbox[6]), 7, (0,0,255), -1) # Nariz
  cv2.line(img,(bbox[5],bbox[6]),(bbox[7],bbox[8]),(0,0,255),2)
  cv2.circle(img,(bbox[7],bbox[8]), 7, (0,0,255), -1) # Ojo izquierdo
  cv2.line(img,(bbox[5],bbox[6]),(bbox[9],bbox[10]),(0,0,255),2)
  cv2.circle(img,(bbox[9],bbox[10]), 7, (0,0,255), -1) # Ojo derecho

  for i in range(0,6):
    x=11+i*2
    y=12+i*2
    cv2.circle(img,(bbox[x],bbox[y]), 7, (0,0,255), -1) # Orejas
    cv2.line(img,(bbox[x],bbox[y]),(bbox[x-4],bbox[y-4]),(0,0,255),2)

  cv2.line(img,(bbox[15],bbox[16]),(bbox[17],bbox[18]),(0,0,255),2)

  ########################    SHIRT     ################################
  cv2.line(img,(bbox[15],bbox[16]+5),(bbox[17],bbox[18]+5),(0,255,0),2)

  for i in range(0,4):
    x=19+i*2
    y=20+i*2
    cv2.circle(img,(bbox[x],bbox[y]), 7, (0,255,0), -1)
    cv2.line(img,(bbox[x],bbox[y]),(bbox[x-4],bbox[y-4]),(0,255,0),2)


  cv2.circle(img,(bbox[27],bbox[28]), 7, (0,255,0), -1)
  cv2.line(img,(bbox[27],bbox[28]),(bbox[15],bbox[16]),(0,255,0),2)

  cv2.circle(img,(bbox[29],bbox[30]), 7, (0,255,0), -1)
  cv2.line(img,(bbox[29],bbox[30]),(bbox[17],bbox[18]),(0,255,0),2)

  cv2.line(img,(bbox[29],bbox[30]+5),(bbox[27],bbox[28]+5),(0,255,0),2)

  #######################     PANTS     ###############################
  cv2.line(img,(bbox[29],bbox[30]),(bbox[27],bbox[28]),(255,0,0),2)
  cv2.circle(img,(bbox[27]-5,bbox[28]), 7, (255,0,0), -1)
  cv2.circle(img,(bbox[29]+5,bbox[30]), 7, (255,0,0), -1)

  for i in range(0,4):
    x=31+i*2
    y=32+i*2
    cv2.circle(img,(bbox[x],bbox[y]), 7, (255,0,0), -1)
    cv2.line(img,(bbox[x],bbox[y]),(bbox[x-4],bbox[y-4]),(255,0,0),2)
  return img

def print_clothes(image,bbox):

  img = image.copy()
  ###################    HEAD   #############################
  diff = int(np.abs(bbox[11]-bbox[13])*0.6)+ int(np.abs(bbox[12]-bbox[14])*0.3)
  cv2.rectangle(img,(bbox[11],min(bbox[12],bbox[14])-diff),(bbox[13],min(bbox[12],bbox[14])+diff),(0,0,255),3)

  ###################    SHIRT   #############################
  xs,ys=[],[]
  for i in range(15,30):
    if i%2==0:
      ys.append(bbox[i])
    else:
      xs.append(bbox[i])

  #reg=int(np.abs(bbox[6]-bbox[16])*0.2)
  reg=int(np.abs(bbox[15]-bbox[17])*0.2)
  cv2.rectangle(img,(np.min(xs)-reg,np.min(ys)-reg),(np.max(xs)+reg,np.max(ys)),(0,255,0),3)

  ###################    PANTS   #############################
  xp,yp=[],[]
  for i in range(29,38):
    if i%2==0:
      yp.append(bbox[i])
    else:
      xp.append(bbox[i])

  cv2.rectangle(img,(np.min(xp)-reg*2,np.min(yp)-reg),(np.max(xp)+reg*2,np.max(yp)),(255,0,0),3)

  return img


def demo(opt):
  os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
  opt.debug = max(opt.debug, 0)
  Detector = detector_factory[opt.task]
  detector = Detector(opt)

  cam = cv2.VideoCapture(0 if opt.demo == 'webcam' else opt.demo)
  detector.pause = False
  while True:
      _, img = cam.read()
      cv2.imshow('input', img)
      ret = detector.run(img)
      time_str = ''
      for stat in time_stats:
        time_str = time_str + '{} {:.3f}s |'.format(stat, ret[stat])
      print(time_str)

      results = ret['results']

      for bbox in results[1]:
        if bbox[4] > opt.vis_thresh:

          points=[]
          for i in range(0,len(bbox)):
            points.append(int(bbox[i]))


          pose_img = print_poses(img,points)
          clothes_img = print_clothes(img,points)



          cv2.imshow('Poses', pose_img)

          cv2.imshow('Clothes', clothes_img)



      if cv2.waitKey(1) == 27:
          return  # esc to quit

if __name__ == '__main__':
  opt = opts().init()
  demo(opt)
