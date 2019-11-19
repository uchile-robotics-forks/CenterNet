from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import cv2
import numpy as np

import sys
sys.path.append("..")


from centernet.src.lib.opts_pose import opts
from centernet.src.lib.detectors.detector_factory import detector_factory



class Clothes_detector():
    def __init__(self,weights):
        opt = opts().init()
        opt.load_model = weights
        os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
        Detector = detector_factory[opt.task]
        self.detector = Detector(opt)
        self.opt = opt

    
    def detect(self,img):
        results = self.detector.run(img)['results']

        det=[]

        for bbox in results[1]:
            if bbox[4] > self.opt.vis_thresh:

                points=[]
                for i in range(0,len(bbox)):
                    points.append(int(bbox[i]))

                det.append(points)
        return det



    def get_points(self,img):

        det = self.detect(img)

        sk_out = []

        if (len(det)>0):
            for i in range(0,len(det)):

                bbox = det[i]

                #####################    HEAD    #############################
                head = []
                for j in range(5,15):
                    head.append(bbox[j])

                ###################    SHIRT   #############################
                shirt = []
                for j in range(15,31):
                    shirt.append(bbox[j])

                ###################    PANTS   #############################
                pants = []
                for j in range(27,39):
                    pants.append(bbox[j])

                sk_out.append([head,shirt,pants])

        return sk_out

    def get_bbox(self,img):

        det = self.detect(img)

        bbox_out = []

        if (len(det)>0):
            for i in range(0,len(det)):

                bbox = det[i]
                #####################    HEAD    #############################
                diff = int(np.abs(bbox[11]-bbox[13])*0.6)+ int(np.abs(bbox[12]-bbox[14])*0.3)
                head =  [bbox[11],min(bbox[12],bbox[14])-diff,bbox[13],min(bbox[12],bbox[14])+diff]

                ###################    SHIRT   #############################
                xs,ys=[],[]
                for j in range(15,31):
                    if j%2==0:
                        ys.append(bbox[j])
                    else:
                        xs.append(bbox[j])

                reg=int(np.abs(bbox[15]-bbox[17])*0.2) #SHOULDERS DISTANCE
                shirt = [np.min(xs)-reg,np.min(ys)-reg,np.max(xs)+reg,np.max(ys)]

                ###################    PANTS   #############################
                xp,yp=[],[]
                for j in range(27,39):
                    if j%2==0:
                        yp.append(bbox[j])
                    else:
                        xp.append(bbox[j])

                pants = [np.min(xp)-reg*2,np.min(yp)-reg,np.max(xp)+reg*2,np.max(yp)]

                bbox_out.append([head,shirt,pants])

        return bbox_out




def main():
    detector = Clothes_detector('../models/multi_pose_dla_3x.pth')

    cam = cv2.VideoCapture(0)
    while True:
        _, img = cam.read()
        cv2.imshow('input', img)

        # det = detector.detect(img)
        # if (len(det)>0):
        #     clothes_img = print_clothes(img,det[0])
        #     cv2.imshow('Clothes', clothes_img)

        bboxes = detector.get_bbox(img)

        clothes_img = img.copy()
        if ( len(bboxes)>0):
            for box in bboxes:
                head, shirt, pants = box[0], box[1], box[2]

                cv2.rectangle(clothes_img,(head[0],head[1]),(head[2],head[3]),(0,0,255),3)
                cv2.rectangle(clothes_img,(shirt[0],shirt[1]),(shirt[2],shirt[3]),(0,255,0),3)
                cv2.rectangle(clothes_img,(pants[0],pants[1]),(pants[2],pants[3]),(255,0,0),3)
                

        cv2.imshow('Clothes', clothes_img)

        skts = detector.get_points(img)

        if ( len(skts)>0):
            for skt in skts:
                head, shirt, pants = skt[0], skt[1], skt[2]
                #print("sizes: [{},{},{}]".format(len(head),len(shirt),len(pants)))

                for i in range(0,len(head)//2):
                    cv2.circle(img,(head[2*i],head[2*i+1]), 7, (0,0,255), -1)

                for i in range(0,len(shirt)//2):
                    cv2.circle(img,(shirt[2*i],shirt[2*i+1]), 7, (0,255,0), -1)

                for i in range(0,len(pants)//2):
                    cv2.circle(img,(pants[2*i],pants[2*i+1]), 7, (255,0,0), -1)
                

        cv2.imshow('Poses', img)

        if cv2.waitKey(1) == 27:
            return  # esc to quit
      



if __name__ == '__main__':
    main()
  
    