# -*- coding: utf-8 -*-
"""
Created on Mon May 20 20:21:25 2019

@author: user
"""

import os
import numpy as np
import cv2
import mrcnn.config
import mrcnn.utils
from mrcnn.model import MaskRCNN
from pathlib import Path

class MaskRCNNConfig(mrcnn.config.Config):
    NAME = "coco_pretrained_model_config"
    IMAGES_PER_GPU = 1 
    GPU_COUNT = 1 
    NUM_CLASSES = 1+80 #COCO dataset has 80 classes + 1 background class
    DETECTION_MIN_CONFIDENCE = 0.6
    
#get only trucks or cars
def get_car_box(boxes,class_ids):
    car_boxes = []
    for i , box in enumerate(boxes):
        if class_ids[i] in [3,8,6]: #car, truck, bus
            car_boxes.append(box)
    return np.array(car_boxes)

#dir
Root_dir = Path("../Mask_RCNN-master")

#logs dir
Model_dir = os.path.join(Root_dir,"logs")

#coco training model
COCO_MODEL_PATH = os.path.join(Root_dir,"mask_rcnn_coco.h5")

#download COCO trained weights if needed
if not os.path.exists(COCO_MODEL_PATH):
    mrcnn.utils.download_trained_weights(COCO_MODEL_PATH)

#image 
IMAGE_dir = os.path.join(Root_dir,"images")

video_source = os.path.join(Root_dir,"test/91117.t.mp4")

#create M-RCNN
model = MaskRCNN(mode="inference",model_dir=Model_dir,
                 config = MaskRCNNConfig())
#load pretrained model 
model.load_weights(COCO_MODEL_PATH,by_name = True)



parked_car_boxes = None
video_capture = cv2.VideoCapture(video_source)

while video_capture.isOpened():
    success, frame = video_capture.read()
    if not success:
        break
    
    rgb_image = frame[:, :, ::-1]
    #model run
    results = model.detect([rgb_image],verbose=0)
    #results[0],results[1]
    r = results[0]

    # columns' decription
    # - r['rois'] are the bounding box of each detected object
    # - r['class_ids'] are the class id (type) of each detected object
    # - r['scores'] are the confidence scores for each detection
    # - r['masks'] are the object masks for each detected object (which gives you the object outline)
    
    if parked_car_boxes is None:
        parked_car_boxes = get_car_box(r['rois'],r['class_ids'])
    else:
        car_boxes = get_car_box(r['rois'],r['class_ids']) 
        overlaps = mrcnn.utils.compute_overlaps(parked_car_boxes,
                                                car_boxes)
        free_space = False
    #print("cars found in frame of video")
    
    #Draw the box of frame
        for parking_area, overlap_area in zip(parked_car_boxes, overlaps):
            max_IoU_overlap = np.max(overlap_area)
            
            y1,x1,y2,x2 = parking_area
            if max_IoU_overlap < 0.15:               #r,g ,b
                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),3)
                free_space = True
            else:                                    #r,g ,b
                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,255),1)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, f"{max_IoU_overlap:0.2}", (x1 + 6, y2 - 6), font, 0.3, (255, 255, 255))
        if free_space:
            
            free_space +=1
        else:
            free_space_frames = 0
    
        cv2.imshow('video',frame)
        
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
            
            
video_capture.release()
cv2.destroyAllWindows()
        






























