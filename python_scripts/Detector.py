# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 20:40:53 2023

@author: Severi Santavirta
"""

# Use Detectron2 to segment videos

from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2 import model_zoo
from pims import PyAVReaderTimed

import scipy.io
import cv2
import numpy as np
import os
import sys
import math

class Detector:
    def __init__(self, model_type = "od", thr = 0.5, device = "cpu"): # Define the predictor model configurations
        self.cfg = get_cfg()
        self.model_type = model_type
        
        # Load model config and pretrained model
        if model_type == "od": # Object detection
            self.cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")
        elif model_type == "is": # Instance segmentation
            self.cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"))
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml")
        elif model_type == "kp": # Keypoint detection
            self.cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_X_101_32x8d_FPN_3x.yaml"))
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Keypoints/keypoint_rcnn_X_101_32x8d_FPN_3x.yaml")
        elif model_type == "lvis": # LVIS instance segmentation
            self.cfg.merge_from_file(model_zoo.get_config_file("LVISv0.5-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_1x.yaml"))
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("LVISv0.5-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_1x.yaml")
        elif model_type == "ps": # Panoptic segmentation
            self.cfg.merge_from_file(model_zoo.get_config_file("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml"))
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml")
        elif model_type == "ps2": # Panoptic segmentation 2 (for some reason works only with cpu for now)
            self.cfg.merge_from_file(model_zoo.get_config_file("Misc/panoptic_fpn_R_101_dconv_cascade_gn_3x.yaml"))
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("Misc/panoptic_fpn_R_101_dconv_cascade_gn_3x.yaml")

        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = thr
        self.cfg.MODEL.DEVICE = device # cuda or cpu
        
        self.predictor = DefaultPredictor(self.cfg)
        self.metadata = MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0])
        
    def getCatalog(self,mdl_type): # Get the ful list of categories available for predictions
        if(mdl_type=="ps"):    
            categories = self.metadata.thing_classes + self.metadata.stuff_classes
        elif(mdl_type=="kp"):
            categories = self.metadata.keypoint_names
        return categories
    
    def onImage(self, imagePath): # Predict image
        image = cv2.imread(imagePath)
        if self.model_type != "ps":
            predictions = self.predictor(image)
    
            viz = Visualizer(image[:,:,::-1], metadata = MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]), instance_mode = ColorMode.IMAGE_BW)
            
            output = viz.draw_instance_predictions(predictions["instances"].to("cpu"))
        else:
            predictions, segmentInfo = self.predictor(image)["panoptic_seg"]
            viz = Visualizer(image[:,:,::-1], MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]))
            output = viz.draw_panoptic_seg_predictions(predictions.to("cpu"), segmentInfo)
            
        cv2.imwrite("C:/video_segmentation/segment/1_result.jpg", output.get_image()[:,:,::-1])
        
    def onVideo(self,videoPath,outDir,model_type,thr): # Predict video
        
        # Preps
        vid_basename = os.path.basename(videoPath)
        thrstr = "%d" % (round(thr*100))
        vid_name = f'{os.path.splitext(vid_basename)[0]}_{model_type}_{thrstr}' 
        
        # Input video info
        vr = PyAVReaderTimed(videoPath)
        frame_width, frame_height = vr.frame_shape[1], vr.frame_shape[0]
        nframes_a, vid_fps = len(vr), vr.frame_rate
        
        # Define output video
        output_fname = f'{outDir}/{model_type}/viz/{vid_name}.mp4'
        
        # Define output data 
        output_data = f'{outDir}/{model_type}/data/{os.path.splitext(vid_basename)[0]}'
    
        output_vid_file = cv2.VideoWriter(filename=output_fname,
                    fourcc=cv2.VideoWriter_fourcc(*"mp4v"), # for .mp4
                    fps=float(vid_fps),
                    frameSize=(frame_width,frame_height), # (width, height),
                    isColor=True )
        
        # Access the categories and their types
        if(model_type=='ps'):
            cats = self.metadata.thing_classes + self.metadata.stuff_classes
            n_things = len(self.metadata.thing_classes)
        elif(model_type=='kp'):
            cats = self.metadata.keypoint_names
            
        
        # Read the input video one frame at a time
        cap = cv2.VideoCapture(videoPath)
        
        if(cap.isOpened()==False):
            print/("Error opening file...")
            return
    
        fr = 0
        digits = len(str(nframes_a))
        (success, image) = cap.read()
        while success:
                
            # Progress bar
            fr = fr + 1
            pr = fr/nframes_a*100
            sys.stdout.write('\r')
            sys.stdout.write("[%-100s] %.1f%%" % ('='*math.floor(pr), pr))
            sys.stdout.flush()
            
            if(self.model_type == ('is')):
                predictions = self.predictor(image) # Predict instances using the model
                
                # Save instance masks
                classes_image = predictions['instances'].get('pred_classes').cpu().numpy().astype('uint8') # Classes identified in this image
                classes_image = classes_image+1 # To make them valid indices for matlab
                
                # Initialize mask array
                mask = np.zeros((frame_height,frame_width),dtype='uint8')
                n=0
                for pred_mask in predictions['instances'].pred_masks:
                    instance = pred_mask.to("cpu").numpy().astype('uint8') # mask of the current instance
                    mask[instance==1] = classes_image[n]
                    n=n+1
                    
                # Create a dictionary and save as .mat file
                mdict = {'mask':mask}
                scipy.io.savemat(f'{output_data}/frame_{fr:0>{digits}}.mat',mdict)
                
                # Visualize
                viz = Visualizer(image[:,:,::-1], metadata = MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]), instance_mode = ColorMode.IMAGE)
                output = viz.draw_instance_predictions(predictions["instances"].to("cpu"))
            elif(self.model_type == ("ps" or "ps2")):
                predictions, segmentInfo = self.predictor(image)["panoptic_seg"]
                
                # Initialize mask array
                mask = np.zeros((frame_height,frame_width),dtype='uint8')
                for segment in segmentInfo:
                    category_id = segment["category_id"]+1
                    instance_id = segment["id"]

                    # Extract the mask for the current instance
                    instance = (predictions == instance_id) * (predictions != -1)
                    instance = instance.to('cpu').numpy().astype('uint8')
                    
                    # Figure our the category id in catalog
                    if(segment['isthing']):
                        mask[instance==1] = category_id
                    else:
                        mask[instance==1] = category_id+n_things

                # Create a dictionary and save as .mat file
                mdict = {'mask':mask}
                scipy.io.savemat(f'{output_data}/frame_{fr:0>{digits}}.mat',mdict)
                
                # Visualization
                viz = Visualizer(image[:,:,::-1], MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]))
                output = viz.draw_panoptic_seg_predictions(predictions.to("cpu"), segmentInfo)
            elif(self.model_type == ('kp')):
                predictions = self.predictor(image) # Predict instances using the model
                
                # x,y,score for each 17 keypoints. The predictions score is not well documented (higher is better).
                # Probably scores > 0.05 are plotted in the visualizations
                instance = predictions['instances'].get('pred_keypoints').to("cpu").numpy() 
    
                # Create a dictionary and save as .mat file
                mdict = {'keypoints':instance}
                scipy.io.savemat(f'{output_data}/frame_{fr:0>{digits}}.mat',mdict)
                
                # Visualize
                viz = Visualizer(image[:,:,::-1], metadata = MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]), instance_mode = ColorMode.IMAGE)
                output = viz.draw_instance_predictions(predictions["instances"].to("cpu"))
                   
            # Save image to video file
            output_vid_file.write(output.get_image()[:,:,::-1])
            (success, image) = cap.read()
                
        # Save the video
        output_vid_file.release()
                