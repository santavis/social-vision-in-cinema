# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 20:56:33 2023

@author: Severi Santavirta
"""

# Panoptic segmentation of the video frames

import os
os.chdir('path/video_segmentation/segment')
from Detector import *
import torch
import scipy.io

# INPUT detector configuration
mdl_type = 'ps' # Choose model type, ps = Panoptic segmentation, take a look at Detectror and MODEL_ZOO
prediction_thresh = 0.50 # Confidence interval for classification 
device = "cuda" # cuda or cpu

# INPUT dataset
dset = 'conjuring' # localizer (Exp. 1), kasky (Exp. 2) or conjuring (Ext. 3) 
input_vids = f"path/video_segmentation/videos/{dset}_eyetracking_mp4" # Directory of input video files
output_vids = f"path/video_segmentation/segment/{dset}" # Directory to store the data and visualize segmentations

# Initialize detector
detector = Detector(model_type=mdl_type, thr = prediction_thresh, device = device)

# Get the catalog
categories = detector.getCatalog(mdl_type)
mdict = {'catalog':categories}
scipy.io.savemat(f'{output_vids}/{mdl_type}/catalog.mat',mdict)

# Loop over all localizer videos
thrstr = "%d" % (round(prediction_thresh*100))
f = os.listdir(input_vids)
for vid_idx, vid in enumerate(f):    
    
    vid_path = os.path.join(input_vids,vid)
    
    # PS    
    if not(os.path.isfile(f'{output_vids}/ps/viz/{os.path.splitext(vid)[0]}_{mdl_type}_{thrstr}.mp4')):
        if not(os.path.exists(f'{output_vids}/ps/data/{os.path.splitext(vid)[0]}')):
            os.mkdir(f'{output_vids}/ps/data/{os.path.splitext(vid)[0]}')
        print("\nPanoptic segmentation: Movie clip %d/%d" % (vid_idx+1,len(f)))
        torch.cuda.empty_cache()
        detector.onVideo(vid_path,output_vids,mdl_type,prediction_thresh)

