# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 14:09:37 2023

@author: Severi Santavirta
"""

# Face detection 

import os
os.chdir('path/gazetime_aoi/detection')
from facedetection_insightface_video import *
import scipy.io
import pickle
import cv2
import numpy as np
import shutil

# INPUT dataset
dset = 'localizer'
input_vids = f"path/videos/{dset}_eyetracking_mp4" # Directory of input video files
output_vids = f"path/video_segmentation/segment/{dset}" # Directory to store the data and visualize segmentations
f = os.listdir(input_vids)

# Read one file for frame size
file_path = vid_path = os.path.join(input_vids,f[1])
vid = cv2.VideoCapture(file_path)
frame_height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))

# Append the catalog
cats = scipy.io.loadmat(f'path/video_segmentation/segment/{dset}/ps/catalog.mat')['catalog']
if(len(cats)<137):
    cats = np.append(cats,['face','eyes','mouth'])
    mdict = {'catalog':cats}
    scipy.io.savemat(f'path/video_segmentation/segment/{dset}/face/catalog.mat',mdict)

for vid_idx, vid in enumerate(f):
    
    vid_path = os.path.join(input_vids,vid)
    
    # Run the face detection
    if not(os.path.isfile(f'{output_vids}/face/viz/{os.path.splitext(vid)[0]}_insightface.mp4')):
    
        run_face_detection(vid_path,f'{output_vids}/ps/viz/{os.path.splitext(vid)[0]}_ps_50.mp4',f'{output_vids}/face/viz')
    
    # Overlay paoptic segmentations with the face masks       
    if not(os.path.isdir(f'{output_vids}/face/data/{os.path.splitext(vid)[0]}')):
        
        # Download the created pickle file and extract the masks and export them to data folder
        os.mkdir(f'{output_vids}/face/data/{os.path.splitext(vid)[0]}')
        pklfile = f'{output_vids}/face/viz/{os.path.splitext(vid)[0]}_insightface.pkl'
        with open(pklfile, 'rb') as pickle_file:
            content = pickle.load(pickle_file)
            
        # Loop through each frame
        fr= 0
        digits = len(str(len(content)))
        for frame in content:
            fr=fr+1
            mask = dets2rect(content[frame],frame_height,frame_width)
            
            # The face masks are overlayed onto the panoptic segmentation to save space
            segment = scipy.io.loadmat(f'path/video_segmentation/segment/{dset}/ps/data/{os.path.splitext(vid)[0]}/frame_{fr:0>{digits}}.mat')['mask']
            segment[mask==1] = 135 # face
            segment[mask==2] = 136 # eyes
            segment[mask==3] = 137 # mouth
            mdict = {'mask':segment}
            scipy.io.savemat(f'{output_vids}/face/data/{os.path.splitext(vid)[0]}/frame_{fr:0>{digits}}.mat',mdict)
            
    # Compress the panoptic segmentation folder for memory management
    if(os.path.isdir(f'{output_vids}/ps/data/{os.path.splitext(vid)[0]}')):
        archived = shutil.make_archive(f'{output_vids}/ps/data/{os.path.splitext(vid)[0]}_compressed.zip','zip', f'{output_vids}/ps/data/{os.path.splitext(vid)[0]}')
        deleted = shutil.rmtree(f'{output_vids}/ps/data/{os.path.splitext(vid)[0]}')


