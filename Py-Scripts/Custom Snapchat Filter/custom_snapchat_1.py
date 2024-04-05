""" Custom Snapchat Filter"""
import csv

import math

""" To-DO
1. Blur background & Image Background- Done
2. swiping & pressing on the screen
3. filters annotating
4. main code implementation
5. method to save and store the current frame in local
6. gradio implementation
"""
import cv2
import numpy as np
import time
import torch
import os
import mediapipe as mp
from Custom_HandTrackingModule import HandTracker
import faceBlendCommon as fbc


device = "CUDA" if torch.cuda.is_available() else "CPU"
print("Device: ", device)

# FaceMesh Configuration
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False,max_num_faces=1,refine_landmarks=True,
                                  min_detection_confidence=0.5,min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_style = mp.solutions.drawing_styles
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

## Info of available filters
filter_config = {
    'anonymous':
        [{'path': "filters/GrassVillage_Anonymous_Halloween.png",
          'anno_path': "filters/GrassVillage_Anonymous_Halloween_Annotation.csv",
          'morph': True}],
    'joker':
        [{'path': "filters/joker.png",
          'anno_path': "filters/joker_annotation.csv",
          'morph': True}],
    'carnival':
        [{'path': "filters/carnival.png",
          'anno_path': "filters/carnival_annotation.csv",
          'morph': True}]
