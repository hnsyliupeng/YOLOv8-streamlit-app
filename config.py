#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   @File Name:     config.py
   @Author:        Luyao.zhang
   @Date:          2023/5/16
   @Description: configuration file
-------------------------------------------------
"""
from pathlib import Path
import sys

# Get the absolute path of the current file
file_path = Path(__file__).resolve()

# Get the parent directory of the current file
root_path = file_path.parent

# Add the root path to the sys.path list if it is not already there
if root_path not in sys.path:
    sys.path.append(str(root_path))

# Get the relative path of the root directory with respect to the current working directory
ROOT = root_path.relative_to(Path.cwd())


# Source
SOURCES_LIST = ["Image", "Video", "Webcam"]


# DL model config
DETECTION_MODEL_DIR = ROOT / 'weights' / 'detection'
YOLOv5 = DETECTION_MODEL_DIR / "yolov5.pt"
YOLOv8 = DETECTION_MODEL_DIR / "yolov8.pt"
YOLOv8fine = DETECTION_MODEL_DIR / "yolov8gold.pt"
#YOLOv8l = DETECTION_MODEL_DIR / "yolov8l.pt"
#YOLOv8x = DETECTION_MODEL_DIR / "yolov8x.pt"
#YOLOv8f = DETECTION_MODEL_DIR / "fiber.pt"

DETECTION_MODEL_LIST = [
    "yolov5.pt",
    "yolov8.pt",
    "yolov8gold.pt",
    #"yolov8l.pt",
    #"yolov8x.pt",
    ]

Segment_MODEL_DIR = ROOT / 'weights' / 'segment'
SAMseg =Segment_MODEL_DIR /"sam_vit_b_01ec64.pth"
SAMFineseg=Segment_MODEL_DIR /"sam_fine.pth"
SAMMEseg=Segment_MODEL_DIR /"medsam_vit_b.pth"
SAMLMseg=Segment_MODEL_DIR /"medlam.pth"
SAMFTseg=Segment_MODEL_DIR /"swinl_only_sam_many2many.pth"
Segment_MODEL_LIST = ["sam_vit_b_01ec64.pth",
                     "sam_fine.pth",
                     "medsam_vit_b.pth",
                     "medlam.pth",
                     "swinl_only_sam_many2many.pth"]

