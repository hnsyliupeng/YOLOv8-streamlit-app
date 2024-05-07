#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   @File Name:     app.py
   @Author:        Luyao.zhang
   @Date:          2023/5/15
   @Description:
-------------------------------------------------
"""
from pathlib import Path
from PIL import Image
import streamlit as st

import config
from utils import load_model, load_segmodel,infer_uploaded_image, infer_uploaded_video, infer_uploaded_webcam

# setting page layout
st.set_page_config(
    page_title="Fiber Detection for YOLOv8",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
    )

# main page heading
st.title("Fiber Detection and Segmentation for YOLOv8")

# sidebar
st.sidebar.header("DL Model For Fiber Detection and Segmentation")

# model options
task_type = st.sidebar.selectbox(
    "Select Detection model",
    ["Detection"]
)
# model options
seg_type = st.sidebar.selectbox(
    "Select Segmentation model",
    ["Segmentation"]
)

model_type = None
segmodel_type= None
if task_type == "Detection":
    model_type = st.sidebar.selectbox(
        "Select Model",
        config.DETECTION_MODEL_LIST
    )
else:
    st.error("Currently only 'Detection' function is implemented")

if seg_type == "Segmentation":
    segmodel_type = st.sidebar.selectbox(
        "Segment model",
        config.Segment_MODEL_LIST
    )
else:
    st.error("Currently only 'Segmentation' function is implemented")

confidence = float(st.sidebar.slider(
    "Select Model Confidence", 30, 100, 50)) / 100

model_path = ""
if model_type:
    model_path = Path(config.DETECTION_MODEL_DIR, str(model_type))
else:
    st.error("Please Select Model in Sidebar")

# load pretrained DL model
try:
    model = load_model(model_path)
except Exception as e:
    st.error(f"Unable to load model. Please check the specified path: {model_path}")

segmodel_path = ""
if segmodel_type:
    segmodel_path = Path(config.Segment_MODEL_DIR, str(segmodel_type))
else:
    st.error("Please Select Model in Sidebar")

# load pretrained DL model
try:
    segmodel = load_segmodel(segmodel_path)
except Exception as e:
    st.error(f"Unable to load model. Please check the specified path: {segmodel_path}")

# image/video options
st.sidebar.header("Image/Video Config")
source_selectbox = st.sidebar.selectbox(
    "Select Source",
    config.SOURCES_LIST
)

source_img = None
if source_selectbox == config.SOURCES_LIST[0]: # Image
    infer_uploaded_image(confidence, model,segmodel_path)
elif source_selectbox == config.SOURCES_LIST[1]: # Video
    infer_uploaded_video(confidence, model)
elif source_selectbox == config.SOURCES_LIST[2]: # Webcam
    infer_uploaded_webcam(confidence, model)
else:
    st.error("Currently only 'Image' and 'Video' source are implemented")