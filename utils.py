#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   @File Name:     utils.py
   @Author:        Luyao.zhang
   @Date:          2023/5/16
   @Description:
-------------------------------------------------
"""
from ultralytics import YOLO
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import streamlit as st
import cv2
import torch
from PIL import Image
import tempfile
import matplotlib.pyplot as plt
import numpy as np

def _display_detected_frames(conf, model, st_frame, image):
    """
    Display the detected objects on a video frame using the YOLOv8 model.
    :param conf (float): Confidence threshold for object detection.
    :param model (YOLOv8): An instance of the `YOLOv8` class containing the YOLOv8 model.
    :param st_frame (Streamlit object): A Streamlit object to display the detected video.
    :param image (numpy array): A numpy array representing the video frame.
    :return: None
    """
    # Resize the image to a standard size
    image = cv2.resize(image, (720, int(720 * (9 / 16))))

    # Predict the objects in the image using YOLOv8 model
    res = model.predict(image, conf=conf)

    # Plot the detected objects on the video frame
    res_plotted = res[0].plot()
    st_frame.image(res_plotted,
                   caption='Detected Video',
                   channels="BGR",
                   use_column_width=True
                   )


@st.cache_resource
def load_model(model_path):
    """
    Loads a YOLO object detection model from the specified model_path.

    Parameters:
        model_path (str): The path to the YOLO model file.

    Returns:
        A YOLO object detection model.
    """
    model = YOLO(model_path)
    return model

def load_segmodel(segmodel_path):
    """
    Loads a SAM object segmentation model from the specified model_path.

    Parameters:
        model_path (str): The path to the YOLO model file.

    Returns:
        A SAM object segmentation model.
    """
    return segmodel_path
def infer_uploaded_image(conf, model,segmodel_path):
    """
    Execute inference for uploaded image
    :param conf: Confidence of YOLOv8 model
    :param model: An instance of the `YOLOv8` class containing the YOLOv8 model.
    :return: None
    """
    source_img = st.sidebar.file_uploader(
        label="Choose an image...",
        type=("jpg", "jpeg", "png", 'bmp', 'webp')
    )

    col1, col2 = st.columns(2)

    with col1:
        if source_img:
            uploaded_image = Image.open(source_img)
            # adding the uploaded image to the page with caption
            st.image(
                image=source_img,
                caption="Uploaded Image",
                use_column_width=True
            )

    if source_img:
        if st.button("Execution"):
            with st.spinner("Running..."):
                res = model.predict(uploaded_image,
                                    conf=conf)
                boxes = res[0].boxes
                for result in res:
                    boxex = result.boxes  # Boxes object for bbox outputs
    
                bbox = boxex.xyxy.tolist()
                bbox = [[int(i) for i in box] for box in bbox]
                Ts = res[0].boxes.xyxy                      
                numfiber=str(Ts.shape[0])
                res_plotted = res[0].plot()[:, :, ::-1]
                image=uploaded_image
                with col2:
                    st.image(res_plotted,
                             caption="Detected Image",
                             use_column_width=True)
                    try:
                        with st.expander("Detection Results"):
                            for box in boxes:
                                st.write(box.xywh)
                         
                        with st.container():
                             st.write("The number of fiber is")
                             st.write(numfiber)
                        with st.container():
                             st.write("Segmentation")
                            # st.write(source_img)
                             #st.write(image)
                             segsamplot(source_img,bbox,segmodel_path)
                    except Exception as ex:
                        st.write("No image is uploaded yet!")
                        st.write(ex)


def infer_uploaded_video(conf, model):
    """
    Execute inference for uploaded video
    :param conf: Confidence of YOLOv8 model
    :param model: An instance of the `YOLOv8` class containing the YOLOv8 model.
    :return: None
    """
    source_video = st.sidebar.file_uploader(
        label="Choose a video..."
    )

    if source_video:
        st.video(source_video)

    if source_video:
        if st.button("Execution"):
            with st.spinner("Running..."):
                try:
                    tfile = tempfile.NamedTemporaryFile()
                    tfile.write(source_video.read())
                    vid_cap = cv2.VideoCapture(
                        tfile.name)
                    st_frame = st.empty()
                    while (vid_cap.isOpened()):
                        success, image = vid_cap.read()
                        if success:
                            _display_detected_frames(conf,
                                                     model,
                                                     st_frame,
                                                     image
                                                     )
                        else:
                            vid_cap.release()
                            break
                except Exception as e:
                    st.error(f"Error loading video: {e}")


def infer_uploaded_webcam(conf, model):
    """
    Execute inference for webcam.
    :param conf: Confidence of YOLOv8 model
    :param model: An instance of the `YOLOv8` class containing the YOLOv8 model.
    :return: None
    """
    try:
        flag = st.button(
            label="Stop running"
        )
        vid_cap = cv2.VideoCapture(0)  # local camera
        st_frame = st.empty()
        while not flag:
            success, image = vid_cap.read()
            if success:
                _display_detected_frames(
                    conf,
                    model,
                    st_frame,
                    image
                )
            else:
                vid_cap.release()
                break
    except Exception as e:
        st.error(f"Error loading video: {str(e)}")

# Plot segmentation 
def segsamplot(source_img,bbox,segmodel_path):

  sam_checkpoint = segmodel_path
  #model_type = "vit_h"
  model_type = "vit_b"
  device = "cpu"

  sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
  sam.to(device=device)
  #image = cv2.resize(image, (720, int(720 * (9 / 16))))
  img = Image.open(source_img)
  #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  image = cv2.cvtColor(np.asarray(img),cv2.COLOR_RGB2BGR)
  #image = cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGB)
  predictor = SamPredictor(sam)
  predictor.set_image(image)
  #bbox = boxes.xyxy.tolist()
 # bbox = [[int(i) for i in box] for box in bbox]
  input_boxes = torch.tensor(bbox, device=predictor.device)

  transformed_boxes = predictor.transform.apply_boxes_torch(input_boxes, image.shape[:2])
  masks, _, _ = predictor.predict_torch(
      point_coords=None,
      point_labels=None,
      boxes=transformed_boxes,
      multimask_output=False,
  )

  for i, mask in enumerate(masks):

      binary_mask = masks[i].squeeze().numpy().astype(np.uint8)

      # Find the contours of the mask
      contours, hierarchy = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

      largest_contour = max(contours, key=cv2.contourArea)

      # Get the new bounding box
      bbox = [int(x) for x in cv2.boundingRect(largest_contour)]

      # Get the segmentation mask for object 
      segmentation = largest_contour.flatten().tolist()

      # Write bounding boxes to file in YOLO format
      with open('BBOX_yolo.txt', 'w') as f:
          for contour in contours:
              # Get the bounding box coordinates of the contour
              x, y, w, h = cv2.boundingRect(contour)
              # Convert the coordinates to YOLO format and write to file
              f.write('0 {:.6f} {:.6f} {:.6f} {:.6f}\n'.format((x+w/2)/image.shape[1], (y+h/2)/image.shape[0], w/image.shape[1], h/image.shape[0]))
              f.write('\n')
      mask=segmentation
      
          # load the image
      #width, height = image_path.size
      #img = Image.open(source_img)
      width, height = img.size

      # convert mask to numpy array of shape (N,2)
      mask = np.array(mask).reshape(-1,2)

      # normalize the pixel coordinates
      mask_norm = mask / np.array([width, height])

      # compute the bounding box
      xmin, ymin = mask_norm.min(axis=0)
      xmax, ymax = mask_norm.max(axis=0)
      bbox_norm = np.array([xmin, ymin, xmax, ymax])

      # concatenate bbox and mask to obtain YOLO format
      yolo = np.concatenate([bbox_norm, mask_norm.reshape(-1)])

      # compute the bounding box
      # write the yolo values to a text file
      with open('yolomask_format.txt', 'w') as f:
          for val in yolo:
              f.write("{:.6f} ".format(val))
          f.write('\n')

      # Print the bounding box and segmentation mask
      print("Bounding box:", bbox)
      #print("Segmentation mask:", segmentation)
      print("yolo",yolo)
  #plt.figure(figsize=(10, 10))
  fig, ax = plt.subplots()
  #with st.image(image):
  im = ax.imshow(image)
  for mask in masks:
     show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
  for box in input_boxes:
     show_box(box.cpu().numpy(), plt.gca())
  plt.axis('off')
  st.pyplot(fig) 

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
   # ax.imshow(mask_image)
    ax.imshow(mask_image)
    #st.pyplot()
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))   