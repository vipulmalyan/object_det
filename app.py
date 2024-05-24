import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image, ExifTags
from urllib.request import urlretrieve

# Load YOLOv5 model
model = torch.hub.load("ultralytics/yolov5", "yolov5s")

# Function to fix image orientation based on EXIF data
def fix_orientation(img):
    try:
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == 'Orientation':
                break

        exif = dict(img._getexif().items())

        if exif[orientation] == 3:
            img = img.rotate(180, expand=True)
        elif exif[orientation] == 6:
            img = img.rotate(270, expand=True)
        elif exif[orientation] == 8:
            img = img.rotate(90, expand=True)
    except (AttributeError, KeyError, IndexError):
        pass  # Ignore if there is no EXIF data

    return img

st.title('Seeing Beyond Clarity: Utilizing Blur Vision in Object Detection System')

option = st.selectbox('Select an option:', 
                      ('Local System - Image', 'Web Address - Image', 'Local System - Video'))

if option == 'Local System - Image':
    uploaded_file = st.file_uploader("Upload Image", type=['png', 'jpg', 'jpeg'])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image = fix_orientation(image)  # Fix orientation

        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        img_path = f"temp_image.{uploaded_file.type.split('/')[1]}"
        image.save(img_path)
        results = model(img_path)
        st.image(np.squeeze(results.render()), use_column_width=True)

elif option == 'Web Address - Image':
    img_url = st.text_input('Enter Image URL')
    if img_url:
        img_path, _ = urlretrieve(img_url, 'downloaded_image.png')
        image = Image.open(img_path)
        image = fix_orientation(image)  # Fix orientation

        st.image(image, caption='Downloaded Image', use_column_width=True)
        
        results = model(img_path)
        st.image(np.squeeze(results.render()), use_column_width=True)

elif option == 'Local System - Video':
    uploaded_file = st.file_uploader("Upload Video", type=['mp4'])
    if uploaded_file is not None:
        video_path = f"temp_video.{uploaded_file.type.split('/')[1]}"
        with open(video_path, 'wb') as f:
            f.write(uploaded_file.read())
        
        cap = cv2.VideoCapture(video_path)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame)
            st.image(np.squeeze(results.render()), use_column_width=True, channels='BGR')
        
        cap.release()

# Styling for background image and footer
page_bg_img = '''
<style>
[data-testid="stAppViewContainer"] {
    background-image: url("https://raw.githubusercontent.com/vipulmalyan/Cloud-Wallpapers/main/Obj%20Det.jpg");
    background-size: cover;
}
</style>
'''
st.markdown(page_bg_img, unsafe_allow_html=True)

hide_st_style = '''
<style> footer {visibility: hidden;} </style>
'''
st.markdown(hide_st_style, unsafe_allow_html=True)

footer_html = """
<style>
.footer {
    position: fixed;
    bottom: 0;
    width: 100%;
    color: white; /* Text color */
    padding: 10px;
    text-align: center; /* Center the text */
    font-size: 18px; /* Adjust the font size */
}
</style>
<div class="footer">Made by Vipul Malyan</div>
"""
st.markdown(footer_html, unsafe_allow_html=True)
