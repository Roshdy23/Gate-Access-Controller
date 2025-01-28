import streamlit as st
import cv2
import numpy as np
import easyocr
import pytesseract
import os
import tempfile

from PIL import Image
from deep_model import run_easy_OCR
from main import run_OCR

st.title("Egyptian License Plate Recognition")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

option = st.radio(
    "Select OCR Method:",
    ('OCR', 'EasyOCR')
)


def check_plate_access(plate_text, plates_file='plates.txt'):
    with open(plates_file, 'r') as file:
        plates = {line.strip() for line in file}

    return plate_text in plates


def process_image(image_path, method):
    text = ""
    if method == "OCR":
        text = run_OCR(image_path)
    elif method == "EasyOCR":
        text = run_easy_OCR(image_path)

    return check_plate_access(text), text


if uploaded_file is not None:
    temp_dir = "temp"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    temp_file_path = os.path.join(temp_dir, uploaded_file.name)

    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.write(f"**Image Path:** {temp_file_path}")

    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    match, detected_text = process_image(temp_file_path, option)

    st.write("**Detected Text:**", detected_text)

    if match:
        st.markdown('<div style="background-color:#00FF00; padding:10px; border-radius:5px;">✅ PASS</div>',
                    unsafe_allow_html=True)
    else:
        st.markdown('<div style="background-color:#FF0000; padding:10px; border-radius:5px;">❌ STOP</div>',
                    unsafe_allow_html=True)