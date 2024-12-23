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

st.title("License Plate Recognition")

# Step 1: Allow the user to upload an image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

# Step 2: User selects OCR method
option = st.radio(
    "Select OCR Method:",
    ('OCR', 'EasyOCR')
)

def check_plate_access(plateStr, plates_file='plates.txt'):
    with open(plates_file, 'r') as file:
        plates = file.readlines()

    plate_found = False
    for plate in plates:
        if plateStr in plate:
            plate_found = True
            break
    return plate_found

# Step 3: Check the Output
def process_image(imagePath, method):
    match = True
    # Use selected OCR method
    if method == "OCR":
        text = run_OCR(imagePath);
        match = check_plate_access(text)
    elif method == "EasyOCR":
        text = run_easy_OCR(imagePath)
        match = check_plate_access(text)

    return match, text

# Check if file is uploaded and process it
if uploaded_file is not None:
    # Ensure the 'temp' directory exists
    temp_dir = "temp"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    # Save the uploaded file to a temporary location
    temp_file_path = os.path.join(temp_dir, uploaded_file.name)

    # Save the file locally
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Display the image path to the user
    st.write(f"**Image Path:** {temp_file_path}")

    # Step 4: Read and Display Uploaded Image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Process Image
    match, detected_text = process_image(temp_file_path, option)

    # Step 5: Display Result
    st.write("**Detected Text:**", detected_text)

    # Step 6: Highlight the Result
    if match:
        st.markdown('<div style="background-color:#D4EDDA; padding:10px; border-radius:5px;">✅ PASS</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div style="background-color:#F8D7DA; padding:10px; border-radius:5px;">❌ STOP</div>', unsafe_allow_html=True)
