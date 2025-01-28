import streamlit as st
from deep_model import run_easy_OCR

st.title("Egyptian License Plate Recognition - Grant/Remove Access")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])


def add_plate_to_allowlist(plate_text, plates_file='plates.txt'):
    with open(plates_file, 'a') as file:
        file.write(f"{plate_text}\n")


def remove_plate_from_allowlist(plate_text, plates_file='plates.txt'):
    with open(plates_file, 'r') as file:
        plates = file.readlines()

    with open(plates_file, 'w') as file:
        for plate in plates:
            if plate.strip() != plate_text:
                file.write(plate)


if uploaded_file is not None:
    # Process image
    temp_file_path = f"temp/{uploaded_file.name}"
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    detected_text = run_easy_OCR(temp_file_path)

    st.write("**Detected Text:**", detected_text)

    action = st.radio("Grant or Remove Access:", ("Grant Access", "Remove Access"))

    if action == "Grant Access":
        if st.button("Add Plate to Allowlist"):
            add_plate_to_allowlist(detected_text)
            st.success(f"Plate {detected_text} added to allowlist.")

    elif action == "Remove Access":
        if st.button("Remove Plate from Allowlist"):
            remove_plate_from_allowlist(detected_text)
            st.success(f"Plate {detected_text} removed from allowlist.")
