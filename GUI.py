import PySimpleGUI as GUI
from PIL import Image
import cv2
import numpy as np
from main import  main
def extract_license_plate(image_path):
    return main(image_path)

def is_plate_allowed(plate_image):
    return True  # Replace with actual logic

layout = [
    [GUI.Text("License Plate Recognition App")],
    [GUI.Image(key="-IMAGE-")],
    [GUI.Button("Select Image")],
    [GUI.Canvas(size=(50, 50), key="-LED-")]
]

window = GUI.Window("License Plate Recognition", layout)

while True:
    event, values = window.read()
    if event == GUI.WIN_CLOSED:
        break
    if event == "Select Image":
        file_path = GUI.popup_get_file("Select an image file")
        if not file_path:
            continue
        plate_image = extract_license_plate(file_path)
        if plate_image is None:
            GUI.popup_error("No license plate found in the image.")
            continue

        led_canvas = window["-LED-"].TKCanvas
        led_canvas.delete("all")
        if is_plate_allowed(plate_image):
            led_canvas.create_oval(10, 10, 40, 40, fill="green")
        else:
            led_canvas.create_oval(10, 10, 40, 40, fill="red")

window.close()
