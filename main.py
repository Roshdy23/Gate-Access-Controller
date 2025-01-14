import cv2
import numpy as np
import argparse
from preProcessing import imagePreprocessing
from plateDetection import plateDetection
from buildDB import buildCharacterDB
from detect_characters import GateAccessController
from segmentPlate import segment_plate

features = []
labels = []
controller = GateAccessController()
# UNCOMMENT THIS TO TRAIN THE MODEL
# controller.train_model()

def run_OCR(input_image_path):

    preprocessed_image, original_image = imagePreprocessing(input_image_path)
    # cv2.imshow("Original image", original_image)

    #buildCharacterDB(features, labels)

    license_plate = plateDetection(preprocessed_image, original_image)
    plateStr = ""

    if license_plate is not None:
        chars = segment_plate(license_plate)
        for i, char in enumerate(chars):
            cv2.imshow(f"Character {i}", char)
            prediction = controller.predict_plate_text(char)
            print(f"Character {i}: {prediction}")

            plateStr += prediction[0]
            if i != len(chars) - 1:
                plateStr += "-"

        # cv2.imshow("Detected License Plate", license_plate)
    else:
        print("No license plate detected.")

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return plateStr