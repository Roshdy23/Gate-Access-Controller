import cv2
import numpy as np
import argparse
from preProcessing import imagePreprocessing
from plateDetection import plateDetection
from buildDB import buildCharacterDB
from segmentPlate import segment_plate

features = []
labels = []

parser = argparse.ArgumentParser()
parser.add_argument("index", type=int)
args = parser.parse_args()

input_image_path = f"./images/plate{args.index}.jpg"

preprocessed_image, original_image = imagePreprocessing(input_image_path)
cv2.imshow("Original image", original_image)


#buildCharacterDB(features, labels)



license_plate = plateDetection(preprocessed_image, original_image)

if license_plate is not None:
    chars = segment_plate(license_plate)
    for i, char in enumerate(chars):
        cv2.imshow(f"Character {i}", char)
    cv2.imshow("Detected License Plate", license_plate)
else:
    print("No license plate detected.")

cv2.waitKey(0)
cv2.destroyAllWindows()
