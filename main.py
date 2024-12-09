import cv2
import numpy as np
import argparse
from preProcessing import imagePreprocessing
from plateDetection import plateDetection
from buildDB import buildCharacterDB

features =[]
labels=[]

parser = argparse.ArgumentParser()
parser.add_argument("index", type=int)
args = parser.parse_args()

input_image_path = f"./images/plate{args.index}.jpg"

preprocessed_image, original_image = imagePreprocessing(input_image_path)
cv2.imshow("Original image", original_image)

buildCharacterDB(features,labels)
print(features)
print(labels)



#cv2.imshow("Preprocessed License Plate", preprocessed_image)

license_plate = plateDetection(preprocessed_image, original_image)


if license_plate is not None:
    cv2.imshow("Detected License Plate", license_plate)
else:
    print("No license plate detected.")

cv2.waitKey(0)
cv2.destroyAllWindows()
