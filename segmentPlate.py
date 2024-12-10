import cv2
import imutils
import numpy as np
from skimage import measure
from skimage.filters import threshold_local
from skimage.morphology import binary_closing


def preprocess_plate(plate):
    # Convert to HSV, extract the value channel
    V = cv2.split(cv2.cvtColor(plate, cv2.COLOR_BGR2HSV))[2]
    # Apply adaptive thresholding
    T = threshold_local(V, 29, offset=15, method="gaussian")
    thresh = (V > T).astype("uint8") * 255
    thresh = cv2.bitwise_not(thresh)

    # Resize the plate to a canonical size
    plate = imutils.resize(plate, width=400)
    thresh = imutils.resize(thresh, width=400)

    # to fill the gaps in each char
    thresh = binary_closing(thresh).astype("uint8") * 255
    return thresh

def add_black_margin(cropped_char, margin_size=10):

    height, width = cropped_char.shape[:2]

    new_height = height + 2 * margin_size
    new_width = width + 2 * margin_size

    margined_char = np.zeros((new_height, new_width), dtype="uint8")

    margined_char[margin_size:margin_size+height, margin_size:margin_size+width] = cropped_char

    return margined_char

def segment_plate(plate):

    preprocessed_plate = preprocess_plate(plate)
    
    labels = measure.label(preprocessed_plate, connectivity=2, background=0)
    cropped_chars = []

    for label in np.unique(labels):
        # background label
        if label == 0:
            continue

        # create a label mask for label
        labelMask = np.zeros(preprocessed_plate.shape, dtype="uint8")
        labelMask[labels == label] = 255


        contours, _ = cv2.findContours(labelMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) > 0:

            max_contour = max(contours, key=cv2.contourArea)
            (x, y, w, h) = cv2.boundingRect(max_contour)

            # Compute the aspect ratio, solidity, and height ratio for the component
            aspectRatio = w / float(h)
            solidity = cv2.contourArea(max_contour) / float(w * h)
            heightRatio = h / float(plate.shape[0])

            # Apply filtering rules
            allowed_aspect = aspectRatio < 1.0
            allowed_solidity = solidity > 0.15
            allowed_height = heightRatio > 0.4 and heightRatio < 0.95


            if allowed_aspect and allowed_solidity and allowed_height:
                cropped_char = preprocessed_plate[y:y + h, x:x + w]
                margined_char = add_black_margin(cropped_char)
                margined_char = cv2.bitwise_not(margined_char)
                cropped_chars.append(margined_char)

    return cropped_chars
