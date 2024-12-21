import cv2
import imutils
import numpy as np
from skimage import measure
from skimage.filters import threshold_local
from skimage.morphology import binary_closing, binary_dilation

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
    thresh = binary_dilation(thresh).astype("uint8") * 255
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
    plate_have_apper_part = plate.shape[0] > 50
    labels = measure.label(preprocessed_plate, connectivity=2, background=0)
    cropped_chars = []

    i = 0

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

            i += 1
            cropped_char = preprocessed_plate[y:y + h, x:x + w]
            # cv2.imshow(f"Cropped Char{i}", cropped_char)
            if(plate_have_apper_part and y < 255 /4):
                continue

            # Compute the aspect ratio, solidity, and height ratio for the component
            aspectRatio = w / float(h)
            solidity = cv2.contourArea(max_contour) / float(w * h)
            heightRatio = h / float(plate.shape[0])

            # Apply filtering rules
            allowed_aspect = aspectRatio < 5.0
            allowed_solidity = solidity > 0.10
            allowed_height = heightRatio > 0.1 and heightRatio < 3

            # # Adjust coordinates based on padding
            x -= 5
            y -= 40
            h += h + 20
            w = w + 2 * 5
            x = max(0, x)
            y = max(0, y)

            cropped_char = preprocessed_plate[y:y + h, x:x + w]
            # cv2.imshow(f"Cropped Char{x}", cropped_char)
            print(i, aspectRatio,solidity,heightRatio,allowed_aspect,allowed_solidity,allowed_height)
            if allowed_aspect and allowed_solidity and allowed_height:
                margined_char = add_black_margin(cropped_char)
                margined_char = cv2.bitwise_not(margined_char)
                cropped_chars.append((margined_char, x, cv2.countNonZero(margined_char)))

    # Sort characters by x-axis position
    cropped_chars = sorted(cropped_chars, key=lambda x: x[1])

    # Filter out overlapping or close characters
    filtered_chars = []
    min_spacing = 30  # Minimum spacing threshold between characters
    for i, (char, x, count) in enumerate(cropped_chars):
        if i == 0:
            filtered_chars.append((char, x, count))
        else:
            prev_char, prev_x, prev_count = filtered_chars[-1]

            if x - prev_x > min_spacing:
                filtered_chars.append((char, x, count))
            elif count > prev_count:  # Replace if current char has more information
                filtered_chars[-1] = (char, x, count)

    return [char[0] for char in filtered_chars]
