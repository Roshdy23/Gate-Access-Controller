import cv2
import imutils
import numpy as np
from skimage import measure
from skimage.filters import threshold_local
from skimage.morphology import binary_closing, binary_dilation, binary_opening


def preprocess_plate(plate):
    value = cv2.split(cv2.cvtColor(plate, cv2.COLOR_BGR2HSV))[2]
    # Apply adaptive thresholding
    threshold = threshold_local(value, 29, offset=15, method="gaussian")
    thresh = (value > threshold).astype("uint8") * 255
    thresh = cv2.bitwise_not(thresh)
    thresh = imutils.resize(thresh, width=400)

    # Fill the gaps in each char
    thresh = binary_closing(thresh).astype("uint8") * 255
    thresh = binary_dilation(thresh).astype("uint8") * 255
    thresh = binary_dilation(thresh).astype("uint8") * 255
    thresh = binary_opening(thresh).astype("uint8") * 255

    return thresh


def add_black_margin(cropped_char, margin_size=10):
    height, width = cropped_char.shape[:2]
    new_height = height + 2 * margin_size
    new_width = width + 2 * margin_size

    margined_char = np.zeros((new_height, new_width), dtype="uint8")
    margined_char[margin_size:margin_size + height, margin_size:margin_size + width] = cropped_char

    return margined_char


def clean_noise(cropped_chars):
    answer = []

    for img in cropped_chars:
        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
       
        mask = np.zeros(img.shape, dtype="uint8")

        # Sort contours by area (number of white pixels)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        def is_near_edge(contour, img_shape, marginx=10, marginy=25):
            x, y, w, h = cv2.boundingRect(contour)
            return x < marginx or y < marginy or (x + w) > (img_shape[1] - marginx) or (y + h) > (
                        img_shape[0] - marginy)

        # Find the largest contour that is not near the edges
        largest_contour = None
        for contour in contours:
            if not is_near_edge(contour, img.shape):
                largest_contour = contour
                break

        if largest_contour is not None:
            cv2.drawContours(mask, [largest_contour], -1, (255, 255, 255), -1)
            largest_contour_y = cv2.boundingRect(largest_contour)[1]

            if len(contours) > 1:
                second_largest_contour = contours[1]
                second_largest_contour_y = cv2.boundingRect(second_largest_contour)[1]
                dif = abs(second_largest_contour_y - largest_contour_y)

                # if the second largest contour is very close to the largest contour (so it is almost the same char)
                if second_largest_contour_y > 25 and dif < 50:
                    cv2.drawContours(mask, [second_largest_contour], -1, (255, 255, 255), -1)
                    combined_contour_area = cv2.contourArea(largest_contour) + cv2.contourArea(second_largest_contour)
                else:
                    combined_contour_area = cv2.contourArea(largest_contour)
            else:
                combined_contour_area = cv2.contourArea(largest_contour)

            for cnt in contours[2:]:
                x, y, w, h = cv2.boundingRect(cnt)
                if (cv2.contourArea(cnt) > 0.1 * combined_contour_area and
                        abs(y - largest_contour_y) < 10 and
                        y < largest_contour_y and
                        h < 15):
                    cv2.drawContours(mask, [cnt], -1, (255, 255, 255), -1)

        cleaned_char = cv2.bitwise_and(img, mask)
        answer.append(cleaned_char)

    return answer


def segment_plate(plate):
    preprocessed_plate = preprocess_plate(plate)
    plate_have_apper_part = plate.shape[0] > 50
    labels = measure.label(preprocessed_plate, connectivity=2, background=0)
    cropped_chars = []

    for label in np.unique(labels):
        # Background label
        if label == 0:
            continue

        # Create a label mask for label
        label_mask = np.zeros(preprocessed_plate.shape, dtype="uint8")
        label_mask[labels == label] = 255

        contours, _ = cv2.findContours(label_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) > 0:
            max_contour = max(contours, key=cv2.contourArea)
            (x, y, w, h) = cv2.boundingRect(max_contour)

            if plate_have_apper_part and y < 255 / 4:
                continue

            # Compute aspect ratio, solidity, and height ratio
            aspect_ratio = w / float(h)
            solidity = cv2.contourArea(max_contour) / float(w * h)
            height_ratio = h / float(plate.shape[0])

            # Filtering rules
            allowed_aspect = aspect_ratio < 5.0
            allowed_solidity = solidity > 0.10
            allowed_height = 0.1 < height_ratio < 3

            # Adjust coordinates based on padding
            x -= 5
            y -= 40
            h += h + 20
            w = w + 2 * 5
            x = max(0, x)
            y = max(0, y)

            cropped_char = preprocessed_plate[y:y + h, x:x + w]
            if allowed_aspect and allowed_solidity and allowed_height:
                margined_char = add_black_margin(cropped_char)
                cropped_chars.append((margined_char, x, cv2.countNonZero(margined_char)))

    # Sort characters by x-axis position
    cropped_chars = sorted(cropped_chars, key=lambda x: x[1])

    # Filter overlapping characters
    filtered_chars = []
    min_spacing = 30
    for i, (char, x, count) in enumerate(cropped_chars):
        if i == 0:
            filtered_chars.append((char, x, count))
        else:
            prev_char, prev_x, prev_count = filtered_chars[-1]

            if x - prev_x > min_spacing:
                filtered_chars.append((char, x, count))
            elif count > prev_count:
                filtered_chars[-1] = (char, x, count)

    best_choice_chars = [char[0] for char in filtered_chars]

    final_chars = clean_noise(best_choice_chars)
    final_chars = [cv2.bitwise_not(char) for char in final_chars]
    return final_chars
