import cv2
import numpy as np

def plateDetection(preprocessed_image, area_threshold=3500):
    edges = cv2.Canny(preprocessed_image, 120, 200)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 2))
    dilated_edges = cv2.dilate(edges, kernel)

    #cv2.imshow("edges", dilated_edges)

    contours, _ = cv2.findContours(dilated_edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    detected_plate = None
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        aspect_ratio = w / h

        if area > area_threshold and 2 < aspect_ratio < 6:
            detected_plate = dilated_edges[y:y+h, x:x+w]
            #cv2.rectangle(dilated_edges, (x, y), (x+w, y+h), (0, 255, 0), 2)
            break  

    return detected_plate