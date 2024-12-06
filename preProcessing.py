import cv2
import numpy as np

def imagePreprocessing(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    
    original_height, original_width = image.shape[:2]
    resize_factor = 640 / original_width
    width = 640
    height = int(original_height * resize_factor)
    resized_image = cv2.resize(image, (width, height))
    
    gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

    equalized_image = cv2.equalizeHist(blurred_image)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    morph_image = cv2.morphologyEx(equalized_image, cv2.MORPH_OPEN, kernel)

    return morph_image, resized_image

