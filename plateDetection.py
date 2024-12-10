import cv2
import numpy as np
from skimage.feature import hog
from sklearn.neighbors import KNeighborsClassifier
import joblib

knn_model = joblib.load("./models/platesDetectionModel/knn_model.pkl")

def extract_hog_features(image):
    features, _ = hog(image, block_norm='L2-Hys', pixels_per_cell=(16, 16), cells_per_block=(2, 2), visualize=True)
    return features

def predict_plate(plate):
    plate_resized = cv2.resize(plate, (128, 64))  
    gray = cv2.cvtColor(plate_resized, cv2.COLOR_BGR2GRAY)
    features = extract_hog_features(gray) 
    prediction = knn_model.predict([features])[0]
    return prediction
    #distances, _ = knn_model.kneighbors([features])
    # 1 <=threshold <= 2.95
    #print(distances[0][0])
    # if 1 <= distances[0][0] <= 3:
    #     return True
    # else:
    #     return False

def getPlate(plates):
    plate_detected = None
    for plate in plates:
        #cv2.imshow("plate", plate)
        if predict_plate(plate) == 1:  
            plate_detected = plate
            break
    return plate_detected

def plateDetection(preprocessed_image, original_image, area_threshold=2000):
    edges = cv2.Canny(preprocessed_image, 120, 200)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 2))
    dilated_edges = cv2.dilate(edges, kernel)
    cv2.imshow("plate", dilated_edges)

    contours, _ = cv2.findContours(dilated_edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    detected_plates = []
    #i = 0
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        aspect_ratio = w / h

        if area > area_threshold and 2 < aspect_ratio < 6:
            #cv2.imshow(f"plate {i}", original_image[y:y+h, x:x+w])
            detected_plates.append(original_image[y:y+h, x:x+w])
            #i = i + 1
    
    detected_plate = getPlate(detected_plates)
    

    return detected_plate
