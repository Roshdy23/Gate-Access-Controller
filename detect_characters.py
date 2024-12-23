import os
import cv2
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
# from skimage.feature import hog
from HOG import HOG
class GateAccessController:
    def __init__(self, dataset_path=r'images\data-set'):
        self.dataset_path = dataset_path
        self.model = svm.SVC(probability=True, random_state=42) 
        self.hog = HOG()


    def extract_hog_features(self, img):
        return self.hog.compute_hog_features(img)

    def get_features_labels(self):
        features = []
        labels = []

        # List all subdirectories (each is a label)
        for img_file in os.listdir(self.dataset_path):
            img_path = os.path.join(self.dataset_path, img_file)
            print("Reading image:", img_path)  # Debugging print statement

            # Read the image
            img = cv2.imread(img_path)
            if img is None:
                print(f"Failed to read image: {img_path}")  # Debugging print statement
                continue
            
            if img is not None and len(img.shape) == 3:
                # Convert to grayscale
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Resize the image
            img_resized = cv2.resize(img, (60, 60))

            # Compute HOG features
            hog_features = self.extract_hog_features(img_resized)
            features.append(hog_features)
            
            # Extract label from filename
            label = img_file.split('.')[0].split('_')[0]
            # Append the label
            labels.append(label)

        print("Number of features:", len(features))  # Debugging print statement
        print("Number of labels:", len(labels))  # Debugging print statement
        return features, labels

    def train_model(self):
        features, labels = self.get_features_labels()
        train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.2, random_state=42)
        self.model.fit(train_features, train_labels)  # Fit the model
        train_accuracy = self.model.score(train_features, train_labels)
        test_accuracy = self.model.score(test_features, test_labels)
        print('Training Accuracy:', train_accuracy)
        print('Testing Accuracy:', test_accuracy)
        return self.model   

    def predict_plate_text(self, img):
        if img is not None and len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        img_resized = cv2.resize(img, (60, 60)) 
        hog_features = self.extract_hog_features(img_resized)
        # Reshape hog_features to 2D array
        hog_features = hog_features.reshape(1, -1)
        prediction = self.model.predict(hog_features)[0]
        confidence = self.model.predict_proba(hog_features)[0].max()
        print('Prediction:', prediction)
        print('Confidence:', confidence)
        return prediction, confidence

# controller = GateAccessController()
# controller.train_model()
# controller.predict_plate_text(cv2.imread('images/un_used_characters/alf_1.jpg'))