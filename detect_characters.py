import os
import cv2
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog

class GateAccessController:
    def __init__(self, dataset_path=r'C:\Users\MOSTAFA\Desktop\data'):
        self.dataset_path = dataset_path
        self.model = svm.SVC(probability=True, random_state=42)  # Use SVM
        self.scaler = StandardScaler()

    charWidth = 60
    charHeight = 60

    def extract_hog_features(self, img):
        # Compute HOG features
        features, _ = hog(img, orientations=9, pixels_per_cell=(8, 8),
                          cells_per_block=(2, 2), block_norm='L2-Hys', visualize=True)
        return features

    def get_features_labels(self):
        features = []
        labels = []

        # List all subdirectories (each is a label)
        for label in os.listdir(self.dataset_path):
            label_path = os.path.join(self.dataset_path, label)

            # Ensure it's a directory
            if os.path.isdir(label_path):
                # List all image files in the subdirectory
                for img_file in os.listdir(label_path):
                    img_path = os.path.join(label_path, img_file)
                    print("Reading image:", img_path)  # Debugging print statement

                    # Read the image
                    img = cv2.imread(img_path)
                    if img is None:
                        print(f"Failed to read image: {img_path}")  # Debugging print statement
                        continue
                    if len(img.shape) == 3:
                        # Convert to grayscale
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                    # Resize the image
                    img_resized = cv2.resize(img, (60, 60))

                    # Compute HOG features
                    hog_features = self.extract_hog_features(img_resized)
                    features.append(hog_features)
                    # Append the label
                    labels.append(label)

        # Scale the features
        features = self.scaler.fit_transform(features)

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
        hog_features = self.scaler.transform([hog_features])
        prediction = self.model.predict(hog_features)[0]
        confidence = self.model.predict_proba(hog_features)[0].max()
        print('Prediction:', prediction)
        print('Confidence:', confidence)
        return prediction, confidence

# controller = GateAccessController()
# controller.train_model()
# controller.predict_plate_text(cv2.imread('images/un_used_characters/alf_1.jpg'))