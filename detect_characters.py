import os
import cv2
from HOG import HOG
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split  # Add this import

class GateAccessController:
    def __init__(self, dataset_path='images/data-set'):
        self.dataset_path = dataset_path
        self.model = None

    def get_features_labels(self):
        features = []
        labels = []
        img_filenames = os.listdir(self.dataset_path)

        for i, fn in enumerate(img_filenames):
            if fn.split('.')[-1] != 'jpg':
                continue

            label = fn.split('.')[0]
            label = label.split('_')[0]
            labels.append(label)

            path = os.path.join(self.dataset_path, fn)
            img = cv2.imread(path)
            if img is not None and len(img.shape) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_resized = cv2.resize(img, (60, 60)) 
            hog = HOG()
            features.append(hog.compute_hog_features(img_resized))
        
        return features, labels      

    def train_model(self):
        features, labels = self.get_features_labels()
        svm_model = svm.LinearSVC(random_state=42)
        train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.2, random_state=42)
        self.model = svm_model.fit(train_features, train_labels)
        accuracy = self.model.score(test_features, test_labels)
        accuracy2 = self.model.score(train_features, train_labels)
        print('Accuracy:', accuracy)
        print('Accuracy:', accuracy2)
        return self.model   

    def predict_plate_text(self, img):
        if img is not None and len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_resized = cv2.resize(img, (60, 60)) 
        hog = HOG()
        features = hog.compute_hog_features(img_resized) 
        prediction = self.model.predict([features])[0]
        print('Prediction:', prediction)
        return prediction

controller = GateAccessController()
controller.train_model()
controller.predict_plate_text(cv2.imread('images/un_used_characters/alf_1.jpg'))