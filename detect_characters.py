import os
import cv2
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

class GateAccessController:
    def __init__(self, dataset_path=r'/home/youssef-roshdy/Public/IP/Gate Access Controller/Gate-Access-Controller/images/data-set/data', n_clusters=100):
        self.dataset_path = dataset_path
        self.model = KNeighborsClassifier(n_neighbors=10, p=2, metric='euclidean')
        self.sift = cv2.SIFT_create()  # Initialize SIFT
        self.n_clusters = n_clusters
        self.kmeans = None
        self.scaler = StandardScaler()

    charWidth = 60
    charHeight = 60

    def extract_sift_descriptors(self):
        descriptors_list = []
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

                    # Compute SIFT features
                    keypoints, descriptors = self.sift.detectAndCompute(img_resized, None)
                    if descriptors is not None:
                        descriptors_list.append(descriptors)
                        # Append the label
                        labels.append(label)

        return descriptors_list, labels

    def create_bovw_model(self, descriptors_list):
        # Stack all descriptors vertically in a numpy array
        all_descriptors = np.vstack(descriptors_list)

        # Perform k-means clustering to create the visual vocabulary
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        self.kmeans.fit(all_descriptors)

    def get_bovw_features(self, descriptors):
        # Predict the cluster for each descriptor
        words = self.kmeans.predict(descriptors)

        # Create a histogram of visual words
        histogram, _ = np.histogram(words, bins=np.arange(self.n_clusters + 1), density=True)

        return histogram

    def get_features_labels(self):
        descriptors_list, labels = self.extract_sift_descriptors()

        # Create the BoVW model
        self.create_bovw_model(descriptors_list)

        features = []
        for descriptors in descriptors_list:
            bovw_features = self.get_bovw_features(descriptors)
            features.append(bovw_features)

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
        keypoints, descriptors = self.sift.detectAndCompute(img_resized, None)
        if descriptors is not None:
            bovw_features = self.get_bovw_features(descriptors)
            bovw_features = self.scaler.transform([bovw_features])
            prediction = self.model.predict(bovw_features)[0]
            confidence = self.model.predict_proba(bovw_features)[0].max()
            print('Prediction:', prediction)
            print('Confidence:', confidence)
            return prediction, confidence
        else:
            print('No features detected')
            return None, None

# controller = GateAccessController()
# controller.train_model()
# controller.predict_plate_text(cv2.imread('images/un_used_characters/alf_1.jpg'))