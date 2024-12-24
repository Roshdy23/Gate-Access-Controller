import os
import cv2
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from HOG import HOG


class GateAccessController:
    def __init__(self, dataset_path=r'images\data-set', model_path='rf_model.pkl'):
        self.dataset_path = dataset_path
        self.model_path = model_path
        self.label_encoder_path = 'label_encoder.pkl'
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)  # Using Random Forest
        self.hog = HOG()
        self.label_encoder = LabelEncoder()
        self.load_model()  # Load model if it exists

    def extract_hog_features(self, img):
        return self.hog.compute_hog_features(img)

    def get_features_labels(self):
        features = []
        labels = []

        # List all files in the dataset directory
        for img_file in os.listdir(self.dataset_path):
            img_path = os.path.join(self.dataset_path, img_file)
            print("Reading image:", img_path)  # Debugging print statement

            # Read the image
            img = cv2.imread(img_path)
            if img is None:
                print(f"Failed to read image: {img_path}")  # Debugging print statement
                continue

            # Convert to grayscale if needed
            if len(img.shape) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Resize the image
            img_resized = cv2.resize(img, (60, 60))

            # Compute HOG features
            hog_features = self.extract_hog_features(img_resized)
            features.append(hog_features)

            # Extract label from filename
            label = img_file.split('.')[0].split('_')[0]
            labels.append(label)

        # Encode labels to integers
        labels = self.label_encoder.fit_transform(labels)

        print("Number of features:", len(features))
        print("Number of labels:", len(labels))
        return np.array(features), np.array(labels)

    def train_model(self):
        features, labels = self.get_features_labels()

        # Split data into training and testing sets
        train_features, test_features, train_labels, test_labels = train_test_split(
            features, labels, test_size=0.2, random_state=42)

        # Train the model
        self.model.fit(train_features, train_labels)

        # Evaluate the model
        train_accuracy = self.model.score(train_features, train_labels)
        test_accuracy = self.model.score(test_features, test_labels)
        print('Training Accuracy:', train_accuracy)
        print('Testing Accuracy:', test_accuracy)

        # Save the model and encoder
        self.save_model()
        return self.model

    def save_model(self):
        joblib.dump(self.model, self.model_path)
        joblib.dump(self.label_encoder, self.label_encoder_path)

    def load_model(self):
        if os.path.exists(self.model_path) and os.path.exists(self.label_encoder_path):
            self.model = joblib.load(self.model_path)
            self.label_encoder = joblib.load(self.label_encoder_path)

    def predict_plate_text(self, img):
        if img is not None and len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        img_resized = cv2.resize(img, (60, 60))
        hog_features = self.extract_hog_features(img_resized)

        # Reshape features for prediction
        hog_features = hog_features.reshape(1, -1)

        # Predict class and confidence
        prediction_idx = self.model.predict(hog_features)[0]
        confidence = max(self.model.predict_proba(hog_features)[0])

        # Decode the prediction back to label
        predicted_label = self.label_encoder.inverse_transform([prediction_idx])[0]
        print('Prediction:', predicted_label)
        print('Confidence:', confidence)
        return predicted_label, confidence


# Example usage
controller = GateAccessController()

# Train the model if it's not already trained
if not os.path.exists('rf_model.pkl'):
    controller.train_model()

