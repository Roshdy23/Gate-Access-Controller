import os
import cv2
import numpy as np
import joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from HOG2 import HOG2

hog = HOG2()

def extract_hog_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    features = hog.compute_hog_features(gray)
    return features

def load_images_with_labels(directory_path, label):
    data = []
    labels = []
    image_paths = [
        os.path.join(directory_path, filename)
        for filename in os.listdir(directory_path)
        if filename.endswith('.jpg') or filename.endswith('.png')
    ]

    for img_path in image_paths:
        image = cv2.imread(img_path)
        if image is not None:
            image_resized = cv2.resize(image, (128, 64))
            features = extract_hog_features(image_resized)
            data.append(features)
            labels.append(label)

    return np.array(data), np.array(labels)

plate_directory = "./egyplates"
non_plate_directory = "./non-plates"

plate_data, plate_labels = load_images_with_labels(plate_directory, label=1)
non_plate_data, non_plate_labels = load_images_with_labels(non_plate_directory, label=0)

data = np.vstack((plate_data, non_plate_data))
labels = np.hstack((plate_labels, non_plate_labels))

pca = PCA(n_components=2)
data_2d = pca.fit_transform(data)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(data_2d, labels)

joblib.dump(knn, 'knn_model.pkl')
joblib.dump(pca, 'pca_model.pkl')
print("KNN model and PCA model saved.")

# def predict_image(image_path):
#     knn = joblib.load("knn_model.pkl")
#     pca = joblib.load("pca_model.pkl")
    
#     image = cv2.imread(image_path)
#     if image is None:
#         print("Error: Could not read the image.")
#         return None

#     image_resized = cv2.resize(image, (128, 64))
#     features = extract_hog_features(image_resized)
    
#     pca_transformed_features = pca.transform([features])
    
#     prediction = knn.predict(pca_transformed_features)[0]
#     return features, prediction

test_plates_data, test_plates_labels = load_images_with_labels("./testImages/plates", label=1)
test_nonplates_data, test_nonplates_labels = load_images_with_labels("./testImages/nonplates", label=0)

test_data = np.vstack((test_plates_data, test_nonplates_data))
test_labels = np.hstack((test_plates_labels, test_nonplates_labels))

test_data_2d = pca.transform(test_data)

test_predictions = knn.predict(test_data_2d)

accuracy = accuracy_score(test_labels, test_predictions)
print(f"Test Accuracy: {accuracy * 100:.2f}%")


plt.figure(figsize=(10, 8))

x_min, x_max = data_2d[:, 0].min() - 1, data_2d[:, 0].max() + 1
y_min, y_max = data_2d[:, 1].min() - 1, data_2d[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])  
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.coolwarm)

colors = ['blue' if label == 0 else 'red' for label in labels]
plt.scatter(data_2d[:, 0], data_2d[:, 1], c=colors, edgecolor='k', s=50)


plt.title("KNN Decision Regions for Plates and Non-Plates")
plt.xlabel("PCA Feature 1")
plt.ylabel("PCA Feature 2")
plt.colorbar()
plt.show()
