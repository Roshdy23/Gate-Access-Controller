import os
import cv2
import numpy as np
from skimage.feature import hog
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import joblib
from sklearn.decomposition import PCA


def extract_hog_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    features, _ = hog(
        gray,
        block_norm='L2-Hys',
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        visualize=True
    )
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

knn = KNeighborsClassifier(n_neighbors=2)
knn.fit(data, labels)

joblib.dump(knn, 'knn_model.pkl')
print("KNN model trained and saved as knn_model.pkl")

pca = PCA(n_components=2)
data_2d = pca.fit_transform(data)

x_min, x_max = data_2d[:, 0].min() - 1, data_2d[:, 0].max() + 1
y_min, y_max = data_2d[:, 1].min() - 1, data_2d[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))

Z = knn.predict(pca.inverse_transform(np.c_[xx.ravel(), yy.ravel()]))
Z = Z.reshape(xx.shape)

plt.figure(figsize=(10, 8))
plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.coolwarm)
plt.scatter(data_2d[:, 0], data_2d[:, 1], c=labels, edgecolor='k', cmap=plt.cm.coolwarm, s=50)
plt.title("KNN Decision Regions for Plates and Non-Plates")
plt.xlabel("PCA Feature 1")
plt.ylabel("PCA Feature 2")
plt.colorbar()

def predict_image(image_path):
    knn = joblib.load("knn_model.pkl")
    
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Could not read the image.")
        return None

    image_resized = cv2.resize(image, (128, 64))
    features = extract_hog_features(image_resized)
    
    prediction = knn.predict([features])[0]
    return features, prediction

test_image_path = './testImages/test2.jpg'
features, result = predict_image(test_image_path)

pca_transformed_test_image = pca.transform([features])

plt.scatter(pca_transformed_test_image[0][0], pca_transformed_test_image[0][1], color='red', s=100, label="Test Image")

if result == 1:
    print("The image is a plate.")
else:
    print("The image is not a plate.")

plt.legend()
plt.show()
