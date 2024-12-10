import os
import cv2
import numpy as np
from skimage.feature import hog
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import joblib

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

plate_directory = "./plates"
non_plate_directory = "./non-plates"

plate_data, plate_labels = load_images_with_labels(plate_directory, label=1)
non_plate_data, non_plate_labels = load_images_with_labels(non_plate_directory, label=0)

data = np.vstack((plate_data, non_plate_data))
labels = np.hstack((plate_labels, non_plate_labels))

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(data, labels)

joblib.dump(knn, 'knn_model.pkl')
print("KNN model trained and saved as knn_model.pkl")

# Analyze distances
# all_distances, _ = knn.kneighbors(data)

# plt.hist(all_distances.flatten(), bins=30, color='blue', alpha=0.7)
# plt.title("Distance Distribution of Training Data")
# plt.xlabel("Distance")
# plt.ylabel("Frequency")
# plt.show()

# Example Test Image
# Uncomment and adjust paths to test the model
# new_image_path = './testImages/test1.jpg'  
# new_image = cv2.imread(new_image_path)
# new_image_resized = cv2.resize(new_image, (128, 64))
# new_features = extract_hog_features(new_image_resized)
# distances, _ = knn.kneighbors([new_features])
# threshold = 2.9
# if distances[0][0] < threshold:
#     print("The image is a plate.")
# else:
#     print("The image is not a plate.")


def predict_image(image_path):
    knn = joblib.load("knn_model.pkl")
    
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Could not read the image.")
        return None

    image_resized = cv2.resize(image, (128, 64))
    features = extract_hog_features(image_resized)
    
    prediction = knn.predict([features])[0]
    return prediction

test_image_path = './testImages/test2.jpg'  
result = predict_image(test_image_path)

if result is not None:
    if result == 1:
        print("The image is a plate.")
    else:
        print("The image is not a plate.")
