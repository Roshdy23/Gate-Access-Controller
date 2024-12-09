import os
import cv2
import numpy as np
from skimage.feature import hog
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import joblib

def extract_hog_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    features, _ = hog(gray, block_norm='L2-Hys', pixels_per_cell=(16, 16), cells_per_block=(2, 2), visualize=True)
    return features

def load_plate_images(directory_path):
    data = []
    image_paths = [os.path.join(directory_path, filename) for filename in os.listdir(directory_path) if filename.endswith('.jpg') or filename.endswith('.png')]
    
    for img_path in image_paths:
        image = cv2.imread(img_path)
        
        if image is not None:
            image_resized = cv2.resize(image, (128, 64)) 
            features = extract_hog_features(image_resized)
            data.append(features)
    
    return np.array(data)

plate_directory = "./Training data for plate detection"

plate_data = load_plate_images(plate_directory)

knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(plate_data, np.ones(len(plate_data)))

joblib.dump(knn, 'knn_model.pkl')
print("KNN model saved as knn_model.pkl")

# new_image_path = './testImages/plate1.jpg'  
# new_image = cv2.imread(new_image_path)

# new_image_resized = cv2.resize(new_image, (128, 64))

# new_features = extract_hog_features(new_image_resized)

# distances, _ = knn.kneighbors([new_features])

# for each image in the training data according to its hog and the other images hog we calculate the distance to get how much its hog is far from other hogs, histogram represents how much a specific distance occured, so in a given interval if an image in this range then it has a high probabilty to be a plate
all_distances, _ = knn.kneighbors(plate_data)

plt.hist(all_distances.flatten(), bins=30, color='blue', alpha=0.7)
#plt.axvline(distances[0][0], color='red', linestyle='dashed', linewidth=2, label=f'Test image distance: {distances[0][0]:.2f}')
plt.title("Distance Distribution of Training Data")
plt.xlabel("Distance")
plt.ylabel("Frequency")
plt.legend()
plt.show()

# threshold = 2.9 

# if distances[0][0] < threshold:
#     print("The image is a plate.")
# else:
#     print("The image is not a plate.")
