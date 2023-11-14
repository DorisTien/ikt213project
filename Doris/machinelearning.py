#traditional method

#This code is based on traditional computer vision techniques using Local Binary Pattern (LBP) 
#features and a Support Vector Machine (SVM) classifier.

import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Step 2: Data Preprocessing and Feature Extraction
def preprocess_image(image_path):
    img = cv2.imread(image_path, 0)  # Read image in grayscale
    img = cv2.resize(img, (100, 100))  # Resize the image
    lbp_image = local_binary_pattern(img, 8, 1, method='uniform')  # Calculate LBP features
    return lbp_image.flatten()

# Step 4: Data Collection and Labeling
# Example image paths and corresponding labels
image_paths = ['C:/Users/doris/Downloads/plastics/1.jpeg', 'C:/Users/doris/Downloads/plastics/2.png', 'C:/Users/doris/Downloads/plastics/n1.webp']  # Example image paths
labels = [1, 1, 0]  # Labels: 1 for plastic images, 0 for non-plastic images

# Step 3, 5, 6: Feature Extraction and Model Training
X = [preprocess_image(img_path) for img_path in image_paths]
y = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the SVM classifier
clf = SVC()
clf.fit(X_train, y_train)

# Step 7: Model Evaluation
y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Step 8: Deployment
# Use the trained model to predict new images
new_image_path = 'C:/Users/doris/Downloads/plastics/3.jpeg'  # Example path to a new image for prediction
new_image_features = preprocess_image(new_image_path)
predicted_label = clf.predict([new_image_features])
if predicted_label == 1:
    print("The image contains plastic.")
else:
    print("The image does not contain plastic.")
