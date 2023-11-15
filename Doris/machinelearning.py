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

image_paths = [#plastic online
    "C:\\Users\\doris\\Downloads\\Assets\\Training\\Plastic\\plastic334.jpg",
    "C:\\Users\\doris\\Downloads\\Assets\\Training\\Plastic\\plastic336.jpg",
    "C:\\Users\\doris\\Downloads\\Assets\\Training\\Plastic\\plastic339.jpg",
    "C:\\Users\\doris\\Downloads\\Assets\\Training\\Plastic\\plastic341.jpg",
    "C:\\Users\\doris\\Downloads\\Assets\\Training\\Plastic\\plastic343.jpg",
    "C:\\Users\\doris\\Downloads\\Assets\\Training\\Plastic\\plastic350.jpg",
    "C:\\Users\\doris\\Downloads\\Assets\\Training\\Plastic\\plastic370.jpg",
    "C:\\Users\\doris\\Downloads\\Assets\\Training\\Plastic\\plastic371.jpg",
    "C:\\Users\\doris\\Downloads\\Assets\\Training\\Plastic\\plastic372.jpg",
    "C:\\Users\\doris\\Downloads\\Assets\\Training\\Plastic\\plastic373.jpg",
    # NON PLASTIC ONLINE
    "C:\\Users\\doris\\Downloads\\Assets\\Training\\Non_Plastic\\00000013.jpg",
    "C:\\Users\\doris\\Downloads\\Assets\\Training\\Non_Plastic\\00000015.jpg",
    "C:\\Users\\doris\\Downloads\\Assets\\Training\\Non_Plastic\\00000022.jpg",
    "C:\\Users\\doris\\Downloads\\Assets\\Training\\Non_Plastic\\00000026.jpg",
    "C:\\Users\\doris\\Downloads\\Assets\\Training\\Non_Plastic\\00000030.jpg",
    "C:\\Users\\doris\\Downloads\\Assets\\Training\\Non_Plastic\\00000031.jpg",
    "C:\\Users\\doris\\Downloads\\Assets\\Training\\Non_Plastic\\00000039.jpg",
    "C:\\Users\\doris\\Downloads\\Assets\\Training\\Non_Plastic\\00000043.jpg",
    "C:\\Users\\doris\\Downloads\\Assets\\Training\\Non_Plastic\\00000045.jpg",
    "C:\\Users\\doris\\Downloads\\Assets\\Training\\Non_Plastic\\00000068.jpg",
    #my pic:plastic
    "C:\\Users\\doris\\Downloads\\Assets\\Training\\mypic\\plast\\20231115_153403.jpg",
"C:\\Users\\doris\\Downloads\\Assets\\Training\\mypic\\plast\\20231115_153426.jpg",
"C:\\Users\\doris\\Downloads\\Assets\\Training\\mypic\\plast\\20231115_153509.jpg",
"C:\\Users\\doris\\Downloads\\Assets\\Training\\mypic\\plast\\20231115_153525.jpg",
"C:\\Users\\doris\\Downloads\\Assets\\Training\\mypic\\plast\\20231115_153547.jpg",
"C:\\Users\\doris\\Downloads\\Assets\\Training\\mypic\\plast\\20231115_153812.jpg",
"C:\\Users\\doris\\Downloads\\Assets\\Training\\mypic\\plast\\20231115_153833.jpg",
#my pic nonplstics
"C:\\Users\\doris\\Downloads\\Assets\\Training\\mypic\\nonplastic\\20231114_184527.jpg",
"C:\\Users\\doris\\Downloads\\Assets\\Training\\mypic\\nonplastic\\20231115_153448.jpg",
"C:\\Users\\doris\\Downloads\\Assets\\Training\\mypic\\nonplastic\\20231115_153517.jpg",
"C:\\Users\\doris\\Downloads\\Assets\\Training\\mypic\\nonplastic\\20231115_153610.jpg",
"C:\\Users\\doris\\Downloads\\Assets\\Training\\mypic\\nonplastic\\20231115_153646.jpg",
"C:\\Users\\doris\\Downloads\\Assets\\Training\\mypic\\nonplastic\\20231115_153710.jpg",
"C:\\Users\\doris\\Downloads\\Assets\\Training\\mypic\\nonplastic\\20231115_153722.jpg",
"C:\\Users\\doris\\Downloads\\Assets\\Training\\mypic\\nonplastic\\20231115_153939.jpg",
"C:\\Users\\doris\\Downloads\\Assets\\Training\\mypic\\nonplastic\\20231115_153944.jpg",
"C:\\Users\\doris\\Downloads\\Assets\\Training\\mypic\\nonplastic\\20231115_153950.jpg"

]

# Replace backslashes with forward slashes
image_paths = [path.replace("\\", "/") for path in image_paths]




labels = [1]*10
for i in range(10):
    labels.append(0)
for i in range(7):
    labels.append(1)
for i in range(10):
    labels.append(0)
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
