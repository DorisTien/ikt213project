import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.utils import to_categorical
from sklearn.metrics import accuracy_score

# Function for locating objects in an image
def locate_object(template_path, image_path):
    # Load the template and image
    template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)  # Ensure grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Ensure grayscale

    # Match the template within the image
    result = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)

    # Find the location of the best match
    _, _, _, max_loc = cv2.minMaxLoc(result)

    # Get the coordinates of the top-left corner of the matched area
    top_left = max_loc
    h, w = template.shape[::-1]  # Reverse dimensions for width and height

    # Draw a rectangle around the matched region
    bottom_right = (top_left[0] + w, top_left[1] + h)
    cv2.rectangle(img, top_left, bottom_right, (0, 255, 0), 2)

    # Display the result
    cv2.imshow('Location Recognition', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Print and return the location in pixel coordinates (x, y)
    location = (top_left[0], top_left[1])
    print("Location in Pixel Coordinates (x, y):", location)
    return location

# Step 2: Data Preprocessing and Feature Extraction
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (100, 100))
    return img

# Step 3: Data Collection and Labeling
image_paths = ['C:/Users/doris/Downloads/plastics/1.jpeg', 'C:/Users/doris/Downloads/plastics/2.png', 'C:/Users/doris/Downloads/plastics/n1.webp']
labels = [1, 1, 0]

# Step 4: Feature Extraction and Model Training
X = [preprocess_image(img_path) for img_path in image_paths]
y = np.array(labels)

# Convert labels to categorical format
label_encoder = LabelEncoder()
y = to_categorical(label_encoder.fit_transform(y))

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert images to numpy arrays
X_train = np.array(X_train)
X_test = np.array(X_test)

# Normalize pixel values to be between 0 and 1
X_train = X_train / 255.0
X_test = X_test / 255.0

# Step 5: Build the CNN Model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(2, activation='softmax'))  # 2 output neurons for binary classification

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Step 6: Train the CNN Model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Step 7: Model Evaluation
y_pred_prob = model.predict(X_test)
y_pred = np.argmax(y_pred_prob, axis=1)
y_true = np.argmax(y_test, axis=1)
print("Accuracy:", accuracy_score(y_true, y_pred))

# Step 8: Deployment
# Use the trained model to predict new images
new_image_path = 'C:/Users/doris/Downloads/Quick Share/20230914_204857.jpg'
new_img = preprocess_image(new_image_path)
new_img = np.expand_dims(new_img, axis=0)
new_img = new_img / 255.0

# Predict using the model
predicted_prob = model.predict(new_img)
predicted_label = np.argmax(predicted_prob)

# If the model predicts the presence of plastics, locate the object
if predicted_label == 1:
    print("The image contains plastic. Performing object detection...")
    template_path = new_image_path  # Replace with the path to your template image
    object_location = locate_object(template_path, new_image_path)
    print("Object Location (x, y):", object_location)
else:
    print("The image does not contain plastic.")
