import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.utils import to_categorical
from sklearn.metrics import accuracy_score

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
new_image_path = 'C:/Users/doris/Downloads/plastics/3.jpeg'
new_img = preprocess_image(new_image_path)
new_img = np.expand_dims(new_img, axis=0)
new_img = new_img / 255.0
predicted_prob = model.predict(new_img)
predicted_label = np.argmax(predicted_prob)
if predicted_label == 1:
    print("The image contains plastic.")
else:
    print("The image does not contain plastic.")
