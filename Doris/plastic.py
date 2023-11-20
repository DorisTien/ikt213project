import numpy as np
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from PIL import Image

# Load the pre-trained VGG16 model
model = VGG16(weights='imagenet')

# Define a function to recognize plastics
def recognize_plastic(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)

    predictions = model.predict(preprocessed_img)
    decoded_predictions = decode_predictions(predictions, top=3)[0]

    print("Predictions:")
    for imagenet_id, label, score in decoded_predictions:
        print(f"{label} ({score:.2f})")

# Path to your image
image_path = "C:/Users/doris/Downloads/Assets/Training/mypic/plast/20231115_153509.jpg" # Change this to the path of your image

recognize_plastic(image_path)

image = Image.open(image_path)

# Display the image
image.show()