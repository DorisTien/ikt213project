import cv2
import numpy as np

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

    # Print the location in pixel coordinates
    print("Location in Pixel Coordinates (x, y):", top_left)

if __name__ == "__main__":
    image_path = "C:/Users/doris/ikt213g23h/assignments/resources/shapes.png"  # Path to the template image
    template_path = "C:/Users/doris/ikt213g23h/assignments/resources/shapes_template.jpg"        # Path to the image in which to search for the template

    locate_object(template_path, image_path)
