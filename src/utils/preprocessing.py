# src/ultis/preprocessing.py

import cv2

def preprocess_image(image, adaptive_threshold=False):
    if len(image.shape) == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image

    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    if adaptive_threshold:
        binary = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        result = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
    else:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        equalized = clahe.apply(blurred)
        result = cv2.cvtColor(equalized, cv2.COLOR_GRAY2RGB)

    return result
