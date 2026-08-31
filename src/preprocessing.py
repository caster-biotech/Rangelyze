import cv2
import numpy as np
import os

def load_and_preprocess(image_path: str) -> np.ndarray:
    """
    Simulates the preparation of a smear slide. 
    Loads the image, corrects color, and enhances contrast for inference.
    """
    # 1. Quality contorl: Sample exist? 
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Error: sample not found in {image_path}")

    # 2. Reading (Placing the slide on the stage)
    image_bgr = cv2.imread(image_path)
    
    # 3. Color correction to RGB
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    
    # 4. Improving illumination without altering colors (CLAHE)
    # Convert to space HSV (Hue, Saturation, Value)
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    
    # Apply CLAHE only to channel V 
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    v_clahe = clahe.apply(v)
    
    # Reconstruct the image with improved contrast
    hsv_clahe = cv2.merge([h, s, v_clahe])
    image_enhanced = cv2.cvtColor(hsv_clahe, cv2.COLOR_HSV2RGB)

    return image_enhanced