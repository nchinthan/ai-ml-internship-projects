import sys
import argparse
import numpy as np
import tensorflow as tf
from PIL import Image

def load_and_preprocess_image(image_path):
    # Load image, convert to grayscale, resize to 28x28
    img = Image.open(image_path).convert('L')
    img = img.resize((28, 28))
    
    # Convert image to numpy array
    img_array = np.array(img).astype('float32') / 255.0
    
    # Invert colors if background is light/white (MNIST expects white text on black background)
    if np.mean(img_array) > 0.5:
        img_array = 1.0 - img_array
        
    img_array = np.expand_dims(img_array, axis=(0, -1))
    return img_array

def main():
    parser = argparse.ArgumentParser(description="Predict handwritten digit from image.")
    parser.add_argument("--image", type=str, required=True, help="Path to digit image file")
    args = parser.parse_args()

    model_path = "models/digit_recognizer.h5"
    
    try:
        model = tf.keras.models.load_model(model_path)
    except Exception as e:
        print(f"Error loading model from {model_path}. Make sure you ran digit_recognizer.ipynb first.")
        sys.exit(1)

    processed_img = load_and_preprocess_image(args.image)
    predictions = model.predict(processed_img)
    predicted_digit = np.argmax(predictions[0])
    confidence = np.max(predictions[0]) * 100

    print(f"\nPredicted Digit: {predicted_digit}")
    print(f"Confidence: {confidence:.2f}%\n")

if __name__ == "__main__":
    main()