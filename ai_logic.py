from PIL import Image
import numpy as np
import tensorflow as tf
import os

# Resolve absolute paths to ensure reliable file loading during runtime
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'crop_doctor_v2.h5')
CLASS_NAMES_PATH = os.path.join(BASE_DIR, 'class_names_v2.txt')

load_error_message = ""

try:
    # Load the model with safe_mode disabled to support custom configurations
    model = tf.keras.models.load_model(MODEL_PATH, compile=False, safe_mode=False)
    
    with open(CLASS_NAMES_PATH, 'r') as f:
        class_names = [line.strip() for line in f.readlines()]
except Exception as e:
    model = None
    class_names = []
    load_error_message = str(e)
    print(f"CRITICAL ERROR LOADING MODEL: {e}")

def analyze_leaf(uploaded_file):
    """
    Processes the uploaded image, normalizes it for the CNN architecture, 
    and returns the predicted class label.
    """
    if model is None:
        return f"System Error: AI model failed to load. Reason: {load_error_message}"

    try:
        # 1. Open and format image to standard RGB
        img = Image.open(uploaded_file).convert('RGB')
        
        # 2. Resize to match expected input dimensions
        img = img.resize((224, 224))
        
        # 3. Normalize pixel values to [0, 1] to match the Colab training pipeline
        img_array = np.array(img, dtype=np.float32) / 255.0
        

        # If Colab used cv2 to train, it learned in BGR. We must flip PIL's RGB to BGR.
        img_array = img_array[:, :, ::-1]
        
        # 4. Expand dimensions to create a batch size of 1
        img_array = np.expand_dims(img_array, axis=0)
        
        # 5. Perform inference
        raw_predictions = model.predict(img_array)[0] 
        
        # 6. Output Masking: Ignore the corrupted 'dataset' class at index 16
        if len(raw_predictions) > 16:
            raw_predictions[16] = 0.0 
            
        # 7. The Confidence Booster
        # Recalculate the percentages out of 100% after removing the corrupted folder
        total_remaining = np.sum(raw_predictions)
        if total_remaining > 0:
            raw_predictions = raw_predictions / total_remaining
            
        # Determine the class with the highest remaining probability
        predicted_class_index = int(np.argmax(raw_predictions))
        confidence = float(np.max(raw_predictions) * 100)
        
        # Print diagnostic information to the terminal
        print("\n" + "="*40)
        print("DIAGNOSTIC: INFERENCE RESULTS")
        print("="*40)
        print(f"Predicted Class Index: {predicted_class_index}")
        print(f"Confidence Level:      {confidence:.2f}%")
        print("="*40 + "\n")

        # If the AI is less than 70% sure, it will reject the image.
        # You can tweak this number (e.g., 60.0 to 85.0) to make it more or less strict.
        if confidence < 70.0:
            return "Unsupported Image: I am only trained to analyze leaves from specific crops (like tomatoes, potatoes, and peppers). Please upload a valid crop leaf."
        
        detected_label = class_names[predicted_class_index]
        
        # Handle out-of-domain images (if your model specifically has this class)
        if "Not_A_Plant" in detected_label:
            return "ERROR: No plant detected"

        # Format the output string for the UI presentation
        formatted_disease = detected_label.replace("___", ": ").replace("__", ": ").replace("_", " ")
        
        return formatted_disease

    except Exception as e:
        return f"Error analyzing image: {e}"