# trainer.py
import cv2
import os
import numpy as np
import json

def train_recognizer():
    data_dir = os.path.join("data", "registered")
    faces = []
    labels = []
    label_mapping = {}  # Numeric label -> Aadhar number
    current_label = 0
    
    # Traverse each registered user's folder
    for aadhar in os.listdir(data_dir):
        user_folder = os.path.join(data_dir, aadhar)
        if not os.path.isdir(user_folder):
            continue
        label_mapping[current_label] = aadhar
        for image_name in os.listdir(user_folder):
            image_path = os.path.join(user_folder, image_name)
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            faces.append(img)
            labels.append(current_label)
        current_label += 1

    if len(faces) == 0:
        print("No training data found. Please register users first.")
        return

    labels = np.array(labels)

    # Create and train the LBPH face recognizer
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, labels)

    # Save the trained model
    if not os.path.exists("models"):
        os.makedirs("models")
    model_path = os.path.join("models", "face_recognizer.yml")
    recognizer.save(model_path)
    print(f"Model saved at {model_path}")

    # Save label mapping to a JSON file
    with open("labels.json", "w") as f:
        json.dump(label_mapping, f)
    print("Label mapping saved in labels.json")

if __name__ == "__main__":
    train_recognizer()
