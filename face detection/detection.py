# detection.py
import cv2
import os
import json

def detect_faces():
    # Load Haar cascade for face detection
    face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    
    # Load the LBPH face recognizer model
    model_path = os.path.join("models", "face_recognizer.yml")
    if not os.path.exists(model_path):
        print("Model not found. Please run the trainer module first.")
        return
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(model_path)
    
    # Load label mapping
    if not os.path.exists("labels.json"):
        print("Label mapping not found.")
        return
    with open("labels.json", "r") as f:
        label_mapping = json.load(f)
    
    cap = cv2.VideoCapture(0)
    print("Starting surveillance detection. Press 'q' to quit.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        
        for (x, y, w, h) in faces:
            face_img = gray[y:y+h, x:x+w]
            # Recognize the face (LBPH returns a label and a confidence; lower confidence is better)
            label, confidence = recognizer.predict(face_img)
            # Adjust the threshold as needed – here, a confidence less than 100 is considered a match.
            if confidence < 100:
                aadhar = label_mapping.get(str(label)) if str(label) in label_mapping else label_mapping.get(label)
                text = f"{aadhar} - Already Voted"
                color = (0, 0, 255)  # Red
            else:
                text = "Unknown"
                color = (0, 255, 0)  # Green

            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        cv2.imshow("Surveillance Detection", frame)
        if cv2.waitKey(1) == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_faces()
