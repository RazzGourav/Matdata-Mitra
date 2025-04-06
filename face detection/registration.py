# registration.py
import cv2
import os

def register_user():
    aadhar = input("Enter your Aadhar number: ").strip()
    save_dir = os.path.join("data", "registered", aadhar)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    else:
        print("User folder already exists – new images will be added.")
    
    # Load Haar cascade for face detection
    face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    
    cap = cv2.VideoCapture(0)
    print("Look at the camera. Press 's' to capture a face image, 'q' to quit.")
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        
        # Draw rectangles around detected faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        cv2.imshow("Registration", frame)
        key = cv2.waitKey(1)
        
        if key == ord('s'):
            # Save each detected face (cropped and in grayscale)
            for (x, y, w, h) in faces:
                face_img = gray[y:y+h, x:x+w]
                count += 1
                file_path = os.path.join(save_dir, f"{aadhar}_{count}.jpg")
                cv2.imwrite(file_path, face_img)
                print(f"Saved {file_path}")
            if count >= 20:
                print("Collected sufficient images for training.")
                break
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    register_user()
