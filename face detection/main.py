# main.py
from registration import register_user
from trainer import train_recognizer
from detection import detect_faces

def main():
    print("Select an option:")
    print("1. Register a new face")
    print("2. Train model")
    print("3. Start surveillance detection")
    choice = input("Enter 1, 2, or 3: ").strip()
    
    if choice == "1":
        register_user()
    elif choice == "2":
        train_recognizer()
    elif choice == "3":
        detect_faces()
    else:
        print("Invalid choice.")

if __name__ == "__main__":
    main()
