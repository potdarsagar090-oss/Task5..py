"""
Face Detection and Recognition System
A friendly, humanized AI application for detecting and recognizing faces
"""

import cv2
import numpy as np
import os
from datetime import datetime
import pickle
from pathlib import Path
import sys

# Fix Unicode encoding for Windows console
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')


class FaceRecognitionSystem:
    def __init__(self):
        self.name = "FaceGuard AI"
        self.version = "1.0"
        
        # Initialize face detection models
        try:
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            self.eye_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_eye.xml'
            )
        except Exception as e:
            print(f"Error loading cascade classifiers: {e}")
            print("Please ensure OpenCV is properly installed.")
            sys.exit(1)
        
        # Face recognizer
        try:
            self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        except Exception as e:
            print(f"Error creating face recognizer: {e}")
            print("Please install: pip install opencv-contrib-python")
            sys.exit(1)
        
        # Storage paths
        self.data_dir = Path("face_data")
        self.data_dir.mkdir(exist_ok=True)
        self.model_path = self.data_dir / "trained_model.yml"
        self.names_path = self.data_dir / "names.pkl"
        
        # Known faces database
        self.known_faces = {}
        self.load_known_faces()
        
        # Session stats
        self.faces_detected_today = 0
        self.recognition_attempts = 0
        self.successful_recognitions = 0
        
    def greet_user(self):
        """Give a friendly welcome message"""
        hour = datetime.now().hour
        
        if 5 <= hour < 12:
            greeting = "Good morning"
        elif 12 <= hour < 17:
            greeting = "Good afternoon"
        elif 17 <= hour < 22:
            greeting = "Good evening"
        else:
            greeting = "Hello"
        
        print("\n" + "="*60)
        print(f"  {greeting}! Welcome to {self.name} v{self.version}")
        print("="*60)
        print("  I'm here to help you detect and recognize faces!")
        print("  Let's make face recognition friendly and accessible.")
        print("="*60 + "\n")
    
    def detect_faces_in_image(self, image_path, show_result=True):
        """
        Detect faces in a static image with friendly feedback
        """
        print(f"[IMAGE] Analyzing image: {image_path}")
        print("[SCAN] Looking for faces...")
        
        # Read the image
        img = cv2.imread(str(image_path))
        
        if img is None:
            print("[ERROR] Oops! Couldn't read that image. Please check the file path.")
            return None
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        self.faces_detected_today += len(faces)
        
        if len(faces) == 0:
            print("[INFO] Hmm, I couldn't find any faces in this image.")
            print("       Try an image with clearer faces or better lighting!")
            return img, []
        
        if len(faces) == 1:
            print(f"[SUCCESS] Great! I found 1 face in the image!")
        else:
            print(f"[SUCCESS] Awesome! I detected {len(faces)} faces in the image!")
        
        # Draw rectangles around faces
        for i, (x, y, w, h) in enumerate(faces, 1):
            # Draw face rectangle
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Add label
            label = f"Face {i}"
            cv2.putText(img, label, (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Try to detect eyes for better accuracy
            roi_gray = gray[y:y+h, x:x+w]
            eyes = self.eye_cascade.detectMultiScale(roi_gray)
            
            if len(eyes) >= 2:
                print(f"   [EYES] Face {i}: Detected eyes - looks like a real face!")
            else:
                print(f"   [FACE] Face {i}: Detected (couldn't confirm eyes)")
        
        if show_result:
            cv2.imshow(f"{self.name} - Detection Result", img)
            print("\n[TIP] Press any key to close the image window...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        return img, faces
    
    def detect_faces_webcam(self):
        """
        Real-time face detection from webcam with friendly interaction
        """
        print("[WEBCAM] Starting webcam for real-time face detection...")
        print("[TIPS] Tips:")
        print("   - Face the camera directly for best results")
        print("   - Make sure you have good lighting")
        print("   - Press 'q' to quit, 's' to save a snapshot")
        print("\n[INIT] Initializing camera...\n")
        
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("[ERROR] Couldn't access your camera. Please check:")
            print("   - Camera is connected")
            print("   - No other app is using it")
            print("   - You've granted camera permissions")
            return
        
        print("[READY] Camera ready! You're live now!\n")
        
        frame_count = 0
        last_face_count = 0
        snapshot_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[WARN] Failed to grab frame. Retrying...")
                continue
            
            frame_count += 1
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )
            
            # Provide friendly feedback when face count changes
            if len(faces) != last_face_count:
                if len(faces) == 0:
                    print("[SCAN] No faces detected - try moving into frame!")
                elif len(faces) == 1:
                    print("[FOUND] Got you! 1 face detected.")
                else:
                    print(f"[FOUND] Cool! {len(faces)} faces in view!")
                last_face_count = len(faces)
            
            # Draw on frame
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # Detect eyes
                roi_gray = gray[y:y+h, x:x+w]
                eyes = self.eye_cascade.detectMultiScale(roi_gray)
                
                # Draw eyes
                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(frame, (x+ex, y+ey), 
                                (x+ex+ew, y+ey+eh), (255, 0, 0), 2)
                
                # Status text
                status = "Real Face" if len(eyes) >= 2 else "Face Detected"
                cv2.putText(frame, status, (x, y-10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Display info on frame
            info_text = f"Faces: {len(faces)} | Press 'q' to quit, 's' for snapshot"
            cv2.putText(frame, info_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            cv2.imshow(f"{self.name} - Live Detection", frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("\n[EXIT] Closing camera. Thanks for using the system!")
                break
            elif key == ord('s'):
                snapshot_count += 1
                filename = f"snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                cv2.imwrite(filename, frame)
                print(f"[SAVE] Snapshot saved as '{filename}'!")
        
        cap.release()
        cv2.destroyAllWindows()
        
        print(f"\n[STATS] Session Summary:")
        print(f"   Total frames processed: {frame_count}")
        print(f"   Snapshots taken: {snapshot_count}")
    
    def collect_training_data(self, person_name, num_samples=30):
        """
        Collect face samples for training with user guidance
        """
        print(f"\n[TRAIN] Let's collect face samples for: {person_name}")
        print(f"[INFO] I'll capture {num_samples} photos of your face.")
        print("\n[TIPS] Tips for best results:")
        print("   - Look at the camera")
        print("   - Try different angles (slightly left, right, up, down)")
        print("   - Keep your face well-lit")
        print("   - Show different expressions")
        print("\n[INPUT] Press 'Enter' when ready...")
        input()
        
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("[ERROR] Can't access camera. Please check your camera connection.")
            return False
        
        print("[READY] Camera is ready! Starting capture...\n")
        
        # Create directory for this person
        person_dir = self.data_dir / person_name
        person_dir.mkdir(exist_ok=True)
        
        samples_collected = 0
        frame_count = 0
        
        while samples_collected < num_samples:
            ret, frame = cap.read()
            if not ret:
                continue
            
            frame_count += 1
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 5)
            
            for (x, y, w, h) in faces:
                samples_collected += 1
                
                # Save face image
                face_img = gray[y:y+h, x:x+w]
                img_path = person_dir / f"{person_name}_{samples_collected}.jpg"
                cv2.imwrite(str(img_path), face_img)
                
                # Draw rectangle
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # Progress indicator
                progress = int((samples_collected / num_samples) * 100)
                progress_text = f"Progress: {samples_collected}/{num_samples} ({progress}%)"
                cv2.putText(frame, progress_text, (10, 30),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Friendly encouragement
                if samples_collected % 10 == 0:
                    print(f"[PROGRESS] Great! {samples_collected} samples collected!")
                
                if samples_collected >= num_samples:
                    print(f"\n[SUCCESS] Perfect! Collected all {num_samples} samples!")
                    break
            
            cv2.imshow("Collecting Face Data - Look at the camera!", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\n[WARN] Collection stopped early.")
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        if samples_collected >= num_samples:
            print(f"\n[DONE] Successfully collected {samples_collected} face samples for {person_name}!")
            return True
        else:
            print(f"\n[WARN] Only collected {samples_collected} samples. You might want to try again.")
            return False
    
    def train_recognizer(self):
        """
        Train the face recognition model with collected data
        """
        print("\n[TRAIN] Training face recognition model...")
        print("[READ] Reading all collected face samples...")
        
        faces = []
        labels = []
        label_map = {}
        current_label = 0
        
        # Read all face samples
        for person_dir in self.data_dir.iterdir():
            if person_dir.is_dir():
                person_name = person_dir.name
                label_map[current_label] = person_name
                
                sample_count = 0
                for img_path in person_dir.glob("*.jpg"):
                    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        faces.append(img)
                        labels.append(current_label)
                        sample_count += 1
                
                print(f"   [DATA] {person_name}: {sample_count} samples loaded")
                current_label += 1
        
        if len(faces) == 0:
            print("[ERROR] No training data found! Please collect some face samples first.")
            return False
        
        print(f"\n[PROCESS] Training on {len(faces)} face images from {len(label_map)} people...")
        
        # Train the model
        self.recognizer.train(faces, np.array(labels))
        
        # Save the model and labels
        self.recognizer.save(str(self.model_path))
        with open(self.names_path, 'wb') as f:
            pickle.dump(label_map, f)
        
        self.known_faces = label_map
        
        print("[SUCCESS] Training complete!")
        print(f"[SAVE] Model saved to: {self.model_path}")
        print(f"\n[INFO] Trained to recognize: {', '.join(label_map.values())}")
        
        return True
    
    def load_known_faces(self):
        """Load previously trained model"""
        if self.model_path.exists() and self.names_path.exists():
            try:
                self.recognizer.read(str(self.model_path))
                with open(self.names_path, 'rb') as f:
                    self.known_faces = pickle.load(f)
                return True
            except:
                return False
        return False
    
    def recognize_faces_webcam(self):
        """
        Real-time face recognition from webcam
        """
        if not self.known_faces:
            print("[WARN] No trained model found!")
            print("       Please train the model first by collecting face samples.")
            return
        
        print("[START] Starting face recognition...")
        print(f"[INFO] I can recognize: {', '.join(self.known_faces.values())}")
        print("[TIP] Press 'q' to quit\n")
        
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("[ERROR] Can't access camera.")
            return
        
        print("[READY] Camera ready! Looking for faces...\n")
        
        recognized_names = set()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 5)
            
            for (x, y, w, h) in faces:
                face_roi = gray[y:y+h, x:x+w]
                
                # Recognize face
                self.recognition_attempts += 1
                label, confidence = self.recognizer.predict(face_roi)
                
                # Lower confidence value = better match (distance metric)
                if confidence < 70:  # Good match threshold
                    name = self.known_faces.get(label, "Unknown")
                    color = (0, 255, 0)  # Green for recognized
                    
                    if name not in recognized_names:
                        recognized_names.add(name)
                        print(f"[FOUND] Hey there, {name}! Nice to see you!")
                        self.successful_recognitions += 1
                else:
                    name = "Unknown"
                    color = (0, 0, 255)  # Red for unknown
                
                # Draw rectangle and name
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                
                # Display name and confidence
                display_text = f"{name} ({int(100-confidence)}%)"
                cv2.putText(frame, display_text, (x, y-10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Display info
            info = f"Recognizing... Press 'q' to quit | Recognized: {len(recognized_names)}"
            cv2.putText(frame, info, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            cv2.imshow(f"{self.name} - Face Recognition", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\n[EXIT] Stopping recognition...")
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        print(f"\n[STATS] Recognition Summary:")
        print(f"   People recognized: {', '.join(recognized_names) if recognized_names else 'None'}")
        print(f"   Recognition attempts: {self.recognition_attempts}")
        print(f"   Successful recognitions: {self.successful_recognitions}")
    
    def show_menu(self):
        """Display interactive menu"""
        while True:
            print("\n" + "="*60)
            print(f"  {self.name} - Main Menu")
            print("="*60)
            print("  1. [IMAGE] Detect faces in an image")
            print("  2. [VIDEO] Live face detection (webcam)")
            print("  3. [TRAIN] Collect training data for a person")
            print("  4. [MODEL] Train face recognition model")
            print("  5. [RECOG] Live face recognition (webcam)")
            print("  6. [STATS] Show statistics")
            print("  7. [HELP]  Help & tips")
            print("  8. [EXIT]  Exit")
            print("="*60)
            
            choice = input("\nWhat would you like to do? (1-8): ").strip()
            
            if choice == '1':
                img_path = input("[INPUT] Enter image path: ").strip()
                self.detect_faces_in_image(img_path)
            
            elif choice == '2':
                self.detect_faces_webcam()
            
            elif choice == '3':
                name = input("[INPUT] Enter person's name: ").strip()
                if name:
                    num_samples = input("[INPUT] Number of samples (default 30): ").strip()
                    num_samples = int(num_samples) if num_samples.isdigit() else 30
                    self.collect_training_data(name, num_samples)
                else:
                    print("[ERROR] Name cannot be empty!")
            
            elif choice == '4':
                self.train_recognizer()
            
            elif choice == '5':
                self.recognize_faces_webcam()
            
            elif choice == '6':
                self.show_statistics()
            
            elif choice == '7':
                self.show_help()
            
            elif choice == '8':
                print("\n" + "="*60)
                print(f"  Thanks for using {self.name}!")
                print("  Have a great day!")
                print("="*60 + "\n")
                break
            
            else:
                print("[ERROR] Invalid choice! Please enter a number between 1 and 8.")
    
    def show_statistics(self):
        """Display usage statistics"""
        print("\n" + "="*60)
        print("  [STATS] System Statistics")
        print("="*60)
        print(f"  Faces detected today: {self.faces_detected_today}")
        print(f"  Recognition attempts: {self.recognition_attempts}")
        print(f"  Successful recognitions: {self.successful_recognitions}")
        
        if self.known_faces:
            print(f"\n  [DATABASE] Known faces: {len(self.known_
