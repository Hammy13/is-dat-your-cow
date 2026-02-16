import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import json
from datetime import datetime

class CowIdentificationSystem:
    def __init__(self, model_path='runs/detect/train/weights/best.pt'):
        # Layer 3: YOLO Detector
        if not Path(model_path).exists():
            print(f"Model not found at {model_path}, using yolo11n.pt")
            model_path = 'yolo11n.pt'
        self.detector = YOLO(model_path)
        
        # Layer 6: Gallery (embedding store)
        self.gallery = {}
        self.load_gallery()
        
        # Layer 4: Tracking
        self.tracks = {}
        self.next_track_id = 0
        
    def load_gallery(self):
        """Load existing cow embeddings"""
        gallery_path = Path('gallery.json')
        if gallery_path.exists():
            with open(gallery_path, 'r') as f:
                self.gallery = json.load(f)
    
    def save_gallery(self):
        """Save cow embeddings"""
        with open('gallery.json', 'w') as f:
            json.dump(self.gallery, f, indent=2)
    
    def preprocess_frame(self, frame):
        """Layer 1: Frame preprocessing"""
        frame = cv2.resize(frame, (640, 640))
        return frame
    
    def detect_cows(self, frame):
        """Layer 3: YOLO Detection"""
        results = self.detector(frame, conf=0.5)
        detections = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                detections.append({'bbox': [x1, y1, x2, y2], 'conf': conf})
        return detections
    
    def extract_features(self, crop):
        """Layer 5: Re-ID Feature Extraction (simplified)"""
        crop_resized = cv2.resize(crop, (128, 128))
        features = cv2.calcHist([crop_resized], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        features = features.flatten()
        features = features / (np.linalg.norm(features) + 1e-6)
        return features
    
    def match_identity(self, features, threshold=0.7):
        """Layer 6: Identity Matching"""
        best_match = None
        best_score = threshold
        
        for cow_id, stored_features in self.gallery.items():
            stored = np.array(stored_features)
            similarity = np.dot(features, stored)
            if similarity > best_score:
                best_score = similarity
                best_match = cow_id
        
        return best_match, best_score
    
    def process_video(self, video_path, output_path='output.mp4'):
        """Main inference pipeline"""
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        logs = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Layer 1: Preprocessing
            processed = self.preprocess_frame(frame.copy())
            
            # Layer 3: Detection
            detections = self.detect_cows(processed)
            
            for det in detections:
                x1, y1, x2, y2 = det['bbox']
                
                # Scale bbox back to original frame
                scale_x = width / 640
                scale_y = height / 640
                x1, x2 = int(x1 * scale_x), int(x2 * scale_x)
                y1, y2 = int(y1 * scale_y), int(y2 * scale_y)
                
                # Extract crop
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                
                # Layer 5: Feature extraction
                features = self.extract_features(crop)
                
                # Layer 6: Identity matching
                cow_id, score = self.match_identity(features)
                
                if cow_id is None:
                    cow_id = f"COW_{len(self.gallery) + 1:03d}"
                    self.gallery[cow_id] = features.tolist()
                
                # Layer 7: Visualization
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{cow_id} ({score:.2f})", (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Logging
                logs.append({
                    'frame': frame_count,
                    'cow_id': cow_id,
                    'confidence': score,
                    'timestamp': datetime.now().isoformat()
                })
            
            out.write(frame)
            frame_count += 1
            
            if frame_count % 30 == 0:
                print(f"Processed {frame_count} frames")
        
        cap.release()
        out.release()
        
        # Save gallery and logs
        self.save_gallery()
        with open('detection_logs.json', 'w') as f:
            json.dump(logs, f, indent=2)
        
        print(f"Processing complete. Output: {output_path}")
        print(f"Total cows identified: {len(self.gallery)}")

if __name__ == "__main__":
    import sys
    
    # Find video files
    video_dir = Path('videos')
    videos = sorted(list(video_dir.glob('*.mp4')) + list(video_dir.glob('*.avi')))
    
    if not videos:
        print("Error: No video files found in 'videos/' folder")
        print("Please add a video file to the videos/ directory")
        sys.exit(1)
    
    # Select video
    if len(sys.argv) > 1:
        # Command line argument provided
        video_name = sys.argv[1]
        video_path = video_dir / video_name
        if not video_path.exists():
            print(f"Error: Video '{video_name}' not found in videos/ folder")
            sys.exit(1)
    else:
        # Show menu
        print("\n" + "="*50)
        print("Available videos:")
        print("="*50)
        for i, video in enumerate(videos, 1):
            print(f"{i}. {video.name}")
        print("="*50)
        
        choice = input("\nSelect video number (or press Enter for first video): ").strip()
        
        if choice == "":
            video_path = videos[0]
        else:
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(videos):
                    video_path = videos[idx]
                else:
                    print("Invalid choice. Using first video.")
                    video_path = videos[0]
            except ValueError:
                print("Invalid input. Using first video.")
                video_path = videos[0]
    
    print(f"\nProcessing video: {video_path.name}")
    output_name = f"output_{video_path.stem}.mp4"
    
    system = CowIdentificationSystem()
    system.process_video(str(video_path), output_name)
    print(f"\nOutput saved as: {output_name}")
