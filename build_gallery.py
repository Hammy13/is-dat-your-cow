import cv2
import numpy as np
from pathlib import Path
import json

def extract_features(crop):
    """Layer 5: Feature extraction"""
    crop_resized = cv2.resize(crop, (128, 128))
    features = cv2.calcHist([crop_resized], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    features = features.flatten()
    features = features / (np.linalg.norm(features) + 1e-6)
    return features

def build_gallery():
    """Layer 6: Build initial gallery from training images"""
    gallery = {}
    train_images = Path('dataset/images/train')
    
    for img_path in train_images.glob('*.jpg'):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        
        # Extract features from full image (simplified)
        features = extract_features(img)
        cow_id = f"COW_{img_path.stem}"
        gallery[cow_id] = features.tolist()
        print(f"Added {cow_id} to gallery")
    
    # Save gallery
    with open('gallery.json', 'w') as f:
        json.dump(gallery, f, indent=2)
    
    print(f"\nGallery built with {len(gallery)} cows")

if __name__ == "__main__":
    build_gallery()
