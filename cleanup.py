import shutil
from pathlib import Path

def cleanup_workspace():
    """Remove unnecessary files and folders"""
    
    print("Cleaning up workspace...\n")
    
    # Files/folders to remove
    to_remove = [
        # Duplicate/temporary frame folders
        'frames_raw',
        'raw_frames',
        
        # Old training runs (keep only latest)
        'runs/detect/train2',
        'runs/detect/train3',
        'runs/detect/train4',
        
        # Cache files
        'dataset/labels/train.cache',
        'dataset/labels/val.cache',
        
        # Duplicate architecture file
        'arch4.puml',
        
        # Download script (no longer needed)
        'download_video.py',
    ]
    
    removed_count = 0
    
    for item in to_remove:
        path = Path(item)
        if path.exists():
            try:
                if path.is_dir():
                    shutil.rmtree(path)
                    print(f"[OK] Removed folder: {item}")
                else:
                    path.unlink()
                    print(f"[OK] Removed file: {item}")
                removed_count += 1
            except Exception as e:
                print(f"[FAIL] Failed to remove {item}: {e}")
        else:
            print(f"[SKIP] Not found: {item}")
    
    print(f"\nCleanup complete! Removed {removed_count} items.")
    print("\nKeeping essential files:")
    print("  - dataset/ (training data)")
    print("  - videos/ (input videos)")
    print("  - runs/detect/train/ (trained model)")
    print("  - *.py (Python scripts)")
    print("  - *.md (documentation)")
    print("  - *.json (gallery & logs)")
    print("  - *.yaml (config)")
    print("  - *.puml (architecture)")
    print("  - output_*.mp4 (results)")

if __name__ == "__main__":
    cleanup_workspace()
