import cv2, os, glob

VIDEO_DIR = "videos"
OUT_DIR = "frames_raw"
FPS_EXTRACT = 1  # frames per second

os.makedirs(OUT_DIR, exist_ok=True)

for vp in glob.glob(os.path.join(VIDEO_DIR, "*.mp4")):
    name = os.path.splitext(os.path.basename(vp))[0]
    out = os.path.join(OUT_DIR, name)
    os.makedirs(out, exist_ok=True)

    cap = cv2.VideoCapture(vp)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    step = max(int(fps / FPS_EXTRACT), 1)

    i = 0
    saved = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if i % step == 0:
            cv2.imwrite(os.path.join(out, f"{name}_{saved:06d}.jpg"), frame)
            saved += 1
        i += 1

    cap.release()
    print(f"{name}: saved {saved} frames")
