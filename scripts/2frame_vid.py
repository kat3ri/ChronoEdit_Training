import os
import cv2
import numpy as np
import imageio

# Import captioning logic
from data_captioning import load_model, generate_caption

# Configuration
source_dir = r"/workspace/ChronoEdit_Training/assets/images/input-images"
target_dir = r"/workspace/ChronoEdit_Training/assets/images/output-images"
output_dir = r"/workspace/dataset/example_dataset/videos"
metadata_dir = r"/workspace/dataset/example_dataset"

fps = 1
max_pairs = None

# Captioning model config
caption_model_name = "Qwen/Qwen3-VL-30B-A3B-Instruct"
max_resolution = 1080

os.makedirs(output_dir, exist_ok=True)

files1 = set([f for f in os.listdir(source_dir) if not f.startswith('.')])
files2 = set([f for f in os.listdir(target_dir) if not f.startswith('.')])
matching = sorted(files1 & files2)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
detector_available = not face_cascade.empty()

metadata_rows = []

# Load captioning model and processor once
model, processor = load_model(caption_model_name)

processed = 0
for fname in matching:
    if max_pairs is not None and processed >= max_pairs:
        print(f"Reached processing limit: {max_pairs} pairs.")
        break

    src_path = os.path.join(source_dir, fname)
    tgt_path = os.path.join(target_dir, fname)
    key = os.path.splitext(fname)[0]
    video_path = os.path.join(output_dir, f"{key}.mp4")

    img1 = cv2.imread(src_path, cv2.IMREAD_UNCHANGED)
    img2 = cv2.imread(tgt_path, cv2.IMREAD_UNCHANGED)
    if img1 is None or img2 is None:
        print(f"Skipping `{fname}`: could not read image.")
        processed += 1
        continue

    if img1.ndim == 3 and img1.shape[2] == 4:
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGRA2BGR)
    if img2.ndim == 3 and img2.shape[2] == 4:
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGRA2BGR)

    height, width = img1.shape[:2]
    target_ar = width / height

    h2, w2 = img2.shape[:2]
    face_cx, face_cy = w2 // 2, h2 // 2
    if detector_available and img2.ndim == 3 and img2.shape[2] >= 3:
        try:
            gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            if len(faces) > 0:
                x, y, fw, fh = max(faces, key=lambda r: r[2] * r[3])
                face_cx = x + fw // 2
                face_cy = y + fh // 2
        except Exception:
            pass

    crop_w = w2
    crop_h = int(round(crop_w / target_ar))
    if crop_h > h2:
        crop_h = h2
        crop_w = int(round(crop_h * target_ar))
    crop_w = min(crop_w, w2)
    crop_h = min(crop_h, h2)

    x1 = int(max(0, min(face_cx - crop_w // 2, w2 - crop_w)))
    y1 = int(max(0, min(face_cy - crop_h // 2, h2 - crop_h)))
    cropped = img2[y1:y1 + crop_h, x1:x1 + crop_w]

    img2_resized = cv2.resize(cropped, (width, height), interpolation=cv2.INTER_AREA)

    def to_rgb(img):
        if img.ndim == 2:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    frame1 = to_rgb(img1)
    frame2 = to_rgb(img2_resized)

    os.makedirs(os.path.dirname(video_path), exist_ok=True)
    with imageio.get_writer(video_path, fps=fps) as writer:
        writer.append_data(frame1)
        writer.append_data(frame2)

    try:
        reader = imageio.get_reader(video_path)
        frame_count = sum(1 for _ in reader)
        reader.close()
    except Exception as e:
        print(f"Warning: could not verify `{video_path}`: {e}")
        frame_count = -1

    if frame_count != 2:
        print(f"Error: `{video_path}` expected 2 frames but found {frame_count}. Removing file.")
        try:
            os.remove(video_path)
        except Exception:
            pass
    else:
        # --- Captioning logic here ---
        # Save temp images for captioning
        temp_input = os.path.join(output_dir, f"{key}_input_tmp.jpg")
        temp_output = os.path.join(output_dir, f"{key}_output_tmp.jpg")
        imageio.imwrite(temp_input, frame1)
        imageio.imwrite(temp_output, frame2)
        try:
            caption, _, _ = generate_caption(
                temp_input, temp_output, model, processor, max_resolution
            )
        except Exception as e:
            print(f"Captioning failed for `{fname}`: {e}")
            caption = "captioning_error"
        # Clean up temp files
        try:
            os.remove(temp_input)
            os.remove(temp_output)
        except Exception:
            pass

        print(f"Created video: `{video_path}` (key=`{key}`) | Caption: {caption}")
        metadata_rows.append((key, f"videos/{os.path.basename(video_path)}", caption, 0))

    processed += 1

metadata_path = os.path.join(metadata_dir, "metadata.csv")
with open(metadata_path, "w", encoding="utf-8", newline="") as mf:
    mf.write("key,video,prompt,umt5\n")
    for k, v, p, u in metadata_rows:
        k_safe = str(k).replace(",", "")
        v_safe = str(v).replace(",", "")
        p_safe = str(p).replace("\t", " ").replace("\n", " ").replace(",", "")
        mf.write(f"{k_safe},{v_safe},{p_safe},{u}\n")

print(f"Wrote metadata: `{metadata_path}`")