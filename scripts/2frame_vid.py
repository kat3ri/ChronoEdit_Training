import os
import cv2
import numpy as np
import imageio

# Import captioning logic
from data_captioning import load_model, generate_caption

# Configuration
source_dir = r"/workspace/data/skin_edit/images/input-images"
target_dir = r"/workspace/data/skin_edit/images/output-images"
output_dir = r"/workspace/data/skin_edit/videos"
metadata_dir = r"/workspace/data/skin_edit"

fps = 1
max_pairs = 5

# Captioning model config
caption_model_name = "Qwen/Qwen3-VL-30B-A3B-Instruct"
max_resolution = 1080

os.makedirs(output_dir, exist_ok=True)

files1 = set([f for f in os.listdir(source_dir) if not f.startswith('.')])
files2 = set([f for f in os.listdir(target_dir) if not f.startswith('.')])
matching = sorted(files1 & files2)

def build_basename_map(directory):
    return {os.path.splitext(f)[0]: f for f in os.listdir(directory) if not f.startswith('.')}

source_map = build_basename_map(source_dir)
target_map = build_basename_map(target_dir)
matching_keys = sorted(source_map.keys() & target_map.keys())


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
detector_available = not face_cascade.empty()

metadata_rows = []

# Load captioning model and processor once
model, processor = load_model(caption_model_name)

print(f"Loading captioning model: {caption_model_name}")
model, processor = load_model(caption_model_name)
print("Captioning model loaded.")

processed = 0
for key in matching_keys:
    print(f"\nProcessing pair: {key}")
    if max_pairs is not None and processed >= max_pairs:
        print(f"Reached processing limit: {max_pairs} pairs.")
        break

    src_path = os.path.join(source_dir, source_map[key])
    tgt_path = os.path.join(target_dir, target_map[key])
    video_path = os.path.join(output_dir, f"{key}.mp4")

    img1 = cv2.imread(src_path, cv2.IMREAD_UNCHANGED)
    img2 = cv2.imread(tgt_path, cv2.IMREAD_UNCHANGED)
    if img1 is None or img2 is None:
        print(f"Skipping `{fname}`: could not read image.")
        processed += 1
        continue

    if img1.ndim == 3 and img1.shape[2] == 4:
        print("Converting img1 from BGRA to BGR")
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGRA2BGR)
    if img2.ndim == 3 and img2.shape[2] == 4:
        print("Converting img2 from BGRA to BGR")
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGRA2BGR)


    height, width = img1.shape[:2]
    target_ar = width / height

    h2, w2 = img2.shape[:2]
    face_cx, face_cy = w2 // 2, h2 // 2
    if detector_available and img2.ndim == 3 and img2.shape[2] >= 3:
        try:
            print("Detecting face in img2...")
            gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            if len(faces) > 0:
                x, y, fw, fh = max(faces, key=lambda r: r[2] * r[3])
                face_cx = x + fw // 2
                face_cy = y + fh // 2
                print(f"Face detected at ({face_cx}, {face_cy})")
            else:
                print("No face detected.")
        except Exception as e:
            print(f"Face detection error: {e}")


    crop_w = w2
    crop_h = int(round(crop_w / target_ar))
    if crop_h > h2:
        crop_h = h2
        crop_w = int(round(crop_h * target_ar))
    crop_w = min(crop_w, w2)
    crop_h = min(crop_h, h2)

    x1 = int(max(0, min(face_cx - crop_w // 2, w2 - crop_w)))
    y1 = int(max(0, min(face_cy - crop_h // 2, h2 - crop_h)))
    print(f"Cropping img2: x1={x1}, y1={y1}, crop_w={crop_w}, crop_h={crop_h}")
    cropped = img2[y1:y1 + crop_h, x1:x1 + crop_w]

    print(f"Resizing cropped img2 to ({width}, {height})")
    img2_resized = cv2.resize(cropped, (width, height), interpolation=cv2.INTER_AREA)

    def to_rgb(img):
        if img.ndim == 2:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    print("Converting images to RGB")
    frame1 = to_rgb(img1)
    frame2 = to_rgb(img2_resized)

    os.makedirs(os.path.dirname(video_path), exist_ok=True)
    print(f"Writing video: {video_path}")
    with imageio.get_writer(video_path, fps=fps) as writer:
        writer.append_data(frame1)
        writer.append_data(frame2)

    try:
        print(f"Verifying video frame count: {video_path}")
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
        except Exception as e:
            print(f"Failed to remove file: {e}")
    else:
        temp_input = os.path.join(output_dir, f"{key}_input_tmp.jpg")
        temp_output = os.path.join(output_dir, f"{key}_output_tmp.jpg")
        print(f"Saving temp images for captioning: {temp_input}, {temp_output}")
        imageio.imwrite(temp_input, frame1)
        imageio.imwrite(temp_output, frame2)
        try:
            print("Generating caption...")
            caption, _, _ = generate_caption(
                temp_input, temp_output, model, processor, max_resolution
            )
            print(f"Caption generated: {caption}")
        except Exception as e:
            print(f"Captioning failed for `{fname}`: {e}")
            caption = "captioning_error"
        try:
            print("Cleaning up temp images.")
            os.remove(temp_input)
            os.remove(temp_output)
        except Exception as e:
            print(f"Failed to remove temp images: {e}")

        print(f"Created video: `{video_path}` (key=`{key}`) | Caption: {caption}")
        metadata_rows.append((key, f"videos/{os.path.basename(video_path)}", caption, 0))

    processed += 1

metadata_path = os.path.join(metadata_dir, "metadata.csv")
print(f"Writing metadata: {metadata_path}")
with open(metadata_path, "w", encoding="utf-8", newline="") as mf:
    mf.write("key,video,prompt,umt5\n")
    for k, v, p, u in metadata_rows:
        k_safe = str(k).replace(",", "")
        v_safe = str(v).replace(",", "")
        p_safe = str(p).replace("\t", " ").replace("\n", " ").replace(",", "")
        mf.write(f"{k_safe},{v_safe},{p_safe},{u}\n")

print(f"Wrote metadata: `{metadata_path}`")