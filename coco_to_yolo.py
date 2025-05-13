import json
import os

# Pfade
json_path = r"C:\Users\miche\OneDrive - Flex\4.4_CAS Machine Intelligence\03_Deep_Learning\Leistungsnachweis\Dataset\3_Augmented2\instances_train_aug.json"
output_dir = 'yolo_labels'
os.makedirs(output_dir, exist_ok=True)

# JSON laden
with open(json_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Dictionaries für schnellen Zugriff
image_id_to_info = {img['id']: img for img in data['images']}
annotations = data['annotations']

# Optional: COCO Kategorien in ein Mapping umwandeln (id -> 0-indexierte class_id)
category_id_to_class_id = {cat['id']: idx for idx, cat in enumerate(data['categories'])}

# YOLO-Dateien erstellen
from collections import defaultdict
annotations_per_image = defaultdict(list)
for ann in annotations:
    annotations_per_image[ann['image_id']].append(ann)

for image_id, anns in annotations_per_image.items():
    img_info = image_id_to_info[image_id]
    file_name = img_info['file_name']
    width, height = img_info['width'], img_info['height']

    yolo_lines = []
    for ann in anns:
        cat_id = ann['category_id']
        class_id = category_id_to_class_id[cat_id]
        bbox = ann['bbox']  # [x_min, y_min, width, height]
        x_center = (bbox[0] + bbox[2] / 2) / width
        y_center = (bbox[1] + bbox[3] / 2) / height
        norm_width = bbox[2] / width
        norm_height = bbox[3] / height

        yolo_line = f"{class_id} {x_center:.6f} {y_center:.6f} {norm_width:.6f} {norm_height:.6f}"
        yolo_lines.append(yolo_line)

    # TXT-Datei speichern
    base_filename = os.path.splitext(file_name)[0]
    txt_path = os.path.join(output_dir, f"{base_filename}.txt")
    with open(txt_path, 'w', encoding='utf-8') as txt_file:
        txt_file.write('\n'.join(yolo_lines))
