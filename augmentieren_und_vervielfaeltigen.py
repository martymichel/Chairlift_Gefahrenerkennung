import os
import json
import numpy as np
from PIL import Image
import albumentations as A
from itertools import product

# Pfade anpassen
images_dir      = r"C:\Users\miche\OneDrive - Flex\4.4_CAS Machine Intelligence\03_Deep_Learning\Leistungsnachweis\Dataset\2_Annotated\images"
annotation_file = r"C:\Users\miche\OneDrive - Flex\4.4_CAS Machine Intelligence\03_Deep_Learning\Leistungsnachweis\Dataset\2_Annotated\instances.json"
output_img_dir  = r"C:\Users\miche\OneDrive - Flex\4.4_CAS Machine Intelligence\03_Deep_Learning\Leistungsnachweis\Dataset\3_Augmented2\images_aug"
output_ann      = r"C:\Users\miche\OneDrive - Flex\4.4_CAS Machine Intelligence\03_Deep_Learning\Leistungsnachweis\Dataset\3_Augmented2\instances_train_aug.json"

# Ausgabeordner erstellen
os.makedirs(output_img_dir, exist_ok=True)

# COCO-Annotation laden
with open(annotation_file) as f:
    coco = json.load(f)

new_images = []
new_annotations = []
ann_id = 1
img_counter = 1

# Parameter-Sets für full-factorial Design
rotate_limits     = [-6, 0, 6]
hue_limits        = [0, 20, 40]
brightness_limits = [0.0, 0.2, 0.4]
flips             = [False, True]

# Alle Kombinationen berechnen
param_grid = list(product(
    rotate_limits,
    hue_limits,
    brightness_limits,
    flips
))

for rot, hue, bri, flip in param_grid:
    # Dynamische Pipeline
    transforms = []
    if rot != 0:
        transforms.append(A.Rotate(limit=rot, p=1.0))
    if hue != 0:
        transforms.append(A.HueSaturationValue(hue_shift_limit=hue, sat_shift_limit=0, val_shift_limit=0, p=1.0))
    if bri != 0:
        transforms.append(A.RandomBrightnessContrast(brightness_limit=bri, contrast_limit=0, brightness_by_max=True, p=1.0))
    if flip:
        transforms.append(A.HorizontalFlip(p=1.0))
    
    pipeline = A.Compose(transforms,
                         bbox_params=A.BboxParams(format='coco', label_fields=['category_ids']))
    
    # Jede Kombination auf alle Bilder anwenden
    for img_info in coco['images']:
        img_path = os.path.join(images_dir, img_info['file_name'])
        image = np.array(Image.open(img_path).convert("RGB"))
        
        # Bboxes und Kategorien pro Bild
        anns = [a for a in coco['annotations'] if a['image_id']==img_info['id']]
        bboxes = [a['bbox'] for a in anns]
        cat_ids = [a['category_id'] for a in anns]
        
        # Augmentation anwenden
        aug = pipeline(image=image, bboxes=bboxes, category_ids=cat_ids)
        aug_img = Image.fromarray(aug['image'])
        
        # Dateiname mit Parametern
        combo_name = f"r{rot}_h{hue}_b{int(bri*100)}_" + \
                     f"f{int(flip)}"
        new_fname = f"{os.path.splitext(img_info['file_name'])[0]}_{combo_name}.jpg"
        aug_img.save(os.path.join(output_img_dir, new_fname))
        
        # Neue Image-Entry
        new_images.append({
            "id": img_counter,
            "file_name": new_fname,
            "width": img_info['width'],
            "height": img_info['height']
        })
        
        # Neue Annotations
        for bbox, cid in zip(aug['bboxes'], aug['category_ids']):
            new_annotations.append({
                "id": ann_id,
                "image_id": img_counter,
                "category_id": cid,
                "bbox": [round(float(x), 2) for x in bbox],
                "iscrowd": 0
            })
            ann_id += 1
        
        img_counter += 1

# COCO-Struktur speichern
aug_coco = {
    "images": new_images,
    "annotations": new_annotations,
    "categories": coco['categories']
}
os.makedirs(os.path.dirname(output_ann), exist_ok=True)
with open(output_ann, "w") as f:
    json.dump(aug_coco, f, indent=2)

print("Full-factorial Augmentation abgeschlossen:")
print(f"- {len(new_images)} Bilder erzeugt")
print(f"- {len(new_annotations)} Annotationen erzeugt")
