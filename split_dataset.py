import os
import json
import shutil
from sklearn.model_selection import train_test_split

# Pfade
base_dir     = "."  # Augmented/
images_dir   = os.path.join(base_dir, "images_aug")
anno_file    = os.path.join(base_dir, "instances_train_aug.json")
split_dir    = os.path.join(base_dir, "splits")

# Split-Verhältnisse
train_frac = 0.7 # Training
val_frac   = 0.2  # Test ist dann automatisch 1 - train_frac - val_frac

# Lade COCO-Annotation
with open(anno_file) as f:
    coco = json.load(f)

# Liste aller Images
all_images = coco["images"]

# Zuerst: Train vs. Rest (Val+Test)
train_imgs, rest = train_test_split(
    all_images,
    train_size=train_frac,
    random_state=42
)

# Dann Rest aufteilen in Val und Test
val_relative = val_frac / (1 - train_frac)
val_imgs, test_imgs = train_test_split(
    rest,
    train_size=val_relative,
    random_state=42
)

# Funktion zum Filtern und Speichern
def save_split(images_subset, split_name):
    # IDs dieses Splits
    ids = {img["id"] for img in images_subset}
    # Filtere Annotationen
    anns = [a for a in coco["annotations"] if a["image_id"] in ids]
    out = {
        "images": images_subset,
        "annotations": anns,
        "categories": coco["categories"]
    }
    # JSON speichern
    out_path = os.path.join(split_dir, "annotations", f"instances_{split_name}.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    # Bilder kopieren
    dest_dir = os.path.join(split_dir, "images", split_name)
    for img in images_subset:
        src = os.path.join(images_dir, img["file_name"])
        dst = os.path.join(dest_dir,   img["file_name"])
        shutil.copy(src, dst)

# Splits speichern
save_split(train_imgs, "train")
save_split(val_imgs,   "val")
save_split(test_imgs,  "test")

print("Split abgeschlossen:")
print(f"- Training:   {len(train_imgs)} Bilder")
print(f"- Validation: {len(val_imgs)} Bilder")
print(f"- Test:       {len(test_imgs)} Bilder")
