import os
from pathlib import Path
from ultralytics import YOLO

def main():
    # Pfad zum Datensatz
    base_path = Path(r"C:\Users\miche\OneDrive - Flex\4.4_CAS Machine Intelligence\03_Deep_Learning\Leistungsnachweis\Dataset\3_yolo_dataset")

    # YAML-Datei automatisch schreiben
    yaml_path = base_path / "yolo_dataset.yaml"
    with open(yaml_path, 'w') as f:
        f.write(f"""\
path: {base_path.as_posix()}
train: images/train
val: images/val

# Falls du eigene Klassen hast, ersetze sie hier
names:
  0: Chair
  1: Human
  2: DANGER
""")

    # Modell laden
    model = YOLO("yolo11n.pt")

    # Training starten
    model.train(
        data=str(yaml_path),
        epochs=15,
        imgsz=640,
        batch=0.9,
        device=0,  # oder 'cpu'
        project="runs/train",
        name="yolo11n_custom",
        workers=0
    )

if __name__ == "__main__":
    main()