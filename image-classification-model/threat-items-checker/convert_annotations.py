import os
import glob

ANNOTATION_DIR = 'E:/facultate/licenta/implementation/diagramsGit/BachelorsThesisFrontend/image-classification-model/OPIXray/train/train_annotation'
OUTPUT_DIR = 'E:/facultate/licenta/implementation/diagramsGit/BachelorsThesisFrontend/image-classification-model/OPIXray/train/annotation'
IMAGE_DIR = 'E:/facultate/licenta/implementation/diagramsGit/BachelorsThesisFrontend/image-classification-model/OPIXray/train/train_image'

os.makedirs(OUTPUT_DIR, exist_ok=True)

class_dict = {
    "RazorBlade": 0,
    "SafetyPin": 1,
    "PaperClip": 2,
    "Pen": 3,
    "ThinNail": 4,
    "Screw": 5,
    "HandGun": 6,
    "VgaConnector": 7,
    "Straight_Knife": 8,
    "MultiPurposeKnife": 9,
    "Key": 10,
    "Plier": 11,
    "Shuriken": 12,
    "Scissor": 13,

}

annotation_files = glob.glob(os.path.join(ANNOTATION_DIR, "*.txt"))

for ann_file in annotation_files:
    if ann_file.endswith('train.txt') or ann_file.endswith('valid.txt') or ann_file.endswith('classes.txt'):
        continue

    base_name = os.path.basename(ann_file)
    output_file = os.path.join(OUTPUT_DIR, base_name)

    with open(ann_file, 'r') as f_in, open(output_file, 'w') as f_out:
        lines = f_in.readlines()
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 6:
                img_name = parts[0]
                class_name = parts[1]
                x1 = float(parts[2])
                y1 = float(parts[3])
                x2 = float(parts[4])
                y2 = float(parts[5])

                try:
                    img_path = os.path.join(IMAGE_DIR, img_name)
                    with Image.open(img_path) as img:
                        img_width, img_height = img.size
                except Exception as e:
                    print(f"Error opening image {img_path}: {e}")
                    # Fallback to default dimensions if there's an error
                    img_width, img_height = 1920, 1080
                    print(f"Using default dimensions {img_width}x{img_height} for {img_name}")

                # Calculate YOLO format values (normalized)
                x_center = ((x1 + x2) / 2) / img_width
                y_center = ((y1 + y2) / 2) / img_height
                width = (x2 - x1) / img_width
                height = (y2 - y1) / img_height

                class_id = class_dict.get(class_name, -1)
                if class_id == -1:
                    print(f"Unknown class '{class_name}' in {ann_file}")
                    continue

                f_out.write(f"{class_id} {x_center} {y_center} {width} {height}\n")

print("completed!")