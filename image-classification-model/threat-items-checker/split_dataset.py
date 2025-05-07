import os
import shutil
import random
from pathlib import Path

class_map = {
    'Knife': 0,
    'Gun': 1,
    'Syringe': 2,
    'Scissors': 3,
    'Screwdriver': 4
}

new_dataset_base = 'E:/facultate/licenta/implementation/diagramsGit/BachelorsThesisFrontend/image-classification-model/threat-items-checker/merged-dataset'
for class_name in class_map:
    os.makedirs(f'{new_dataset_base}/{class_name}/train/images', exist_ok=True)
    os.makedirs(f'{new_dataset_base}/{class_name}/train/labels', exist_ok=True)
    os.makedirs(f'{new_dataset_base}/{class_name}/test/images', exist_ok=True)
    os.makedirs(f'{new_dataset_base}/{class_name}/test/labels', exist_ok=True)

print("Created directory structure")

base_dir = 'E:/facultate/licenta/implementation/diagramsGit/BachelorsThesisFrontend/image-classification-model/threat-items-checker/mobilenet_v2_model/weapons_data/guns-knives-yolo'

for split_name in ['train', 'test', 'valid']:
    images_dir = f'{base_dir}/{split_name}/images'
    labels_dir = f'{base_dir}/{split_name}/labels'

    if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
        print(f"Checking alternative path structure for {split_name}...")
        images_dir = f'{base_dir}/guns-knives-yolo/{split_name}/images'
        labels_dir = f'{base_dir}/guns-knives-yolo/{split_name}/labels'

    if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
        print(f"Directory not found for {split_name}. Skipping...")
        continue

    print(f"Processing {split_name} data from {images_dir}")

    for label_file in Path(labels_dir).glob('*.txt'):
        with open(label_file, 'r') as f:
            lines = f.readlines()

        if any(line.strip().split()[0] == '1' for line in lines):
            image_file = None
            for ext in ['.jpg', '.jpeg', '.png']:
                img_path = f'{images_dir}/{label_file.stem}{ext}'
                if os.path.exists(img_path):
                    image_file = img_path
                    break

            if not image_file:
                print(f"Image not found for {label_file.name}")
                continue

            target_split = 'train' if split_name != 'test' else 'test'
            dest_base = f'{new_dataset_base}/Gun/{target_split}'

            dest_img = f'{dest_base}/images/{Path(image_file).name}'
            try:
                shutil.copy(image_file, dest_img)
                print(f"Copied gun image {image_file} to {dest_img}")
            except Exception as e:
                print(f"Error copying {image_file}: {e}")
                continue

            updated = []
            for line in lines:
                parts = line.strip().split()
                if parts[0] == '1':  # Gun class
                    parts[0] = str(class_map['Gun'])
                    updated.append(' '.join(parts))

            dest_label = f'{dest_base}/labels/{label_file.name}'
            with open(dest_label, 'w') as out:
                out.write('\n'.join(updated))

        if any(line.strip().split()[0] == '0' for line in lines):
            image_file = None
            for ext in ['.jpg', '.jpeg', '.png']:
                img_path = f'{images_dir}/{label_file.stem}{ext}'
                if os.path.exists(img_path):
                    image_file = img_path
                    break

            if not image_file:
                print(f"Image not found for {label_file.name}")
                continue

            target_split = 'train' if split_name != 'test' else 'test'
            dest_base = f'{new_dataset_base}/Knife/{target_split}'

            dest_img = f'{dest_base}/images/{Path(image_file).name}'
            try:
                shutil.copy(image_file, dest_img)
                print(f"Copied knife image {image_file} to {dest_img}")
            except Exception as e:
                print(f"Error copying {image_file}: {e}")
                continue

            updated = []
            for line in lines:
                parts = line.strip().split()
                if parts[0] == '0':  # Knife class
                    parts[0] = str(class_map['Knife'])
                    updated.append(' '.join(parts))

            dest_label = f'{dest_base}/labels/{label_file.name}'
            with open(dest_label, 'w') as out:
                out.write('\n'.join(updated))

dataset_base = f'{base_dir}/guns-knives-yolo/Dataset/train'
for class_name in ['Scissors', 'Screwdriver', 'Syringe']:
    img_dir = f'{dataset_base}/{class_name}/images'
    label_dir = f'{dataset_base}/{class_name}/labels'

    if not os.path.exists(img_dir):
        print(f"Image directory not found: {img_dir}")
        continue

    if not os.path.exists(label_dir):
        label_dir = f'{dataset_base}/{class_name}/new_labels'
        if not os.path.exists(label_dir):
            print(f"Label directory not found for {class_name}. Checked both labels and new_labels.")
            continue

    print(f"Processing {class_name} data from {img_dir}")

    img_files = []
    for ext in ['.jpg', '.jpeg', '.png']:
        img_files.extend(list(Path(img_dir).glob(f'*{ext}')))

    if not img_files:
        print(f"No images found in {img_dir}")
        continue

    random.shuffle(img_files)
    split_index = int(len(img_files) * 0.8)

    for i, img_path in enumerate(img_files):
        label_path = Path(label_dir) / f"{img_path.stem}.txt"
        if not label_path.exists():
            print(f"Label not found for {img_path.name}")
            continue

        split = 'train' if i < split_index else 'test'
        dest_img_path = f'{new_dataset_base}/{class_name}/{split}/images/{img_path.name}'
        dest_label_path = f'{new_dataset_base}/{class_name}/{split}/labels/{label_path.name}'

        try:
            shutil.copy(img_path, dest_img_path)
            print(f"Copied {class_name} image {img_path.name}")
        except Exception as e:
            print(f"Error copying {img_path}: {e}")
            continue

        try:
            with open(label_path, 'r') as f:
                updated = []
                for line in f:
                    parts = line.strip().split()
                    parts[0] = str(class_map[class_name])
                    updated.append(' '.join(parts))

            with open(dest_label_path, 'w') as f:
                f.write('\n'.join(updated))
        except Exception as e:
            print(f"Error processing label {label_path}: {e}")

print("Merging and restructuring complete!")