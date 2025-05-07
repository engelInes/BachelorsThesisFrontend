import os
import random

import torch
from pathlib import Path

from sklearn.model_selection import train_test_split
from ultralytics import YOLO
import cv2
import numpy as np
import albumentations as A
import shutil
import sys

SEED = 42

np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def check_and_prepare_dataset(dataset_path):
    """
    Verifies and prepares the dataset structure required by YOLO
    """
    print(f"\n===== Checking Dataset Structure =====")

    train_img_dir = os.path.join(dataset_path, 'train', 'images')
    train_label_dir = os.path.join(dataset_path, 'train', 'labels')
    valid_img_dir = os.path.join(dataset_path, 'valid', 'images')
    valid_label_dir = os.path.join(dataset_path, 'valid', 'labels')
    test_img_dir = os.path.join(dataset_path, 'test', 'images')
    test_label_dir = os.path.join(dataset_path, 'test', 'labels')

    required_dirs = [
        train_img_dir, train_label_dir,
        valid_img_dir, valid_label_dir,
    ]

    for directory in required_dirs:
        if not os.path.exists(directory):
            print(f"Creating missing directory: {directory}")
            os.makedirs(directory, exist_ok=True)

    for directory in [test_img_dir, test_label_dir]:
        if not os.path.exists(directory):
            print(f"Optional directory not found (this is OK): {directory}")

    if os.path.exists(train_img_dir) and not os.listdir(valid_img_dir):
        print("Validation directory is empty. Creating validation set from 10% of training data...")
        create_validation_set(dataset_path)

    train_img_count = len([f for f in os.listdir(train_img_dir) if f.endswith(('.jpg', '.png', '.jpeg'))])
    train_label_count = len([f for f in os.listdir(train_label_dir) if f.endswith('.txt')])
    valid_img_count = len([f for f in os.listdir(valid_img_dir) if f.endswith(('.jpg', '.png', '.jpeg'))])
    valid_label_count = len([f for f in os.listdir(valid_label_dir) if f.endswith('.txt')])

    print(f"Training images: {train_img_count}, labels: {train_label_count}")
    print(f"Validation images: {valid_img_count}, labels: {valid_label_count}")

    if train_img_count == 0 or valid_img_count == 0:
        print("ERROR: No images found in training or validation directories")
        return False

    yaml_path = os.path.join(dataset_path, 'dataset.yaml')
    print(f"Creating/updating dataset.yaml at {yaml_path}")

    with open(yaml_path, 'w') as f:
        f.write(f"# Dataset configuration for X-ray threat detection\n\n")
        f.write(f"# Path to dataset root directory\n")
        f.write(f"path: {dataset_path}\n\n")
        f.write(f"# Training/validation sets\n")
        f.write(f"train: train/images\n")
        f.write(f"val: valid/images\n")
        f.write(f"test: test/images\n\n")
        f.write(f"# Number of classes\n")
        f.write(f"nc: 6\n\n")
        f.write(f"# Class names\n")
        f.write(f"names: ['gun', 'knife', 'pin', 'razor', 'shuriken', 'snail']\n")

    print(f"Dataset structure verified and prepared successfully")
    return True


def create_validation_set(dataset_path, validation_split=0.1):
    """
    Creates a validation set by moving a portion of training data
    """
    train_img_dir = os.path.join(dataset_path, 'train', 'images')
    train_label_dir = os.path.join(dataset_path, 'train', 'labels')
    valid_img_dir = os.path.join(dataset_path, 'valid', 'images')
    valid_label_dir = os.path.join(dataset_path, 'valid', 'labels')

    train_images = [f for f in os.listdir(train_img_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]

    train_subset, val_subset = train_test_split(
        train_images,
        test_size=validation_split,
        random_state=SEED
    )

    val_images = val_subset
    num_val_images = len(val_images)
    print(f"Moving {num_val_images} images to validation set")
    # num_val_images = max(1, int(len(train_images) * validation_split))
    # print(f"Moving {num_val_images} images to validation set")

    # import random
    # random.shuffle(train_images)
    # val_images = train_images[:num_val_images]

    for img_file in val_images:
        img_src = os.path.join(train_img_dir, img_file)
        img_dst = os.path.join(valid_img_dir, img_file)
        shutil.move(img_src, img_dst)

        label_file = os.path.splitext(img_file)[0] + '.txt'
        label_src = os.path.join(train_label_dir, label_file)
        label_dst = os.path.join(valid_label_dir, label_file)
        if os.path.exists(label_src):
            shutil.move(label_src, label_dst)


def select_training_subset(dataset_path, percentage=1):
    """
    Creates a subset of training data by moving a portion of images to a temporary folder

    Args:
        dataset_path: Path to the dataset directory
        percentage: Percentage of training data to use (1-100)

    Returns:
        Tuple of (temp_train_dir, temp_label_dir) containing the subset paths
    """
    print(f"\n===== Creating Training Subset ({percentage}% of data) =====")

    # Validate percentage
    percentage = max(1, min(100, percentage))

    # Create temp directory for subset
    subset_dir = os.path.join(dataset_path, 'subset')
    subset_img_dir = os.path.join(subset_dir, 'images')
    subset_label_dir = os.path.join(subset_dir, 'labels')

    # Clean any existing subset
    if os.path.exists(subset_dir):
        shutil.rmtree(subset_dir)

    os.makedirs(subset_img_dir, exist_ok=True)
    os.makedirs(subset_label_dir, exist_ok=True)

    # Source directories
    train_img_dir = os.path.join(dataset_path, 'train', 'images')
    train_label_dir = os.path.join(dataset_path, 'train', 'labels')

    train_images = [f for f in os.listdir(train_img_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]

    # Calculate how many images to include
    total_images = len(train_images)
    subset_size = 2000 #max(1, int(total_images * percentage / 100))

    print(f"Total training images: {total_images}")
    print(f"Selecting {subset_size} images ({percentage}%) for training subset")

    # Randomly select images
    random.seed(SEED)
    selected_images = random.sample(train_images, subset_size)

    # Copy selected images and their labels to subset directory
    for img_file in selected_images:
        img_src = os.path.join(train_img_dir, img_file)
        img_dst = os.path.join(subset_img_dir, img_file)
        shutil.copy(img_src, img_dst)

        label_file = os.path.splitext(img_file)[0] + '.txt'
        label_src = os.path.join(train_label_dir, label_file)
        label_dst = os.path.join(subset_label_dir, label_file)
        if os.path.exists(label_src):
            shutil.copy(label_src, label_dst)

    print(f"Created training subset with {len(os.listdir(subset_img_dir))} images")

    # Create subset YAML file
    subset_yaml_path = os.path.join(subset_dir, 'subset.yaml')
    with open(subset_yaml_path, 'w') as f:
        f.write(f"# Subset configuration for X-ray threat detection\n\n")
        f.write(f"# Path to dataset root directory\n")
        f.write(f"path: {dataset_path}\n\n")
        f.write(f"# Training/validation sets\n")
        f.write(f"train: subset/images\n")
        f.write(f"val: valid/images\n")
        f.write(f"test: test/images\n\n")
        f.write(f"# Number of classes\n")
        f.write(f"nc: 6\n\n")
        f.write(f"# Class names\n")
        f.write(f"names: ['gun', 'knife', 'pin', 'razor', 'shuriken', 'snail']\n")

    return subset_yaml_path
def augment_dataset(dataset_path, augmentation_factor=2):
    """
    Augment the dataset by creating modified copies of images and labels
    """
    print(f"\n===== Starting Dataset Augmentation (Factor: {augmentation_factor}) =====")

    images_path = os.path.join(dataset_path, 'train', 'images')
    labels_path = os.path.join(dataset_path, 'train', 'labels')

    if not os.path.exists(images_path) or not os.path.exists(labels_path):
        print(f"Train folders not found: {images_path} / {labels_path}")
        return

    transform = A.Compose([
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.7),
        A.Blur(blur_limit=5, p=0.5),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=15, p=0.5, border_mode=cv2.BORDER_CONSTANT),
        A.RandomScale(scale_limit=0.15, p=0.5),
        A.RandomGamma(gamma_limit=(80, 120), p=0.5),
        A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.3),
        # A.RandomBrightnessContrast(p=0.7),
        # A.Blur(blur_limit=5, p=0.5),
        # A.GaussNoise(p=0.5),
        # A.HorizontalFlip(p=0.5),
        # A.Rotate(limit=15, p=0.5, border_mode=cv2.BORDER_CONSTANT),
        # A.RandomScale(scale_limit=0.15, p=0.5),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

    orig_image_files = [f for f in os.listdir(images_path) if f.endswith(('.jpg', '.png', '.jpeg'))]

    print(f"Found {len(orig_image_files)} original images")
    print(f"Target: ~{len(orig_image_files) * augmentation_factor} new augmented images")

    augmented_count = 0
    skipped_count = 0

    for idx, img_file in enumerate(orig_image_files):
        try:
            img_path = os.path.join(images_path, img_file)
            # label_path = os.path.join(labels_path, Path(img_file).stem + '.txt')
            base_name = os.path.splitext(img_file)[0]
            label_path = os.path.join(labels_path, base_name + '.txt')

            if not os.path.exists(label_path):
                continue

            image = cv2.imread(img_path)
            if image is None:
                continue

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            bboxes = []
            class_labels = []

            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(float(parts[0]))
                        x, y, w, h = map(float, parts[1:5])

                        x = max(0.001, min(0.999, x))
                        y = max(0.001, min(0.999, y))
                        w = min(w, 2 * (min(x, 1 - x)))
                        h = min(h, 2 * (min(y, 1 - y)))

                        bboxes.append([x, y, w, h])
                        class_labels.append(class_id)

            if not bboxes:
                continue

            for aug_idx in range(augmentation_factor):
                try:
                    augmented = transform(image=image, bboxes=bboxes, class_labels=class_labels)

                    if not augmented['bboxes']:
                        continue

                    aug_img_file = f"{base_name}_aug{aug_idx}{os.path.splitext(img_file)[1]}"
                    aug_img_path = os.path.join(images_path, aug_img_file)
                    aug_label_path = os.path.join(labels_path, f"{base_name}_aug{aug_idx}.txt")
                    # aug_img_file = f"{Path(img_file).stem}_aug{aug_idx}{Path(img_file).suffix}"
                    # aug_img_path = os.path.join(images_path, aug_img_file)
                    # aug_label_path = os.path.join(labels_path, f"{Path(img_file).stem}_aug{aug_idx}.txt")

                    cv2.imwrite(aug_img_path, cv2.cvtColor(augmented['image'], cv2.COLOR_RGB2BGR))

                    with open(aug_label_path, 'w') as f:
                        for bbox, class_id in zip(augmented['bboxes'], augmented['class_labels']):
                            x, y, w, h = bbox
                            f.write(f"{class_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

                    augmented_count += 1

                except Exception as e:
                    print(f"Error during augmentation of {img_file}, aug {aug_idx}: {str(e)}")
                    skipped_count += 1
                    continue

            if (idx + 1) % 10 == 0:
                print(f"Progress: {idx + 1}/{len(orig_image_files)} images processed")

        except Exception as e:
            print(f"Error processing {img_file}: {str(e)}")
            skipped_count += 1
            continue

    print(f"\n===== Dataset Augmentation Complete =====")
    print(f"Added {augmented_count} augmented images")
    print(f"Skipped {skipped_count} problematic augmentations")


def train_model(dataset_path, epochs=20, img_size=640, batch_size=8, project_name="xray_threat_detector"):
    """
    Train the YOLO model on the dataset and save checkpoints

    Args:
        dataset_path: Path to the dataset directory
        epochs: Number of training epochs
        img_size: Input image size
        batch_size: Batch size for training
        project_name: Name of the project for saving results
    """
    print(f"\n===== Starting Model Training =====")

    if not check_and_prepare_dataset(dataset_path):
        print("Dataset preparation failed. Please fix the issues and try again.")
        return None

    yaml_path = select_training_subset(dataset_path, percentage=10)
    yaml_path = yaml_path
    #yaml_path = os.path.join(dataset_path, 'dataset.yaml')
    if not os.path.exists(yaml_path):
        print(f"ERROR: dataset.yaml not found at {yaml_path}")
        return None

    model = YOLO('yolov8n.pt')

    device = 0 if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    print(f"Dataset: {dataset_path}")
    print(f"Epochs: {epochs}")
    print(f"Image size: {img_size}")
    print(f"Batch size: {batch_size}")

    try:
        results = model.train(
            data=yaml_path,
            epochs=epochs,
            imgsz=img_size,
            batch=batch_size,
            name=project_name,
            verbose=True,
            device=device,
            cache=True,
            patience=15,
            save=True,
            save_period=3,
            lr0=0.01,
            lrf=0.001,
            cos_lr=True,
            optimizer='Adam',
            momentum=0.937,
            weight_decay=0.0005,
            warmup_epochs=3.0,
            warmup_momentum=0.8,
            warmup_bias_lr=0.1,
            augment=True,
            mixup=0.1,
            mosaic=0.8,
            copy_paste=0.1,
            degrees=10.0,
            translate=0.1,
            scale=0.5,
            shear=0.0,
            perspective=0.0,
            flipud=0.0,
            fliplr=0.5,
            hsv_h=0.015,
            hsv_s=0.5,
            hsv_v=0.3,
            close_mosaic=10,
        )

        best_model_path = os.path.join(model.trainer.save_dir, 'weights', 'best.pt')
        last_model_path = os.path.join(model.trainer.save_dir, 'weights', 'last.pt')

        if os.path.exists(best_model_path):
            output_model_path = os.path.join(os.path.dirname(dataset_path), 'xray_threat_detector.pt')
            shutil.copy(best_model_path, output_model_path)
            print(f"Best model saved to: {output_model_path}")

        print(f"\n===== Training Complete =====")
        print(f"Training results saved to: {model.trainer.save_dir}")
        print(f"Best model path: {best_model_path}")
        print(f"Last model path: {last_model_path}")

        subset_dir = os.path.join(dataset_path, 'subset')
        if os.path.exists(subset_dir):
            print(f"Cleaning up temporary subset directory: {subset_dir}")
            shutil.rmtree(subset_dir)

        return results

    except Exception as e:
        print(f"Error during training: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def evaluate_model(model_path, dataset_path):
    """
    Evaluate the trained model on validation set
    """
    print(f"\n===== Evaluating Model =====")

    model = YOLO(model_path)

    valid_path = os.path.join(dataset_path, 'valid', 'images')

    if not os.path.exists(valid_path):
        print(f"Validation path not found: {valid_path}")
        return

    try:
        results = model.val(data=os.path.join(dataset_path, 'dataset.yaml'))

        print("\n===== Evaluation Results =====")
        print(f"mAP50-95: {results.box.map}")
        print(f"mAP50: {results.box.map50}")
        print(f"Precision: {results.box.p}")
        print(f"Recall: {results.box.r}")

        return results

    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    dataset_path = "./dataset/roboflow"

    do_augment = input("Do you want to augment the dataset? (y/n): ").lower() == 'y'
    if do_augment:
        aug_factor = int(input("Enter augmentation factor (1-5 recommended): "))
        augment_dataset(dataset_path, augmentation_factor=aug_factor)

    epochs = int(input("Enter number of training epochs (10-100 recommended): "))
    batch_size = int(input("Enter batch size (4-16 recommended): "))

    train_model(
        dataset_path=dataset_path,
        epochs=epochs,
        img_size=640,
        batch_size=batch_size,
        project_name="xray_threat_detector"
    )