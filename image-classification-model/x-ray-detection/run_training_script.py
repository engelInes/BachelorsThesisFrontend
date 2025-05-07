#!/usr/bin/env python
# Direct execution script for X-Ray threat detection training
# This script runs immediately with preset values

from train_model import augment_dataset, train_model
import os
import sys

# Set the dataset path - change this to your actual path
dataset_path = "E:/facultate/licenta/implementation/diagramsGit/BachelorsThesisFrontend/image-classification-model/x-ray-detection/dataset/roboflow"

# Configuration parameters
AUGMENT = False  # Set to True to augment the dataset
AUGMENTATION_FACTOR = 2  # Number of augmented copies per original image
EPOCHS = 10  # Number of training epochs
IMG_SIZE = 640  # Input image size
BATCH_SIZE = 8  # Training batch size
PROJECT_NAME = "xray_threat_detector"  # Project name for saving results


def main():
    print("X-Ray Threat Detection - Training Script")
    print(f"Dataset path: {dataset_path}")

    # Verify dataset path exists
    if not os.path.exists(dataset_path):
        print(f"ERROR: Dataset path does not exist: {dataset_path}")
        print("Please update the dataset_path variable in this script")
        return

    # Check if the expected subdirectories exist
    train_img_dir = os.path.join(dataset_path, 'train', 'images')
    train_label_dir = os.path.join(dataset_path, 'train', 'labels')
    valid_img_dir = os.path.join(dataset_path, 'valid', 'images')
    valid_label_dir = os.path.join(dataset_path, 'valid', 'labels')

    for subdir in [train_img_dir, train_label_dir, valid_img_dir, valid_label_dir]:
        if not os.path.exists(subdir):
            print(f"WARNING: Expected subdirectory not found: {subdir}")
            user_input = input(f"Create directory {subdir}? (y/n): ")
            if user_input.lower() == 'y':
                os.makedirs(subdir, exist_ok=True)
                print(f"Created directory: {subdir}")
            else:
                print("Directory structure incomplete. Exiting.")
                sys.exit(1)

    # Perform dataset augmentation if enabled
    if AUGMENT:
        print(f"Performing dataset augmentation with factor: {AUGMENTATION_FACTOR}")
        try:
            augment_dataset(dataset_path, augmentation_factor=AUGMENTATION_FACTOR)
        except Exception as e:
            print(f"ERROR during augmentation: {str(e)}")
            print("Continuing with training using non-augmented dataset...")

    # Train the model
    print(f"Starting model training with epochs={EPOCHS}, batch_size={BATCH_SIZE}")
    train_model(
        dataset_path=dataset_path,
        epochs=EPOCHS,
        img_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        project_name=PROJECT_NAME
    )

    print("\nTraining process completed. Check the output directory for results.")


if __name__ == "__main__":
    main()