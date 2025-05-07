import numpy as np
import cv2
import PIL.Image as Image
import os
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
import matplotlib.pylab as plt
import tensorflow as tf
import tensorflow_hub as hub
from pathlib import Path

from keras.mixed_precision import set_global_policy
from sklearn.model_selection import train_test_split
import subprocess
import sys
import json
import random
import datetime


def install_package(package):
    try:
        __import__(package)
    except ImportError:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])


def setup_kaggle():
    kaggle_dir = os.path.expanduser("~/.kaggle")
    os.makedirs(kaggle_dir, exist_ok=True)

    token_path = os.path.join(kaggle_dir, "kaggle.json")
    if not os.path.exists(token_path):
        print("\n=== Kaggle API Setup ===")
        print("To use Kaggle's API, you need to place your API token at ~/.kaggle/kaggle.json")
        print("1. Go to https://www.kaggle.com/settings/account")
        print("2. Scroll down to 'API' section and click 'Create New API Token'")
        print("3. Save the downloaded 'kaggle.json' file")

        token_location = input("Enter the full path to your downloaded kaggle.json file: ")

        if os.path.exists(token_location):
            import shutil
            shutil.copy(token_location, token_path)
            os.chmod(token_path, 0o600)
            print("Kaggle API token configured successfully!")
        else:
            print(f"Error: File not found at {token_location}")
            print("Please download your token and try again.")
            sys.exit(1)

    return True
def download_dataset():
    print("\n=== Downloading Dataset ===")

    data_dir = 'weapons_data'
    os.makedirs(data_dir, exist_ok=True)

    if os.path.exists(os.path.join(data_dir, 'guns-knives-yolo')):
        print("Dataset already exists. Skipping download.")
        return data_dir

    print("Downloading dataset from Kaggle...")
    try:
        subprocess.check_call(['kaggle', 'datasets', 'download', '-d', 'iqmansingh/guns-knives-object-detection'])

        print("Extracting dataset...")
        import zipfile
        with zipfile.ZipFile('guns-knives-object-detection.zip', 'r') as zip_ref:
            zip_ref.extractall(data_dir)
        os.remove('guns-knives-object-detection.zip')
        print("Dataset downloaded and extracted successfully!")

        return data_dir
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        return None
def load_merged_data(base_dir='E:/facultate/licenta/implementation/diagramsGit/BachelorsThesisFrontend/image-classification-model/threat-items-checker/merged-dataset'):
    """
    Load data from the merged dataset structure with multiple classes
    """
    print("\nLoading image data from merged dataset...")
    X, y = [], []

    class_names = ["Knife", "Gun", "Syringe", "Scissors", "Screwdriver"]
    class_mapping = {name: idx for idx, name in enumerate(class_names)}

    total_images = 0
    class_counts = {cls: 0 for cls in class_names}

    for class_name in class_names:
        class_dir = Path(f"{base_dir}/{class_name}")
        if not class_dir.exists():
            print(f"Warning: Class directory {class_dir} not found")
            continue

        for split in ['train', 'test']:
            img_dir = class_dir / split / 'images'
            label_dir = class_dir / split / 'labels'

            if not img_dir.exists() or not label_dir.exists():
                print(f"Warning: {split} directory for {class_name} not properly set up")
                continue

            image_list = list(img_dir.glob('*.jpg')) + list(img_dir.glob('*.jpeg')) + list(img_dir.glob('*.png'))

            print(f"Found {len(image_list)} {class_name} images in {split} set")
            total_images += len(image_list)

            for image_path in image_list:
                # Read image
                try:
                    image = cv2.imread(str(image_path))
                    if image is None:
                        print(f"Skipping unreadable image: {image_path}")
                        continue

                    label_file = label_dir / (image_path.stem + ".txt")
                    if not label_file.exists():
                        print(f"Warning: No label found for {image_path}")
                        continue

                    with open(label_file, 'r') as f:
                        label_content = f.read().strip()

                    if not label_content:
                        print(f"Warning: Empty label file: {label_file}")
                        continue

                    label_parts = label_content.split()
                    if not label_parts:
                        print(f"Warning: Invalid label format in {label_file}")
                        continue

                    class_id = int(label_parts[0])

                    resized_img = cv2.resize(image, IMAGE_SHAPE)
                    X.append(resized_img)
                    y.append(class_id)
                    class_counts[class_name] += 1

                except Exception as e:
                    print(f"Error processing {image_path}: {e}")

    print(f"Total images loaded: {total_images}")
    print(f"Class distribution: {class_counts}")

    return np.array(X), np.array(y), class_names


def create_data_yaml(output_path=None):
    """Create a YAML file for the dataset configuration"""
    class_names = ["Knife", "Gun", "Syringe", "Scissors", "Screwdriver"]

    base_dir = 'E:/facultate/licenta/implementation/diagramsGit/BachelorsThesisFrontend/image-classification-model/threat-items-checker/merged-dataset'

    if output_path is None:
        output_path = os.path.join(base_dir, 'data.yaml')

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    data = {
        'path': base_dir,
        'train': {class_name: os.path.join(base_dir, class_name, 'train') for class_name in class_names},
        'test': {class_name: os.path.join(base_dir, class_name, 'test') for class_name in class_names},
        'nc': len(class_names),
        'names': class_names
    }

    import yaml
    with open(output_path, 'w') as f:
        yaml.dump(data, f)

    print(f"Created dataset configuration at {output_path}")
    return data

def predict_image(model, image_path, class_names):
    """Predict the class of an image using the trained model"""
    try:
        img = Image.open(image_path).resize(IMAGE_SHAPE)
        img_array = np.array(img) / 255.0

        if len(img_array.shape) == 2:
            img_array = np.stack((img_array,) * 3, axis=-1)
        elif img_array.shape[2] == 1:
            img_array = np.concatenate([img_array] * 3, axis=2)
        elif img_array.shape[2] == 4:
            img_array = img_array[:, :, :3]

        result = model.predict(img_array[np.newaxis, ...])
        predicted_class_index = np.argmax(result)
        confidence = np.max(result)

        return {
            'class_id': int(predicted_class_index),
            'class_name': class_names[predicted_class_index],
            'confidence': float(confidence)
        }
    except Exception as e:
        print(f"Error predicting image: {e}")
        return None


def build_model(num_classes, image_shape):
    """Build and compile the model with MobileNetV2"""
    feature_extractor_url = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"
    feature_extractor = hub.KerasLayer(
        feature_extractor_url,
        input_shape=image_shape + (3,),
        trainable=False,
        name="feature_extractor"
    )

    model = tf.keras.Sequential([
        feature_extractor,
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(num_classes)
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy", tf.keras.metrics.SparseTopKCategoricalAccuracy(k=1, name="top1_accuracy")]
    )

    return model


def setup_checkpoints(checkpoint_dir='checkpoints'):
    """Set up checkpoint directories and manager"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    checkpoint_path = os.path.join(checkpoint_dir, f"training_{timestamp}")
    os.makedirs(checkpoint_path, exist_ok=True)

    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(checkpoint_path, "model-{epoch:02d}-{val_accuracy:.4f}.h5"),
        save_weights_only=False,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True,
        verbose=1
    )

    log_dir = os.path.join("logs", timestamp)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=1,
        write_graph=True,
        update_freq='epoch'
    )

    metadata = {
        "timestamp": timestamp,
        "checkpoint_path": checkpoint_path,
        "tensorboard_log": log_dir
    }

    with open(os.path.join(checkpoint_path, "metadata.json"), "w") as f:
        json.dump(metadata, f)

    print(f"Checkpoint directory set up at: {checkpoint_path}")
    print(f"TensorBoard logs will be saved to: {log_dir}")
    print(f"View TensorBoard with: tensorboard --logdir={log_dir}")

    return checkpoint_callback, tensorboard_callback, checkpoint_path


def find_latest_checkpoint(checkpoint_dir='checkpoints'):
    """Find the latest checkpoint directory and best model"""
    if not os.path.exists(checkpoint_dir):
        return None, None

    checkpoint_dirs = [d for d in os.listdir(checkpoint_dir) if os.path.isdir(os.path.join(checkpoint_dir, d))]
    if not checkpoint_dirs:
        return None, None

    checkpoint_dirs.sort(reverse=True)
    latest_dir = os.path.join(checkpoint_dir, checkpoint_dirs[0])

    model_files = [f for f in os.listdir(latest_dir) if f.endswith('.h5')]
    if not model_files:
        return latest_dir, None

    model_files.sort(key=lambda x: float(x.split('-')[-1].replace('.h5', '')), reverse=True)
    best_model = os.path.join(latest_dir, model_files[0])

    return latest_dir, best_model


def train_model(X, y, class_names, epochs=10, batch_size=16, resume_training=False):
    """Train the model with the provided data and checkpointing"""
    print("\n=== Preparing and Training Model ===")
    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        try:
            for gpu in physical_devices:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("Memory growth enabled for GPU")
        except RuntimeError as e:
            print(f"Error enabling memory growth: {e}")


    # set_global_policy('mixed_float16')
    # print("Using mixed precision training")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")

    X_train_scaled = X_train / 255.0
    X_test_scaled = X_test / 255.0

    X_train_scaled = X_train_scaled.astype(np.float32)
    X_test_scaled = X_test_scaled.astype(np.float32)

    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.RandomBrightness(0.1),
        tf.keras.layers.RandomContrast(0.1),
    ])

    initial_epoch = 0
    if resume_training:
        latest_dir, best_model = find_latest_checkpoint()
        if best_model:
            print(f"Resuming training from checkpoint: {best_model}")
            model = tf.keras.models.load_model(best_model)

            try:
                epoch_str = os.path.basename(best_model).split('-')[1]
                initial_epoch = int(epoch_str)
                print(f"Resuming from epoch {initial_epoch}")
            except:
                print("Could not determine initial epoch, starting from 0")
        else:
            print("No checkpoint found. Starting fresh training.")
            model = build_model(len(class_names), IMAGE_SHAPE)
    else:
        model = build_model(len(class_names), IMAGE_SHAPE)

    model.summary()

    checkpoint_callback, tensorboard_callback, checkpoint_path = setup_checkpoints()

    with open(os.path.join(checkpoint_path, "class_names.json"), "w") as f:
        json.dump(class_names, f)

    callbacks = [
        checkpoint_callback,
        tensorboard_callback,
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=5, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6
        ),
        tf.keras.callbacks.CSVLogger(
            os.path.join(checkpoint_path, 'training_log.csv'),
            separator=',', append=True
        )
    ]

    print("\nStarting model training...")
    history = model.fit(
        tf.data.Dataset.from_tensor_slices((X_train_scaled, y_train))
        .map(lambda x, y: (data_augmentation(x), y))
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE),
        epochs=epochs,
        initial_epoch=initial_epoch,
        validation_data=(X_test_scaled, y_test),
        callbacks=callbacks,
        verbose=1
    )

    print("\nEvaluating model on test data...")
    test_results = model.evaluate(X_test_scaled, y_test, verbose=1)
    print(f"Test Loss: {test_results[0]:.4f}")
    print(f"Test Accuracy: {test_results[1]:.4f}")

    model_path = 'weapons_detection_model'
    model.save(model_path)

    performance = {
        "test_loss": float(test_results[0]),
        "test_accuracy": float(test_results[1]),
        "training_history": {
            "accuracy": [float(x) for x in history.history['accuracy']],
            "val_accuracy": [float(x) for x in history.history['val_accuracy']],
            "loss": [float(x) for x in history.history['loss']],
            "val_loss": [float(x) for x in history.history['val_loss']]
        }
    }

    with open(os.path.join(model_path, "performance.json"), "w") as f:
        json.dump(performance, f)

    with open(os.path.join(model_path, "class_names.json"), "w") as f:
        json.dump(class_names, f)

    print(f"Model and metadata saved to {model_path}")
    print(f"Checkpoints saved to {checkpoint_path}")

    return model, history, X_test_scaled, y_test, checkpoint_path


def list_available_checkpoints():
    """List all available checkpoints for user selection"""
    if not os.path.exists('checkpoints'):
        print("No checkpoints found.")
        return None

    checkpoint_dirs = [d for d in os.listdir('checkpoints') if os.path.isdir(os.path.join('checkpoints', d))]
    if not checkpoint_dirs:
        print("No checkpoints found.")
        return None

    print("\nAvailable checkpoint directories:")
    for i, directory in enumerate(sorted(checkpoint_dirs, reverse=True)):
        if directory.startswith('training_'):
            timestamp = directory.replace('training_', '')
            try:
                formatted_date = datetime.datetime.strptime(timestamp, "%Y%m%d-%H%M%S").strftime("%Y-%m-%d %H:%M:%S")
                print(f"[{i + 1}] {formatted_date} ({directory})")
            except:
                print(f"[{i + 1}] {directory}")
        else:
            print(f"[{i + 1}] {directory}")

    choice = input("\nEnter checkpoint number to load (or press Enter to skip): ")
    if not choice:
        return None

    try:
        index = int(choice) - 1
        if 0 <= index < len(checkpoint_dirs):
            return os.path.join('checkpoints', sorted(checkpoint_dirs, reverse=True)[index])
    except:
        pass

    print("Invalid selection.")
    return None


def load_model_from_checkpoint(checkpoint_dir=None):
    """Load a model from a specified checkpoint directory or let user select one"""
    if not checkpoint_dir:
        checkpoint_dir = list_available_checkpoints()
        if not checkpoint_dir:
            return None, None

    model_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.h5')]
    if not model_files:
        print(f"No model files found in {checkpoint_dir}")
        return None, None

    model_files.sort(key=lambda x: float(x.split('-')[-1].replace('.h5', '')), reverse=True)
    best_model_path = os.path.join(checkpoint_dir, model_files[0])

    print(f"Loading model from {best_model_path}")
    model = tf.keras.models.load_model(best_model_path)

    class_names_path = os.path.join(checkpoint_dir, "class_names.json")
    if os.path.exists(class_names_path):
        with open(class_names_path, 'r') as f:
            class_names = json.load(f)
    else:
        class_names = ["Knife", "Gun", "Syringe", "Scissors", "Screwdriver"]
        print("Warning: class_names.json not found, using default class names")

    print("Model loaded successfully!")
    return model, class_names


def visualize_results(history, X_test, y_test, model, class_names, checkpoint_path=None):
    """Visualize training history and sample predictions"""
    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()

    plt.tight_layout()

    if checkpoint_path:
        plt.savefig(os.path.join(checkpoint_path, 'training_history.png'))
    else:
        plt.savefig('training_history.png')

    plt.show()

    from sklearn.metrics import confusion_matrix, classification_report
    predictions = model.predict(X_test)
    pred_classes = np.argmax(predictions, axis=1)

    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_test, pred_classes)

    import seaborn as sns
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')

    if checkpoint_path:
        plt.savefig(os.path.join(checkpoint_path, 'confusion_matrix.png'))
    else:
        plt.savefig('confusion_matrix.png')

    plt.show()

    print("\nClassification Report:")
    print(classification_report(y_test, pred_classes, target_names=class_names))

    if checkpoint_path:
        with open(os.path.join(checkpoint_path, 'classification_report.txt'), 'w') as f:
            f.write(classification_report(y_test, pred_classes, target_names=class_names))

    plt.figure(figsize=(15, 12))

    indices = random.sample(range(len(X_test)), min(9, len(X_test)))

    for i, idx in enumerate(indices):
        plt.subplot(3, 3, i + 1)
        plt.imshow(cv2.cvtColor(X_test[idx].astype(np.uint8), cv2.COLOR_BGR2RGB))

        true_class = class_names[y_test[idx]]
        pred_class = class_names[pred_classes[idx]]

        title_color = "green" if true_class == pred_class else "red"
        plt.title(f"True: {true_class}\nPred: {pred_class}", color=title_color)
        plt.axis('off')

    plt.tight_layout()

    if checkpoint_path:
        plt.savefig(os.path.join(checkpoint_path, 'sample_predictions.png'))
    else:
        plt.savefig('sample_predictions.png')

    plt.show()


def run_inference_demo(model, class_names):
    """Run inference on test images provided by the user"""
    print("\n=== Weapon Detection Demo ===")
    while True:
        test_image_path = input("\nEnter path to test image (or 'q' to quit): ")
        if test_image_path.lower() == 'q':
            break

        if not os.path.exists(test_image_path):
            print(f"Error: File not found at {test_image_path}")
            continue

        try:
            result = predict_image(model, test_image_path, class_names)

            if result:
                print(f"Detected: {result['class_name']} (Confidence: {result['confidence']:.2f})")

                img = cv2.imread(test_image_path)
                img = cv2.resize(img, (640, 480))
                cv2.putText(
                    img,
                    f"{result['class_name']} ({result['confidence']:.2f})",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
                )

                cv2.imshow("Prediction", img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
        except Exception as e:
            print(f"Error during inference: {e}")


if __name__ == "__main__":
    # for package in ["kaggle", "tensorflow-hub", "pillow", "matplotlib", "sklearn", "pyyaml", "seaborn"]:
    #     install_package(package)

    global IMAGE_SHAPE
    IMAGE_SHAPE = (224, 224)

    base_dir = 'E:/facultate/licenta/implementation/diagramsGit/BachelorsThesisFrontend/image-classification-model/threat-items-checker/merged-dataset'
    if not os.path.exists(base_dir):
        print("\nMerged dataset not found.")
        #setup_kaggle()
        #download_dataset()
        #print("\nPlease run the dataset merging script first.")
        sys.exit(1)

    data_yaml_path = os.path.join(base_dir, 'data.yaml')
    if not os.path.exists(data_yaml_path):
        dataset_config = create_data_yaml(data_yaml_path)
    else:
        import yaml

        with open(data_yaml_path, 'r') as f:
            dataset_config = yaml.safe_load(f)

    os.makedirs('checkpoints', exist_ok=True)

    print("\n=== Weapons Detection System ===")
    print("1. Train new model")
    print("2. Resume training from checkpoint")
    print("3. Load model from checkpoint")
    print("4. Use previously saved model")
    print("5. Exit")

    choice = input("\nEnter your choice (1-5): ")

    if choice == '1':
        X, y, class_names = load_merged_data()

        if len(X) == 0:
            print("No data loaded. Please check the dataset structure.")
            sys.exit(1)

        epochs = int(input("Enter number of training epochs (default: 10): ") or "10")
        model, history, X_test, y_test, checkpoint_path = train_model(
            X, y, class_names, epochs=epochs, batch_size=16
        )

        visualize_results(history, X_test, y_test, model, class_names, checkpoint_path)

        run_inference_demo(model, class_names)

    elif choice == '2':
        X, y, class_names = load_merged_data()

        if len(X) == 0:
            print("No data loaded. Please check the dataset structure.")
            sys.exit(1)

        epochs = int(input("Enter additional training epochs (default: 5): ") or "5")
        model, history, X_test, y_test, checkpoint_path = train_model(
            X, y, class_names, epochs=epochs, resume_training=True
        )

        visualize_results(history, X_test, y_test, model, class_names, checkpoint_path)

        run_inference_demo(model, class_names)

    elif choice == '3':
        model, class_names = load_model_from_checkpoint()

        if model is None:
            print("Failed to load model from checkpoint.")
            sys.exit(1)

        run_inference_demo(model, class_names)

    elif choice == '4':
        if not os.path.exists('weapons_detection_model'):
            print("No saved model found at 'weapons_detection_model'.")
            sys.exit(1)

        try:
            model = tf.keras.models.load_model('weapons_detection_model')

            if os.path.exists('weapons_detection_model/class_names.json'):
                with open('weapons_detection_model/class_names.json', 'r') as f:
                    class_names = json.load(f)
            else:
                class_names = ["Knife", "Gun", "Syringe", "Scissors", "Screwdriver"]
                print("Warning: class_names.json not found, using default class names")

            print("Model loaded successfully!")

            run_inference_demo(model, class_names)
        except Exception as e:
            print(f"Error loading model: {e}")
            sys.exit(1)

    elif choice == '5':
        print("Exiting...")
        sys.exit(0)

    else:
        print("Invalid choice. Exiting...")
        sys.exit(1)