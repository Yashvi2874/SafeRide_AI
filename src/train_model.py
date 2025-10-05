"""
Model Training Script for SafeRide_AI
This script provides functionality to retrain the YOLOv8 model for improved accuracy
in detecting driver distractions like phones, books, and laptops.
"""

import os
import cv2
import numpy as np
from ultralytics import YOLO
import argparse
import yaml

def create_dataset_structure():
    """Create the directory structure for the custom dataset."""
    dataset_dirs = [
        "datasets/custom_dataset/images/train",
        "datasets/custom_dataset/images/val",
        "datasets/custom_dataset/labels/train",
        "datasets/custom_dataset/labels/val"
    ]
    
    for dir_path in dataset_dirs:
        os.makedirs(dir_path, exist_ok=True)
        print(f"Created directory: {dir_path}")

def create_dataset_yaml():
    """Create the dataset configuration YAML file."""
    dataset_config = {
        'path': 'datasets/custom_dataset',
        'train': 'images/train',
        'val': 'images/val',
        'nc': 3,
        'names': ['cell phone', 'book', 'laptop']
    }
    
    with open('datasets/custom_dataset.yaml', 'w') as f:
        yaml.dump(dataset_config, f, default_flow_style=False)
    
    print("Created dataset configuration file: datasets/custom_dataset.yaml")

def prepare_training_data():
    """Prepare training data by organizing images and annotations."""
    print("Preparing training data...")
    print("Please organize your custom dataset with the following structure:")
    print("""
    datasets/custom_dataset/
    ├── images/
    │   ├── train/
    │   │   ├── image1.jpg
    │   │   ├── image2.jpg
    │   │   └── ...
    │   └── val/
    │       ├── image101.jpg
    │       ├── image102.jpg
    │       └── ...
    └── labels/
        ├── train/
        │   ├── image1.txt
        │   ├── image2.txt
        │   └── ...
        └── val/
            ├── image101.txt
            ├── image102.txt
            └── ...
    """)
    print("Label files should be in YOLO format (class_id center_x center_y width height)")

def train_model(model_size='n', epochs=100, batch_size=16, img_size=640):
    """
    Train the YOLOv8 model with custom dataset.
    
    Args:
        model_size (str): Model size (n, s, m, l, x) - n is smallest/fastest
        epochs (int): Number of training epochs
        batch_size (int): Batch size for training
        img_size (int): Image size for training
    """
    try:
        # Load a pretrained model
        model_name = f"yolov8{model_size}.pt"
        print(f"Loading pretrained model: {model_name}")
        model = YOLO(model_name)
        
        # Train the model
        print("Starting training process...")
        print(f"Parameters: epochs={epochs}, batch_size={batch_size}, img_size={img_size}")
        
        results = model.train(
            data='datasets/custom_dataset.yaml',
            epochs=epochs,
            batch=batch_size,
            imgsz=img_size,
            cache=True,  # Cache images for faster training
            device=0 if torch.cuda.is_available() else None,  # Use GPU if available
            verbose=True
        )
        
        print("Training completed successfully!")
        print(f"Best model saved to: {results.save_dir}/weights/best.pt")
        
        return results
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        return None

def validate_model(model_path='datasets/custom_dataset/weights/best.pt'):
    """Validate the trained model on validation set."""
    try:
        print(f"Loading trained model: {model_path}")
        model = YOLO(model_path)
        
        # Validate the model
        print("Validating model on validation set...")
        metrics = model.val()
        
        print("Validation Results:")
        print(f"mAP50: {metrics.box.map50:.4f}")
        print(f"mAP50-95: {metrics.box.map:.4f}")
        print(f"Precision: {metrics.box.p:.4f}")
        print(f"Recall: {metrics.box.r:.4f}")
        
        return metrics
        
    except Exception as e:
        print(f"Error during validation: {str(e)}")
        return None

def update_detection_model(new_model_path):
    """Update the detection.py file to use the new trained model."""
    detection_file = "detection.py"
    
    try:
        with open(detection_file, 'r') as f:
            content = f.read()
        
        # Replace the model loading line
        old_model_line = 'model = YOLO("yolov8n.pt")'
        new_model_line = f'model = YOLO("{new_model_path}")'
        
        if old_model_line in content:
            content = content.replace(old_model_line, new_model_line)
            
            with open(detection_file, 'w') as f:
                f.write(content)
            
            print(f"Updated detection.py to use new model: {new_model_path}")
        else:
            print("Could not find model loading line in detection.py")
            
    except Exception as e:
        print(f"Error updating detection.py: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8 model for SafeRide_AI")
    parser.add_argument('--action', choices=['setup', 'train', 'validate', 'update'], 
                       default='setup', help='Action to perform')
    parser.add_argument('--model_size', default='n', help='Model size (n, s, m, l, x)')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--img_size', type=int, default=640, help='Image size')
    parser.add_argument('--new_model_path', default='datasets/custom_dataset/weights/best.pt', 
                       help='Path to new trained model')
    
    args = parser.parse_args()
    
    if args.action == 'setup':
        create_dataset_structure()
        create_dataset_yaml()
        prepare_training_data()
    elif args.action == 'train':
        train_model(args.model_size, args.epochs, args.batch_size, args.img_size)
    elif args.action == 'validate':
        validate_model(args.new_model_path)
    elif args.action == 'update':
        update_detection_model(args.new_model_path)

if __name__ == "__main__":
    # Import torch here to avoid import issues
    try:
        import torch
    except ImportError:
        print("PyTorch is required for training. Please install it with: pip install torch")
        exit(1)
    
    main()