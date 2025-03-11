import os
import yaml
import shutil
import random
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from ultralytics import YOLO

class VehicleMakeModelTrainer:
    def __init__(self, data_dir, output_dir, epochs=100, batch_size=16, image_size=640, pretrained_weights=None):
        """
        Initialize the Vehicle Make and Model trainer
        
        Args:
            data_dir (str): Directory containing the dataset
            output_dir (str): Directory to save model and results
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
            image_size (int): Image size for training
            pretrained_weights (str): Path to pretrained weights (optional)
        """
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.epochs = epochs
        self.batch_size = batch_size
        self.image_size = image_size
        self.pretrained_weights = pretrained_weights
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Path for the prepared dataset
        self.dataset_path = os.path.join(self.output_dir, 'dataset')
        os.makedirs(self.dataset_path, exist_ok=True)
        
        # Initialize class mapping
        self.class_mapping = {}
        self.num_classes = 0
        
    def prepare_dataset(self, split_ratio=0.2):
        """
        Prepare the dataset for YOLO training
        
        Args:
            split_ratio (float): Validation split ratio
        """
        print(f"Preparing dataset from {self.data_dir}")
        
        # Create directories for YOLO format
        train_dir = os.path.join(self.dataset_path, 'train')
        val_dir = os.path.join(self.dataset_path, 'val')
        
        for dir_path in [train_dir, val_dir]:
            os.makedirs(os.path.join(dir_path, 'images'), exist_ok=True)
            os.makedirs(os.path.join(dir_path, 'labels'), exist_ok=True)
        
        # Parse the dataset directory structure
        # Assuming format: data_dir/make/model/images.jpg
        makes = [d for d in os.listdir(self.data_dir) if os.path.isdir(os.path.join(self.data_dir, d))]
        
        class_idx = 0
        image_paths = []
        
        # Collect all image paths and build class mapping
        for make_idx, make in enumerate(makes):
            make_dir = os.path.join(self.data_dir, make)
            models = [d for d in os.listdir(make_dir) if os.path.isdir(os.path.join(make_dir, d))]
            
            for model in models:
                class_name = f"{make}_{model}"
                self.class_mapping[class_idx] = class_name
                
                model_dir = os.path.join(make_dir, model)
                images = [f for f in os.listdir(model_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                
                for img in images:
                    img_path = os.path.join(model_dir, img)
                    image_paths.append((img_path, class_idx))
                
                class_idx += 1
        
        self.num_classes = class_idx
        print(f"Found {len(image_paths)} images across {self.num_classes} classes")
        
        # Save the class mapping
        with open(os.path.join(self.output_dir, 'class_mapping.yaml'), 'w') as f:
            yaml.dump(self.class_mapping, f)
        
        # Split into train and validation sets
        train_paths, val_paths = train_test_split(image_paths, test_size=split_ratio, random_state=42, stratify=[p[1] for p in image_paths])
        
        print(f"Training set: {len(train_paths)} images")
        print(f"Validation set: {len(val_paths)} images")
        
        # Function to copy and create YOLO labels
        def process_image_set(image_set, target_dir):
            for idx, (img_path, class_id) in enumerate(image_set):
                # Copy image
                img_filename = f"{idx:06d}.jpg"
                target_img_path = os.path.join(target_dir, 'images', img_filename)
                
                # Open and convert image if needed
                img = Image.open(img_path)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                img.save(target_img_path)
                
                # Create YOLO label
                img_width, img_height = img.size
                label_filename = f"{idx:06d}.txt"
                target_label_path = os.path.join(target_dir, 'labels', label_filename)
                
                # Create a simple bounding box covering most of the image
                # In a real scenario, you'd use actual bounding box annotations
                x_center, y_center = 0.5, 0.5
                width, height = 0.9, 0.9  # 90% of image width/height
                
                with open(target_label_path, 'w') as f:
                    # YOLO format: class_id x_center y_center width height
                    f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")
        
        # Process training and validation sets
        process_image_set(train_paths, train_dir)
        process_image_set(val_paths, val_dir)
        
        # Create YAML config for YOLO
        yaml_config = {
            'path': os.path.abspath(self.dataset_path),
            'train': 'train/images',
            'val': 'val/images',
            'names': self.class_mapping
        }
        
        with open(os.path.join(self.dataset_path, 'data.yaml'), 'w') as f:
            yaml.dump(yaml_config, f)
            
        print("Dataset preparation complete")
        return os.path.join(self.dataset_path, 'data.yaml')
    
    def train_model(self, data_yaml_path):
        """
        Train the YOLO model for vehicle make and model recognition
        
        Args:
            data_yaml_path (str): Path to the data.yaml file
        """
        print("Starting model training...")
        
        # Initialize the model
        if self.pretrained_weights:
            model = YOLO(self.pretrained_weights)
            print(f"Using pretrained weights: {self.pretrained_weights}")
        else:
            model = YOLO('yolov8n.pt')  # Start with YOLOv8 nano
            print("Using default YOLOv8n weights")
        
        # Train the model
        results = model.train(
            data=data_yaml_path,
            epochs=self.epochs,
            batch=self.batch_size,
            imgsz=self.image_size,
            patience=20,  # Early stopping patience
            save=True,
            project=self.output_dir,
            name='vehicle_make_model',
            device='0' if torch.cuda.is_available() else 'cpu'
        )
        
        # Run validation
        val_results = model.val()
        print(f"Validation results: {val_results}")
        
        # Save the final model
        model_path = os.path.join(self.output_dir, 'vehicle_make_model.pt')
        model.export(format='torchscript')
        
        print(f"Model training complete. Model saved to {model_path}")
        return model_path

    def evaluate_model(self, model_path, test_image_dir=None):
        """
        Evaluate the trained model on sample images
        
        Args:
            model_path (str): Path to the trained model
            test_image_dir (str): Directory containing test images (optional)
        """
        print("Evaluating model...")
        
        # Load the trained model
        model = YOLO(model_path)
        
        # If test directory is not provided, use validation images
        if test_image_dir is None:
            test_image_dir = os.path.join(self.dataset_path, 'val', 'images')
        
        # Get sample images
        test_images = [os.path.join(test_image_dir, f) for f in os.listdir(test_image_dir) 
                       if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if len(test_images) == 0:
            print("No test images found")
            return
        
        # Randomly select a few images for visualization
        sample_images = random.sample(test_images, min(5, len(test_images)))
        
        # Predict and visualize
        results = model(sample_images, verbose=True)
        
        for idx, (img_path, result) in enumerate(zip(sample_images, results)):
            # Plot the result
            fig, ax = plt.subplots(figsize=(10, 10))
            img = Image.open(img_path)
            ax.imshow(np.array(img))
            
            boxes = result.boxes.xyxy.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy().astype(int)
            confs = result.boxes.conf.cpu().numpy()
            
            for box, cls, conf in zip(boxes, classes, confs):
                x1, y1, x2, y2 = box
                class_name = self.class_mapping.get(cls, f"Class {cls}")
                
                # Draw bounding box
                rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor='red', linewidth=2)
                ax.add_patch(rect)
                
                # Add label
                ax.text(x1, y1-10, f"{class_name} {conf:.2f}", color='red', fontsize=12, 
                        bbox=dict(facecolor='white', alpha=0.8))
            
            plt.axis('off')
            plt.tight_layout()
            
            # Save the visualization
            save_path = os.path.join(self.output_dir, f"sample_prediction_{idx}.jpg")
            plt.savefig(save_path)
            plt.close()
            
            print(f"Saved prediction visualization to {save_path}")
        
        print("Model evaluation complete")

def data_augmentation(data_dir, augmented_dir, num_augmentations=5):
    """
    Apply data augmentation to increase dataset size
    
    Args:
        data_dir (str): Source data directory
        augmented_dir (str): Directory to save augmented images
        num_augmentations (int): Number of augmented images to generate per original image
    """
    import albumentations as A
    
    print(f"Augmenting dataset from {data_dir} to {augmented_dir}")
    
    # Create augmentation pipeline
    transform = A.Compose([
        A.RandomBrightnessContrast(p=0.8),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(p=0.8, rotate_limit=15, shift_limit=0.1, scale_limit=0.1),
        A.RGBShift(p=0.5),
        A.Blur(blur_limit=3, p=0.2),
        A.ISONoise(p=0.2),
        A.ColorJitter(p=0.2),
    ])
    
    # Create directory structure
    os.makedirs(augmented_dir, exist_ok=True)
    
    # Process each make and model
    makes = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    
    for make in makes:
        make_dir = os.path.join(data_dir, make)
        aug_make_dir = os.path.join(augmented_dir, make)
        os.makedirs(aug_make_dir, exist_ok=True)
        
        models = [d for d in os.listdir(make_dir) if os.path.isdir(os.path.join(make_dir, d))]
        
        for model in models:
            model_dir = os.path.join(make_dir, model)
            aug_model_dir = os.path.join(aug_make_dir, model)
            os.makedirs(aug_model_dir, exist_ok=True)
            
            # Process each image
            images = [f for f in os.listdir(model_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            for img_file in images:
                # Copy original image
                shutil.copy(
                    os.path.join(model_dir, img_file),
                    os.path.join(aug_model_dir, img_file)
                )
                
                # Load image
                img_path = os.path.join(model_dir, img_file)
                img = np.array(Image.open(img_path).convert('RGB'))
                
                # Generate augmentations
                for i in range(num_augmentations):
                    augmented = transform(image=img)
                    aug_img = Image.fromarray(augmented['image'])
                    
                    # Save augmented image
                    base_name, ext = os.path.splitext(img_file)
                    aug_path = os.path.join(aug_model_dir, f"{base_name}_aug{i}{ext}")
                    aug_img.save(aug_path)
    
    print("Data augmentation complete")
    return augmented_dir

if __name__ == "__main__":
    import torch
    
    parser = argparse.ArgumentParser(description='Train vehicle make and model recognition model')
    parser.add_argument('--data_dir', type=str, default='/home/plato/Documents/Projects/Car Detection/data', help='Directory with vehicle images')
    parser.add_argument('--output_dir', type=str, default='/home/plato/Documents/Projects/Car Detection/vehicle_model_output', help='Output directory')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--image_size', type=int, default=640, help='Image size for training')
    parser.add_argument('--pretrained', type=str, default=None, help='Path to pretrained weights')
    parser.add_argument('--augment', action='store_true', help='Apply data augmentation')
    parser.add_argument('--aug_factor', type=int, default=5, help='Augmentation factor')
    
    args = parser.parse_args()
    
    # Apply data augmentation if requested
    if args.augment:
        print("Applying data augmentation...")
        augmented_dir = os.path.join(args.output_dir, 'augmented_data')
        data_dir = data_augmentation(args.data_dir, augmented_dir, args.aug_factor)
    else:
        data_dir = args.data_dir
    
    # Initialize trainer
    trainer = VehicleMakeModelTrainer(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        image_size=args.image_size,
        pretrained_weights=args.pretrained
    )
    
    # Prepare dataset
    data_yaml_path = trainer.prepare_dataset()
    
    # Train model
    model_path = trainer.train_model(data_yaml_path)
    
    # Evaluate model
    trainer.evaluate_model(model_path)
    
    print(f"Complete! Trained model saved to {model_path}")