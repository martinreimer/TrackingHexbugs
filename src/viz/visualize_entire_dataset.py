'''
Code visualizes bounding boxes from YOLO label files on images, saving the results for both training and validation datasets.

Setup:

Define directories for input images, labels, and output visualizations.
Ensure output directories exist.
YOLO Label Parsing:

Convert YOLO format bounding box annotations (<class-id> <x-center> <y-center> <width> <height>) to pixel coordinates (x_min, y_min, x_max, y_max) based on image dimensions.
Visualization:

For each image in the training and validation directories:
Read the image and corresponding YOLO label file.
Parse each bounding box, draw it on the image, and label it with its class ID.
Save the visualized image to the output directory.
Output:

Annotated images with drawn bounding boxes are saved for both training and validation datasets.
'''
import os
import cv2
import matplotlib.pyplot as plt

# Define paths
merged_data_dir = '../data/final-dataset'

train_img_dir = os.path.join(merged_data_dir, 'train', 'images')
train_label_dir = os.path.join(merged_data_dir, 'train', 'labels')
val_img_dir = os.path.join(merged_data_dir, 'val', 'images')
val_label_dir = os.path.join(merged_data_dir, 'val', 'labels')

# Output directories for visualizations
train_output_dir = os.path.join(merged_data_dir, 'vis', 'train')
val_output_dir = os.path.join(merged_data_dir, 'vis', 'val')

# Create output directories if they don't exist
os.makedirs(train_output_dir, exist_ok=True)
os.makedirs(val_output_dir, exist_ok=True)

# Function to convert YOLO format to bounding box coordinates
def yolo_to_bbox(yolo_bbox, img_width, img_height):
    class_id, x_center, y_center, width, height = yolo_bbox
    x_center, y_center, width, height = float(x_center), float(y_center), float(width), float(height)
    
    x_min = int((x_center - width / 2) * img_width)
    x_max = int((x_center + width / 2) * img_width)
    y_min = int((y_center - height / 2) * img_height)
    y_max = int((y_center + height / 2) * img_height)
    
    return class_id, x_min, y_min, x_max, y_max

# Function to process images and labels
def process_images_and_labels(img_dir, label_dir, output_dir):
    image_files = [f for f in os.listdir(img_dir) if f.endswith('.jpg')]
    for image_file in image_files:
        # Get the corresponding label file
        label_file = image_file.replace('.jpg', '.txt')
        label_path = os.path.join(label_dir, label_file)
        image_path = os.path.join(img_dir, image_file)
        
        # Read the image
        image = cv2.imread(image_path)
        if image is None:
            print('Could not read image:', image_file)
            continue
        img_height, img_width, _ = image.shape

        # Read the label file
        with open(label_path, 'r') as file:
            lines = file.readlines()

        # Parse the bounding boxes and draw them on the image
        for line in lines:
            yolo_bbox = line.strip().split()
            bbox = yolo_to_bbox(yolo_bbox, img_width, img_height)
            class_id, x_min, y_min, x_max, y_max = bbox
            
            # Draw the bounding box
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
            cv2.putText(image, str(class_id), (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
        
        # Save the visualized image
        output_image_path = os.path.join(output_dir, image_file)
        cv2.imwrite(output_image_path, image)

# Process train images and labels
process_images_and_labels(train_img_dir, train_label_dir, train_output_dir)

# Process val images and labels
process_images_and_labels(val_img_dir, val_label_dir, val_output_dir)

print("Visualization complete.")
