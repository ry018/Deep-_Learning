import tensorflow as tf
import os
import cv2
import numpy as np

# Define paths to the SAR and Optical image folders
sar_image_folder = r'R:\SAR_img_colourization_using_DL\Dataset\TRAIN\SAR_1K'
optical_image_folder = r'R:\SAR_img_colourization_using_DL\Dataset\TRAIN\OPT_1K'

# Define the output size for the images
image_size = (256, 256)

# Function to apply histogram equalization
def histogram_equalization(image):
    return cv2.equalizeHist(image)

# Function to apply edge detection
def edge_detection(image):
    edges = cv2.Canny(image, 100, 200)
    return edges

# Function to load and preprocess SAR images
def preprocess_sar_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Load SAR as grayscale
    image = cv2.resize(image, image_size)
    image = histogram_equalization(image)  # Optional: Apply histogram equalization
    image = edge_detection(image)  # Optional: Apply edge detection
    image = image.astype(np.float32) / 127.5 - 1.0  # Normalize to [-1, 1]
    image = np.expand_dims(image, axis=-1)  # Add a channel dimension to match optical image
    return image

# Function to load and preprocess optical images
def preprocess_optical_image(image_path):
    image = cv2.imread(image_path)  # Load Optical as color
    image = cv2.resize(image, image_size)
    image = image.astype(np.float32) / 127.5 - 1.0  # Normalize to [-1, 1]
    return image

# Function to load and preprocess image pairs
def load_and_preprocess_pair(sar_path, optical_path):
    sar_image = preprocess_sar_image(sar_path)
    optical_image = preprocess_optical_image(optical_path)
    return sar_image, optical_image

def augment_image(sar_image, optical_image):
    # Augmentation options: Random flips, rotations, etc.
    if np.random.rand() > 0.5:
        sar_image = np.fliplr(sar_image)
        optical_image = np.fliplr(optical_image)
    if np.random.rand() > 0.5:
        sar_image = np.flipud(sar_image)
        optical_image = np.flipud(optical_image)
    
    # Add more augmentations as needed
    return sar_image, optical_image

def preprocess_dataset(sar_folder, optical_folder):
    sar_images = sorted(os.listdir(sar_folder))
    optical_images = sorted(os.listdir(optical_folder))
    
    dataset_sar = []
    dataset_optical = []
    
    for sar_img_name, optical_img_name in zip(sar_images, optical_images):
        sar_path = os.path.join(sar_folder, sar_img_name)
        optical_path = os.path.join(optical_folder, optical_img_name)
        
        # Load and preprocess the image pair
        sar_image, optical_image = load_and_preprocess_pair(sar_path, optical_path)
        
        # Apply data augmentation
        sar_image, optical_image = augment_image(sar_image, optical_image)
        
        # Store the images in separate lists
        dataset_sar.append(sar_image)
        dataset_optical.append(optical_image)
    
    # Convert lists to numpy arrays
    dataset_sar = np.array(dataset_sar)
    dataset_optical = np.array(dataset_optical)
    
    return dataset_sar, dataset_optical

# Run preprocessing on your dataset
sar_images, optical_images = preprocess_dataset(sar_image_folder, optical_image_folder)

# Save the preprocessed datasets if needed
np.save('preprocessed_sar_images1K.npy', sar_images)
np.save('preprocessed_optical_images1K.npy', optical_images)

print("Dataset preprocessed and saved.")
# Run preprocessing on your dataset
sar_images, optical_images = preprocess_dataset(sar_image_folder, optical_image_folder)

# Define paths for saving the preprocessed datasets
sar_save_path = 'preprocessed_sar_images1K.npy'
optical_save_path = 'preprocessed_optical_images1K.npy'

# Save the preprocessed datasets
np.save(sar_save_path, sar_images)
np.save(optical_save_path, optical_images)

# Print the location where the preprocessed data are saved
print(f"Preprocessed SAR images saved at: {os.path.abspath(sar_save_path)}")
print(f"Preprocessed Optical images saved at: {os.path.abspath(optical_save_path)}")

print("Dataset preprocessed and saved.")
