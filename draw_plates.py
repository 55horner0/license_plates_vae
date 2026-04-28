# prepare_plates_for_draw.py
import numpy as np
import cv2
import os
from glob import glob
import matplotlib.pyplot as plt

def load_and_preprocess_plates(plates_dir, target_size=(28, 28), visualize=False):
    """
    Load cropped license plates and prepare them for DRAW model
    
    Args:
        plates_dir: Path to folder containing cropped license plate images
        target_size: (height, width) to resize images to
        visualize: Whether to show sample images after processing
    
    Returns:
        numpy array of flattened images ready for DRAW
    """
    # Find all image files
    image_paths = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.PNG']:
        image_paths.extend(glob(os.path.join(plates_dir, ext)))
    
    if len(image_paths) == 0:
        print(f"Error: No images found in {plates_dir}")
        print(f"Please check the path: {os.path.abspath(plates_dir)}")
        return None
    
    print(f"Found {len(image_paths)} license plate images")
    
    images = []
    failed = []
    
    for i, path in enumerate(image_paths):
        try:
            # Read image
            img = cv2.imread(path)
            if img is None:
                failed.append(path)
                continue
                
            # Convert to grayscale
            if len(img.shape) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Resize to target size
            img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
            
            # Normalize to [0, 1] range
            img = img.astype(np.float32) / 255.0
            
            # Optional: Apply thresholding to make it binary-like
            # This can help with license plate text
            # img = (img > 0.5).astype(np.float32)
            
            # Flatten to 1D array
            img_flat = img.flatten()
            images.append(img_flat)
            
            if visualize and i < 5:  # Show first 5 images
                plt.figure(figsize=(2, 2))
                plt.imshow(img, cmap='gray')
                plt.title(f"Sample {i+1}: {os.path.basename(path)}")
                plt.axis('off')
                plt.show()
                
        except Exception as e:
            failed.append(f"{path}: {str(e)}")
            continue
    
    if len(images) == 0:
        print("Error: No valid images could be loaded!")
        return None
    
    print(f"Successfully loaded {len(images)} images")
    if failed:
        print(f"Failed to load {len(failed)} images")
    
    # Convert to numpy array
    images_array = np.array(images)
    print(f"Data shape: {images_array.shape}")
    print(f"Data range: [{images_array.min():.3f}, {images_array.max():.3f}]")
    
    return images_array

def save_plate_stats(images, output_dir):
    """Save statistics about the dataset"""
    stats = {
        'num_samples': len(images),
        'image_dim': images.shape[1],
        'mean': float(images.mean()),
        'std': float(images.std()),
        'min': float(images.min()),
        'max': float(images.max())
    }
    
    # Save stats as text file
    stats_file = os.path.join(output_dir, 'plate_dataset_stats.txt')
    with open(stats_file, 'w') as f:
        for key, value in stats.items():
            f.write(f"{key}: {value}\n")
    
    print(f"\nDataset Statistics saved to {stats_file}")
    print("-" * 40)
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    # Plot histogram of pixel values
    plt.figure(figsize=(10, 4))
    plt.hist(images.flatten(), bins=50, alpha=0.7)
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.title('Distribution of Pixel Values')
    plt.savefig(os.path.join(output_dir, 'pixel_distribution.png'))
    plt.show()

if __name__ == "__main__":
    # === CONFIGURATION - CHANGE THESE PATHS ===
    # Path to your cropped license plates folder
    CROPPED_PLATES_DIR = "license_plates_cropped_whole"
    
    # Output directory for prepared data
    OUTPUT_DIR = "draw_plate_data_prepared"
    
    # Target size for DRAW model (28x28 is default, can change)
    TARGET_SIZE = (28, 28)  # (height, width)
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load and preprocess images
    print("Loading and preprocessing license plate images...")
    print(f"Looking for images in: {os.path.abspath(CROPPED_PLATES_DIR)}")
    
    plate_images = load_and_preprocess_plates(
        CROPPED_PLATES_DIR, 
        target_size=TARGET_SIZE,
        visualize=True
    )
    
    if plate_images is not None:
        # Save the processed data
        output_file = os.path.join(OUTPUT_DIR, "license_plates_data.npy")
        np.save(output_file, plate_images)
        print(f"\n✓ Data saved to: {output_file}")
        
        # Save statistics about the dataset
        save_plate_stats(plate_images, OUTPUT_DIR)
        
        # Also save a few sample images for verification
        sample_dir = os.path.join(OUTPUT_DIR, "samples")
        os.makedirs(sample_dir, exist_ok=True)
        for i in range(min(10, len(plate_images))):
            img = plate_images[i].reshape(TARGET_SIZE)
            plt.imsave(os.path.join(sample_dir, f"sample_{i}.png"), img, cmap='gray')
        
        print(f"\n✓ Sample images saved to: {sample_dir}")
        print("\n=== NEXT STEPS ===")
        print(f"1. Update your DRAW script to load: {output_file}")
        print("2. Run the training script")
    else:
        print("\n❌ Failed to prepare data. Please check:")
        print("  - The path to your cropped plates is correct")
        print("  - The folder contains image files (.jpg, .png, etc.)")
        print("  - You have opencv-python installed (pip install opencv-python)")
        

# verify_data.py
import numpy as np
import matplotlib.pyplot as plt

# Load the prepared data
data_file = "draw_plate_data_prepared/license_plates_data.npy"
data = np.load(data_file)

print(f"Data shape: {data.shape}")
print(f"Data type: {data.dtype}")
print(f"Value range: [{data.min():.4f}, {data.max():.4f}]")

# Display the first 16 images
fig, axes = plt.subplots(4, 4, figsize=(8, 8))
img_size = int(np.sqrt(data.shape[1]))

for i, ax in enumerate(axes.flat):
    if i < len(data):
        img = data[i].reshape(img_size, img_size)
        ax.imshow(img, cmap='gray')
        ax.axis('off')
        ax.set_title(f'Plate {i+1}')
    else:
        ax.axis('off')

plt.suptitle('Sample License Plates for DRAW Training')
plt.tight_layout()
plt.savefig('data_verification.png')
plt.show()

print("\n✓ Data verification complete!")
print("If you can see license plate images above, the data is ready for DRAW training.")