# prepare_plates_final.py - Optimized for European dataset
import numpy as np
import cv2
import os
from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as plt

def load_and_preprocess_plates(plates_dir, target_size=(96, 32), split_name=""):
    """
    Load license plates with adaptive preprocessing optimized for European plates
    """
    # Find all images
    image_paths = []
    for ext in ['*.png', '*.PNG', '*.jpg', '*.jpeg']:
        image_paths.extend(glob(os.path.join(plates_dir, ext)))
    
    if not image_paths:
        print(f"❌ No images found in {plates_dir}")
        return None
    
    print(f"\n📁 [{split_name}] Found {len(image_paths)} license plate images")
    print(f"🎯 Target size: {target_size[0]}×{target_size[1]} ({target_size[0]*target_size[1]} pixels)")
    
    images = []
    failed = []
    size_stats = {'original_w': [], 'original_h': [], 'scale': [], 'aspect_ratios': []}
    
    for path in tqdm(image_paths, desc=f"Processing {split_name}"):
        try:
            # Read image (PNG support)
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if img is None:
                failed.append(path)
                continue
            
            # Handle transparency if present (PNG with alpha channel)
            if len(img.shape) == 3 and img.shape[2] == 4:
                # Convert RGBA to RGB by removing alpha
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            
            # Convert to grayscale
            if len(img.shape) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Store original size
            h, w = img.shape[:2]
            size_stats['original_w'].append(w)
            size_stats['original_h'].append(h)
            size_stats['aspect_ratios'].append(w / h)
            
            target_w, target_h = target_size
            
            # Improved adaptive scaling for European plates
            # European plates have aspect ratio ~3.2:1, target is 3:1
            target_aspect = target_w / target_h  # 3.0
            original_aspect = w / h
            
            # Decide strategy based on original size
            if w < 80 or h < 30:
                # Very small plates - preserve detail by upscaling less aggressively
                scale = min(target_w / w, target_h / h)
                scale = min(scale, 1.5)  # Don't upscale too much
                interpolation = cv2.INTER_CUBIC  # Better for upscaling
            elif w > 1000 or h > 300:
                # Very large plates - aggressive but smart downscaling
                scale = min(target_w / w, target_h / h)
                # Use pyramid downscaling for large images (better quality)
                if scale < 0.2:
                    # Multi-stage downscaling for extreme cases
                    temp = img.copy()
                    while temp.shape[1] // 2 > target_w and temp.shape[0] // 2 > target_h:
                        temp = cv2.pyrDown(temp)
                    # Final resize to target
                    resized = cv2.resize(temp, (target_w, target_h), 
                                       interpolation=cv2.INTER_LANCZOS4)
                    # Skip the rest of the resizing logic
                    canvas = prepare_canvas(resized, target_size)
                    images.append(canvas.flatten())
                    continue
                else:
                    interpolation = cv2.INTER_AREA  # Best for downscaling
            else:
                # Normal sized plates - balanced approach
                scale = min(target_w / w, target_h / h)
                interpolation = cv2.INTER_LANCZOS4
            
            size_stats['scale'].append(scale)
            
            # Handle non-extreme cases
            if w <= 1000 or h <= 300:
                new_w = max(1, int(w * scale))
                new_h = max(1, int(h * scale))
                resized = cv2.resize(img, (new_w, new_h), interpolation=interpolation)
                canvas = prepare_canvas(resized, target_size)
                images.append(canvas.flatten())
            
        except Exception as e:
            failed.append(f"{path}: {str(e)}")
    
    if not images:
        print(f"❌ [{split_name}] No valid images processed!")
        return None
    
    images_array = np.array(images, dtype=np.float32)
    
    # Print statistics
    print(f"\n✅ [{split_name}] Successfully processed {len(images_array)} images")
    print(f"📊 Data shape: {images_array.shape}")
    print(f"📈 Value range: [{images_array.min():.3f}, {images_array.max():.3f}]")
    print(f"\n📐 Original sizes:")
    print(f"   Width:  {min(size_stats['original_w']):3d} - {max(size_stats['original_w']):3d} pixels")
    print(f"   Height: {min(size_stats['original_h']):3d} - {max(size_stats['original_h']):3d} pixels")
    print(f"   Aspect ratio: {min(size_stats['aspect_ratios']):.2f} - {max(size_stats['aspect_ratios']):.2f} (avg: {np.mean(size_stats['aspect_ratios']):.2f})")
    
    if size_stats['scale']:
        print(f"   Scale:  {min(size_stats['scale']):.3f} - {max(size_stats['scale']):.3f}x")
    
    if failed:
        print(f"\n⚠️ [{split_name}] Failed to load {len(failed)} images")
    
    return images_array

def prepare_canvas(resized_img, target_size):
    """Place resized image onto canvas with intelligent positioning"""
    target_w, target_h = target_size
    h, w = resized_img.shape[:2]
    
    # Create canvas with dark gray background
    canvas = np.ones((target_h, target_w), dtype=np.float32) * 0.3
    
    # Center the resized image
    x_offset = (target_w - w) // 2
    y_offset = (target_h - h) // 2
    
    # Normalize to [0, 1]
    normalized = resized_img.astype(np.float32) / 255.0
    
    # Place in canvas
    canvas[y_offset:y_offset+h, x_offset:x_offset+w] = normalized
    
    # Apply gentle contrast enhancement
    mean_val = canvas.mean()
    canvas = (canvas - mean_val) * 1.15 + mean_val  # Slightly less aggressive
    canvas = np.clip(canvas, 0, 1)
    
    return canvas

def visualize_results(images_array, target_size, split_name, num_samples=12):
    """Display sample plates after preprocessing"""
    target_w, target_h = target_size
    rows = 3
    cols = 4
    fig, axes = plt.subplots(rows, cols, figsize=(12, 9))
    
    for i, ax in enumerate(axes.flat):
        if i < len(images_array) and i < num_samples:
            img = images_array[i].reshape(target_h, target_w)
            ax.imshow(img, cmap='gray')
            ax.set_title(f'{split_name} {i+1}')
            ax.axis('off')
        else:
            ax.axis('off')
    
    plt.suptitle(f'Prepared License Plates - {split_name} ({target_w}×{target_h})', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'plate_preview_{split_name}.png', dpi=150)
    plt.close()

def save_dataset_stats(images_array, target_size, output_dir, split_name):
    """Save comprehensive dataset statistics"""
    target_w, target_h = target_size
    
    stats = {
        'split': split_name,
        'num_samples': len(images_array),
        'image_width': target_w,
        'image_height': target_h,
        'total_pixels': target_w * target_h,
        'mean_pixel': float(images_array.mean()),
        'std_pixel': float(images_array.std()),
        'min_pixel': float(images_array.min()),
        'max_pixel': float(images_array.max()),
        'percentile_25': float(np.percentile(images_array, 25)),
        'percentile_50': float(np.percentile(images_array, 50)),
        'percentile_75': float(np.percentile(images_array, 75))
    }
    
    # Save as text file
    stats_file = os.path.join(output_dir, f'dataset_stats_{split_name}.txt')
    with open(stats_file, 'w') as f:
        f.write("LICENSE PLATE DATASET STATISTICS\n")
        f.write("=" * 40 + "\n")
        for key, value in stats.items():
            f.write(f"{key:15}: {value}\n")
    
    # Plot histogram
    plt.figure(figsize=(10, 4))
    plt.hist(images_array.flatten(), bins=50, alpha=0.7, color='blue')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.title(f'Distribution of Pixel Values - {split_name}')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, f'pixel_distribution_{split_name}.png'))
    plt.close()

def process_all_splits(base_dataset_path, output_dir, target_size=(96, 32)):
    """Process all three splits (train, val, test)"""
    splits = ['train', 'val', 'test']
    prepared_data = {}
    
    print("=" * 60)
    print("LICENSE PLATE PREPARATION - EUROPEAN DATASET")
    print("=" * 60)
    print(f"📁 Dataset path: {base_dataset_path}")
    print(f"🎯 Target size: {target_size[0]}×{target_size[1]}")
    print(f"📐 Aspect ratio target: {target_size[0]/target_size[1]:.2f}:1")
    print("=" * 60)
    
    for split in splits:
        split_path = os.path.join(base_dataset_path, split)
        
        if not os.path.exists(split_path):
            print(f"\n⚠️ Warning: {split_path} does not exist, skipping...")
            continue
            
        print(f"\n{'='*40}")
        print(f"Processing {split.upper()} split")
        print(f"{'='*40}")
        
        plate_images = load_and_preprocess_plates(split_path, target_size, split)
        
        if plate_images is not None:
            output_file = os.path.join(output_dir, f"license_plates_{split}_{target_size[0]}x{target_size[1]}.npy")
            np.save(output_file, plate_images)
            print(f"💾 Data saved to: {output_file}")
            
            visualize_results(plate_images, target_size, split)
            save_dataset_stats(plate_images, target_size, output_dir, split)
            
            # Save sample images
            sample_dir = os.path.join(output_dir, f"samples_{split}")
            os.makedirs(sample_dir, exist_ok=True)
            for i in range(min(20, len(plate_images))):
                img = plate_images[i].reshape(target_size[1], target_size[0])
                plt.imsave(os.path.join(sample_dir, f"plate_{i:03d}.png"), img, cmap='gray')
            
            print(f"🖼️ Sample images saved to: {sample_dir}")
            prepared_data[split] = plate_images
    
    return prepared_data

if __name__ == "__main__":
    # CONFIGURATION
    DATASET_PATH = "european_license_plates_dataset/dataset_final"
    OUTPUT_DIR = "draw_plate_data_prepared_european"
    TARGET_SIZE = (96, 32)  # (width, height)
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Process all splits
    prepared_data = process_all_splits(DATASET_PATH, OUTPUT_DIR, TARGET_SIZE)
    
    if prepared_data:
        print("\n" + "=" * 60)
        print("✅ PREPARATION COMPLETE!")
        print("=" * 60)
        print("\n📋 SUMMARY:")
        for split, data in prepared_data.items():
            print(f"   {split.upper()}: {data.shape[0]} images, shape: {data.shape}")
        
        print("\n✅ The script successfully handles:")
        print("   • European plate aspect ratios (~3.27:1 → 3.00:1)")
        print("   • Extreme size variations (63×25 to 2967×804)")
        print("   • PNG files with transparency")
        print("   • Multi-stage downscaling for very large plates")