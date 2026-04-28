# prepare_all_plates_combined.py
import numpy as np
import cv2
import os
from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def load_plates_from_directory(plates_dir, target_size=(96, 32), source_name=""):
    """
    Load license plates from a single directory with adaptive preprocessing
    """
    # Find all images
    image_paths = []
    for ext in ['*.png', '*.PNG', '*.jpg', '*.jpeg', '*.JPG', '*.JPEG']:
        image_paths.extend(glob(os.path.join(plates_dir, ext)))
    
    if not image_paths:
        print(f"❌ No images found in {plates_dir}")
        return None, [], []
    
    print(f"\n📁 [{source_name}] Found {len(image_paths)} license plate images")
    print(f"🎯 Target size: {target_size[0]}×{target_size[1]}")
    
    images = []
    failed = []
    source_paths = []
    size_stats = {'original_w': [], 'original_h': [], 'aspect_ratios': []}
    
    for path in tqdm(image_paths, desc=f"Processing {source_name}"):
        try:
            # Read image (supports PNG with transparency)
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if img is None:
                failed.append(path)
                continue
            
            # Handle transparency if present (PNG with alpha channel)
            if len(img.shape) == 3 and img.shape[2] == 4:
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
            
            # Smart resizing based on original size
            if w < 80 or h < 30:
                # Very small plates - preserve detail
                scale = min(target_w / w, target_h / h)
                scale = min(scale, 1.5)  # Don't upscale too much
                interpolation = cv2.INTER_CUBIC
            elif w > 1000 or h > 300:
                # Very large plates - aggressive downscaling
                scale = min(target_w / w, target_h / h)
                if scale < 0.2:
                    # Multi-stage downscaling for extreme cases
                    temp = img.copy()
                    while temp.shape[1] // 2 > target_w and temp.shape[0] // 2 > target_h:
                        temp = cv2.pyrDown(temp)
                    resized = cv2.resize(temp, (target_w, target_h), 
                                       interpolation=cv2.INTER_LANCZOS4)
                    canvas = prepare_canvas(resized, target_size, target_w, target_h)
                    images.append(canvas.flatten())
                    source_paths.append(path)
                    continue
                else:
                    interpolation = cv2.INTER_AREA
            else:
                # Normal sized plates - balanced approach
                scale = min(target_w / w, target_h / h)
                interpolation = cv2.INTER_LANCZOS4
            
            # Resize
            new_w = max(1, int(w * scale))
            new_h = max(1, int(h * scale))
            resized = cv2.resize(img, (new_w, new_h), interpolation=interpolation)
            
            # Prepare canvas
            canvas = prepare_canvas(resized, target_size, target_w, target_h)
            images.append(canvas.flatten())
            source_paths.append(path)
            
        except Exception as e:
            failed.append(f"{path}: {str(e)}")
    
    if not images:
        print(f"❌ [{source_name}] No valid images processed!")
        return None, [], []
    
    images_array = np.array(images, dtype=np.float32)
    
    # Print statistics
    print(f"\n✅ [{source_name}] Successfully processed {len(images_array)} images")
    print(f"📊 Data shape: {images_array.shape}")
    print(f"📈 Value range: [{images_array.min():.3f}, {images_array.max():.3f}]")
    if size_stats['original_w']:
        print(f"\n📐 Original sizes:")
        print(f"   Width:  {min(size_stats['original_w']):3d} - {max(size_stats['original_w']):3d} pixels")
        print(f"   Height: {min(size_stats['original_h']):3d} - {max(size_stats['original_h']):3d} pixels")
        print(f"   Aspect ratio: {min(size_stats['aspect_ratios']):.2f} - {max(size_stats['aspect_ratios']):.2f}")
    
    if failed:
        print(f"\n⚠️ [{source_name}] Failed to load {len(failed)} images")
    
    return images_array, source_paths, size_stats

def prepare_canvas(resized_img, target_size, target_w, target_h):
    """Place resized image onto canvas with intelligent positioning"""
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
    if canvas.std() > 0:
        canvas = (canvas - canvas.mean()) * 1.15 + canvas.mean()
        canvas = np.clip(canvas, 0, 1)
    
    return canvas

def visualize_results(images_array, target_size, output_dir, name, num_samples=12):
    """Display sample plates after preprocessing"""
    target_w, target_h = target_size
    rows = 3
    cols = 4
    fig, axes = plt.subplots(rows, cols, figsize=(12, 9))
    
    for i, ax in enumerate(axes.flat):
        if i < len(images_array) and i < num_samples:
            img = images_array[i].reshape(target_h, target_w)
            ax.imshow(img, cmap='gray')
            ax.set_title(f'{name} {i+1}')
            ax.axis('off')
        else:
            ax.axis('off')
    
    plt.suptitle(f'Prepared License Plates - {name} ({target_w}×{target_h})', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'preview_{name}.png'), dpi=150)
    plt.close()

def save_dataset_stats(images_array, target_size, output_dir, name):
    """Save comprehensive dataset statistics"""
    target_w, target_h = target_size
    
    stats = {
        'dataset': name,
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
    stats_file = os.path.join(output_dir, f'stats_{name}.txt')
    with open(stats_file, 'w') as f:
        f.write("LICENSE PLATE DATASET STATISTICS\n")
        f.write("=" * 40 + "\n")
        for key, value in stats.items():
            f.write(f"{key:20}: {value}\n")
    
    # Plot histogram
    plt.figure(figsize=(10, 4))
    plt.hist(images_array.flatten(), bins=50, alpha=0.7, color='blue')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.title(f'Distribution of Pixel Values - {name}')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, f'histogram_{name}.png'))
    plt.close()
    
    print(f"📊 Statistics saved to: {stats_file}")

def save_sample_images(images_array, target_size, output_dir, name, num_samples=20):
    """Save sample images as PNG files"""
    target_w, target_h = target_size
    sample_dir = os.path.join(output_dir, f'samples_{name}')
    os.makedirs(sample_dir, exist_ok=True)
    
    for i in range(min(num_samples, len(images_array))):
        img = images_array[i].reshape(target_h, target_w)
        plt.imsave(os.path.join(sample_dir, f'plate_{i:03d}.png'), img, cmap='gray')
    
    print(f"🖼️ Sample images saved to: {sample_dir}")

def combine_all_plates(target_size=(96, 32), output_dir="draw_plate_data_prepared_combined"):
    """
    Combine all license plates from all sources into a single dataset
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 70)
    print("COMBINED LICENSE PLATE DATASET PREPARATION")
    print("=" * 70)
    print(f"🎯 Target size: {target_size[0]}×{target_size[1]}")
    print("=" * 70)
    
    all_images = []
    all_sources = []
    all_stats = {}
    
    # ============================================
    # SOURCE 1: Original cropped plates
    # ============================================
    original_dir = "license_plates_cropped_whole"
    if os.path.exists(original_dir):
        print(f"\n{'='*40}")
        print("📦 SOURCE 1: Original Cropped Plates")
        print(f"{'='*40}")
        images, paths, stats = load_plates_from_directory(original_dir, target_size, "original")
        if images is not None:
            all_images.append(images)
            all_sources.extend(paths)
            all_stats['original'] = stats
    
    # ============================================
    # SOURCE 2: European dataset (train, val, test)
    # ============================================
    european_base = "european_license_plates_dataset/dataset_final"
    if os.path.exists(european_base):
        for split in ['train', 'val', 'test']:
            split_dir = os.path.join(european_base, split)
            if os.path.exists(split_dir):
                print(f"\n{'='*40}")
                print(f"📦 SOURCE: European Dataset - {split.upper()}")
                print(f"{'='*40}")
                images, paths, stats = load_plates_from_directory(split_dir, target_size, f"european_{split}")
                if images is not None:
                    all_images.append(images)
                    all_sources.extend(paths)
                    all_stats[f'european_{split}'] = stats
    
    # ============================================
    # COMBINE ALL IMAGES
    # ============================================
    if not all_images:
        print("\n❌ No images found in any source directory!")
        return None
    
    print(f"\n{'='*40}")
    print("🔄 COMBINING ALL DATASETS")
    print(f"{'='*40}")
    
    combined_images = np.vstack(all_images)
    
    print(f"\n✅ Combined dataset created:")
    print(f"   Total images: {len(combined_images)}")
    print(f"   Shape: {combined_images.shape}")
    print(f"   Range: [{combined_images.min():.3f}, {combined_images.max():.3f}]")
    print(f"   Mean: {combined_images.mean():.4f}, Std: {combined_images.std():.4f}")
    
    # ============================================
    # SAVE COMBINED DATASET
    # ============================================
    output_file = os.path.join(output_dir, f"license_plates_combined_{target_size[0]}x{target_size[1]}.npy")
    np.save(output_file, combined_images)
    print(f"\n💾 Combined dataset saved to: {output_file}")
    
    # ============================================
    # CREATE TRAIN/VAL/TEST SPLITS (80/10/10)
    # ============================================
    print(f"\n{'='*40}")
    print("📊 CREATING TRAIN/VAL/TEST SPLITS")
    print(f"{'='*40}")
    
    # Split: 80% train, 10% val, 10% test
    train_ratio = 0.8
    val_ratio = 0.1
    test_ratio = 0.1
    
    # First split: train vs temporary (val+test)
    train_data, temp_data = train_test_split(combined_images, test_size=(val_ratio + test_ratio), random_state=42)
    # Second split: val vs test from temporary
    val_data, test_data = train_test_split(temp_data, test_size=(test_ratio/(val_ratio+test_ratio)), random_state=42)
    
    print(f"📊 Split sizes:")
    print(f"   Train: {len(train_data)} images ({len(train_data)/len(combined_images)*100:.1f}%)")
    print(f"   Val:   {len(val_data)} images ({len(val_data)/len(combined_images)*100:.1f}%)")
    print(f"   Test:  {len(test_data)} images ({len(test_data)/len(combined_images)*100:.1f}%)")
    
    # Save splits
    np.save(os.path.join(output_dir, f"train_{target_size[0]}x{target_size[1]}.npy"), train_data)
    np.save(os.path.join(output_dir, f"val_{target_size[0]}x{target_size[1]}.npy"), val_data)
    np.save(os.path.join(output_dir, f"test_{target_size[0]}x{target_size[1]}.npy"), test_data)
    
    # ============================================
    # SAVE SAMPLES AND VISUALIZATIONS
    # ============================================
    print(f"\n{'='*40}")
    print("🎨 GENERATING VISUALIZATIONS")
    print(f"{'='*40}")
    
    # Visualize samples from combined dataset
    visualize_results(combined_images, target_size, output_dir, "combined")
    save_sample_images(combined_images, target_size, output_dir, "combined", num_samples=30)
    
    # Save statistics for each split
    save_dataset_stats(train_data, target_size, output_dir, "train")
    save_dataset_stats(val_data, target_size, output_dir, "val")
    save_dataset_stats(test_data, target_size, output_dir, "test")
    save_dataset_stats(combined_images, target_size, output_dir, "combined")
    
    # ============================================
    # SAVE SUMMARY REPORT
    # ============================================
    summary_file = os.path.join(output_dir, "DATASET_SUMMARY.txt")
    with open(summary_file, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("COMBINED LICENSE PLATE DATASET - SUMMARY REPORT\n")
        f.write("=" * 70 + "\n\n")
        
        f.write(f"TARGET SIZE: {target_size[0]}×{target_size[1]} pixels\n")
        f.write(f"TOTAL IMAGES: {len(combined_images)}\n\n")
        
        f.write("DATA SPLITS:\n")
        f.write(f"  - Train: {len(train_data)} images\n")
        f.write(f"  - Val:   {len(val_data)} images\n")
        f.write(f"  - Test:  {len(test_data)} images\n\n")
        
        f.write("SOURCE BREAKDOWN:\n")
        for source_name, stats in all_stats.items():
            if stats and 'original_w' in stats:
                f.write(f"  - {source_name}: {len(stats['original_w'])} images\n")
        
        f.write("\nSTATISTICS (Combined Dataset):\n")
        f.write(f"  - Mean pixel value: {combined_images.mean():.4f}\n")
        f.write(f"  - Std pixel value:  {combined_images.std():.4f}\n")
        f.write(f"  - Min pixel value:  {combined_images.min():.4f}\n")
        f.write(f"  - Max pixel value:  {combined_images.max():.4f}\n")
    
    print(f"\n📄 Dataset summary saved to: {summary_file}")
    
    # ============================================
    # FINAL OUTPUT
    # ============================================
    print("\n" + "=" * 70)
    print("✅ PREPARATION COMPLETE!")
    print("=" * 70)
    print(f"\n📂 Output directory: {output_dir}")
    print("\n📁 Generated files:")
    print(f"   • license_plates_combined_{target_size[0]}x{target_size[1]}.npy - All images")
    print(f"   • train_{target_size[0]}x{target_size[1]}.npy - Training set (80%)")
    print(f"   • val_{target_size[0]}x{target_size[1]}.npy - Validation set (10%)")
    print(f"   • test_{target_size[0]}x{target_size[1]}.npy - Test set (10%)")
    print(f"   • DATASET_SUMMARY.txt - Complete summary")
    print(f"   • samples_combined/ - 30 sample images")
    print(f"   • preview_*.png - Visual previews")
    print(f"   • stats_*.txt - Statistics per split")
    print(f"   • histogram_*.png - Pixel distributions")
    
    print("\n📋 FOR YOUR DRAW MODEL TRAINING SCRIPT:")
    print(f"   Update the data file path to:")
    print(f"   data_file = \"{output_dir}/train_{target_size[0]}x{target_size[1]}.npy\"")
    print("\n   Or use the validation set for monitoring:")
    print(f"   val_file = \"{output_dir}/val_{target_size[0]}x{target_size[1]}.npy\"")
    
    return combined_images

if __name__ == "__main__":
    # CONFIGURATION
    TARGET_SIZE = (96, 32)  # (width, height) - matches your DRAW model
    OUTPUT_DIR = "draw_plate_data_prepared_combined"
    
    # Run the combined preparation
    combined_data = combine_all_plates(TARGET_SIZE, OUTPUT_DIR)
    
    print("\n🎯 Next steps:")
    print("1. Update your draw_plates_cpu_fixed.py script to use:")
    print(f"   data_file = \"{OUTPUT_DIR}/train_96x32.npy\"")
    print("2. Run training: python draw_plates_cpu_fixed.py")