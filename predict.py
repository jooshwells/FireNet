from ultralytics import YOLO
import os
import glob

def predict():
    # Load model
    model = YOLO('runs/detect/firearm_detector/weights/best.pt')

    # Get all images in the validation folder
    val_images_path = "datasets/yolo_dataset/images/val"
    # Use glob to find all .jpg files
    image_files = glob.glob(os.path.join(val_images_path, "*.jpg"))
    
    print(f"Found {len(image_files)} images. Scanning for detections...")

    # Create a folder to save positive detections
    output_dir = "inference_results"
    os.makedirs(output_dir, exist_ok=True)

    detections_found = 0

    # Loop through the first 20 images (or all of them if you prefer)
    for i, image_path in enumerate(image_files):
        
        # Lower confidence to 0.25 to catch obscure guns
        results = model.predict(image_path, conf=0.25, verbose=False)
        result = results[0]

        # If boxes are detected (len(result.boxes) > 0)
        if len(result.boxes) > 0:
            detections_found += 1
            filename = os.path.basename(image_path)
            save_path = os.path.join(output_dir, f"detected_{filename}")
            
            # Save the image with drawn boxes
            result.save(filename=save_path)
            print(f"  -> Gun detected in {filename}! Saved to {output_dir}")
    
    print(f"\nScan complete. Found firearms in {detections_found} images.")
    print(f"Check the '{output_dir}' folder to see them.")

if __name__ == "__main__":
    predict()