# Prediction Script Using Trained Model
# Joshua Wells and Connor Reynolds
# CAP 5415  

from ultralytics import YOLO
import os
import cv2
import glob

def predict():
    # Load model
    model = YOLO('weights/best.pt') # Using pre-trained weights. Change to your own weights under runs/<model-name> if self-trained

    # Get all images in the validation folder
    val_images_path = "datasets/yolo_dataset/images/val"
    # Use glob to find all .jpg files
    image_files = glob.glob(os.path.join(val_images_path, "*.jpg"))
    
    print(f"Found {len(image_files)} images. Scanning for detections...")

    # Create a folder to save positive detections
    output_dir = "inference_results"
    os.makedirs(output_dir, exist_ok=True)

    detections_found = 0

    for i, image_path in enumerate(image_files):
        
        results = model.predict(image_path, conf=0.25, verbose=False)
        result = results[0]

        # If boxes are detected (len(result.boxes) > 0)
        if len(result.boxes) > 0:
            img = result.orig_img.copy()

            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                
                box_color = (0, 255, 0)  # Green
                thickness = 3
                
                cls_id = int(box.cls[0])
                class_name = model.names[cls_id]

                cv2.rectangle(img, (x1, y1), (x2, y2), box_color, thickness)

                conf = float(box.conf[0])
                label = f"{class_name}: {conf:.2f}"
                
                t_size = cv2.getTextSize(label, 0, fontScale=0.6, thickness=1)[0]
                cv2.rectangle(img, (x1, y1 - t_size[1] - 4), (x1 + t_size[0], y1), box_color, -1)
                
                cv2.putText(img, label, (x1, y1 - 4), 0, 0.6, (0, 0, 0), 2)

            filename = os.path.basename(image_path)
            save_path = os.path.join(output_dir, f"{filename}")
            cv2.imwrite(save_path, img)

            print(f"  -> Gun detected, saved finding to {filename}")
            detections_found +=1
        
    
    print(f"\nScan complete. Found firearms in {detections_found} images.")
    print(f"Check the '{output_dir}' folder to see them.")

if __name__ == "__main__":
    predict()