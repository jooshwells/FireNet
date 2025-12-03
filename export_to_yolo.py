import fiftyone as fo
import fiftyone.types as types
import os

# --- 1. Load Your Final Dataset ---
dataset_name = "firearm-detection-final"
print(f"Loading dataset '{dataset_name}'...")

if not fo.dataset_exists(dataset_name):
    print(f"Error: Dataset '{dataset_name}' not found.")
else:
    dataset = fo.load_dataset(dataset_name)
    dataset.persistent = True # Make sure it's persistent

    # --- 2. Split into Train and Validation ---
    print(f"Splitting dataset {len(dataset)} samples...")
    
    # Clear any old tags to ensure fresh split
    dataset.untag_samples("train")
    dataset.untag_samples("val")

    # Define split ratio (80% train, 20% val)
    split_ratio = 0.8
    train_size = int(len(dataset) * split_ratio)

    # Get a shuffled view of the sample IDs
    shuffled_ids = dataset.shuffle().values("id")

    # Get the IDs for each split
    train_ids = shuffled_ids[:train_size]
    val_ids = shuffled_ids[train_size:]

    # Use select() to get views and tag them
    train_view = dataset.select(train_ids)
    train_view.tag_samples("train")

    val_view = dataset.select(val_ids)
    val_view.tag_samples("val")
    
    # Save the tags to the underlying dataset
    dataset.save()

    print(f"Total samples: {len(dataset)}")
    print(f"Training samples (tagged 'train'): {len(train_view)}")
    print(f"Validation samples (tagged 'val'): {len(val_view)}")
    
    # --- 3. Export to YOLOv5 Format ---
    
    # Define the specific classes we want to detect.
    target_classes = ["Handgun", "Shotgun", "Rifle"]
    
    # Define export path
    base_export_dir = "./datasets" 
    yolo_dir = os.path.join(base_export_dir, "yolo_dataset")
    label_field = "ground_truth" # The field where we stored detections

    print(f"Exporting to YOLO format at: {yolo_dir}")

    # Export the training split
    print("Exporting TRAIN split...")
    train_view.export(
        export_dir=yolo_dir,
        dataset_type=types.YOLOv5Dataset,
        label_field=label_field,
        split="train",
        classes=target_classes,
    )

    # Export the validation split
    print("Exporting VALIDATION split...")
    val_view.export(
        export_dir=yolo_dir,
        dataset_type=types.YOLOv5Dataset,
        label_field=label_field,
        split="val",
        classes=target_classes,
    )

    print("\n--- Export Complete! ---")
    print(f"Your dataset.yaml and images are located in {yolo_dir}")