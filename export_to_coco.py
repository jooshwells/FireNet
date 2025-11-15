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
    
    # Clear any old tags
    dataset.untag_samples("train")
    dataset.untag_samples("val")

    # Define split ratio
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
    
    # --- 3. Run the export (this is from the next step) ---
    # The rest of the script from my previous answer will now work perfectly
    # using 'train_view' and 'val_view'.
    
    # (Example: Define export paths)
    base_export_dir = "./datasets" # <-- UPDATE THIS PATH
    coco_train_dir = os.path.join(base_export_dir, "coco_train")
    coco_val_dir = os.path.join(base_export_dir, "coco_val")
    label_field = "ground_truth"

    # Export the training split
    print(f"Exporting TRAIN split to: {coco_train_dir}")
    train_view.export(
        export_dir=coco_train_dir,
        dataset_type=types.COCODetectionDataset,
        label_field=label_field,
    )

    # Export the validation split
    print(f"Exporting VALIDATION split to: {coco_val_dir}")
    val_view.export(
        export_dir=coco_val_dir,
        dataset_type=types.COCODetectionDataset,
        label_field=label_field,
    )

    print("\n--- Export Complete! ---")