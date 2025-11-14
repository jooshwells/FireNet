import fiftyone as fo
import fiftyone.utils.openimages as foou
import fiftyone.zoo as foz

# --- Define our classes ---
positive_classes = ["Handgun", "Shotgun", "Rifle"]

# (CRITICAL) Use "distractor" classes, not "Cheese"
negative_classes = [
    "Tool", "Drill", "Hammer", "Wrench", 
    "Flashlight", "Remote control", "Backpack"
]

# --- 1. Load Positive Samples ---
print("Loading positive samples...")
positive_dataset = foz.load_zoo_dataset(
    "open-images-v7",
    split="train",
    label_types=["detections"],
    classes=positive_classes,
    max_samples=1000,  # <-- Increased for a better test
    dataset_name="oiv7-firearms-positive-temp"
)
print(f"Loaded {len(positive_dataset)} positive samples.")


# --- 2. Load Negative Samples (with contamination prevention) ---
print("Loading negative samples...")
negative_dataset = foz.load_zoo_dataset(
    "open-images-v7",
    split="train",
    label_types=["detections"],
    classes=negative_classes,
    # (IMPORTANT) This prevents loading images that also have firearms
    exclude_classes=positive_classes, 
    max_samples=2000,  # <-- Keep a 1:2 or 1:3 ratio
    dataset_name="oiv7-firearms-negative-temp"
)
print(f"Loaded {len(negative_dataset)} negative samples.")

# --- 3. Clear the "distractor" annotations ---
print("Clearing labels from negative samples...")
negative_dataset.clear_sample_field("ground_truth")


# --- 4. Create a new, final dataset and merge them ---
print("Merging datasets...")
dataset_name = "firearm-detection-final"
if fo.dataset_exists(dataset_name):
    fo.delete_dataset(dataset_name)
    print(f"Deleted old dataset: {dataset_name}")

final_firearm_dataset = fo.Dataset(dataset_name)
final_firearm_dataset.persistent = True # Keep this dataset by default

# Add all samples from both datasets
final_firearm_dataset.add_samples(positive_dataset)
final_firearm_dataset.add_samples(negative_dataset)

# Clean up the temporary datasets
fo.delete_dataset("oiv7-firearms-positive-temp")
fo.delete_dataset("oiv7-firearms-negative-temp")

print(f"Final dataset created with {len(final_firearm_dataset)} total samples.")

# (BEST PRACTICE) Use __name__ == "__main__" to safely launch the app
if __name__ == "__main__":
    session = fo.launch_app(final_firearm_dataset)
    print("App launched. Press Ctrl+C in this terminal to close.")
    session.wait(-1) # Use -1 to wait indefinitely