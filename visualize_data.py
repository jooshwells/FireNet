import fiftyone as fo

final_firearm_dataset = fo.load_dataset("firearm-detection-final")

# (BEST PRACTICE) Use __name__ == "__main__" to safely launch the app
if __name__ == "__main__":
    session = fo.launch_app(final_firearm_dataset)
    print("App launched. Press Ctrl+C in this terminal to close.")
    session.wait(-1) # Use -1 to wait indefinitely