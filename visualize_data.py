# Data Visualization Script Using FiftyOne
# Joshua Wells and Connor Reynolds
# CAP 5415  

import fiftyone as fo

final_firearm_dataset = fo.load_dataset("firearm-detection-final")

if __name__ == "__main__":
    session = fo.launch_app(final_firearm_dataset)
    print("App launched. Press Ctrl+C in this terminal to close.")
    session.wait(-1) # Use -1 to wait indefinitely