# FireNet: Localized Firearm Detection
## Course project for CAP5415.
- Developed by Connor Reynolds and Joshua Wells

### Instructions
- To start, you need to get a local copy of our database using the create_and_visualize.py script.
- Once this script has been run, run the export_to_yolo.py script to create the datasets directory with the files needed by the training and prediction steps.
- If you would like to train the model yourself, feel free to run train_yolo.py to get your own set of weights, then update predict.py accordingly to use your new weights.
- After you have either created your own weights or have decided to use ours, run the predict.py script, which will place results in the inference_results directory.
    - If you would like to see where the model guessed right or wrong, use the error_analysis.py script. If you are using your own weights, make sure to update the reference to them in this script too.

### Necessary Libraries
- FiftyOne
- Ultralytics
- CV2