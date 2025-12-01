import os
import cv2
from ultralytics import YOLO
import glob

def get_ground_truth_pixels(label_path, img_w, img_h):
    """
    Reads a YOLO .txt file and converts normalized coordinates to pixels.
    Returns a list of dicts: [{'class_id': 0, 'bbox': [x1, y1, x2, y2]}]
    """
    gt_data = []
    
    if not os.path.exists(label_path):
        return gt_data

    with open(label_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        cls_id = int(parts[0])
        
        # Normalized Center-XYWH
        nx, ny, nw, nh = map(float, parts[1:])

        # Convert to Pixel Top-Left-XY-Bottom-Right-XY
        # 1. Un-normalize
        w_px = nw * img_w
        h_px = nh * img_h
        x_center_px = nx * img_w
        y_center_px = ny * img_h

        # 2. Calculate corners
        x1 = int(x_center_px - (w_px / 2))
        y1 = int(y_center_px - (h_px / 2))
        x2 = int(x_center_px + (w_px / 2))
        y2 = int(y_center_px + (h_px / 2))

        gt_data.append({
            'class_id': cls_id,
            'bbox': [x1, y1, x2, y2]
        })
        
    return gt_data

def get_prediction_pixels(model, image_path):
    """
    Runs inference and returns predicted boxes in pixels.
    Returns a list of dicts: [{'class_id': 0, 'bbox': [x1, y1, x2, y2], 'conf': 0.85}]
    """
    # Run inference
    results = model.predict(image_path, conf=0.25, verbose=False)
    result = results[0]
    
    pred_data = []
    
    # Iterate through detections
    for box in result.boxes:
        # .xyxy gives specific pixel coordinates [x1, y1, x2, y2]
        coords = box.xyxy[0].cpu().numpy().astype(int).tolist()
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        
        pred_data.append({
            'class_id': cls_id,
            'bbox': coords,
            'conf': conf
        })
        
    return pred_data

def calculate_iou(boxA, boxB):
    '''
    Calculates the Intersection over Union (IoU) of two boxes
    Parameters: boxA (ground truth box) and boxB (predicted box)
    '''
    # box = [x1, y1, x2, y2]
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

true_positives = 0
false_positives = 0
false_negatives = 0
localization_error = 0

# Folder to save any errored predictions
output_dir = "error_analysis_error_review"
os.makedirs(output_dir, exist_ok=True)

image_paths = [os.path.normpath(i) for i in glob.glob("datasets/yolo_dataset/images/val/*.jpg")]
label_paths = [os.path.normpath(i) for i in glob.glob("datasets/yolo_dataset/labels/val/*.txt")]

model = YOLO('runs/detect/firearm_detector/weights/best.pt')

for image_path, label_path in zip(image_paths, label_paths):
    print(f"Image: {image_path}")

    img = cv2.imread(image_path)
    h, w, _ = img.shape

    gt_list = get_ground_truth_pixels(label_path, w, h)
    pred_list = get_prediction_pixels(model, image_path)

    print("Ground Truth:", gt_list)
    print("Predictions:", pred_list)

    # For errored image annotations
    box_color_gt = (0, 255, 0)  # Green
    box_color_pred = (0, 0, 255)  # Red
    thickness = 3

    idx = 0
    iou_thresh = 0.5
    is_wrong = False
    while idx < len(gt_list) and idx < len(pred_list):
        boxA = gt_list[idx]
        boxB = pred_list[idx]

        # Prediction matches same class as ground truth
        if(boxA['class_id'] == boxB['class_id']):
            iou = calculate_iou(boxA['bbox'], boxB['bbox'])
            print(f"IOU for boxA: {boxA['class_id']} and boxB: {boxB['class_id']} = {iou}")

            # Prediction was correct
            if (iou > iou_thresh):
                true_positives += 1
            # Prediction had localization problem
            else:
                false_positives += 1
                is_wrong = True
                localization_error += 1
        # Prediction predicted wrong class
        else:
            false_positives += 1 # Wrong class predicted
            false_negatives += 1 # Correct class not correctly detected
            is_wrong = True

        # Draw bounding boxes if prediction wrong
        if is_wrong:
            x1, y1, x2, y2 = boxA['bbox']
            cv2.rectangle(img, (x1, y1), (x2, y2), box_color_gt, thickness)
            x1, y1, x2, y2 = boxB['bbox']
            cv2.rectangle(img, (x1, y1), (x2, y2), box_color_pred, thickness)

        idx += 1

    # Get any leftover false positives
    for idx in range(len(pred_list)):
        if idx >= len(gt_list):
            false_positives += 1
            is_wrong = True
            x1, y1, x2, y2 = pred_list[idx]['bbox']
            cv2.rectangle(img, (x1, y1), (x2, y2), box_color_pred, thickness)

    # Get any leftover false negatives
    for idx in range(len(gt_list)):
        if idx >= len(pred_list):
            false_negatives += 1
            is_wrong = True
            x1, y1, x2, y2 = gt_list[idx]['bbox']
            cv2.rectangle(img, (x1, y1), (x2, y2), box_color_gt, thickness)

    # Save errored prediction images with bounding boxes
    if is_wrong:
        filename = os.path.basename(image_path)
        save_path = os.path.join(output_dir, f"{filename}")
        cv2.imwrite(save_path, img)

    print("--------------------")
    

        