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
false_negatives = 0
localization_error = 0

image_paths = [os.path.normpath(i) for i in glob.glob("datasets/yolo_dataset/images/val/*.jpg")]
label_paths = [os.path.normpath(i) for i in glob.glob("datasets/yolo_dataset/labels/val/*.txt")]

for image_path, label_path in zip(image_paths, label_paths):
    print(f"Image: {image_path}")

    img = cv2.imread(image_path)
    h, w, _ = img.shape

    model = YOLO('runs/detect/firearm_detector/weights/best.pt')
    gt_list = get_ground_truth_pixels(label_path, w, h)
    pred_list = get_prediction_pixels(model, image_path)

    print("Ground Truth:", gt_list)
    print("Predictions:", pred_list)

    idx = 0

    while idx < len(gt_list) and idx < len(pred_list):
        boxA = gt_list[idx]
        boxB = pred_list[idx]
        
        if(boxA['class_id'] == boxB['class_id']):
            print(f"IOU for boxA: {boxA['class_id']} and boxB: {boxB['class_id']} = {calculate_iou(boxA['bbox'], boxB['bbox'])}")
        idx += 1
    
    print("--------------------")
    

        