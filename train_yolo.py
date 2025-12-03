# Training Script for YOLOv8 Nano
# Joshua Wells and Connor Reynolds
# CAP 5415  

from ultralytics import YOLO

def main():
    print("Loading YOLOv8-Nano model...")
    model = YOLO('yolov8n.pt') # trying the medium model

    data_yaml_path = "./datasets/yolo_dataset/dataset.yaml"

    print("Starting training...")

    results = model.train(
        data=data_yaml_path,
        epochs=100,
        patience=25,
        imgsz=640,
        batch=16,
        name='firearm_detector',
        device='0',
        workers=1,
        cache=False,
        
        dropout=0.25,  
        cls=1.0,       
        mixup=0.15,    
        copy_paste=0.1 
    )

    print("Evaluating model...")
    metrics = model.val()
    
    # Print the specific metric for "map50-95" (the standard academic metric for detection)
    print(f"Mean Average Precision (mAP): {metrics.box.map}")

if __name__ == '__main__':
    main()