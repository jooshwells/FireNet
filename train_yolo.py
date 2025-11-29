from ultralytics import YOLO

def main():
    print("Loading YOLOv8-Nano model...")
    model = YOLO('yolov8n.pt') 

    data_yaml_path = "./datasets/yolo_dataset/dataset.yaml"

    print("Starting training...")
    results = model.train(
        data=data_yaml_path,
        epochs=15,             
        imgsz=640,            
        batch=16,              
        name='firearm_detector', # Name of the sub-folder where results will be saved
        device='0',             # Use '0' for GPU, or 'cpu' if you don't have CUDA
        workers=1,
        cache=False
    )

    print("Evaluating model...")
    metrics = model.val()
    
    # Print the specific metric for "map50-95" (the standard academic metric for detection)
    print(f"Mean Average Precision (mAP): {metrics.box.map}")

if __name__ == '__main__':
    main()