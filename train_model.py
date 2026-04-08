from ultralytics import YOLO
import torch

def main():

    # Check GPU
    if torch.cuda.is_available():
        device = 0
        print("Using GPU:", torch.cuda.get_device_name(0))
    else:
        device = "cpu"
        print("CUDA not available, using CPU")

    # Load YOLOv8 pretrained model
    model = YOLO("yolov8n.pt")

    # Train
    model.train(
        data="DATASETS/data.yaml",   # <-- YOUR YAML PATH
        epochs=50,
        imgsz=640,
        batch=8,
        device=device,
        workers=2,
        project="pothole_training",
        name="pothole_detector"
    )

    print("Training Complete!")

if __name__ == "__main__":
    main()