from ultralytics import YOLO
import os

def train_segmentation(dataset_path, output_path, epochs, imgsz):
    os.makedirs(output_path, exist_ok=True)

    model = YOLO("yolo11s-seg.pt")

    model.train(
        data=dataset_path,
        epochs=epochs,
        imgsz=imgsz,
        project=output_path,
        name="segmentation_exp",
        save=True
    )

    print(f"Training completed. Model saved in: {output_path}")

if __name__ == "__main__":
    dataset_yaml = "/home/traxian/Ahmed_intern/samworth/data/processed_data/samworth-walkers.v3i.yolov11/data.yaml"
    output_dir = "/home/traxian/Ahmed_intern/samworth/models"
    train_segmentation(dataset_yaml, output_dir, epochs=300, imgsz=640)
