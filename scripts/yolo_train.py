from pathlib import Path
import shutil
import os
import torch
from ultralytics import YOLO
from src.data.yolo_preparation import prepare_yolo_dataset
from config.yolo_config import Config


def train_yolov8():
    """
    Huấn luyện YOLOv8 trên dataset đã chuẩn bị.
    """
    print("Bắt đầu huấn luyện YOLOv8...")

    # Chuẩn bị dataset và lấy đường dẫn tới file YAML
    dataset_yaml_path = prepare_yolo_dataset(Config)
    
    # Load pre-trained YOLOv8 model
    model = YOLO('yolov8n.pt')
    print("Khởi tạo model YOLOv8 thành công")
    
    # Huấn luyện
    results = model.train(
        data=dataset_yaml_path,
        epochs=Config.epochs,
        imgsz=Config.img_size,
        batch=Config.batch_size,
        name='yolov8_receipts_text_detection',
        device='0' if torch.cuda.is_available() else 'cpu',
        patience=10  
    )
    
    print("Đã hoàn thành huấn luyện YOLOv8!")

    # Sao lưu model tốt nhất
    model_path = Path('runs/detect/yolov8_receipts_text_detection/weights/best.pt')
    if model_path.exists():
        shutil.copy(str(model_path), os.path.join(Config.output_dir, 'yolov8_best.pt'))
        print(f"Đã lưu model tốt nhất tại {os.path.join(Config.output_dir, 'yolov8_best.pt')}")
    else:
        print("Không tìm thấy file model best.pt!")

    return model


# Chạy thử huấn luyện
if __name__ == "__main__":
    print("Thử huấn luyện với YOLOv8...")
    model = train_yolov8()
