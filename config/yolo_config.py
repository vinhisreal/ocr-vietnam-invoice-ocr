import os
class Config:
    data_dir = '/kaggle/input/vietnamese-receipts-mc-ocr-2021'
    image_dir = '/kaggle/input/vietnamese-receipts-mc-ocr-2021/preprocessor/preprocessor/imgs'
    label_dir = '/kaggle/input/vietnamese-receipts-mc-ocr-2021/dataset/text_detector/txt'
    
    output_dir = '/kaggle/working/output'
    yolo_dataset_dir = '/kaggle/working/dataset_yolo'
    
    batch_size = 16
    img_size = 640
    epochs = 50
    
    train_ratio = 0.8
    val_ratio = 0.1
    test_ratio = 0.1

os.makedirs(Config.output_dir, exist_ok=True)
os.makedirs(Config.yolo_dataset_dir, exist_ok=True)