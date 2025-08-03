import os
import glob
import shutil
import random
import cv2
from tqdm import tqdm

def convert_to_yolo_format(image_path, label_path, output_dir, class_id=0):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Không thể đọc ảnh {image_path}")
        return False
    
    img_height, img_width = img.shape[:2]
    
    try:
        with open(label_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"Lỗi khi đọc file {label_path}: {str(e)}")
        return False

    os.makedirs(output_dir, exist_ok=True)
    image_name = os.path.basename(image_path)
    output_label_path = os.path.join(output_dir, os.path.splitext(image_name)[0] + '.txt')
    
    with open(output_label_path, 'w', encoding='utf-8') as out_file:
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                coords = [float(coord) for coord in line.split(',') if coord]
                if len(coords) != 8:
                    print(f"Sai định dạng bounding box trong {label_path}: {line}")
                    continue
                
                x_coords = [coords[i] for i in range(0, len(coords), 2)]
                y_coords = [coords[i] for i in range(1, len(coords), 2)]

                x_min, x_max = min(x_coords), max(x_coords)
                y_min, y_max = min(y_coords), max(y_coords)

                x_center = (x_min + x_max) / (2 * img_width)
                y_center = (y_min + y_max) / (2 * img_height)
                width = (x_max - x_min) / img_width
                height = (y_max - y_min) / img_height

                if width <= 0 or height <= 0:
                    continue

                # Clamp giá trị vào [0,1]
                x_center = max(0, min(1, x_center))
                y_center = max(0, min(1, y_center))
                width = max(0, min(1, width))
                height = max(0, min(1, height))

                yolo_line = f"{class_id} {x_center} {y_center} {width} {height}\n"
                out_file.write(yolo_line)

            except Exception as e:
                print(f"Lỗi khi xử lý line trong {label_path}: {line} - {str(e)}")
                continue
    
    return True


def prepare_yolo_dataset(Config):
    for split in ['train', 'val', 'test']:
        for subdir in ['images', 'labels']:
            os.makedirs(os.path.join(Config.yolo_dataset_dir, split, subdir), exist_ok=True)

    all_images = glob.glob(os.path.join(Config.image_dir, '*.jpg'))
    print(f"Tìm thấy {len(all_images)} ảnh.")
    
    random.shuffle(all_images)

    total_images = len(all_images)
    train_count = int(total_images * Config.train_ratio)
    val_count = int(total_images * Config.val_ratio)

    train_images = all_images[:train_count]
    val_images = all_images[train_count:train_count + val_count]
    test_images = all_images[train_count + val_count:]

    print(f"Chia thành: {len(train_images)} train, {len(val_images)} val, {len(test_images)} test")

    datasets = {
        'train': train_images,
        'val': val_images,
        'test': test_images
    }

    for split, images in datasets.items():
        print(f"Đang xử lý tập {split}...")
        success_count = 0

        for img_path in tqdm(images):
            img_filename = os.path.basename(img_path)
            img_name_no_ext = os.path.splitext(img_filename)[0]

            label_path = os.path.join(Config.label_dir, img_name_no_ext + '.txt')

            if not os.path.exists(label_path):
                continue

            dst_img_path = os.path.join(Config.yolo_dataset_dir, split, 'images', img_filename)
            try:
                shutil.copy(img_path, dst_img_path)
            except Exception as e:
                print(f"Lỗi khi copy ảnh {img_path}: {str(e)}")
                continue

            success = convert_to_yolo_format(
                img_path,
                label_path,
                os.path.join(Config.yolo_dataset_dir, split, 'labels')
            )

            if success:
                success_count += 1

        print(f"✅ {success_count}/{len(images)} ảnh trong tập {split} đã được xử lý thành công.")

    # Ghi file dataset.yaml
    yaml_content = f"""
path: {Config.yolo_dataset_dir}
train: train/images
val: val/images
test: test/images
nc: 1
names: ['text']
"""
    with open(os.path.join(Config.yolo_dataset_dir, 'dataset.yaml'), 'w') as f:
        f.write(yaml_content.strip())

    print(f"🎉 Dataset YOLOv8 đã sẵn sàng tại: {Config.yolo_dataset_dir}")
    return os.path.join(Config.yolo_dataset_dir, 'dataset.yaml')
