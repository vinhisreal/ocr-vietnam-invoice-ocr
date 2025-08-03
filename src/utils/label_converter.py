import cv2, os
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