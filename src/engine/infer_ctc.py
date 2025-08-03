import cv2
import torch
from src.utils.preprocessing import preprocess_image
from src.utils.augmentations import get_ocr_transforms
from src.engine.ctc_utils import ctc_greedy_decode
def predict_from_image_path(model, image_path, tokenizer, device, preprocess=True):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Không thể đọc ảnh từ {image_path}")
        return ""
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    if preprocess:
        image = preprocess_image(image)
    
    transform = get_ocr_transforms(is_train=False)
    image_tensor = transform(image=image)['image']
    
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.unsqueeze(0).to(device)
        log_probs = model(image_tensor)
        prediction = ctc_greedy_decode(log_probs, tokenizer)[0]
    
    return prediction