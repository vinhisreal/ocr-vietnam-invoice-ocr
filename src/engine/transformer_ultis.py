import torch
import torch.nn.functional as F

def collate_fn(batch):
    """Hàm custom collate_fn để xử lý các chuỗi có độ dài khác nhau"""
    images, target_inputs, target_outputs, texts = zip(*batch)
    
    heights = [img.shape[1] for img in images]
    widths = [img.shape[2] for img in images]
    
    if len(set(heights)) > 1 or len(set(widths)) > 1:
        print(f"Warning: Không đồng nhất kích thước trong batch: heights={heights}, widths={widths}")
        
        target_height = max(set(heights), key=heights.count)
        target_width = max(set(widths), key=widths.count)
        
        resized_images = []
        for img in images:
            if img.shape[1] != target_height or img.shape[2] != target_width:
                resized = F.interpolate(img.unsqueeze(0), size=(target_height, target_width), 
                                        mode='bilinear', align_corners=False).squeeze(0)
                resized_images.append(resized)
            else:
                resized_images.append(img)
        
        images = resized_images
    
    images = torch.stack(images)
    target_inputs = torch.stack(target_inputs)
    target_outputs = torch.stack(target_outputs)
    
    return images, target_inputs, target_outputs, texts

def evaluate(model, dataloader, tokenizer, device, max_length=100, max_samples=None):
    """Đánh giá mô hình Transformer-based decoder"""
    model.eval()
    
    all_predictions = []
    all_targets = []
    
    print("Đang đánh giá... ", end="", flush=True)
    
    with torch.no_grad():
        for idx, (images, _, _, texts) in enumerate(dataloader):
            if idx % 5 == 0:
                print(".", end="", flush=True)
                
            images = images.to(device)
            
            predictions = []
            for i in range(images.size(0)):
                pred_text = model.generate(images[i:i+1], tokenizer, max_length=max_length)
                predictions.append(pred_text)
            
            all_predictions.extend(predictions)
            all_targets.extend(texts)
            
            if max_samples and len(all_predictions) >= max_samples:
                all_predictions = all_predictions[:max_samples]
                all_targets = all_targets[:max_samples]
                break
    
    print(" Hoàn thành!")
    print(f"Đã đánh giá {len(all_predictions)} mẫu")
    
    correct = sum(p == t for p, t in zip(all_predictions, all_targets))
    accuracy = correct / len(all_predictions) if all_predictions else 0
    
    total_cer = 0
    for pred, target in zip(all_predictions, all_targets):
        distance = levenshtein_distance(pred, target)
        target_len = max(len(target), 1)  # Tránh chia cho 0
        total_cer += distance / target_len
    avg_cer = total_cer / len(all_predictions) if all_predictions else 1
    
    return accuracy, avg_cer, all_predictions, all_targets
