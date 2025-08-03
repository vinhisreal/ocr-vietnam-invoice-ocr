import torch, os
from torch.utils.data import Dataset, DataLoader
from src.engine.train_transformer import train_cnn_transformer_decoder
from src.engine.evaluator import evaluate_transformer
from src.data.dataset import VietnameseOCRDataset, collate_fn
from src.utils.tokenizer import SimpleTokenizer
from src.models.transformer_model import CNNTransformerDecoder
from src.utils.augmentations import get_ocr_transforms
from src.utils.preprocessing import preprocess_image
from src.utils.config import load_config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    config = load_config("../config/config.yaml")
    data_config = config["data"]

    images_dir = data_config["images_dir"]
    train_file = data_config["train_file"]
    val_file = data_config["val_file"]
    
    output_dir = os.path.join("checkpoints", "transformer")
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(images_dir):
        print(f"Thư mục ảnh không tồn tại: {images_dir}")
        return
    if not os.path.exists(train_file):
        print(f"File train không tồn tại: {train_file}")
        return
    if not os.path.exists(val_file):
        print(f"File validation không tồn tại: {val_file}")
        return
    
    tokenizer = SimpleTokenizer()
    
    train_dataset = VietnameseOCRDataset(
        images_dir=images_dir,
        annotation_file=train_file,
        tokenizer=tokenizer,
        transform=get_ocr_transforms(height=32, width=320, is_train=True),
        preprocess=True,
        adaptive_threshold=False,
        max_target_length=100
    )
    
    val_dataset = VietnameseOCRDataset(
        images_dir=images_dir,
        annotation_file=val_file,
        tokenizer=tokenizer,
        transform=get_ocr_transforms(height=32, width=320, is_train=False),
        preprocess=True,
        adaptive_threshold=False,
        max_target_length=100
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False,
        collate_fn=collate_fn
    )
    
    model = CNNTransformerDecoder(
        vocab_size=tokenizer.vocab_size,
        input_channels=3,
        hidden_dim=256,
        nhead=4,
        num_encoder_layers=3,
        num_decoder_layers=3,
        dim_feedforward=1024,
        dropout=0.1
    )
    
    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Tổng số tham số của mô hình: {total_params:,}")
    print(f"Số tham số có thể train: {trainable_params:,}")
    
    model, history = train_cnn_transformer_decoder(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        tokenizer=tokenizer,
        output_dir=output_dir,
        epochs=30,
        lr=0.0005,
        weight_decay=1e-5,
        device=device,
        patience=10
    )
    
    checkpoint = torch.load(os.path.join(output_dir, 'best_model.pth'))
    best_model = CNNTransformerDecoder(
        vocab_size=tokenizer.vocab_size,
        input_channels=3,
        hidden_dim=256,
        nhead=4,
        num_encoder_layers=3,
        num_decoder_layers=3,
        dim_feedforward=1024,
        dropout=0.1
    )
    
    best_model.load_state_dict(checkpoint['model_state_dict'])
    best_model = best_model.to(device)
    
    # Đánh giá mô hình tốt nhất
    accuracy, cer, predictions, targets = evaluate_transformer(
        best_model, val_loader, tokenizer, device, max_samples=200
    )
    
    print(f"\nĐánh giá mô hình tốt nhất:")
    print(f"Accuracy: {accuracy:.4f}, CER: {cer:.4f}")
    
    print("Quá trình huấn luyện và đánh giá hoàn tất!")

if __name__ == "__main__":
    main()