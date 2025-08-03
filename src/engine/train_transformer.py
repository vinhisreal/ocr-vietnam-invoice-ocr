import os
import torch
import time
from torch.cuda.amp import autocast, GradScaler
import time
from tqdm.auto import tqdm
from src.engine.evaluator import evaluate_transformer
import torch.nn as nn

def train_cnn_transformer_decoder(model, train_loader, val_loader, tokenizer, output_dir='output', 
                 epochs=30, lr=0.0005, weight_decay=1e-5, device='cuda', patience=10):
    """Hàm huấn luyện mô hình CNN + Transformer Decoder"""
    os.makedirs(output_dir, exist_ok=True)
    
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_idx)
    
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=lr, 
        betas=(0.9, 0.999),
        weight_decay=weight_decay
    )
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=True
    )
    
    if hasattr(torch.amp, 'GradScaler'):
        scaler = torch.amp.GradScaler('cuda' if device == 'cuda' else 'cpu')
    else:
        scaler = GradScaler(enabled=(device == 'cuda'))
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'accuracy': [],
        'cer': [],
        'learning_rate': []
    }
    
    best_cer = float('inf')
    no_improve_epochs = 0
    best_epoch = -1
    
    accumulation_steps = 2
    
    for epoch in range(epochs):
        print(f"\n{'='*20} Epoch {epoch+1}/{epochs} {'='*20}")
        start_time = time.time()
        
        model.train()
        train_loss = 0.0
        batch_count = 0
        optimizer.zero_grad()
        
        progress_bar = tqdm(train_loader, desc=f"Training")
        for i, (images, target_inputs, target_outputs, _) in enumerate(progress_bar):
            images = images.to(device)
            target_inputs = target_inputs.to(device)
            target_outputs = target_outputs.to(device)
            
            if hasattr(torch.amp, 'autocast'):
                ctx_manager = torch.amp.autocast('cuda' if device == 'cuda' else 'cpu')
            else:
                ctx_manager = autocast(enabled=(device == 'cuda'))
            
            with ctx_manager:
                tgt_mask = model.create_square_subsequent_mask(target_inputs.size(1), device)
                
                output = model(images, target_inputs, tgt_mask, tokenizer.pad_token_idx)
                
                output_flat = output.reshape(-1, output.size(-1))
                target_flat = target_outputs.reshape(-1)
                
                loss = criterion(output_flat, target_flat)
                loss = loss / accumulation_steps
            
            if device == 'cuda':
                scaler.scale(loss).backward()
                
                if (i + 1) % accumulation_steps == 0 or (i + 1 == len(train_loader)):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else:
                loss.backward()
                
                if (i + 1) % accumulation_steps == 0 or (i + 1 == len(train_loader)):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()
            
            current_lr = optimizer.param_groups[0]['lr']
            train_loss += loss.item() * accumulation_steps
            batch_count += 1
            progress_bar.set_postfix(loss=f"{loss.item() * accumulation_steps:.4f}", lr=f"{current_lr:.7f}")
        
        avg_train_loss = train_loss / batch_count if batch_count > 0 else float('inf')
        history['train_loss'].append(avg_train_loss)
        history['learning_rate'].append(optimizer.param_groups[0]['lr'])
        
        model.eval()
        val_loss = 0.0
        val_batch_count = 0
        
        with torch.no_grad():
            for images, target_inputs, target_outputs, _ in tqdm(val_loader, desc="Validation"):
                images = images.to(device)
                target_inputs = target_inputs.to(device)
                target_outputs = target_outputs.to(device)
                
                tgt_mask = model.create_square_subsequent_mask(target_inputs.size(1), device)
                
                output = model(images, target_inputs, tgt_mask, tokenizer.pad_token_idx)
                
                output_flat = output.reshape(-1, output.size(-1))
                target_flat = target_outputs.reshape(-1)
                
                loss = criterion(output_flat, target_flat)
                val_loss += loss.item()
                val_batch_count += 1
        
        avg_val_loss = val_loss / val_batch_count if val_batch_count > 0 else float('inf')
        history['val_loss'].append(avg_val_loss)
        
        scheduler.step(avg_val_loss)
        
        accuracy, cer, predictions, targets = evaluate_transformer(
            model, val_loader, tokenizer, device, max_samples=100
        )
        
        history['accuracy'].append(accuracy)
        history['cer'].append(cer)
        
        epoch_time = time.time() - start_time
        print(f"\nEpoch {epoch+1}/{epochs} - Time: {epoch_time:.1f}s")
        print(f"  Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        print(f"  Accuracy: {accuracy:.4f}, CER: {cer:.4f}")
        print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.7f}")
        
        print("\nMẫu dự đoán:")
        for i in range(min(5, len(predictions))):
            print(f"  Mẫu {i+1}:")
            print(f"  Dự đoán: '{predictions[i]}'")
            print(f"  Thực tế: '{targets[i]}'")
            print()
        
        if cer < best_cer:
            best_cer = cer
            best_epoch = epoch + 1
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'accuracy': accuracy,
                'cer': cer,
                'history': history,
            }, os.path.join(output_dir, 'best_model.pth'))
            print(f"  Đã lưu mô hình tốt nhất với CER: {cer:.4f}!")
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
            print(f"  Không cải thiện CER. Còn {patience - no_improve_epochs} epochs nữa sẽ dừng early stopping.")
        
        # Lưu checkpoint
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'accuracy': accuracy,
                'cer': cer,
                'history': history,
                'best_cer': best_cer,
                'best_epoch': best_epoch
            }, os.path.join(output_dir, f'checkpoint_epoch_{epoch+1}.pth'))
            print(f"  Lưu checkpoint tại epoch {epoch+1}")
        
        # Early stopping
        if no_improve_epochs >= patience:
            print(f"\nEarly stopping sau {patience} epochs không cải thiện.")
            break
    
    print(f"\nKết thúc huấn luyện sau {epoch+1} epochs")
    print(f"Mô hình tốt nhất tại epoch {best_epoch} với CER: {best_cer:.4f}")
    
    return model, history