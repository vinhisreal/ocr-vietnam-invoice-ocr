import os
import torch
from torch.nn import CTCLoss
from torch.cuda.amp import autocast, GradScaler
import time
from tqdm.auto import tqdm
from src.engine.evaluator import evaluate_ctc

def train_cnnt(model, train_loader, val_loader, tokenizer, output_dir='output', 
               epochs=30, lr=0.0005, weight_decay=1e-5, device='cuda', patience=10):
    os.makedirs(output_dir, exist_ok=True)
    
    criterion = CTCLoss(blank=tokenizer.blank_token, reduction='mean', zero_infinity=True)
    
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
        for i, (images, targets, _) in enumerate(progress_bar):
            images = images.to(device)
            targets = targets.to(device)
            
            if hasattr(torch.amp, 'autocast'):
                ctx_manager = torch.amp.autocast('cuda' if device == 'cuda' else 'cpu')
            else:
                ctx_manager = autocast(enabled=(device == 'cuda'))
            
            with ctx_manager:
                log_probs = model(images)
                
                if i == 0 and epoch == 0:
                    print(f"Log_probs shape: {log_probs.shape}")
                
                input_lengths = torch.full((log_probs.size(1),), log_probs.size(0), device=device)
                target_lengths = torch.sum(targets != tokenizer.blank_token, dim=1)
                
                valid_targets = target_lengths > 0
                if valid_targets.sum() == 0:
                    print("Batch không có target hợp lệ, skip")
                    continue
                
                if valid_targets.sum() < images.size(0):
                    log_probs = log_probs[:, valid_targets, :]
                    targets = targets[valid_targets]
                    target_lengths = target_lengths[valid_targets]
                    input_lengths = torch.full((valid_targets.sum(),), log_probs.size(0), device=device)
                
                flat_targets = []
                for j, length in enumerate(target_lengths):
                    flat_targets.extend(targets[j, :length].tolist())
                
                try:
                    flat_targets = torch.tensor(flat_targets, device=device)
                    loss = criterion(log_probs, flat_targets, input_lengths, target_lengths)
                    loss = loss / accumulation_steps
                except Exception as e:
                    print(f"Error in CTC loss: {e}")
                    print(f"Log probs shape: {log_probs.shape}")
                    print(f"Target lengths: {target_lengths}")
                    print(f"Input lengths: {input_lengths}")
                    print(f"Flat targets length: {len(flat_targets)}")
                    print(f"Batch size: {images.size(0)}, Valid targets: {valid_targets.sum()}")
                    continue
            
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
            for images, targets, texts in tqdm(val_loader, desc="Validation"):
                images = images.to(device)
                targets = targets.to(device)
                
                log_probs = model(images)
                
                if epoch == 0 and val_batch_count == 0:
                    print(f"Validation log_probs shape: {log_probs.shape}")
                
                input_lengths = torch.full((images.size(0),), log_probs.size(0), device=device)
                target_lengths = torch.sum(targets != tokenizer.blank_token, dim=1)
                
                valid_targets = target_lengths > 0
                if valid_targets.sum() == 0:
                    continue
                
                if valid_targets.sum() < images.size(0):
                    log_probs = log_probs[:, valid_targets, :]
                    targets = targets[valid_targets]
                    input_lengths = input_lengths[valid_targets]
                    target_lengths = target_lengths[valid_targets]
                
                flat_targets = []
                for i, length in enumerate(target_lengths):
                    flat_targets.extend(targets[i, :length].tolist())
                
                try:
                    flat_targets = torch.tensor(flat_targets, device=device)
                    loss = criterion(log_probs, flat_targets, input_lengths, target_lengths)
                    val_loss += loss.item()
                    val_batch_count += 1
                except Exception as e:
                    print(f"Lỗi trong validation loss: {e}")
                    continue
        
        avg_val_loss = val_loss / val_batch_count if val_batch_count > 0 else float('inf')
        history['val_loss'].append(avg_val_loss)
        
        scheduler.step(avg_val_loss)
        
        accuracy, cer, predictions, targets = evaluate_ctc(
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
        
        if no_improve_epochs >= patience:
            print(f"\nEarly stopping sau {patience} epochs không cải thiện.")
            break
    
    print(f"\nKết thúc huấn luyện sau {epoch+1} epochs")
    print(f"Mô hình tốt nhất tại epoch {best_epoch} với CER: {best_cer:.4f}")
    
    return model, history