import torch
from src.utils.metrics import levenshtein_distance

def evaluate_ctc(model, dataloader, tokenizer, device, decode_method='greedy', max_samples=None):
    model.eval()
    all_predictions, all_targets = [], []

    print("Đang đánh giá CTC... ", end="", flush=True)

    with torch.no_grad():
        for idx, (images, targets, texts) in enumerate(dataloader):
            if idx % 5 == 0:
                print(".", end="", flush=True)

            images = images.to(device)
            log_probs = model(images)  # [B, T, C]
            predictions = tokenizer.decode(log_probs, method=decode_method)  # CTC decode

            all_predictions.extend(predictions)
            all_targets.extend(texts)

            if max_samples and len(all_predictions) >= max_samples:
                all_predictions = all_predictions[:max_samples]
                all_targets = all_targets[:max_samples]
                break

    print(" Hoàn thành!")
    return _compute_metrics(all_predictions, all_targets)


def evaluate_transformer(model, dataloader, tokenizer, device, max_length=100, max_samples=None):
    model.eval()
    all_predictions, all_targets = [], []

    print("Đang đánh giá Transformer... ", end="", flush=True)

    with torch.no_grad():
        for idx, (images, _, _, texts) in enumerate(dataloader):
            if idx % 5 == 0:
                print(".", end="", flush=True)

            images = images.to(device)
            predictions = [model.generate(images[i:i+1], tokenizer, max_length=max_length)
                           for i in range(images.size(0))]

            all_predictions.extend(predictions)
            all_targets.extend(texts)

            if max_samples and len(all_predictions) >= max_samples:
                all_predictions = all_predictions[:max_samples]
                all_targets = all_targets[:max_samples]
                break

    print(" Hoàn thành!")
    return _compute_metrics(all_predictions, all_targets)


def _compute_metrics(predictions, targets):
    assert len(predictions) == len(targets)
    correct = sum(p == t for p, t in zip(predictions, targets))
    accuracy = correct / len(predictions) if predictions else 0

    total_cer = sum(
        levenshtein_distance(p, t) / max(len(t), 1)
        for p, t in zip(predictions, targets)
    )
    avg_cer = total_cer / len(predictions) if predictions else 1

    print(f"Đã đánh giá {len(predictions)} mẫu | Accuracy: {accuracy:.4f} | CER: {avg_cer:.4f}")
    return accuracy, avg_cer, predictions, targets
