import torch
import torch.nn.functional as F


def collate_fn_ctc(batch):
    """
    Gom batch lại, resize ảnh nếu có kích thước khác nhau, và stack lại thành tensor.
    """
    images, targets, texts = zip(*batch)

    heights = [img.shape[1] for img in images]
    widths = [img.shape[2] for img in images]

    if len(set(heights)) > 1 or len(set(widths)) > 1:
        print(f"[WARNING] Batch có ảnh không đồng nhất kích thước: heights={heights}, widths={widths}")

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
    targets = torch.stack(targets)

    return images, targets, texts

def ctc_greedy_decode(log_probs: torch.Tensor, tokenizer, blank_token: int = 0):
    """
    Thực hiện greedy decoding cho output của mô hình CTC.
    Args:
        log_probs (Tensor): Tensor [T, B, C] (time, batch, vocab size)
        tokenizer: tokenizer có .decode(list[int]) để convert index → text
        blank_token (int): chỉ số của blank token trong vocab
    Returns:
        List[str]: Danh sách các chuỗi đã decode cho từng sample trong batch.
    """
    pred_indices = torch.argmax(log_probs, dim=2).cpu().numpy()  # [T, B]
    T, B = pred_indices.shape

    results = []
    for b in range(B):
        indices = pred_indices[:, b]
        collapsed = []
        prev = -1
        for idx in indices:
            if idx != blank_token and idx != prev:
                collapsed.append(idx)
            prev = idx
        text = tokenizer.decode(collapsed)
        results.append(text)

    return results
