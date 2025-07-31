# src/data/dataset.py

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from src.utils.preprocessing import preprocess_image


class VietnameseOCRDataset(Dataset):
    def __init__(
        self,
        images_dir,
        annotation_file,
        tokenizer,
        decoder_type="ctc",  # "ctc" hoặc "transformer"
        transform=None,
        preprocess=True,
        adaptive_threshold=False,
        max_samples=None,
        max_target_length=100,
    ):
        assert decoder_type in ["ctc", "transformer"], "decoder_type must be 'ctc' or 'transformer'"
        self.images_dir = images_dir
        self.annotation_file = annotation_file
        self.tokenizer = tokenizer
        self.decoder_type = decoder_type
        self.transform = transform
        self.preprocess = preprocess
        self.adaptive_threshold = adaptive_threshold
        self.max_target_length = max_target_length

        self.samples = self._load_samples(max_samples)

    def _load_samples(self, max_samples):
        with open(self.annotation_file, "r", encoding="utf-8") as f:
            lines = f.readlines()

        samples = []
        skipped = 0
        for line in lines:
            parts = line.strip().split("\t")
            if len(parts) == 2:
                image_name, text = parts
                image_path = os.path.join(self.images_dir, image_name)
                if os.path.exists(image_path) and text.strip():
                    samples.append((image_path, text))
                    if max_samples and len(samples) >= max_samples:
                        break
                else:
                    skipped += 1

        print(f"[INFO] Loaded {len(samples)} samples from {self.annotation_file}")
        if skipped > 0:
            print(f"[WARNING] Skipped {skipped} samples due to missing images or empty text")
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, text = self.samples[idx]

        image = cv2.imread(image_path)
        if image is None:
            print(f"[ERROR] Failed to read image: {image_path}")
            image = np.zeros((32, 100, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if self.preprocess:
                image = preprocess_image(image, self.adaptive_threshold)

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        # ===== Xử lý label =====
        if self.decoder_type == "ctc":
            target = torch.tensor(self.tokenizer.encode(text), dtype=torch.long)
            return image, target, text

        elif self.decoder_type == "transformer":
            target = torch.tensor(
                self.tokenizer.encode(text, max_length=self.max_target_length),
                dtype=torch.long,
            )
            target_input = target[:-1]  # Bỏ EOS
            target_output = target[1:]  # Bỏ SOS
            return image, target_input, target_output, text