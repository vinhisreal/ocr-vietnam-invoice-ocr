# src/utils/tokenizer.py

class CTC_SimpleTokenizer:
    def __init__(self):
        chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        chars += "àáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ"
        chars += "ÀÁẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬÈÉẺẼẸÊẾỀỂỄỆÌÍỈĨỊÒÓỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÙÚỦŨỤƯỨỪỬỮỰỲÝỶỸỴĐ"
        chars += " .,;:!?()-/\\\"'[]{}@#$%^&*+=_<>|~`₫$€¥£¢°×÷√≤≥±≠∞≈"
        
        self.blank_token = 0
        self.char_to_idx = {char: idx + 1 for idx, char in enumerate(chars)}
        self.idx_to_char = {idx + 1: char for idx, char in enumerate(chars)}
        self.idx_to_char[self.blank_token] = ''
        self.vocab_size = len(self.char_to_idx) + 1
        print(f"[CTC] Vocab size: {self.vocab_size}")
    
    def encode(self, text, max_length=100):
        indices = [self.char_to_idx.get(char, 0) for char in text]
        return indices[:max_length] + [0] * (max_length - len(indices))

    def decode(self, indices):
        return ''.join([self.idx_to_char.get(idx, '') for idx in indices if idx != 0])


class Transformer_SimpleTokenizer:
    def __init__(self):
        self.special_tokens = ["<PAD>", "<SOS>", "<EOS>", "<UNK>"]
        self.pad_token_idx = 0
        self.sos_token_idx = 1
        self.eos_token_idx = 2
        self.unk_token_idx = 3

        chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        chars += "àáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ"
        chars += "ÀÁẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬÈÉẺẼẸÊẾỀỂỄỆÌÍỈĨỊÒÓỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÙÚỦŨỤƯỨỪỬỮỰỲÝỶỸỴĐ"
        chars += " .,;:!?()-/\\\"'[]{}@#$%^&*+=_<>|~`₫$€¥£¢°×÷√≤≥±≠∞≈"

        self.char_to_idx = {char: idx + len(self.special_tokens) for idx, char in enumerate(chars)}
        self.idx_to_char = {idx + len(self.special_tokens): char for idx, char in enumerate(chars)}

        for idx, token in enumerate(self.special_tokens):
            self.char_to_idx[token] = idx
            self.idx_to_char[idx] = token

        self.vocab_size = len(self.char_to_idx)
        print(f"[TFM] Vocab size: {self.vocab_size}")

    def encode(self, text, max_length=100, add_special_tokens=True):
        indices = []
        if add_special_tokens:
            indices.append(self.sos_token_idx)
        indices += [self.char_to_idx.get(c, self.unk_token_idx) for c in text]
        if add_special_tokens:
            indices.append(self.eos_token_idx)
        indices = indices[:max_length]
        indices += [self.pad_token_idx] * (max_length - len(indices))
        return indices

    def decode(self, indices, remove_special_tokens=True):
        result = []
        for idx in indices:
            if remove_special_tokens and idx in [self.pad_token_idx, self.sos_token_idx, self.eos_token_idx]:
                continue
            result.append(self.idx_to_char.get(idx, "<UNK>"))
        return ''.join(result)
