# src/models/model_transformer.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer

from src.models.cnn_backbone import CNNBackbone
from src.utils.positional_encoding import PositionalEncoding


class CNNTransformerDecoder(nn.Module):
    def __init__(self, vocab_size, input_channels=3, hidden_dim=256, nhead=4,
                 num_encoder_layers=3, num_decoder_layers=3, dim_feedforward=1024, dropout=0.1):
        super(CNNTransformerDecoder, self).__init__()

        self.backbone = CNNBackbone(input_channels=input_channels, hidden_dim=hidden_dim)

        encoder_layer = TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers)

        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.pos_encoder = PositionalEncoding(hidden_dim, dropout)

        decoder_layer = TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers)

        self.fc_out = nn.Linear(hidden_dim, vocab_size)
        self.vocab_size = vocab_size

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu') \
                    if isinstance(m, nn.Conv2d) else nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def create_square_subsequent_mask(self, sz, device):
        mask = torch.triu(torch.ones(sz, sz, device=device)) == 1
        mask = mask.float().masked_fill(~mask, float('-inf')).masked_fill(mask, float(0.0))
        return mask

    def forward(self, src, tgt, tgt_mask=None, pad_idx=0):
        memory = self.encode(src)
        if tgt_mask is None:
            tgt_mask = self.create_square_subsequent_mask(tgt.size(1), tgt.device)
        tgt_key_padding_mask = (tgt == pad_idx)

        tgt_emb = self.embedding(tgt)
        tgt_emb = self.pos_encoder(tgt_emb)

        out = self.decoder(tgt_emb, memory, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask)
        out = self.fc_out(out)
        return out

    def encode(self, src):
        x = self.backbone(src)  # (B, C, H=1, W) → (B, W, C)
        x = self.encoder(x)     # (B, W, C)
        return x

    def generate(self, src, tokenizer, max_length=100, beam_size=1):
        self.eval()
        with torch.no_grad():
            memory = self.encode(src)
            sos = tokenizer.sos_token_idx
            eos = tokenizer.eos_token_idx

            if beam_size == 1:
                output = torch.LongTensor([[sos]]).to(src.device)
                for _ in range(max_length):
                    tgt_mask = self.create_square_subsequent_mask(output.size(1), output.device)
                    tgt_emb = self.pos_encoder(self.embedding(output))
                    out = self.decoder(tgt_emb, memory, tgt_mask=tgt_mask)
                    logits = self.fc_out(out[:, -1])
                    next_token = torch.argmax(logits, dim=-1).unsqueeze(1)
                    output = torch.cat([output, next_token], dim=1)
                    if next_token.item() == eos:
                        break
                return tokenizer.decode(output.squeeze().tolist())
            else:
                raise NotImplementedError("Beam search not implemented.")
