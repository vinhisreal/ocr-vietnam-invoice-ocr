# src/models/ctc_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.cnn_backbone import CNNBackbone  
import math
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from src.utils.positional_encoding import PositionalEncoding
from src.models.cnn_backbone import CNNBackbone

class CNNT(nn.Module):
    def __init__(self, vocab_size, input_channels=3, hidden_dim=256, nhead=4, 
                 num_encoder_layers=4, dim_feedforward=1024, dropout=0.1):
        super(CNNT, self).__init__()
        
        self.cnn = CNNBackbone(input_channels, hidden_dim)
        self.pos_encoder = PositionalEncoding(hidden_dim, dropout)

        encoder_layer = TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="relu",
            batch_first=False
        )
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_encoder_layers)

        self.fc_out = nn.Linear(hidden_dim, vocab_size)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.cnn(x)              # [B, C, H, W]
        features = features.squeeze(2)      # Remove H = 1 → [B, C, W]
        features = features.permute(2, 0, 1)  # [W, B, C]

        features = self.pos_encoder(features)
        encoded = self.transformer_encoder(features)

        logits = self.fc_out(encoded)       # [W, B, vocab_size]
        return F.log_softmax(logits, dim=2)  # for CTCLoss
