"""
pose estimation transformer model

Author: Saifullah Ijaz
Date: 03/09/2024
"""
import torch
import math
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0)]

class PoseEstimationTransformer(nn.Module):
    def __init__(self, input_dim=4, d_model=256, nhead=2, num_encoder_layers=2, dim_feedforward=512, dropout=0.1):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)
        self.translation_head = nn.Linear(d_model, 3)
        self.rotation_head = nn.Linear(d_model, 6)

    def forward(self, src_list):
        translations = []
        rotations = []
        for src in src_list:
            src = self.input_projection(src).unsqueeze(1)
            
            src = self.pos_encoder(src)
            
            output = self.transformer_encoder(src)
            
            output = output.mean(dim=0)
            
            translation = self.translation_head(output)
            rotation = self.rotation_head(output)
            
            translations.append(translation.squeeze(0))
            rotations.append(rotation.squeeze(0))

        return torch.stack(translations), torch.stack(rotations)