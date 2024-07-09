import torch
import torch.nn as nn
import math
from .transformer import Transformer
from utils import generate_padding_mask, generate_future_mask, combine_padding_mask
'''
    transformer应用于机器翻译
'''

class MachineTranslation(nn.Module):
    def __init__(self, source_vocab_dim, target_vocab_dim, d_model, n_head, n_encoder_layers, n_decoder_layers, d_feedforward, pad_token,
                 dropout=0.1):
        super(MachineTranslation, self).__init__()
        self.source_embedding = nn.Embedding(source_vocab_dim, d_model)
        self.target_embedding = nn.Embedding(target_vocab_dim, d_model)
        self.transformer = Transformer(d_model, n_head, n_encoder_layers, n_decoder_layers, d_feedforward, dropout)
        self.fc_out = nn.Linear(d_model, target_vocab_dim)
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model
        self.pad_token = pad_token

    def forward(self, source, target):
        # embedding
        source_embedded = self.dropout(self.source_embedding(source) * math.sqrt(self.d_model))
        target_embedded = self.dropout(self.target_embedding(target) * math.sqrt(self.d_model))
        # positional encoding

        # 生成mask
        source_padding_mask = generate_padding_mask(source, self.pad_token)
        target_padding_mask = generate_padding_mask(target, self.pad_token)
        target_future_mask = generate_future_mask(target.shape[1])

        # transformer层
        transformer_output = self.transformer(source_embedded, target_embedded, source_padding_mask,
                                              target_padding_mask, target_future_mask)
        # 没做softmax后面用cross-entropy
        output = self.fc_out(transformer_output)


        return output