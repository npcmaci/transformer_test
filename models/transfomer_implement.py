import torch
import torch.nn as nn
import math
from .transformer import Transformer
from .positional_encoding import PositionalEncoding
from utils import generate_padding_mask, generate_future_mask
'''
    transformer应用于机器翻译
'''


class MachineTranslation(nn.Module):
    def __init__(self, source_vocab_dim, target_vocab_dim, d_model, n_head, n_encoder_layers, n_decoder_layers,
                 d_feedforward, pad_token, dropout=0.1,
                 configs=None):
        super(MachineTranslation, self).__init__()
        if configs is None:
            configs = {'use_padding_mask': None, 'use_shared_embedding': None, 'use_shared_pre_softmax': None}
        self.source_embedding = nn.Embedding(source_vocab_dim, d_model)
        self.target_embedding = nn.Embedding(target_vocab_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        self.transformer = Transformer(d_model, n_head, n_encoder_layers, n_decoder_layers, d_feedforward, dropout)
        self.fc_out = nn.Linear(d_model, target_vocab_dim)
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model
        self.pad_token = pad_token
        self.configs = configs

    def forward(self, source, target):
        # embedding
        if self.configs['use_shared_embedding'] is not None and self.configs['use_shared_embedding'] is not True:
            source_embedded = self.target_embedding(source) * math.sqrt(self.d_model)
            target_embedded = self.target_embedding(target) * math.sqrt(self.d_model)
        else:
            source_embedded = self.source_embedding(source) * math.sqrt(self.d_model)
            target_embedded = self.target_embedding(target) * math.sqrt(self.d_model)

        # positional encoding
        source_pe = self.positional_encoding(source_embedded)
        target_pe = self.positional_encoding(target_embedded)

        # 生成mask
        if self.configs['use_padding_mask'] is not None and self.configs['use_padding_mask'] is True:
            source_padding_mask = generate_padding_mask(source, self.pad_token).to(source.device)
            target_padding_mask = generate_padding_mask(target, self.pad_token).to(source.device)
        else:
            source_padding_mask = None
            target_padding_mask = None
        target_future_mask = generate_future_mask(target.shape[1]).to(source.device)

        # transformer层
        transformer_output = self.transformer(source_pe, target_pe, source_padding_mask,
                                              target_padding_mask, target_future_mask)

        # pre_softmax层
        if self.configs['use_shared_pre_softmax'] is not None and self.configs['use_shared_pre_softmax'] is True:
            output = torch.matmul(transformer_output, self.target_embedding.weight.T)
        else:
            output = self.fc_out(transformer_output)

        return output
