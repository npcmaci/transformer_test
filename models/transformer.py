import torch
import torch.nn as nn
from .transformer_decoder import TransformerDecoder
from .transformer_encoder import TransformerEncoder

'''
    transformer核心结构
'''

class Transformer(nn.Module):
    def __init__(self, d_model, n_head, n_encoder_layers, n_decoder_layers, d_feedforward, dropout = 0.1):
        super(Transformer, self).__init__()
        self.encoder = TransformerEncoder(d_model, n_head, n_encoder_layers, d_feedforward, dropout)
        self.decoder = TransformerDecoder(d_model, n_head, n_decoder_layers, d_feedforward, dropout)
        self.d_model = d_model
        # 初始化权重
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, source, target, source_padding_mask = None,
                target_padding_mask = None, target_future_mask = None):
        memory = self.encoder(source, source_padding_mask)
        output = self.decoder(memory, target, source_padding_mask, target_padding_mask, target_future_mask)

        return output


