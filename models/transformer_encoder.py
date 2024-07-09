import torch.nn as nn
from .multihead_attention_block import MultiheadAttentionBlock
from .feedforward_block import FeedforwardBlock
from utils import generate_padding_mask, generate_future_mask, combine_padding_mask

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, n_head, d_feedforward, dropout = 0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attention = MultiheadAttentionBlock(d_model, n_head, dropout)
        self.feedforward = FeedforwardBlock(d_model, d_feedforward, dropout)
    def forward(self, source, source_padding_mask = None):
        output = self.self_attention(source, source, source, source_padding_mask)
        output = self.feedforward(output)

        return output

class TransformerEncoder(nn.Module):
    def __init__(self, d_model, n_head, n_encoder_layers, d_feedforward, dropout = 0.1):
        super(TransformerEncoder, self).__init__()
        self.encoder_layers = nn.ModuleList(
            TransformerEncoderLayer(d_model, n_head, d_feedforward, dropout) for _ in range(n_encoder_layers)
        )
    def forward(self, source, source_padding_mask = None):
        output = source
        for encoder_layer in self.encoder_layers:
            output = encoder_layer(source, source_padding_mask)

        return output