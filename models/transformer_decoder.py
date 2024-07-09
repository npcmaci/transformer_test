import torch.nn as nn
from .multihead_attention_block import MultiheadAttentionBlock
from .feedforward_block import FeedforwardBlock
from utils import generate_padding_mask, generate_future_mask, combine_padding_mask

class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, n_head, d_feedforward, dropout=0.1):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attention = MultiheadAttentionBlock(d_model, n_head, dropout)
        self.cross_attention = MultiheadAttentionBlock(d_model, n_head, dropout)
        self.feedforward = FeedforwardBlock(d_model, d_feedforward, dropout)

    def forward(self, memory, target, memory_padding_mask = None, target_padding_mask = None, target_future_mask = None):
        output = self.self_attention(target, target, target, combine_padding_mask(memory_padding_mask, target_padding_mask), target_future_mask)
        output = self.cross_attention(output, memory, memory, memory_padding_mask)
        output = self.feedforward(output)

        return output

class TransformerDecoder(nn.Module):
    def __init__(self, d_model, n_head, n_decoder_layers, d_feedforward, dropout=0.1):
        super(TransformerDecoder, self).__init__()
        self.decoder_layers = nn.ModuleList(
            TransformerDecoderLayer(d_model, n_head, d_feedforward, dropout) for _ in range(n_decoder_layers)
        )

    def forward(self, memory, target, memory_padding_mask=None, target_padding_mask=None, target_future_mask=None):
        output = target
        for decoder_layer in self.decoder_layers:
            output = decoder_layer(memory, target, memory_padding_mask, target_padding_mask, target_future_mask)

        return output