import torch
import math
import torch.nn as nn
import torch.nn.functional as F


class MultiheadAttentionBlock(nn.Module):
    def __init__(self, d_model, n_head, dropout=0.1):
        assert d_model % n_head == 0
        super(MultiheadAttentionBlock, self).__init__()
        self.LQ = nn.Linear(d_model, d_model)
        self.LK = nn.Linear(d_model, d_model)
        self.LV = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_model // n_head

    def forward(self, Q_source, K_source, V_source, padding_mask=None, future_mask=None):
        batch_size = Q_source.size(0)

        # 线性变换并分成多头 (batch_size, n_head, seq_length, d_head)
        Q = self.LQ(Q_source).view(batch_size, -1, self.n_head, self.d_head).transpose(1, 2)
        K = self.LK(K_source).view(batch_size, -1, self.n_head, self.d_head).transpose(1, 2)
        V = self.LV(V_source).view(batch_size, -1, self.n_head, self.d_head).transpose(1, 2)

        # 分头算注意力 (batch_size, n_head, seq_length, seq_length)
        attention_weights = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_head)

        # mask mask.shape = (batch_size, 1, seq_length, seq_length)
        if padding_mask is not None:
            attention_weights = attention_weights.masked_fill(padding_mask, float('-inf'))

        if future_mask is not None:
            attention_weights = attention_weights.masked_fill(future_mask, float('-inf'))

        # softmax (batch_size, n_head, seq_length, seq_length)
        attention_weights = F.softmax(attention_weights, dim=-1)

        # 合并 (batch_size, seq_length, d_model)
        combined_weights = (torch.matmul(attention_weights, V).transpose(1, 2).contiguous()
                            .view(batch_size, -1, self.d_model))

        # 输出 (batch_size, seq_length, d_model)
        output = self.out_linear(combined_weights)

        # Add & Norm
        output = self.layer_norm(Q_source + output)

        return output

