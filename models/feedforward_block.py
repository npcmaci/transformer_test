import torch.nn as nn

'''
    multihead层后的线性层
'''

class FeedforwardBlock(nn.Module):
    def __init__(self, d_model, d_feedforward, dropout=0.1):
        super(FeedforwardBlock, self).__init__()
        self.linear1 = nn.Linear(d_model, d_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_feedforward, d_model)
        self.activation = nn.ReLU()
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        x2 = self.linear1(x)
        x2 = self.activation(x2)
        x2 = self.linear2(x2)

        # add & norm
        output = x + x2
        output = self.norm(output)
        return output

