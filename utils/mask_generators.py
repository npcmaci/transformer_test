import torch


def generate_future_mask(size):
    future_mask = torch.triu(torch.ones(size, size, dtype=torch.bool), diagonal=1)

    return future_mask


def generate_padding_mask(source, pad_token = 0):
    # (batch_size, 1, 1, seq_length)
    if source.dim() == 2:
        padding_mask = (source == pad_token).unsqueeze(1).unsqueeze(2)
    else: # dim = 3, one-hot 编码
        padding_mask = torch.argmax(source, dim=-1) == pad_token
        padding_mask = padding_mask.unsqueeze(1).unsqueeze(2)
    # (batch_size, 1, seq_length, seq_length)
    padding_mask = padding_mask.expand(-1, -1, padding_mask.shape[3], -1)

    return padding_mask


def combine_padding_mask(mask1, mask2):
    a = mask1[:, :, 0, :].unsqueeze(3)
    c = a.expand(-1, -1, mask2.shape[-1], -1)

    return c
