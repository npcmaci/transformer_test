import torch

def generate_future_mask(size):
    future_mask = torch.triu(torch.ones(size, size, dtype=torch.bool), diagonal=1)

    return future_mask

def generate_padding_mask(source, pad_token=0):
    padding_mask = (source == pad_token).unsqueeze(1).unsqueeze(2)
    # (batch_size, 1, seq_length, seq_length)
    padding_mask = padding_mask | padding_mask.transpose(1, 2)

    return padding_mask

def combine_padding_mask(mask1, mask2):
    a = mask2[:, :, 0, :].unsqueeze(3)
    b = mask1[:, :, :, 0].unsqueeze(2)
    c = a | b

    return c