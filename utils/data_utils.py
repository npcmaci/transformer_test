import os
import numpy as np


# BPE所需的数据文件
def prepare_training_data(dataset, filename):
    file_path = os.path.join('data', filename + '_train.txt')
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in dataset['train']['translation']:
            f.write(item['en'] + '\n')
            f.write(item['de'] + '\n')


def byte_piece_encode(texts, sp, max_length, start_end_token=True):
    encoded_texts = []
    for text in texts:
        encoded_text = sp.encode_as_ids(text)
        if start_end_token is True:
            encoded_text = [sp.piece_to_id('<s>')] + encoded_text + [sp.piece_to_id('</s>')]

        if len(encoded_text) < max_length:
            encoded_text += [sp.piece_to_id('<pad>')] * (max_length - len(encoded_text))
        else:
            if start_end_token is True:
                encoded_text = encoded_text[:max_length - 1] + [sp.piece_to_id('</s>')]
            else:
                encoded_text = encoded_text[:max_length]
        encoded_texts.append(encoded_text)
    return np.array(encoded_texts, dtype=np.int64)


def one_hot_encode(indices, vocab_size):
    one_hot = np.zeros((len(indices), vocab_size), dtype=np.int64)
    for i, index in enumerate(indices):
        one_hot[i, index] = 1
    return one_hot
