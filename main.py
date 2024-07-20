import torch
import torch.nn
from torch.utils.data import DataLoader
import numpy as np
import os
from datasets import load_dataset
import sentencepiece as spm
from utils import byte_piece_encode, one_hot_encode
from utils import TransformerTrainDataset, TransformerEvaluationDataset
from utils import train_model
from models import MachineTranslation


def main():
    # 超参数
    epoch = 10
    batch_size = 32
    d_model = 512
    n_head = 8
    n_encoder_layers = 6
    n_decoder_layers = 6
    d_feedforward = 2048
    dropout = 0.1
    lr = 0.01
    batch_size = 32
    num_samples = 100
    max_length = 64
    vocab_size = 37000

    ####################################################
    # 数据集
    dataset = load_dataset('wmt14', 'de-en')
    # BPE
    file_path = os.path.join('data', 'bpe.model')
    sp = spm.SentencePieceProcessor(model_file=file_path)
    encoded_train_source = byte_piece_encode([item['en'] for item in dataset['train']['translation'][:1000]], sp, max_length, False)
    encoded_train_target = byte_piece_encode([item['de'] for item in dataset['train']['translation'][:1000]], sp, max_length, True)
    encoded_valid_source = byte_piece_encode([item['en'] for item in dataset['validation']['translation'][:num_samples]], sp, max_length, False)
    encoded_valid_target = byte_piece_encode([item['de'] for item in dataset['validation']['translation'][:num_samples]], sp, max_length, True)
    # # one-hot (不需要)
    # encoded_train_source = np.array([one_hot_encode(sentence, vocab_size) for sentence in encoded_train_source])
    # encoded_train_target = np.array([one_hot_encode(sentence, vocab_size) for sentence in encoded_train_target])
    # encoded_valid_source = np.array([one_hot_encode(sentence, vocab_size) for sentence in encoded_valid_source])
    # encoded_valid_target = np.array([one_hot_encode(sentence, vocab_size) for sentence in encoded_valid_target])
    # dataset & dataloader
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset = TransformerTrainDataset(encoded_train_source, encoded_train_target)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataset = TransformerTrainDataset(encoded_valid_source, encoded_valid_target)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    #####################################################
    # 模型定义
    model = MachineTranslation(
        source_vocab_dim=37000,
        target_vocab_dim=37000,
        d_model=512,
        n_head=8,
        n_encoder_layers=6,
        n_decoder_layers=6,
        d_feedforward=2048,
        pad_token=sp.piece_to_id('<pad>'),
        dropout=0.1,
        configs={
            'use_padding_mask': True,
            'use_shared_embedding': True,
            'use_shared_pre_softmax': True
        }
    )
    # 训练
    train_model(model, train_dataloader, valid_dataloader, epoch, pad_token=sp.piece_to_id('<pad>'), device=device)
    # 分析

    print()


if __name__ == '__main__':
    main()

