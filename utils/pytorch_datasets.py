import torch
from torch.utils.data import Dataset


class TransformerTrainDataset(Dataset):
    def __init__(self, source_texts, target_texts):
        self.source_texts = source_texts
        self.target_texts = target_texts

    def __len__(self):
        return len(self.source_texts)

    def __getitem__(self, idx):
        return torch.tensor(self.source_texts[idx]), torch.tensor(self.target_texts[idx])


#测试集所用数据集还没定义好
class TransformerEvaluationDataset(Dataset):
    def __init__(self, source_texts, target_texts):
        self.source_texts = source_texts
        self.target_texts = target_texts

    def __len__(self):
        return len(self.source_texts)

    def __getitem__(self, idx):
        return torch.tensor(self.source_texts[idx]), torch.tensor(self.target_texts[idx])