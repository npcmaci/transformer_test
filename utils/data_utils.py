import os


# BPE所需的数据文件
def prepare_training_data(dataset, filename):
    file_path = os.path.join('data', filename + '_train.txt')
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in dataset['train']['translation']:
            f.write(item['en'] + '\n')
            f.write(item['de'] + '\n')
