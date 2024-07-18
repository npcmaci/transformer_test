### 数据集引入
#### 安装依赖
```commandline
pip install transformers datasets
```
#### 数据预处理
```python
from datasets import load_dataset
from utils import prepare_training_data
import sentencepiece as spm

# 加载WMT 2014 英语-德语数据集
dataset = load_dataset('wmt14', 'de-en')
prepare_training_data(dataset, 'WMT_de_en')
# 生成 BPM token
spm.SentencePieceTrainer.Train('--input=data/WMT_de_en_train.txt --model_prefix=bpe --vocab_size=37000 --model_type=bpe')
```
