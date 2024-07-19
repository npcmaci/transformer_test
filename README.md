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
import os

# 需要改成项目路径
os.chdir('your disk position') 
data_dir = 'data'
if not os.path.exists(data_dir):
    os.makedirs(data_dir)
# 加载WMT 2014 英语-德语数据集
dataset = load_dataset('wmt14', 'de-en')
prepare_training_data(dataset, 'WMT_de_en')
# 生成 BPM token
spm.SentencePieceTrainer.Train(
    '--input=data/WMT_de_en_train.txt '
    '--model_prefix=bpe '
    '--vocab_size=37000 '
    '--model_type=bpe '
    '--user_defined_symbols=<pad>,<s>,</s> '
    '--control_symbols=<unk> '
    '--remove_extra_whitespace=true '
    '--character_coverage=0.9995 '
    '--shuffle_input_sentence=true'
)
```
