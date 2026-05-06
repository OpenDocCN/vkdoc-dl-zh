# 第 5 章 虚拟助手中的 NLP

## 编码器-解码器训练

使用 `corrected_cust` 和 `bots_templ_list_shift` 列作为输入和输出，你可以训练编码器-解码器模型。鉴于你使用的是大型架构，本次练习将使用 Google Colab 和 GPU。[Colaboratory](https://colab.research.google.com/) 是一个谷歌研究项目，旨在帮助传播机器学习教育和研究。它是一个 Jupyter notebook 环境，无需设置即可使用，并且完全在云端运行。你可以快速上手并设置一个 Colab Jupyter notebook。

首先，将数据上传到 Google Drive。然后，你可以使用清单 5-38 中的代码在 Google Colab 中挂载数据。这会将所有文件加载到驱动器中。运行代码时，你需要通过点击链接输入授权码。参见清单 5-38。

**清单 5-38.**

```python
from google.colab import drive
drive.mount('/content/drive')
```

会出现一个类似图 5-19 的屏幕，你需要输入链接中显示的授权码。然后参见清单 5-39。

**图 5-19.** 挂载文件的授权

**清单 5-39.**

```python
import pandas as pd
t1 = pd.read_csv(base_fl_csv)
```

你在清单 5-40 中为句子添加开始和结束标签。

**清单 5-40.** 开始和结束标签

```python
t1["bots_templ_list_shift"] = 'start ' + t1["bots_templ_list_shift"] + ' end'
```

你在清单 5-41 中创建分词器，一个用于编码器，一个用于解码器。

**清单 5-41.** 创建分词器

```python
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

tokenizer = Tokenizer()
tokenizer1 = Tokenizer()
```

现在，你对客户和机器人文本（分别是编码器输入和解码器输入）中的句子进行分词，并根据语料库中句子的最大长度进行填充。参见清单 5-42 和 5-43。

**清单 5-42.**

```python
en_col_tr = list(t1["corrected_cust"].str.split())
de_col_tr = list(t1["bots_templ_list_shift"].str.split())

tokenizer.fit_on_texts(en_col_tr)
en_tr1 = tokenizer.texts_to_sequences(en_col_tr)

tokenizer1.fit_on_texts(de_col_tr)
de_tr1 = tokenizer1.texts_to_sequences(de_col_tr)
```

**清单 5-43.**

```python
def get_max_len(list1):
    len_list = [len(i) for i in list1]
    return max(len_list)

max_len1 = get_max_len(en_tr1)
max_len2 = get_max_len(de_tr1)

en_tr2 = pad_sequences(en_tr1, maxlen=max_len1, dtype='int32', padding='post')
de_tr2 = pad_sequences(de_tr1, maxlen=max_len2, dtype='int32', padding='post')

de_tr2.shape, en_tr2.shape, max_len1, max_len2
((7760, 27), (7760, 28), 28, 27)
```

你可以看到编码器和解码器输入是二维形状。由于你直接将它们输入 LSTM（没有嵌入层），你需要将它们转换为三维数组。这是通过将词序列转换为独热编码形式来完成的。参见清单 5-44 和 5-45。

**清单 5-44.**

```python
from keras.utils import to_categorical

en_tr3 = to_categorical(en_tr2)
de_tr3 = to_categorical(de_tr2)

en_tr3.shape, de_tr3.shape
((7760, 28, 733), (7760, 27, 1506))
```

**清单 5-45.**

```python
from keras.utils import to_categorical

en_tr3 = to_categorical(en_tr2)
de_tr3 = to_categorical(de_tr2)

en_tr3.shape, de_tr3.shape
((7760, 28, 733), (7760, 27, 1506))
```

数组现在是三维的。请注意，到目前为止你只定义了输入。现在必须定义输出。模型的输出是时间步为 t+1 的解码器。你希望根据上一个词预测下一个词。参见清单 5-46。

**清单 5-46.**

```python
import numpy as np
from scipy.ndimage.interpolation import shift

de_target3 = np.roll(de_tr3, -1, axis=1)
de_target3[:, -1, :] = 0
```

在清单 5-47 中保存编码器和解码器令牌的数量以定义模型输入。

**清单 5-47.**

```python
num_encoder_tokens = en_tr3.shape[2]
num_decoder_tokens = de_tr3.shape[2]
```

在采用教师强制（teacher forcing）的编码器-解码器架构中，训练和推理步骤存在差异。代码源自 [`blog.keras.io/a-ten-minute-introduction-to-sequence-to-sequence-learning-in-keras.html`](https://blog.keras.io/a-ten-minute-introduction-to-sequence-to-sequence-learning-in-keras.html) 上的这篇文章。首先，你将看到训练步骤。这里你为编码器输入（客户文本）的每一步定义一个包含 100 个隐藏节点的 LSTM。你保留编码器的最终细胞状态和隐藏状态，并丢弃每个 LSTM 细胞的输出。解码器接收来自机器人文本的输入，并使用编码器输出的初始状态进行初始化。由于编码器有 100 个隐藏节点，解码器也有 100 个隐藏节点。每个细胞的解码器输出被传递到一个全连接层。网络使用解码器输入的时间调整滞后进行训练。基本上，t-1 时刻的编码器值和解码器输入预测 t 时刻的机器人文本。参见清单 5-48。

**清单 5-48.**

```python
from keras.models import Model
from keras.layers import Input, LSTM, Dense

encoder_inputs = Input(shape=(None, num_encoder_tokens))
encoder = LSTM(100, dropout=.2, return_state=True)
encoder_outputs, state_h, state_c = encoder(encoder_inputs)
# 我们丢弃 encoder_outputs，只保留状态。
encoder_states = [state_h, state_c]

decoder_inputs = Input(shape=(None, num_decoder_tokens))
decoder_lstm = LSTM(100, return_sequences=True, return_state=True, dropout=0.2)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
decoder_dense = Dense(num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# 定义将要构建的模型
```



