## 第 5 章 虚拟助手中的自然语言处理

首先，将所有保存的模型导入到新会话中。参见清单 5-52。

**清单 5-52.**
```python
import pandas as pd
from keras.models import Model
from tensorflow.keras.models import load_model
```

然后加载编码器和解码器模型。参见清单 5-53。

**清单 5-53.**
```python
encoder_model = load_model(dest_folder + 'enc_model_collab_211_redo1')
decoder_model = load_model(dest_folder + 'dec_model_collab_211_redo1')
```

加载编码器和解码器的分词器以及用于将模板转换为句子的字典文件。使用相关的 `dest_folder`。这是你从 Google Colab 下载模型文件的位置。参见清单 5-54。

**清单 5-54.**
```python
import pickle
import numpy as np

pkl_file = open(dest_folder + 'tokenizer_en_redo_1.pkl', 'rb')
tokenizer = pickle.load(pkl_file)
pkl_file = open(dest_folder + 'tokenizer_de_redo.pkl', 'rb')
tokenizer1 = pickle.load(pkl_file)
pkl_file = open('dict_templ.pkl', 'rb')
df_sents = pickle.load(pkl_file)
bus_sch = np.load("bus_stops.npz", allow_pickle=True)  # 确保使用 `.npz` 文件！
bus_sch1 = list(bus_sch['arr_0'])
```