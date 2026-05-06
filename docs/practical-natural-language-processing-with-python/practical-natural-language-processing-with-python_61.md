# 第 5 章 虚拟助手中的自然语言处理

***清单 5-50.***

```
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam

model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
model.fit([en_tr3, de_tr3], de_target3,
          batch_size=30,
          epochs=30,
          validation_split=0.2)
```

现在你想测试模型是否工作，因此保存模型对象。在此处使用相关的 `root_dir` 文件夹。参见清单 5-51。

***清单 5-51.***

```
from tensorflow.keras.models import save_model

dest_folder = root_dir + '/collab_models/'
encoder_model.save(dest_folder + 'enc_model_collab_211_redo1')
decoder_model.save(dest_folder + 'dec_model_collab_211_redo1')

import pickle
dest_folder = root_dir + '/collab_models/'
output = open(dest_folder + 'tokenizer_en_redo.pkl', 'wb')
# 使用协议 0 对字典进行 pickle。
pickle.dump(tokenizer, output)

dest_folder = root_dir + '/collab_models/'
output1 = open(dest_folder + 'tokenizer_de_redo.pkl', 'wb')
pickle.dump(tokenizer1, output1)
```

打开一个新的笔记本并编写推理部分的执行代码。此代码在本地编写，并从 Google Colab 获取文件。代码包含三个部分：给定文本的预处理（替换适当的词和正则表达式）、将文本转换为所需的数组格式，以及模型推理。

一旦拟合了模型，你就一次运行一个词的推理代码。起始词被初始化为词的起始标记。请注意，在代码块末尾，解码器状态会为下一个词重新初始化。

