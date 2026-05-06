# 高级文本生成

在本应用中，我们将使用列夫·托尔斯泰著名小说《战争与和平》中的文本（`https://cs.stanford.edu/people/karpathy/char-rnn/warpeace_input.txt`）。这部小说自然使用了大量特殊字符，例如问号/感叹号、引号和句号。在之前的示例中，我们剔除了这些字符，因为我们只想生成名字。在这个项目中，我不会移除所有此类特殊字符，因为我们希望创作另一部小说，或至少一段包含所有这些字符的文字。

在如此庞大的语料库上训练一个复杂的 LSTM 模型所需的时间非常长；你需要做好定期存储模型训练状态的准备。我将向你展示如何在每个 epoch 结束时保存模型的状态。这样，如果在训练过程中断连，你可以从之前已知的检查点继续训练。此外，我还将增加另一个功能，即在每个 epoch 结束时，通过让模型基于一个固定种子进行预测来测试其性能。我们会将预测结果存储到 Google Drive 的一个文件中。这样，你可以在后台持续训练模型，并通过定期检查磁盘上预测文件的内容来评估其性能。

### 创建项目

创建一个新的 Colab 项目，并将其重命名为 `LargeCorpusTextGeneration`。导入所需的库。

```python
import sys
import requests
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Dense, Activation, Dropout, LSTM
```

你将在训练过程中每个 epoch 结束时保存检查点数据和模型的预测结果。为此，你需要挂载 Google Drive 并指定用于保存数据的相应文件夹。

```python
from google.colab import drive
drive.mount('/content/drive')
```

挂载过程中，系统会要求你进行授权。当驱动器挂载完成后，你需要设置用于存储文件的文件夹。

```bash
cd '/content/drive/My Drive/TextGenerationDemo'
```

## 加载文本

我们通过发起以下 HTTP 请求将小说文本加载到项目中：

```python
r = requests.get("https://cs.stanford.edu/people/karpathy/char-rnn/warpeace_input.txt")
```

我们从响应对象中将小说文本读取到一个局部变量中。

```python
raw_txt = r.text
```

我们获取文本中唯一字符的列表，并打印语料库大小和输出类别数量：

```python
chars = sorted(list(set(raw_txt)))
print("Corpus: {}".format(len(raw_txt)))
print("Categories: {}".format(len(chars)))
```

你将看到以下输出：

```
Corpus: 3258246
Categories: 87
```

这部小说包含超过 300 万个字符，文本中有 87 个唯一字符。这两个数字都远大于我们婴儿名字模型中的数字。

## 处理数据

与之前的示例类似，我们需要将所有唯一字符映射为整数，以便模型处理。我们还需要提供反向映射来解释模型的输出。我们通过创建以下两个数组来实现这一点：

```python
ix_to_char = {ix:char for ix, char in enumerate(chars)}
char_to_ix = {char:ix for ix, char in enumerate(chars)}
```

然后，我们使用以下代码段将整个文本分割成序列：

```python
maxlen = 10
x_data = []
y_data = []
for i in range(0, len(raw_txt) - maxlen, 1):
    in_seq  = raw_txt[i: i + maxlen]
    out_seq = raw_txt[i + maxlen]
    x_data.append([char_to_ix[char] for char in in_seq])
    y_data.append([char_to_ix[out_seq]])
nb_chars = len(x_data)
print('Number of sequences:', int(len(x_data)/maxlen))
```

这段代码与你之前看到的示例类似。当序列大小为 10 时，创建的序列数量为 325,823。这是上述代码的输出：

```
Number of sequences: 325823
```

现在，我们对输入数据进行缩放和重塑，使其适合输入我们的网络。

```python
#### scale and transform data
x = np.reshape(x_data, (nb_chars, maxlen, 1))
n_vocab = len(chars)
x = x/float(n_vocab)
```

我们通过调用 `to_categorical` 方法将输出转换为其类别。

```python
y = tf.keras.utils.to_categorical(y_data)
```

你可以使用 print 语句检查输入和输出的大小：

```python
print("The shape of x_training data : ", x.shape)
print("The shape of y_training data : ", y.shape)
```

输出为

```
The shape of x_training data :  (3258236, 10, 1)
The shape of y_training data :  (3258236, 86)
```

如你所见，我们有大量的序列。网络的输入数量将为 10，输出为 86 个类别。

## 定义模型

我们使用以下代码定义模型：

```python
Model = tf.keras.Sequential([
    tf.keras.layers.LSTM(800, input_shape = (len(x[1]), 1), return_sequences = True),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.LSTM(800, return_sequences = True),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.LSTM(800),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(len(y[1]), activation = 'softmax')
])
```

模型定义与之前示例中使用的相同，只是考虑到输入文本的语料库大小，我增加了每一层的节点数量。

模型使用典型的交叉熵和 Adam 优化器进行编译。

```python
Model.compile(loss = 'categorical_crossentropy', optimizer = 'adam')
```

现在，我将描述本项目最重要的部分，即在每个 epoch 结束时保存模型的状态及其预测结果。

## 创建检查点

为了创建检查点，你需要为训练方法创建一个自定义回调函数。我们首先为检查点文件指定名称。

```python
filepath = "model_weights_saved.hdf5"
```

我们使用 `ModelCheckpoint` 方法来创建一个回调方法：

```python
checkpoint = ModelCheckpoint(filepath, monitor = 'loss', verbose = 1, save_best_only = True, mode = 'min')
```

在回调函数中，我们将监控损失值，并保存损失值最小时的模型训练权重。

我们创建一个变量来列出回调函数的数量，在本例中只有一个。

```python
model_callbacks = [checkpoint]
```



### `CustomCallback` 类

现在我们将创建另一个回调函数，用于写入模型的预测结果。我们会在每个 epoch 结束时将预测结果存储到一个文本文件中。我们创建一个全局变量来跟踪 epoch 编号。

```
epoch_number = 0
```

我们声明一个用于存储预测结果的文件名：

```
filename = 'predictions.txt'
```

如果文件已存在，我们覆盖其内容。

```
file = open(filename , 'w')
file.truncate()
file.close()
```

我们按如下方式声明自定义类：

```
class CustomCallback(tf.keras.callbacks.Callback):
```

我们使用以下语句定义一个名为 `on_epoch_end` 的方法：

```
def on_epoch_end(self , epoch , logs = None):
```

该方法将在每个 epoch 结束时被调用。在方法体中，我们首先增加全局 epoch 计数：

```
global epoch_number
epoch_number = epoch_number + 1
```

我们以追加模式打开预测文件，以便添加我们的预测结果：

```
filename = 'predictions.txt'
file = open(filename , 'a')
```

我们声明种子文本：

```
seed = "looking fo"
```

我们通过一个简单的 for 循环从该种子文本创建一个模式：

```
pattern = []
for i in seed:
value = char_to_ix[i]
pattern.append(value)
```

我们首先将 epoch 编号写入文件：

```
file.seek(0)
file.write("\n\n Epoch number :
{}\n\n".format(epoch_number))
```

我们设置一个循环来进行 100 次预测：

```
for i in range(100):
```

我们重塑并缩放输入数据：

```
X = np.reshape(pattern ,
(1, len(pattern) , 1))
X = X/float(n_vocab)
```

我们让模型预测给定种子文本的下一个字符：

```
int_prediction = Model.predict(X ,
verbose = 0)
```

我们选取概率最大的字符，并将其复制到预测变量中：

```
index = np.argmax(int_prediction)
prediction = ix_to_char[index]
```

我们将预测的字符写入文件：

```
file.write(prediction)
```

我们将预测的字符添加到模式中，并提取最后十个字符（我们种子文本的大小）以创建新模式：

```
pattern.append(index)
pattern = pattern[1:len(pattern)]
```

现在我们迭代进行下一次预测。在预测 100 次之后，我们关闭文件：

```
file.close()
```

### 模型训练

创建两个回调函数后，我们现在使用以下语句训练模型：

```
Model.fit(x, y , batch_size = 200,
epochs = 10 ,
callbacks = [CustomCallback() ,
model_callbacks])
```

我们定义了一个足够大的批次大小，即 200 个序列。在 `fit` 调用中指定了两个回调函数。自定义回调函数存储预测结果，模型回调函数存储模型状态。

当我运行训练时，在 GPU 上每个 epoch 大约需要 550 秒。幸运的是，结果在每个 epoch 结束时都会保存。因此，无需担心超时或断开连接，因为我能够看到模型直到上次中断时的性能，并且还可以使用“结果”部分之后讨论的代码，从上一个断点继续进一步训练。

### 结果

第一个、第五个和第十个 epoch 之后的预测结果如下：

```
Epoch number : 1
r the soldiers were all the same time the soldiers were all the same time the soldiers were all the
Predictions after 5 epochs:
Epoch number : 5
r the first time to the countess was a serious and the servants and the servants and the servants an
Predictions after 10 epochs:
Epoch number : 10
r the first time they had been sent to the countess was a still more than the countess was a still m
```

你可以注意到，模型的性能在每个 epoch 都在不断提升。

### 训练续接

要从最后一个已知的检查点继续训练，请使用以下代码：

```
try:
Model.load_weights(filepath)
except Exception as error:
print("Error loading in model : {}".format(error))
```

我们只需从存储的检查点文件加载权重，然后调用模型的 `fit` 方法来继续训练。

```
Model.fit(x, y , batch_size = 200, epochs = 10 ,
callbacks = [CustomCallback() ,
model_callbacks])
```

请注意，epoch 编号将从全局 epoch 的最后一个值继续。

以下是我继续训练总共 50 个 epoch 时的一些结果。

```
Epoch number : 20
r the first time to the countess was a small conversation with the state of a strange and the counte
Epoch number : 30
r the first time the soldiers who were all the same time he had seen and was about to see the counte
Epoch number : 40
r the first time the staff officer who had been at the same time he had seen him to the countess was
Epoch number : 50
r the first time the streets of the countess was a man of his soul and the same time he had seen and
```

如你所见，随着训练的进行，模型的输出质量不断提升。

### 一些观察

为了进行一些实验并减少训练时间，我将每个 LSTM 层中的节点数从 800 减少到 100。这确实将训练时间从大约 10 分钟缩短到了 2 分钟。在运行训练 100 个 epoch 后，以下是几个时间点的结果：

```
Epoch number : 1
and the same and the same and the same and the same and the same and the same and the same and the
Epoch number : 25
and the service and the service and the service and the service and the service and the service and
Epoch number : 50
and the same to the same to the same to the same to the same to the same to the same to the same to
Epoch number : 75
and the strength to the same and the strength to the same and the strength to the same and the stre
Epoch number : 100
and the same and the same and the same and the same and the same and the same and the same and the
```

如你所见，尽管 epoch 数量更多，但模型不再学习。我们可以得出结论，理解大型文本需要更多的内存，这通过增加每个 LSTM 层中的节点数来实现。

我进行了另一个实验，将每个 LSTM 层中的节点数增加到 500。结果，训练时间增加到大约 11 分钟/epoch。以下是结果：

```
Epoch number : 5
the same time to the same time to the same time to the same time to the same time to the same time
Epoch number : 10
the countess was a strange and the same things and the same things and the same things and the same
Epoch number : 15
the countess and the same time the soldiers and the same time the soldiers and the same time the so
Epoch number : 20
, and the same time the countess was still the staff of the countess was still the staff of the coun
```

我们看到通过增加节点数有所改进。在第 20 个 epoch，它甚至生成了一个逗号。继续进一步训练可能会进一步提升性能。



## 完整源代码

`LargeCorpusTextGeneration` 的完整源代码如代码清单 7-3 所示。

```python
import sys
import requests
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Dense, Activation, Dropout, LSTM
from google.colab import drive

drive.mount('/content/drive')
cd '/content/drive/My Drive/TextGenerationDemo'

r = requests.get("https://cs.stanford.edu/people/karpathy/char-rnn/warpeace_input.txt")
raw_txt = r.text
chars = sorted(list(set(raw_txt)))
print("Corpus: {}".format(len(raw_txt)))
print("Categories: {}".format(len(chars)))

ix_to_char = {ix:char for ix, char in enumerate(chars)}
char_to_ix = {char:ix for ix, char in enumerate(chars)}

maxlen = 10
x_data = []
y_data = []

for i in range(0, len(raw_txt) - maxlen, 1):
    in_seq  = raw_txt[i: i + maxlen]
    out_seq = raw_txt[i + maxlen]
    x_data.append([char_to_ix[char] for char in in_seq])
    y_data.append([char_to_ix[out_seq]])

nb_chars = len(x_data)
print('Number of sequences:', int(len(x_data)/maxlen))

#### scale and transform data
x = np.reshape(x_data, (nb_chars, maxlen, 1))
n_vocab = len(chars)
x = x/float(n_vocab)
x.shape

y = tf.keras.utils.to_categorical(y_data)
print("The shape of x_training data : ", x.shape)
print("The shape of y_training data : ", y.shape)

Model = tf.keras.Sequential([
    tf.keras.layers.LSTM(800, input_shape=(len(x[1]), 1), return_sequences=True),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.LSTM(800, return_sequences=True),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.LSTM(800),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(len(y[1]), activation='softmax')
])

Model.compile(loss='categorical_crossentropy', optimizer='adam')

filepath = "model_weights_saved.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
model_callbacks = [checkpoint]

epoch_number = 0
filename = 'predictions.txt'
file = open(filename, 'w')
file.truncate()
file.close()

class CustomCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        global epoch_number
        epoch_number = epoch_number + 1
        filename = 'predictions.txt'
        file = open(filename, 'a')
        seed = "looking fo"
        pattern = []
        for i in seed:
            value = char_to_ix[i]
            pattern.append(value)
        file.seek(0)
        file.write("\n\n Epoch number : {}\n\n".format(epoch_number))
        for i in range(100):
            X = np.reshape(pattern, (1, len(pattern), 1))
            X = X/float(n_vocab)
            int_prediction = Model.predict(X, verbose=0)
            index = np.argmax(int_prediction)
            prediction = ix_to_char[index]
            #sys.stdout.write(prediction)
            file.write(prediction)
            pattern.append(index)
            pattern = pattern[1:len(pattern)]
        file.close()

Model.fit(x, y, batch_size=2000, epochs=10, callbacks=[CustomCallback(), model_callbacks])

try:
    Model.load_weights(filepath)
except Exception as error:
    print("Error loading in model : {}".format(error))

Model.fit(x, y, batch_size=200, epochs=25, callbacks=[CustomCallback(), model_callbacks])
```

*代码清单 7-3* `LargeCorpusTextGeneration` 完整源代码

## 拓展阅读

Andrej Karpathy 在文本生成方面所做的工作值得一提，该工作发表在他著名的博客 *循环神经网络的非凡有效性* ([`http://karpathy.github.io/2015/05/21/rnn-effectiveness/`](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)) 中。在他的实验中，他使用了莎士比亚文学、维基百科文章、LaTeX（代数几何语言文本），甚至 Linux 源代码。经过详尽训练后产生的结果非常出色。他能够生成奇妙的数学方程，这些方程在大多数情况下语法正确。他的模型能够生成几乎可以编译的计算机源代码。这可以作为一个很好的证明，表明使用 LSTM 进行文本生成在实际中能够生成高质量的文本。

从我讨论的两个示例中，你可以很容易地理解，训练一个 LSTM 来生成高质量文本需要大量的资源和时间。要获得最佳结果，还需要你进行一些实验。以下是你在微调文本生成应用时需要考虑的一些技巧：

*   为节省训练时间，通过移除不需要的字符来减小词汇量。
*   添加更多 LSTM 和 Dropout 层，并在每层中增加更多 LSTM 单元。
*   尝试调整超参数，例如批次大小、优化器和序列长度，看看哪种效果最好。
*   尝试使用大量的训练轮次。
*   使用大型文本语料库。

## 本章小结

在本章中，你学习了一种新的神经网络架构，即 RNN。LSTM 是 RNN 的一种特殊情况。传统的 DNN 不具备记忆能力，而 LSTM 同时具有长期和短期记忆。因此，对于像文本生成这样记忆至关重要的场景，LSTM 表现出色。你学习了如何创建基于 LSTM 的网络来生成婴儿名字，甚至在学习了著名作家的小说后生成高质量的文本段落。

在下一章中，你将学习另一种用于语言翻译的语言模型，例如从英语翻译成法语，或从西班牙语翻译成日语。

