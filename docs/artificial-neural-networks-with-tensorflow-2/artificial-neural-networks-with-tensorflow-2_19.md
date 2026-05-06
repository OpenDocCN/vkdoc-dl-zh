# 生成婴儿名字

在本项目中，你将使用著名文本生成博客文章——*循环神经网络的不合理有效性*（`http://karpathy.github.io/2015/05/21/rnn-effectiveness/`）中给出的数据集。该数据集包含多个已知的婴儿名字，每个名字由换行符分隔。你将使用这些名字来训练 LSTM 模型，使其记住数据集中定义的序列。训练完成后，你将要求模型根据其学习到的语义预测一些新名字。

### 创建项目

创建一个新的 Colab 项目，并将其重命名为 `TextGenerationBabyNames`。导入所需的库。

```
import sys
import re
import requests
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Dense, Activation, Dropout, LSTM
```

## 下载文本

要下载数据集，请对以下命令中指定的 URL 发起 HTTP 请求：`r = requests.get('https://cs.stanford.edu/people/karpathy/namesGenUnique.txt')`

请求成功后，你可以通过将响应中的文本复制到本地变量中来提取数据集：

```
raw_txt = r.text
```

通过调用 `raw_txt` 上的 `len` 方法来检查读取的数据长度：

```
len(raw_txt)
```

你将得到输出 `52127`，表明数据集中总共有 52,127 个字符。要查看数据的样子，只需在终端上打印 `raw_txt`：

```
raw_txt
```

该命令及其输出如图 7-10 所示。

![../images/495303_1_En_7_Chapter/495303_1_En_7_Fig10_HTML.jpg](img/495303_1_En_7_Fig10_HTML.jpg)

**图 7-10** 数据文件内容

你会看到一个长列表，其中包含由换行符分隔的名字。你可以使用 `print` 语句将它们逐行打印出来，查看前几个名字：

```
print(raw_txt[:100])
```

你将看到以下输出：

```
jka
Dillie
Ryine
Cherita
Dasher
Chailine
Frennide
Gremaley
Patj
Handi
Gully
Wennie
Ferentra
Jixandli
```

程序打印了前 100 个字符。这些名字的长度显然是可变的。

## 处理文本

为了将数据输入到我们的模型中进行训练，我们必须去掉 `\n` 字符。我们使用以下命令将其替换为空格：

```
raw_txt = raw_txt.replace('\n', ' ')
```

现在，我们通过创建一个集合来提取 `raw_txt` 中的所有唯一字符：

```
set(raw_txt)
```

部分输出如图 7-11 所示。

![../images/495303_1_En_7_Chapter/495303_1_En_7_Fig11_HTML.jpg](img/495303_1_En_7_Fig11_HTML.jpg)

**图 7-11** 唯一字符集

如你所见，该集合包含空格、短划线（`-`）、点（`.`）、冒号（`:`）和数字（0 到 9）等字符。在训练模型之前，应从集合中移除这些字符，因为它们对模型将要生成的新名字没有用处。我们使用正则表达式来移除这些字符。

```
raw_txt = re.sub('[-.0-9:]', '', raw_txt)
```

此外，在生成的婴儿名字中，我们不需要同时存在大写和小写字符。因此，我们通过调用 `lower` 方法将所有字符转换为小写：

```
raw_txt1 = raw_txt.lower()
set(raw_txt1)
```

尝试再次打印该集合，你会注意到它只包含小写字母和空格字符。

你可能会想，如果我们最终只需要一组小写字母和空格字符，为什么还要进行整个数据处理练习。为了生成婴儿名字，目标字符集只包含字母和用于分隔名字的空格。然而，在更高级的文本生成应用中，例如生成包含数学公式的文档、法律文件、科学摘要等，你的目标字符集会大得多，包含各种字符。但是，采用较大的目标字符集也会导致训练时间呈指数级增长。因此，通常我们会从原始文本中剔除一些不需要的字符。这种移除不需要字符的文本处理可以加快训练速度。

你现在可以检查这个新集合的大小：

```
len1 = len(set(raw_txt1))
print(len1)
```

它会在你的终端上打印 `27`。

由于模型处理的是数字而不是字母，你需要将字母映射到不同的数字。此外，当模型输出预测时，它会发送一组数字，这些数字必须转换回字母才能理解。因此，我们为这些映射创建两个数组。这通过以下代码片段完成：

```
chars = sorted(list(set(raw_txt1)))
arr = np.arange(0, len1)
char_to_ix = {}
ix_to_char = {}
for i in range(len1):
    char_to_ix[chars[i]] = arr[i]
    ix_to_char[arr[i]] = chars[i]
```

`char_to_ix` 数组将提供从集合中的字符到唯一整数的映射，而 `ix_to_char` 将提供从整数到字符的反向映射。尝试打印 `ix_to_char` 数组，你将看到如图 7-12 所示的部分输出。

![../images/495303_1_En_7_Chapter/495303_1_En_7_Fig12_HTML.jpg](img/495303_1_En_7_Fig12_HTML.jpg)

**图 7-12** 预处理后的唯一字符集

现在，你将使用以下代码片段创建输入和输出序列：

```
maxlen = 5
x_data = []
y_data = []
for i in range(0, len(raw_txt1) - maxlen, 1):
    in_seq = raw_txt1[i: i + maxlen]
    out_seq = raw_txt1[i + maxlen]
    x_data.append([char_to_ix[char] for char in in_seq])
    y_data.append([char_to_ix[out_seq]])
nb_chars = len(x_data)
print('Text corpus: {}'.format(nb_chars))
print('Sequences # ', int(len(x_data) / maxlen))
```

请注意，我们定义了一个长度为 5 的序列。因此，前五个字符将是输入，第六个字符将是目标。在下一个循环中，第 2 到第 6 个字符将是输入序列，第 7 个字符将是目标，依此类推。因此，在 `for` 循环中，我们创建了用于训练模型的 `x_data`，`y_data` 是训练期间使用的目标值。前面代码片段的输出如下：

```
Text corpus: 52038
Sequences #  10407
```

数据集包含 52,038 个字符，被划分为 10,407 个序列，每个序列长度为 5。

接下来，我们将数据转换为 numpy 数组以输入到我们的模型，并将训练数据归一化到 0 到 1 的范围内。

```
x = np.reshape(x_data, (nb_chars, maxlen, 1))
x = x / float(len(chars))
```

我们将目标序列转换为分类列。

```
y = tf.keras.utils.to_categorical(y_data)
y[:1]
```

在前面的语句中，当你打印转换后 `y_data` 中的某个项目时，你会看到如图 7-13 所示的输出。

![../images/495303_1_En_7_Chapter/495303_1_En_7_Fig13_HTML.jpg](img/495303_1_En_7_Fig13_HTML.jpg)

**图 7-13** 分类后的目标数据示例

在这个数组中，其中一个值为 1，其余为 0。值 1 对应于 `char_to_ix` 数组中该特定索引处的字符。

你现在可以打印 `x_data` 的形状：

```
x.shape
```

输出为

```
(52038, 5, 1)
```

这表明输入有 52,038 个序列，每个序列长度为 5。你也可以通过调用 `y.shape` 来检查 `y` 的大小。

```
y.shape
```

输出为

```
(52038, 27)
```

输出中有 27 个类别。



## 定义模型

我们按如下方式定义模型：

```python
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(256,
                         input_shape=(maxlen, 1),
                         return_sequences=True),
    tf.keras.layers.LSTM(256,
                         return_sequences=True),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.LSTM(64),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(len(y[1]),
                          activation='softmax')
])
```

模型摘要如下所示：

```
Model: "sequential"
____________________________________________________
Layer (type)          Output Shape        Param #
====================================================
lstm (LSTM)           (None, 5, 256)      264192
____________________________________________________
lstm_1 (LSTM)        (None, 5, 256)       525312
____________________________________________________
dropout (Dropout)    (None, 5, 256)       0
____________________________________________________
lstm_2 (LSTM)        (None, 64)           82176
____________________________________________________
dropout_1 (Dropout)  (None, 64)           0
____________________________________________________
dense (Dense)        (None, 27)           1755
====================================================
Total params: 873,435
Trainable params: 873,435
Non-trainable params: 0
```

如您所见，该模型包含三个 LSTM 层，每层由 500 个节点组成。每个 LSTM 层之后是一个 dropout 率为 20% 的 dropout 层。最后一层是一个包含 27 个节点的 Dense 层，通过 softmax 激活函数进行分类。请注意，我们的数据有 27 个分类输出。模型的视觉表示如图 7-14 所示。

![../images/495303_1_En_7_Chapter/495303_1_En_7_Fig14_HTML.jpg](img/495303_1_En_7_Fig14_HTML.jpg)

**图 7-14** 模型架构层

## 编译

我们使用分类交叉熵和 Adam 优化器来编译模型。

```python
model.compile(loss='categorical_crossentropy',
              optimizer='adam')
```

请注意，对于这类语言模型问题，没有测试数据集。我们对整个数据集进行建模，以预测序列中每个分类字符的概率。模型完美预测下一个字符的准确率对我们来说并不重要。相反，我们关注的是最小化所选的损失函数。因此，我们试图在泛化能力和过拟合之间取得平衡，避免死记硬背。

## 创建检查点

训练 LSTM 网络通常需要很长时间。由于网络本身的特性，每个 epoch 后的损失可能会增加或减少。最低的损失最终会为我们带来最佳的预测结果。因此，我们需要捕获产生最低损失的 epoch 的模型权重。这可以通过使用 `ModelCheckPoint` 方法并在每个 epoch 后设置回调来实现。

```python
filepath = "model_weights_babynames.hdf5"
checkpoint = ModelCheckpoint(filepath,
                             monitor='loss', verbose=1,
                             save_best_only=True, mode='min')
model_callbacks = [checkpoint]
```

您之前在第 4 章的一个项目中已经使用过诸如提前停止之类的回调，因此这是另一种类型的回调，它将在每个 epoch 后被调用。

## 训练

现在我们通过调用 `fit` 方法来训练模型。

```python
model.fit(x, y, epochs=300, batch_size=32,
          callbacks=model_callbacks)
```

我将 epoch 数设置为 10，批次大小设置为 32 个序列。稍后我将向您展示增加这两个变量值的效果。当我在 GPU 上运行此代码时，每个 epoch 的训练时间大约为 6 秒，这对我们来说是合理的。我想在此说明，我如此关注训练时间的原因是，在大型语料库上训练 LSTM 并拥有大量分类输出时，即使您在一组 GPU 上使用分布式训练，也需要数小时才能完成。

训练结束后，让我们尝试进行一些预测。

### 预测

我们首先需要创建一个输入序列。目前，我们从原始数据库中定义一个输入序列。它通过以下代码片段定义：

```python
pattern = []
seed = 'handi'
for i in seed:
    value = char_to_ix[i]
    pattern.append(value)
```

我们使用的序列是“handi”——长度为 5。请注意，我们的模型定义期望输入序列大小为 5。种子序列中的每个字符都通过我们之前创建的 `char_to_ix` 数组转换为其整数值。

现在，我们将设置一个 for 循环，以预测给定序列“handi”之后的 100 个字符。作为参考，我们首先打印种子，并将词汇表中的字符数设置为 `n_vocab` 变量。

```python
print(seed)
n_vocab = len(chars)
```

我们设置循环来进行 100 次预测。

```python
for i in range(100):
```

我们首先使用以下两条语句重塑此输入模式并归一化其内容：

```python
X = np.reshape(pattern, (1, len(pattern), 1))
X = X / float(n_vocab)
```

我们将此输入馈送到模型，并要求它预测给定模式之后的下一个字符：

```python
int_prediction = model.predict(X, verbose=0)
```

我们提取具有最大预测概率的字符的索引，并使用我们之前创建的 `ix_to_char` 数组将其转换为字符。

```python
index = np.argmax(int_prediction)
prediction = ix_to_char[index]
```

我们在终端上打印预测的字符：

```python
sys.stdout.write(prediction)
```

我们将此字符附加到我们的模式中，并通过提取最后五个字符（即我们的输入序列长度）来重新创建一个新模式。使用这个新模式，我们要求模型进行下一次预测。同样，我们将要求模型以我们定义为种子的输入序列开始，进行 100 次预测。

```python
pattern.append(index)
pattern = pattern[1:len(pattern)]
```

用于生成 100 个字符的整个 for 循环在代码清单 7-1 中给出，供您快速参考。

```python
print(seed)
n_vocab = len(chars)
for i in range(100):
    X = np.reshape(pattern, (1, len(pattern), 1))
    X = X / float(n_vocab)
    int_prediction = model.predict(X, verbose=0)
    index = np.argmax(int_prediction)
    prediction = ix_to_char[index]
    sys.stdout.write(prediction)
    pattern.append(index)
    pattern = pattern[1:len(pattern)]
```

**代码清单 7-1** 用于预测 100 个字符的循环



### 完整源码 – `TextGenerationBabyNames`

婴儿名字生成项目的完整源码见代码清单 7-2。

```
import sys
import re
import requests
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks
import ModelCheckpoint
from tensorflow.keras.layers
import Dense, Activation,
Dropout, LSTM
r = requests.get('https://cs.stanford.edu/people/karpathy/namesGenUnique.txt')
raw_txt = r.text
len(raw_txt)
raw_txt
print(raw_txt[:100])
set(raw_txt)
len(set(raw_txt))
raw_txt = raw_txt.replace('\n' , ' ')
set(raw_txt)
len(set(raw_txt))
raw_txt = re.sub('[-.0-9:]' , '' , raw_txt)
len(set(raw_txt))
raw_txt1 = raw_txt.lower()
set(raw_txt1)
len1 = len(set(raw_txt1))
print (len1)
len1
chars = sorted(list(set(raw_txt1)))
arr = np.arange(0, len1)
char_to_ix = {}
ix_to_char = {}
for i in range(len1):
char_to_ix[chars[i]] = arr[i]
ix_to_char[arr[i]] = chars[i]
char_to_ix
ix_to_char
#print("Total length of file  : {}".format(len(raw_txt1)))
maxlen = 5
x_data = []
y_data = []
for i in range(0, len(raw_txt1) - maxlen, 1):
in_seq  = raw_txt1[i: i + maxlen]
out_seq = raw_txt1[i + maxlen]
x_data.append([char_to_ix[char]
for char in in_seq])
y_data.append([char_to_ix[out_seq]])
nb_chars = len(x_data)
print('Text corpus: {}'.format(nb_chars))
print('Sequences # ', int(len(x_data) / maxlen))
#y_data[:5]
#x_data[1][:]
x = np.reshape(x_data , (nb_chars , maxlen , 1))
x = x/float(len(chars))
y = tf.keras.utils.to_categorical(y_data)
y[:1]
x.shape
y.shape
model = tf.keras.Sequential([
tf.keras.layers.LSTM(256,
input_shape = (maxlen, 1),
return_sequences = True),
tf.keras.layers.LSTM(256,
return_sequences = True),
tf.keras.layers.Dropout(0.2),
tf.keras.layers.LSTM(64),
tf.keras.layers.Dropout(0.2),
tf.keras.layers.Dense(len(y[1]),
activation='softmax')
])
model.summary()
model.compile(loss = 'categorical_crossentropy',
optimizer = 'adam')
filepath = "model_weights_babynames.hdf5"
checkpoint = ModelCheckpoint(filepath,
monitor = 'loss', verbose = 1,
save_best_only = True, mode = 'min')
model_callbacks = [checkpoint]
model.fit(x,y, epochs = 300, batch_size = 32 ,
callbacks = model_callbacks)
pattern = []
seed = 'handi'
for i in seed:
value = char_to_ix[i]
pattern.append(value)
print(seed)
n_vocab = len(chars)
for i in range(100):
X = np.reshape(pattern , (1, len(pattern) , 1))
X = X/float(n_vocab)
int_prediction = model.predict(X , verbose = 0)
index = np.argmax(int_prediction)
prediction = ix_to_char[index]
sys.stdout.write(prediction)
pattern.append(index)
pattern = pattern[1:len(pattern)]
from google.colab import drive
drive.mount('/content/drive')
cd 'My Drive'
model.save('baby_names_model.h5')
from tensorflow.keras.models import load_model
saved_model = load_model('baby_names_model.h5')
pattern = []
seed = 'bgajm'
for i in seed:
value = char_to_ix[i]
pattern.append(value)
print(seed)
n_vocab = len(chars)
for i in range(100):
X = np.reshape(pattern , (1, len(pattern) , 1))
X = X/float(n_vocab)
int_prediction = saved_model.predict
(X , verbose = 0)
index = np.argmax(int_prediction)
prediction = ix_to_char[index]
sys.stdout.write(prediction)
pattern.append(index)
pattern = pattern[1:len(pattern)]
Listing 7-2
TextGenerationBabyNames full source
```

运行代码后，我得到了如图 7-15 所示的输出。

![../images/495303_1_En_7_Chapter/495303_1_En_7_Fig15_HTML.jpg](img/495303_1_En_7_Fig15_HTML.jpg)

图 7-15

种子 `"handi"` 生成的婴儿名字

网络可能已经很好地生成了有意义的名字。现在，你可以尝试增加训练轮数，看看预测结果是否有改进。

当我将模型重新训练 50 轮，批次大小为 128 时，对于相同的种子 `"handi"`，我得到了如图 7-16 所示的输出。

![../images/495303_1_En_7_Chapter/495303_1_En_7_Fig16_HTML.jpg](img/495303_1_En_7_Fig16_HTML.jpg)

图 7-16

50 轮训练后的婴儿名字

根据个人对名字的偏好，这看起来可能更好。至少，名字没有重复。

这里需要理解的关键点是：通过增加训练轮数、增加 LSTM 层节点数、增加 LSTM 层数、调整序列长度等操作，可能会提高预测质量。事实上，理论上，经过充分训练的大型网络，对于原始文本中的已知种子，能够生成与原始文本完全相同的输出。这相当于说，一个拥有图像式记忆的人能够按原始顺序复现原文。通过这个讨论，我们了解到 LSTM 神经网络可以有效地用于自主生成模仿原文的高质量文本。

就我们生成婴儿名字的应用而言，使用网络生成数据库中已存在的名字是没有意义的。要生成数据库中不存在但听起来与现有名字相似的名字，你需要用原始文本中不存在的输入序列作为网络的种子。通常，人们会生成一个随机种子。我尝试了随机种子 `"bgajm"`，得到了如图 7-17 所示的输出。

![../images/495303_1_En_7_Chapter/495303_1_En_7_Fig17_HTML.jpg](img/495303_1_En_7_Fig17_HTML.jpg)

图 7-17

使用随机种子生成的名字

在进入更复杂的文本生成问题之前，我想指出网络训练时间对批次大小的另一个重要依赖关系。如果批次大小较小，覆盖整个文本语料库中所有字符所需的时间会更长，从而导致训练时间增加。同时，增加批次大小会需要更多的系统内存资源。因此，在获得网络最佳训练时间的同时，需要在批次大小和可分配资源之间进行权衡。考虑到 LSTM 训练时间较长，建议在训练后保存模型，以便后续用于不同的种子。接下来，我将展示如何保存和复用模型进行后续预测。

### 保存/复用模型

你将把训练好的模型保存到 Google Drive。为此，你需要挂载驱动器。

```
from google.colab import drive
drive.mount('/content/drive')
```

挂载过程中，系统会要求你输入授权码。挂载驱动器后，切换到你想保存模型的文件夹。

```
cd 'drive/My Drive/TextGenerationDemo'
```

现在，调用模型的 `save` 方法，将其保存到所需的文件名。

```
model.save('baby_names_model.h5')
```

你可以在之后的任何时间点通过调用 `load_model` 重新加载保存的模型。

```
from tensorflow.keras.models import load_model
saved_model = load_model('baby_names_model.h5')
```

加载后的模型存储在变量 `saved_model` 中，可用于后续的预测或进一步训练。

通过这个简单的文本生成示例介绍之后，让我们进入一个更实际的例子，使用更大的文本语料库。



