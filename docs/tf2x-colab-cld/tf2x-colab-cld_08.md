# 8. 自动文本生成

前馈神经网络通常非常适合分类和回归问题。CNNs 非常适合复杂的图像分类。但是，前馈网络和 CNNs 的激活只在一个方向上流动，从输入层到输出层。由于信号只在一个方向上流动，如果数据中的模式随时间变化，前馈和卷积网络就不太理想。因此，我们需要一个不同的网络架构来处理受时间影响的数据。

**循环神经网络**（RNN）看起来很像前馈神经网络，但它也有指向后方的连接。也就是说，某一层的输出可以作为输入回到网络中较早的层。一层向网络中较早的层提供信息的能力意味着 RNN 内置了一个反馈循环机制，使其能够作为预测引擎。因此，RNNs 作为预测器非常出色，因为它们自然地随着时间变化而很好地工作。

RNN 会记住过去，其决策会受到它从过去学到的内容的影响。前馈和卷积网络只记住它们在训练期间学到的内容。例如，前馈图像分类器在训练期间学习图像的外观，然后使用这些知识在生产中对其他图像进行分类。而 RNNs 在训练期间也会学习类似的内容，但它们还会记住它们学到的内容，因此它们可以在数据变化时做出良好的决策。

章节的笔记本位于以下网址：[`https://github.com/paperd/tensorflow`](https://github.com/paperd/tensorflow)。

## 自然语言处理

机器学习中的一个迷人的进步是教会机器如何理解人类交流的能力。专注于理解人类如何交流的机器学习领域是自然语言处理。**自然语言处理**（NLP）是机器学习中的一个领域，专注于计算机理解、分析、操作和可能生成人类语言的能力。RNNs 是神经网络的一个重要变体，在 NLP 中得到了广泛的应用。

RNNs 非常适合 NLP，因为它们的标准输入是一个单词，而不是像前馈和卷积网络那样作为标准输入的整个样本。因此，RNNs 具有处理不同长度句子的灵活性，这是由于它们的固定结构，顺序神经网络无法实现。由于它们的灵活结构，RNNs 还可以共享在不同文本位置学习到的特征。

RNN 的反馈循环能力使其能够解析句子中的每个单词并对它进行激活。然后，单词的激活值可以反馈到解析句子的层。因此，激活值会告知句子从每个单词中学到了什么！然后，这个循环对每个单词继续进行，直到网络理解了整个句子。在机器学习的术语中，RNN 将句子中的每个单词视为在特定时间‘t’发生的单独输入，并使用这个输入‘t-1’的激活值作为对原始句子的反馈。

启用 GPU（如果尚未启用）：

1.  在右上角菜单中点击 *运行时*。

1.  从下拉菜单中选择 *更改运行时类型*。

1.  从 *硬件加速器* 下拉菜单中选择 *GPU*。

1.  点击 *保存*。

测试 GPU 是否激活：

```py
import tensorflow as tf
# display tf version and test if GPU is active
tf.__version__, tf.test.gpu_device_name()
```

导入 *tensorflow* 库。如果显示 ‘/device:GPU:0’，则 GPU 已激活。如果显示 ‘..’，则常规 CPU 已激活。

## 使用 RNN 生成文本

如前所述，RNN 通常用于自然语言任务。通常，我们可以通过字符或单词来模拟自然语言任务。我们首先构建一个 *字符级* 模型来生成文本。在下一章中，我们将构建一个 *单词级* 模型来预测情感。

### 文本文件

我们将使用查尔斯·狄更斯的书籍 *双城记*。为了方便，我们已下载了纯文本 UTF-8 并进行了处理，这样您就不必自己处理了。要获取 *处理过的* 文本文件，只需按照以下简单步骤操作：

1.  前往此书籍的 GitHub 网址：[`github.com/paperd/tensorflow`](https://github.com/paperd/tensorflow)*.*

1.  定位文件：点击 *第八章*，点击 *数据*，然后点击 *two_cities.txt*。

1.  点击 *原始* 按钮。

1.  复制文本（*Ctrl+A+C*）。

1.  将其粘贴到 *记事本* 或其他基本文本编辑器中（*Ctrl+V*）。

1.  将其保存在您的计算机上为 [*two_cities.txt*](https://github.com/paperd/tensorflow)。

1.  将文件拖放到您的 Google Drive 的 *Colab Notebooks* 文件夹中。

注意

请获取处理过的版本，因为我们的代码示例基于这个版本。我们希望将学习重点放在文本生成而不是文本处理上。

如果您想获取原始文本文件并自行处理，请访问网址[`www.gutenberg.org/ebooks/98`](https://www.gutenberg.org/ebooks/98)并按照以下步骤操作：

1.  点击 *纯文本 UTF-8*（[`github.com/paperd/tensorflow`](https://github.com/paperd/tensorflow)）作为格式选择。

1.  复制文本（Ctrl+A+C）。

1.  将其粘贴到 [*记事本*](https://github.com/paperd/tensorflow) 或其他基本文本编辑器中。

1.  点击 [*另存为…*](https://github.com/paperd/tensorflow)。

1.  将“另存为类型”更改为 [*所有文件*](https://github.com/paperd/tensorflow)。

1.  将其保存在您的计算机上为 [*two_cities.txt*](https://github.com/paperd/tensorflow)。

1.  处理文本文件。

Project Gutenberg 处理原始文本的出色指南可在[`jss367.github.io/Getting-text-from-Project-Gutenberg.html`](https://jss367.github.io/Getting-text-from-Project-Gutenberg.html)*.*。

注意

如果你亲自处理原始文本文件，*不要* 使用本章中的代码示例，因为你的处理可能与我们代码中的不一致。我们只是想让你有机会练习文本处理。

### 挂载 Google Drive

我们必须挂载 Google Drive 以启用对文本文件的访问：

```py
from google.colab import drive
drive.mount('/content/drive')
```

执行代码单元格后，点击 URL，选择一个 Google 账户，点击 *允许* 按钮，复制授权代码并将其粘贴到文本框 *输入您的授权代码:* 中，然后按键盘上的 *Enter* 键。

注意

确保你的文件在 Google Drive 的 *Colab Notebooks* 目录中！

### 将语料库读入内存

在 NLP 中，一个文本文档通常被称为语料库。**语料库**是一组书面文本，特别是特定作者的全部作品或特定主题的写作集合。

注意

所有代码都是基于文本文件的 *处理* 版本。

将语料库读入内存：

```py
two_cities = 'drive/My Drive/Colab Notebooks/two_cities.txt'
with open(two_cities) as f:
corpus = f.read()
```

### 验证语料库

显示语料库开头的文本：

```py
print (corpus[:74])
```

由于我们从开头开始，验证起来相当容易。但验证结尾需要更多的工作。

获取语料库的长度：

```py
len(corpus)
```

现在，我们知道语料库在哪里结束了。

通过一些尝试和错误，我们可以显示结尾的著名引言：

```py
print (corpus[757116:])
```

如果你想探索其他在线 NLP 书籍，一个很好的起点是以下 URL 的 *Project Gutenberg*：[`www.gutenberg.org/`](http://www.gutenberg.org/)。

### 创建词汇表

由于我们的目标是使用字符级模型生成文本，我们训练模型来预测序列中的下一个字符。然后我们可以反复调用模型来生成更长的文本序列。

创建包含在语料库中的唯一字符的词汇表：

```py
# unique characters in the corpus
vocab = sorted(set(corpus))
print ('{} unique characters'.format(len(vocab)))
```

我们在 *vocab* 中存储了 74 个唯一字符。

### 向量化文本

算法处理的是数字，而不是文本。因此，我们必须设计语料库的数值表示。一个简单的解决方案是将文本向量化。**文本向量化**是将文本转换为数值表示的过程。

首先，我们创建一个名为 *int_map* 的字典来存储唯一字符的整数字典映射。接下来，创建一个名为 *char_map* 的 numpy 数组来存储每个整数字符映射。numpy 数组允许我们将编码后的整数字典映射转换回它们的字符表示。一旦我们将语料库向量化，我们就可以为 TensorFlow 构建输入管道。

### 创建整数字典映射

创建字典：

```py
# create a dictionary with integer representations of characters
int_map = {key : value for value, key in enumerate(vocab)}
int_map['a']
```

我们使用字典推导式创建 *int_map*，它包含语料库的整数字典映射。**字典推导式**是使用简单表达式创建字典的方法。字典推导式的形式为 *{key: value for (key, value) in iterable}*。在我们的情况下，键是语料库中的唯一字符，值是唯一字符的整数字典映射。因此，整数 *45* 代表字母 *a*。

### 创建字符映射

创建包含字符映射的 numpy 数组：

```py
# create numpy array to hold character mappings of integers
import numpy as np
char_map = np.array(vocab)
char_map[45]
```

看起来我们的映射是有效的。

### 映射序列

按照列表 8-1 映射一个序列。

```py
# create variable to hold line break
br = '\n'
# simple sequence
sequence = 'hello world'
print ('original sequence:', sequence, br)
# map to integer representations
maps = np.array([int_map[c] for c in sequence])
print ('integer mappings:', maps, br)
# map integer representations back into characters
s = [char_map[i] for i in maps]
# create string from list of characters
s = ''.join(s)
print ('translation:', s)
Listing 8-1
Map a text sequence
```

序列 "hello world" 使用 *int_map* 向量化，并使用 *char_map* 转换回原始序列。

### 向量化语料库

现在我们有了映射算法，我们准备向量化语料库：

```py
# vectorize the corpus
encoded = np.array([int_map[c] for c in corpus])
encoded[:20], char_map[encoded[:20]]
```

Numpy 数组 *encoded* 存储了向量化后的语料库。我们显示 20 个整数映射及其字符等效项，以验证一切正常。

### 预测下一个字符

在 RNN 中，由于内置的反馈循环，每个神经元都有多个激活。**时间步**是指一个神经元的单个激活。在训练的每个时间步，我们的目标是根据一个或一系列字符预测下一个可能的字符。因此，输入到模型中的必须是字符序列。但是，字符序列必须构建成与我们的模型兼容。

### 创建训练输入序列

为了创建训练实例，我们将编码后的语料库划分为输入序列。每个输入序列包含语料库中的 *seq_length* 个字符。*seq_length* 是我们希望单个输入序列中字符的最大长度。我们将语料库划分为等长序列以提高性能。

对于每个输入序列，样本包含输入序列，相应的目标包含向右移动一个字符的输入序列。目标是这样创建的，因为我们试图预测序列中的下一个字符。因此，我们将文本划分为 *seq_length + 1* 的块。

首先将编码后的语料库转换为 TensorFlow 张量：

```py
# initialize maximum length sequence for a single input
seq_length = 100
# create training dataset
ds = tf.data.Dataset.from_tensor_slices(encoded)
ds
```

### 显示样本

显示训练数据集的一些样本：

```py
for i in ds.take(6):
print (i.numpy(), ':', char_map[i])
```

我们使用之前创建的 *char_map* 数组将每个字符的整数表示转换回其字符状态。

### 批量序列

创建批量序列：

```py
sequences = ds.batch(seq_length + 1, drop_remainder=True)
```

批量方法使我们能够轻松地将单个字符转换为所需大小的序列。使用参数 *drop_remainder=True* 确保所有批次大小相同。

显示第一个批次：

```py
for i in sequences.take(1):
print (char_map[i], br)
print ('batch size:', len(i))
```

批量大小为 *101*，以考虑目标集。

注意

每个目标通过将输入序列向右移动一个字符来表示。

### 创建样本和目标

构建一个创建样本和目标集的函数：

```py
def create_sample_target(piece):
sample = piece[:-1]
target = piece[1:]
return sample, target
```

函数将输入序列向右移动**1**个字符，以形成每个批次的样本和目标文本。因此，每个样本包含前 100 个字符，每个目标包含从第二个字符到第 101 个字符。

创建数据集：

```py
dataset = sequences.map(create_sample_target)
```

将函数映射到**序列**上，以创建由所需样本和目标集组成的语料库。

显示第一个输入序列的样本和目标：

```py
for sample, target in  dataset.take(1):
print ('sample:', char_map[sample], br)
print ('target:', char_map[target])
```

如所需，目标比样本提前一个字符，这样算法就可以从目标中学习如何预测下一个字符。

### 时间步预测

让我们查看样本和目标集的前五个时间步，如列表 8-2 所示。

```py
for i, (input_idx, target_idx) in enumerate(
zip(sample[:5], target[:5])):
print('Step:', i)
print(' input:', input_idx.numpy(),
char_map[input_idx])
print(' expected output:', target_idx.numpy(),
char_map[target_idx])
if i < 4: print()
Listing 8-2
Time step prediction
```

样本和目标向量的每个索引都作为一个单独的时间步处理。也就是说，样本和目标中处理的每个字符都是一个时间步。因此，对于时间步 0 的输入，模型接收*A*的索引（input_idx），并试图预测下一个字符*空格*的索引。在下一个时间步，模型重复相同的过程。但 RNN 模型除了当前输入字符外，还考虑了前一步的上下文。输出验证样本和目标集已正确创建。

### 创建训练批次

数据集已经分割成可管理的文本序列。但在将数据馈送到模型之前，我们需要打乱数据并将其打包成批次。

设置批量和缓冲区大小：

```py
BATCH_SIZE = 64
BUFFER_SIZE = 10000
```

打乱、批处理、缓存和预取：

```py
corpus_ds = (dataset
.shuffle(BUFFER_SIZE)
.batch(BATCH_SIZE, drop_remainder=True)
.cache().prefetch(1))
corpus_ds
```

TensorFlow 数据设计用于处理无限序列。因此，它不会尝试在内存中打乱整个序列。相反，它维护一个缓冲区，其中打乱元素。我们将*BUFFER_SIZE*设置为 10000，以给 TensorFlow 一个相当大的缓冲区大小，但不是太大，以免引起内存问题。我们注意到我们的数据集包含 64 个批量大小的训练样本和 100 个序列长度的目标。

### 构建模型

首先初始化一些重要的变量。我们将*vocab_size*设置为语料库中唯一字符的数量。我们将*嵌入维度*设置为 256。**词嵌入**是 NLP 中的一种学习技术，其中词汇表中的单词或短语被映射到实数向量。

实际上，我们使用维度在 50 到 500 之间的词嵌入向量。我们使用 256，因为我们认为它在处理时间和性能之间是一个很好的折衷。增加词嵌入的数量可以提供更好的性能。但更高的嵌入维度计算成本更高。我们将*rnn_units*设置为 1024，这代表从一层输出的神经元数量。

初始化变量：

```py
# length of the vocabulary in chars
vocab_size = len(vocab)
# the embedding dimension
embedding_dim = 256
# number of RNN units
rnn_units = 1024
```

按照列表 8-3 创建模型。

```py
# generate seed for reproducibility
tf.random.set_seed(0)
np.random.seed(0)
# clear any previous models
tf.keras.backend.clear_session()
# import libraries
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense,\
Embedding
from tensorflow.keras import losses
# create the model
model = Sequential([
Embedding(vocab_size, embedding_dim,
batch_input_shape=[BATCH_SIZE, None]),
GRU(rnn_units, return_sequences=True,
stateful=True, recurrent_initializer="glorot_uniform"),
Dense(vocab_size)
])
Listing 8-3
Create the model
```

第一层是一个嵌入层，其输入为词汇大小、嵌入维度和批次的输入形状。嵌入层的输出连接到第二层，即一个具有 1024 个神经元（由*rnn_units*参数标识）的 GRU 层。为了保留在此层学到的内容，我们设置*return_sequences=True*和*stateful=True*。我们还告诉 GRU 层通过设置*recurrent_initializer='glorot_uniform'*从均匀分布中抽取样本。GRU 层的输出连接到最终密集层，其输入为词汇大小。

**门控循环单元**（GRUs）是 RNN 中的一个门控机制，与具有遗忘门的长期短期记忆（LSTM）类似，但参数比 LSTM 少，并且没有输出门。

深入讨论 GRUs，请参考以下 URL：[`https://arxiv.org/ftp/arxiv/papers/1701/1701.05923.pdf`](https://arxiv.org/ftp/arxiv/papers/1701/1701.05923.pdf)。

### 显示模型摘要

检查模型：

```py
model.summary()
```

第一层是一个嵌入层。因此，通过将 74 个词汇大小乘以 256 个嵌入维度，计算出可学习参数的数量，总共为 18,944。

第二层是一个 GRU 层。可学习参数的数量基于公式 *3* `×` *(n*^(*2*) `×` *mn + 2n)*，其中 *m* 是输入维度，*n* 是输出维度。乘以 *3* 是因为 GRU 有三个操作集，需要这些大小的权重矩阵。将 *n* 乘以 *2* 是因为 RNN 的反馈循环。因此，我们得到 3,938,304 个可学习参数。以下是结果分解：

+   * 3 `×` (1024² + 1024 `×` 256 + 2 `×` 1024)

+   * 3 `×` (1048576 + 262144 + 2048)

+   * 3 `×` 1312768

+   * 3,938,304

如我们所见，计算第二层的可学习参数相当复杂。所以让我们逻辑地分解它。GRU 层是一个具有反馈循环的前馈层。前馈网络的可学习参数通过将前一层的输出（256 个神经元）与当前层的神经元（1024 个神经元）相乘来计算。在前馈网络中，我们还需要考虑当前层的 1024 个神经元。但由于 RNN 的反馈机制，我们将当前层的 1024 个神经元乘以 2。最后，当前层的 1024 个神经元被反馈，产生 1024²个可学习参数。GRU 使用三组操作（隐藏状态、重置门和更新门），需要这些大小的权重矩阵，因此我们将可学习参数乘以 3。

第三层是密集层。因此，通过将输出维度 74 乘以输入维度 1024，并加上 74 以计算该层的神经元数量，总共为 75,850。

### 检查输出形状

显示数据集中第一个批次的形状：

```py
for sample, target in corpus_ds.take(1):
example_batch_predictions = model(sample)
example_batch_predictions.shape
```

第一个批次具有预期的 *batch_size* 为 *64*，*sequence_length* 为 *100*，*vocab_size* 为 *74*。请注意，密集层的模型摘要输出形状为 (64, None, 74)。序列长度没有包括，因为模型可以在任何长度的输入上运行。

### 计算损失

我们从输出分布中采样以预测字符索引。输出分布由字符词汇上的 logits 定义。**logit** 是一个介于 0 和 1 之间，以及负无穷和正无穷的值，由 logit 函数导出。简单来说，logit 是一个预测。logit 函数是 sigmoid 函数的逆函数，因为它限制了 Y 轴上的值在 0 到 1 之间，而不是 X 轴。由于我们的模型返回 logits，我们需要设置 *from_logits* 标志来计算损失。

构建一个计算损失的函数：

```py
def loss(labels, logits):
return losses.sparse_categorical_crossentropy(
labels, logits, from_logits=True)
```

调用函数：

```py
pre_trained_loss = loss(target, example_batch_predictions)
```

该模型期望一个由批大小、序列长度和词汇大小组成的 3D 张量。

让我们看看形状是否正确：

```py
print('pred shape: ', example_batch_predictions.shape)
print('scalar_loss: ', pre_trained_loss.numpy().mean())
```

太好了！预测形状符合预期，批大小为 64，序列长度为 100，词汇大小为 74。我们还显示了预训练模型的平均损失。

### 编译模型

编译模型：

```py
model.compile(loss=loss,
optimizer='adam')
```

### 配置检查点

使用 RNN，我们希望保存模型在每个时间步学到的内容。一种方法是通过回调方法保存包含此信息的检查点。**检查点**捕获模型使用的所有 TensorFlow 参数（或 tf.Variable.objects）的确切值。

列表 8-4 保存了检查点，这样我们就可以回忆起 RNN 学到了什么。

```py
import os
# directory where the checkpoints will be saved
checkpoint_dir = './training_checkpoints'
# name of the checkpoint files
checkpoint_files = os.path.join(checkpoint_dir,
'ckpt_{epoch}')
# callback method
checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
filepath=checkpoint_files,
save_weights_only=True)
Listing 8-4
Configure and save checkpoints
```

### 训练模型

训练模型 10 个 epoch：

```py
EPOCHS = 10
history = model.fit(corpus_ds, epochs=EPOCHS,
callbacks=[checkpoint_callback])
```

我们可以添加更多的 epoch 来提高性能。我们告诉模型使用*checkpoint_callback*来保存检查点。

### 为文本创建重建模型

现在我们已经训练了一个 RNN，我们重新构建模型以一次预测一个字符。我们分三步进行重建：

1.  从检查点恢复权重。

1.  以 1 个批处理大小重建。

1.  加载权重并将模型重塑以确保具有 1 个批处理大小的张量。

#### 从检查点恢复权重

从训练期间建立的检查点恢复权重。我们恢复检查点以获取 RNN 在每个时间步学到的内容。

恢复：

```py
tf.train.latest_checkpoint(checkpoint_dir)
```

#### 以 1 个批处理大小重建

由于我们构建的序列是向前看一个字符，所以我们一次预测一个字符。由于 RNN 状态是从时间步到时间步传递的，一旦构建完成，模型只能接受一个固定的批处理大小。

按照列表 8-5 所示，以 1 个批处理大小重建模型（而不是 64）。

```py
# generate seed for reproducibility
tf.random.set_seed(0)
np.random.seed(0)
# clear any previous models
tf.keras.backend.clear_session()
# set batch size to 1
BATCH_SIZE = 1
# Rebuild model
model = Sequential([
Embedding(vocab_size, embedding_dim,
batch_input_shape=[BATCH_SIZE, None]),
GRU(rnn_units, return_sequences=True,
stateful=True, recurrent_initializer="glorot_uniform"),
Dense(vocab_size)
])
Listing 8-5
Rebuild the model
```

#### 加载权重和重塑

加载从训练的 RNN 保存的权重并将模型重塑以确保张量具有 1 个批处理大小：

```py
model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
model.build(tf.TensorShape([1, None]))
```

### 模型摘要

检查模型：

```py
model.summary()
```

注意，输出形状的第一个参数为 1，表示批处理大小已按预期更改。

### 创建生成文本的组件

要创建新的文本，创建一个函数并初始化一组变量以供函数使用。为了准备 TensorFlow 使用的起始字符串，在传递给函数之前对其进行矢量化并重塑。

#### 创建函数

创建如列表 8-6 所示的文本生成函数。

```py
def create_text(model, input_eval, temperature, start_string):
# Empty string to store our results
new_text = []
# Here batch size == 1
model.reset_states()
for i in range(n):
# model encoded input
predictions = model(input_eval)
# remove batch dimension so we can manipulate predictions
predictions = tf.squeeze(predictions, 0)
# divide predictions by temperature
predictions = predictions / temperature
# use a categorical distribution to predict character
# returned by model
predicted_id = tf.random.categorical(
predictions, num_samples=1)[-1,0].numpy()
# pass predicted character as next input to model
# with previous hidden state
input_eval = tf.expand_dims([predicted_id], 0)
# append generated characters to text
new_text.append(char_map[predicted_id])
return (start_string + ''.join(new_text))
Listing 8-6
Create the text function
```

函数接受模型、矢量化起始字符串、温度和原始起始字符串。它首先初始化一个列表来保存创建的新文本并重置模型的各个状态。函数继续通过迭代*n*次（我们希望创建的字符数）。

**温度**是神经网络的一个超参数，用于通过在应用 softmax 激活之前缩放 logits 来控制预测的随机性。更简单地说，温度表示在计算 softmax 激活之前将 logits 除以多少。

在迭代过程中，该函数对编码后的起始字符串进行建模，并将结果放入 *predictions*。然后它移除额外的 *1* 维度，以便可以将 *predictions* 的内容除以 *temperature*。函数的下一个任务是使用分类分布来预测模型返回的下一个字符。函数需要将 *1* 维度添加回去，以便可以将预测的字符作为下一个输入传递给模型，同时带上之前的隐藏状态。新生成的字符被附加到 *new_text* 数组中。这个过程会重复进行，直到循环结束。

#### 初始化变量

初始化变量：

```py
n = 500
temp = 0.3
start_string = 'Tale'
```

将 *n* 设置为我们希望创建的字符数。接着设置温度和起始字符串。低温会导致文本更加可预测，而高温会导致文本更加令人惊讶。你可以尝试找到最佳设置。你也可以尝试不同的起始字符串。我们选择 *Tale* 是因为我们知道语料库中包含这个名字。

#### 向量化并重新塑形起始字符串

向量化起始字符串，因为模型只识别数字。重新塑形向量化后的起始字符串以供 TensorFlow 使用。也就是说，向起始字符串添加 1 维度，以便模型可以处理它。显示形状以验证一切正常。

将起始字符串向量化并重新塑形以供 TensorFlow 使用，如列表 8-7 所示。

```py
# vectorize starting string
input_vectorized = [int_map[s] for s in start_string]
print ('original shape:', end=' ')
print (str(np.array(input_vectorized).ndim) + 'D', br)
# reshape string for TensorFlow model consumption
input_vectorized = tf.expand_dims(input_vectorized, 0)
print ('new shape:', input_vectorized.shape)
Listing 8-7
Vectorize and reshape the starting string
```

新形状是 (1, 4)。*1* 维度表示批处理大小为 1。*4* 维度表示 start_string 的长度。通过尝试不同的起始字符串来观察不同文本的生成。

### 创建新文本

设置随机种子并调用函数：

```py
tf.random.set_seed(0)
np.random.seed(0)
print (create_text(model, input_vectorized, temp, start_string))
```

哇！尽管句子毫无意义，但模型确实创造了实际的句子。如果你仔细想想，我们刚才所做的是多么令人惊叹。模型吞噬了一个语料库，并能够从中学习。
