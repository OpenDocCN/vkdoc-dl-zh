# 9. 情感分析

我们已经展示了如何训练一个字符级 RNN 来创建原始文本。现在，我们创建一个 *词级* RNN 来分析情感。

**情感分析**是使用自然语言处理文本分析工具对文本数据中的极性、情感和意图进行解释和分类。极性可以是积极的、消极的或中性的。情感可以跨越广泛的感受，如愤怒、快乐、挫败感和悲伤等。意图也可以跨越广泛的动机，如感兴趣或不感兴趣。情感分析的一个常见应用是通过在线反馈识别客户对产品、品牌或服务的情感。一般应用包括社交媒体监控、品牌监控、客户服务、客户反馈和市场研究。

要了解情感分析的精彩讨论，请参阅以下 URL：

[情感分析](https://monkeylearn.com/sentiment-analysis/%2523:%253F:text%253DSentiment%2520analysis%2520is%2520the%2520interpretation,or%2520services%2520in%2520online%2520feedback)

情感分析是一个非常常见的自然语言处理任务。技术上，它通过计算识别和分类文本语料库中表达的意见来确定态度或情感。通常，情感分析用于确定对特定主题或产品的正面、负面或中性意见。

各章节的笔记本位于以下 URL：[GitHub 上的 tensorflow](https://github.com/paperd/tensorflow)。

## IMDb 数据集

用于练习自然语言处理的流行数据集是 IMDb 评论数据集。**IMDb** 是二元情感分类的基准数据集。该数据集包含 50,000 条标记为正面（1）或负面（0）的电影评论。评论经过预处理，每个评论都被编码为整数形式的单词索引序列。评论中的单词通过其在数据集中的整体频率进行索引。50,000 条评论分为 25,000 条用于训练和 25,000 条用于测试。因此，我们可以使用分类或其他深度学习算法来预测正面和负面评论的数量。

IMDb 流行是因为它易于使用，相对容易处理，并且对机器学习爱好者来说挑战性足够。我们喜欢与 IMDb 合作，因为它与电影数据一起工作非常有趣。

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

## 将 IMDb 作为 TFDS 加载

加载 IMDb 的推荐方式是作为 TFDS：

```py
import tensorflow_datasets as tfds
imdb, info = tfds.load(
'imdb_reviews/subwords8k', with_info=True,
as_supervised=True, shuffle_files=True)
```

我们使用 *imdb_reviews/subwords8k* TFDS，因此我们在较小的词汇量上训练模型。*subwords8k* 子集的词汇量为 8,000，这意味着我们在 8,000 个在评论中最常用的单词上训练模型。这也意味着我们不需要构建自己的词汇字典！我们可以使用这个子集并获得良好的性能，并且显著减少训练时间。加载 TFDS 还为我们提供了访问 *tfds.features.text.SubwordTextEncoder* 的权限，这是 TFDS 文本编码器。

我们将 *with_info=True* 设置为启用对数据集和编码器的信息访问。我们将 *as_supervised=True* 设置为使返回的 TFDS 具有两个元组结构（输入，标签），这与 *builder.info.supervised_keys* 一致。如果设置为 *False*（默认值），则返回的 TFDS 将包含所有特征的字典。我们将 *shuffle_files=True* 设置为因为通常可以提高性能。

### 显示键

通过将数据集作为 TFDS 加载，我们可以访问其键：

```py
imdb.keys()
```

我们可以看到数据集被分为测试、训练和无监督样本。

### 将数据集分为训练集和测试集

由于我们正在构建一个监督模型，我们只对训练和测试样本感兴趣：

```py
train, test = imdb['train'], imdb['test']
```

### 显示第一个样本

探索数据集的一个样本总是一个好主意：

```py
br ='\n'
for sample, target in train.take(1):
print ('encoded review:')
print (sample, br)
print ('target:', target.numpy())
```

第一个训练示例包含一个编码的评论和一个标签。评论已经编码为一个整数张量，数据类型为 *int64*。标签是一个标量值，可以是 0（负面）或 1（正面），数据类型为 *int64*。

审查张量的形状表示了它包含的单词数量。为了提高可读性，我们使用 *numpy* 方法将目标张量转换为数值。

### 显示 TFDS 的信息

*info* 对象为我们提供了访问元数据的方式：

```py
info
```

### 浏览元数据

查看训练和测试分割中的示例数量：

```py
train_size = info.splits['train'].num_examples
test_size = info.splits['test'].num_examples
train_size, test_size
```

查看监督键：

```py
info.supervised_keys
```

查看特征信息：

```py
info.features
```

查看 TFDS 的名称及其描述的一部分：

```py
info.name, info.description[0:25]
```

我们甚至可以截取引用字符串以获取标题：

```py
info.citation[184:242]
```

### 创建编码器

编码器内置在 TFDS 的 *SubwordTextEncoder* 中。有了编码器，我们可以轻松地进行解码（整数到文本）和编码（文本到整数）。我们从数据集的 *info* 对象访问编码器。

根据我们加载到内存中的 IMDb 数据集创建一个编码器：

```py
encoder = info.features['text'].encoder
```

现在编码器已经构建完成，我们可以用它来将字符串向量化，并将向量化字符串解码回文本字符串。

测试编码器：

```py
sample_string = 'What a Beautiful Day!'
encoded_string = encoder.encode(sample_string)
print ('Encoded string:', encoded_string)
original_string = encoder.decode(encoded_string)
print ('Original string:', original_string)
```

### 在样本上使用编码器

创建一个返回可读形式标签评分的函数：

```py
def rev(d):
if tf.math.equal(d, 0): return 'negative review'
elif tf.math.equal(d, 1): return 'positive review'
```

显示如列表 9-1 所示的第一个评论。

```py
for sample, target in train.take(1):
print ('review:', end=' ')
text = encoder.decode(sample)
print (text[0:100])
print ('opinion:', end=' ')
print ('\'' + rev(target) + '\'')
Listing 9-1
Display the first review
```

显示如列表 9-2 所示的多个评论。

```py
n = 6
for i, sample in enumerate(train.take(n)):
if i > 0:
print ('review', str(i+1) +':', end=' ')
text = encoder.decode(sample[0])
print (text[0:100])
print ('opinion:', end=' ')
print ('\'' + rev(sample[1]) + '\'')
if i < n-1:
print ()
Listing 9-2
Display multiple reviews
```

我们跳过了第一条评论，因为我们已经看过它了。

显示词汇量大小：

```py
print('Vocabulary size: {}'.format(encoder.vocab_size))
```

### 完成输入管道

创建编码字符串（或评论）的批次，以极大地提高性能。由于机器学习算法期望相同大小的批次，请使用`*padded_batch*`方法对序列进行零填充，以便每个评论的长度与批次中最长字符串的长度相同。

初始化变量：

```py
BUFFER_SIZE = 10000
BATCH_SIZE = 64
```

如列表 9-3 所示，在适当的地方进行洗牌、分批、缓存和预取训练和测试集。

```py
train_ds = (train
.shuffle(BUFFER_SIZE)
.padded_batch(BATCH_SIZE)
.cache().prefetch(1))
test_ds = (test
.padded_batch(BATCH_SIZE)
.cache().prefetch(1))
Listing 9-3
Finish the input pipeline
```

检查张量：

```py
train_ds, test_ds
```

查询以下 URL 获取有关填充字符张量的更新：

[TensorFlow 文本分类 RNN 教程](http://www.tensorflow.org/tutorials/text/text_classification_rnn)

### 创建模型

播种种子、导入库、清除以前的模型，并创建模型，如列表 9-4 所示。

```py
import numpy as np
# generate seed for reproducibility
tf.random.set_seed(0)
np.random.seed(0)
# import libraries
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense,\
Embedding
# clear any previous models
tf.keras.backend.clear_session()
# build the model
embed_size = 128
model = Sequential([
Embedding(encoder.vocab_size, embed_size, mask_zero=True,
input_shape=[None]),
GRU(128, return_sequences=True),
GRU(128),
Dense(1, activation="sigmoid")
])
Listing 9-4
Create the model
```

第一层是嵌入层。`*embedding*`层用于为传入的单词创建词向量。在训练过程中，通过使相似类别彼此更接近的方式学习词类别的表示（或词向量）。因此，词向量可以存储单词之间的关系，如`*good*`和`*great*`。由于我们的模型学习单词关系，词向量是密集的。因此，词向量不会像我们对 one-hot 编码那样用大量的零进行填充。

嵌入层接受词汇大小、嵌入大小和输入形状。我们将`*mask_zero=True*`设置为通知模型忽略填充标记，所有下游层。忽略填充标记可以提高性能。

接下来的两层是 GRU 层，最后一层是单神经元密集输出层。输出层使用 sigmoid 激活来输出评论表达对电影正面或负面情绪的估计概率。

### 模型摘要

检查模型：

```py
model.summary()
```

第一层是嵌入层。因此，通过将词汇大小 8185 乘以嵌入维度（embed_size）128，得到总共 1,047,680 个可学习的参数。

第二层是 GRU 层。可学习的参数数量基于公式`*3* `×` *(n*^(*2*) `×` *mn + 2n)*，其中*m*是输入维度，*n*是输出维度。乘以`*3*`是因为 GRU 需要这些大小的权重矩阵的三套操作。将*n*乘以`*2*`是因为 RNN 的反馈循环。因此，我们得到 99,072 个可学习的参数。

下面是如何分解结果：

+   * 3 `×` (128² + 128 `×` 128 + 2 `×` 128)

+   * 3 `×` (16384 + 16384 + 256)

+   * 3 `×` 33024

+   * 99,072

如我们所见，计算第二层的可学习参数相当复杂。所以让我们逻辑上分解它。一个 GRU 层是一个具有反馈循环的前馈层。前馈网络的可学习参数是通过将前一层（128 个神经元）的输出与当前层的神经元（128 个神经元）相乘来计算的。在前馈网络中，我们还需要考虑该层的 128 个神经元。但由于 RNN 的反馈机制，我们需要将这层的 128 个神经元乘以 2。最后，当前层的 128 个神经元被反馈，产生 128²个可学习参数。GRU 使用三组操作（隐藏状态、重置门和更新门），需要权重矩阵，所以我们把可学习参数乘以 3。

第三层是一个 GRU。由于 n 和 m 与第二层完全相同，我们得到 99,072 个可学习参数。所以计算是相同的。

最终层是密集层。因此，通过将输出维度 1 乘以输入维度 128，并加 1 来考虑该层的神经元数量，总共得到 129 个可学习参数。

### 编译模型

编译：

```py
model.compile(loss='binary_crossentropy', optimizer="adam",
metrics=['accuracy'])
```

### 训练模型

我们发现两个 epoch 提供了相当好的准确度，无需调整。然而，你的结果可能会有所不同。所以你可以尽情地实验。但请记住，训练文本模型需要大量的训练时间！

```py
# to suppress unimportant error messages
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
history = model.fit(train_ds, epochs=2, validation_data=test_ds)
```

### 在测试数据上泛化

虽然模型拟合信息在训练过程中提供了验证损失和准确度值，但始终明确地在测试数据上评估模型是一个好主意，因为准确度和损失值可能会有所不同：

```py
test_loss, test_acc = model.evaluate(test_ds)
```

### 可视化训练性能

如列表 9-5 所示进行可视化。

```py
import matplotlib.pyplot as plt
# history.history contains the training record
history_dict = history.history
acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']
epochs = range(1, len(acc) + 1)
plt.figure(figsize=(12,9))
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.ylim((0.5,1))
plt.show()
# clear previous figure
plt.clf()
plt.figure(figsize=(12,9))
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
Listing 9-5
Visualize training performance
```

### 从虚构评论中进行预测

让我们从我们虚构的评论中进行预测。首先，创建一个返回预测的函数。由于我们创建了自己的评论，该函数必须将文本评论转换为 TensorFlow 可以消费的形式。

创建函数：

```py
def predict_review(text):
encoded_text = encoder.encode(text)
encoded_text = tf.cast(encoded_text, tf.float32)
prediction = model.predict(tf.expand_dims(encoded_text, 0))
return prediction
```

函数接受一个文本评论。它首先对评论进行编码。然后，将编码后的评论转换为 float32。函数通过进行预测并将结果返回到调用环境结束。我们向编码后的文本添加*1*维，以便它可以被 TensorFlow 模型消费。

使用虚构的评论测试函数：

```py
review = ('Just loved it. My kids thought the movie was cool. '
'Even my wife liked it.')
pred = predict_review(review)
pred, pred.shape
```

我们有一个预测。预测值大于 0.5 表示评论是正面的。否则，评论是负面的。

让我们通过创建另一个函数来使预测更加易于接受：

```py
def palatable(pred):
score = tf.squeeze(pred, 0).numpy()
return score[0]
```

函数将预测转换为 numpy 标量。

调用函数：

```py
score = palatable(pred)
score, score.shape
```

函数从预测中移除了*1*维。

让我们进一步创建一个函数，该函数返回纯英文的评论：

```py
def impression(score):
if score >= 0.5:
return 'positive impression'
else:
return 'negative impression'
```

调用函数：

```py
impression(score)
```

如预期，评论是正面的。

让我们再试一次：

```py
review = ('The movie absolutely sucked. '
'No character development. '
'Dialogue just blows.')
pred = predict_review(review)
score = palatable(pred)
print (impression(score))
```

如预期，评论是负面的。

### 在测试数据批次上进行预测

我们也可以从测试集中进行预测。让我们使用*predict*方法对第一个测试批次进行预测。由于测试数据已经以张量形式存在，我们不需要进行编码。

从测试集批次中进行预测并显示如清单 9-6 所示的第一个评论。

```py
# get predictions from 1st test batch
for sample, target in test_ds.take(1):
y_pred_64 = model.predict(sample)
# display first review from this batch
print ('review:', end=' ')
print (encoder.decode(sample[0])[177:307])
# display first label from this batch
print ('label:', end=' ')
print (target[0].numpy(), br)
# display number of examples in the batch
print ('samples and target in first batch:', end=' ')
len(sample), len(target)
Listing 9-6
Make predictions based on a batch from the test set
```

我们从*test_ds*中获取第一个批次。我们使用*predict*方法进行预测，并将它们放置在 y_pred_64 变量中。变量*y_pred_64*包含 64 个预测，因为批次大小是 64。然后我们显示该批次的第一个评论及其相关标签。记住，标签*1*表示评论是积极的，标签*0*表示它是消极的。最后，我们显示样本大小和目标的大小，以验证我们第一个批次中有 64 个示例。

获取第一个预测：

```py
print (y_pred_64[0])
```

使其易于接受：

```py
impression(y_pred_64[0])
```

将预测与实际标签进行比较：

```py
impression(y_pred_64[0]), impression(target[0].numpy())
```

如果预测与实际标签匹配，则它是正确的。

清单 9-7 显示了前五个预测的预测效果。

```py
for i in range(5):
p = impression(y_pred_64[i])
t = impression(target[i].numpy())
print (i, end=': ')
if p == t: print ('correct')
else: print ('incorrect')
Listing 9-7
Prediction efficacy for five predictions
```

### 第一批次的预测准确率

创建一个函数将印象转换回 1 或 0 的标签：

```py
def convert_label(feeling):
if feeling == 'positive impression':
return 1
else: return 0
```

返回整个第一个批次的预测准确率，如清单 9-8 所示。

```py
ls = []
n = len(target)
for i, _ in enumerate(range(n)):
t = target[i].numpy() # labels
p = convert_label(impression(y_pred_64[i])) # predictions
if t == p: ls.append(True)
correct = ls.count(True)
acc = correct / n
batch_accuracy = str(int(np.round(acc, 2) * 100)) + '%'
print ('accuracy for the first batch:', batch_accuracy)
Listing 9-8
Prediction accuracy for the first batch
```

我们首先遍历第一个批次，并将标签与预测进行比较。如果预测正确，我们将此信息添加到列表中。然后我们继续计算正确预测的数量。最后，我们将正确预测的数量除以批次大小，以获得整体预测准确率。

## 利用预训练嵌入

令人惊讶的是，我们可以在 IMDb 数据集上重用预训练模型中的模块。*TensorFlow Hub 项目*是一个包含数百个可重用机器学习模块的库。**模块**是 TensorFlow 图的一个自包含部分，包括其权重和资产，可以在称为迁移学习的过程中的不同任务之间重用。**迁移学习**是一种机器学习方法，其中为某个任务开发的模型被用作不同任务中模型的起点。

您可以通过浏览以下 URL 来浏览库：

[`http://tfhub.dev`](http://tfhub.dev)

一旦你找到了一个模块，将 URL 复制到你的模型中。该模块及其预训练权重将自动下载。使用预训练模型的一个巨大优势是我们不需要从头开始创建和训练自己的模型！

### 加载 IMDb 数据集

由于我们使用的是预训练模型，我们可以使用完整词汇表加载 IMDb 数据集：

```py
data, info = tfds.load('imdb_reviews', as_supervised=True,
with_info=True, shuffle_files=True)
```

我们使用完整词汇表，因为我们不需要担心用它进行训练！

显示元数据：

```py
info
```

### 构建输入管道

创建训练集和测试集：

```py
train, test = data['train'], data['test']
```

批量和预取：

```py
batch_size = 32
train_set = train.repeat().batch(batch_size).prefetch(1)
test_set = test.batch(batch_size).prefetch(1)
```

检查张量：

```py
train_set, test_set
```

### 创建预训练模型

导入 TF Hub 库并创建一个骨架模型来容纳预训练模块，如清单 9-9 所示。

```py
import tensorflow_hub as hub
# clear any previous models
tf.keras.backend.clear_session()
model = tf.keras.Sequential([
hub.KerasLayer(
'https://tfhub.dev/google/tf2-preview/nnlm-en-dim50/1',
dtype=tf.string, input_shape=[], output_shape=[50]),
Dense(128, activation="relu"),
Dense(1, activation="sigmoid")
])
Listing 9-9
Create the model
```

*hub.KerasLayer* 下载句子编码模块。每个输入到该层的字符串都会自动编码为 50D 向量。因此，每个向量代表 50 个单词。每个单词都是基于在 70 亿词的 Google 新闻语料库上预训练的嵌入矩阵进行嵌入的。接下来添加了两个密集层，以提供基本情感分析模型。使用 TF Hub 方便且高效，因为我们可以使用从预训练模型中学到的知识。

### 编译模型

编译模型：

```py
model.compile(loss='binary_crossentropy', optimizer="adam", metrics=['accuracy'])
```

### 训练模型

训练模型：

```py
# to suppress unimportant error messages
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
history = model.fit(train_set,
steps_per_epoch=train_size // batch_size,
epochs=5, validation_data=test_set)
```

训练时间显著减少！

### 进行预测

基于第一个 *test_set* 批次进行预测：

```py
for sample, target in test_set.take(1):
y_pred_32 = model.predict(sample)
```

由于批大小为 32，因此每个批次有 32 个预测。

如列表 9-10 所示，通过索引显示测试集中的误分类。

```py
for i in range(batch_size):
p = convert_label(impression(y_pred_32[i]))
l = target[i].numpy()
if p != l:
print ('pred:', p, 'actual:', l, 'indx:', i)
Listing 9-10
Misclassifications in the first batch by index
```

### 计算第一个批次的预测准确度

列表 9-11 展示了计算第一个批次预测准确度的代码。

```py
ls = []
n = len(target)
for i, _ in enumerate(range(n)):
t = target[i].numpy() # labels
p = convert_label(impression(y_pred_32[i])) # predictions
if t == p: ls.append(True)
correct = ls.count(True)
acc = correct / n
batch_accuracy = str(int(np.round(acc, 2) * 100)) + '%'
print ('accuracy for the first batch:', batch_accuracy)
Listing 9-11
Calculate prediction accuracy for the first batch
```

代码不是寻找误分类，而是找到正确的预测。代码首先将实际标签与预测标签进行比较。如果实际标签被正确预测，则将布尔值 *True* 添加到列表中。一旦遍历完第一个批次，就计算列表中的元素数量。这个数量除以批大小 32，以确定准确度，并以百分比的形式显示。

## 使用 Keras 探索 IMDb

由于 Keras 在工业界非常流行，我们演示了如何使用 keras.datasets 训练 IMDb。我们使用 *keras.datasets.imdb.load_data* 函数以格式就绪的方式加载数据集，以便在神经网络和深度学习模型中使用。

加载 Keras IMDb 有一些优势。首先，单词已经被编码为整数。其次，编码的单词按照其在数据集中的绝对流行度排列。因此，每条评论中的句子由一系列整数组成。第三，第一次调用 *imdb.load_data* 时，将 IMDb 下载到您的计算机上，并存储在您的家目录下 *~/.keras/datasets/imdb.pkl* 作为 32 兆字节的文件。imdb.load_data 函数还提供了额外的参数，包括要加载的前置单词数量（在返回的数据中，整数较低的单词标记为零），要跳过的前置单词数量（以避免像 *the* 这样的单词），以及支持的评论最大长度。

加载 Keras IMDb：

```py
train, test = tf.keras.datasets.imdb.load_data()
```

函数将数据加载到训练和测试元组中。因此 *train[0]* 包含训练评论，*train[1]* 包含训练标签。而 *test[0]* 包含测试评论，*test[1]* 包含测试标签。每个评论都表示为一个整数 numpy 数组，每个整数代表一个单词。标签包含整数标签列表（0 表示负面评论，1 表示正面评论）。

为了提高可读性，创建变量来表示评论和标签：

```py
train_reviews, train_labels = train[0], train[1]
test_reviews, test_labels = test[0], test[1]
```

显示训练和测试评论样本的形状：

```py
train_reviews.shape, test_reviews.shape
```

如预期的那样，我们有 25,000 条训练和 25,000 条测试评论。

显示训练和测试标签的形状：

```py
train_labels.shape, test_labels.shape
```

如预期，我们有 25,000 个训练标签和 25,000 个测试标签。

### 探索训练样本

显示标签类别和唯一单词数量：

```py
print ('categories:', np.unique(train_labels))
print ('number of unique words:',
len(np.unique(np.hstack(train_reviews))))
```

数据集由两个类别标记，代表每个评论的情感。训练样本包含 88,585 个唯一单词。

让我们看看最长训练评论中有多少单词：

```py
longest = np.amax([len(i) for i in train_reviews])
print ('longest review:', longest)
```

我们创建一个包含每个评论单词数的列表，然后找到单词数最多的评论的长度。

获取最长评论的索引：

```py
mid_result = np.where([len(i) for i in train_reviews] == longest)
longest_index = mid_result[0][0]
longest_index
```

我们使用*np.where*函数来查找索引。我们使用了双重索引，因为该函数返回一个包含我们所需索引的列表的元组。

### 创建解码函数

创建一个函数，将评论解码为可读的英文形式，如列表 9-12 所示。

```py
def readable(review):
index = tf.keras.datasets.imdb.get_word_index()
reverse_index = dict([(value, key)\
for (key, value) in index.items()])
return ' '.join( [reverse_index.get(i - 3, '?')\
for i in review])
Listing 9-12
Function that decodes a review
```

函数使用*tf.keras.datasets.imdb.get_word_index*实用工具来获取单词及其唯一分配的整数的字典。然后，该函数创建另一个包含从第一个字典中的值和键分组作为键和值的键值对字典。最后，它根据其 ID（或键）返回单词。索引偏移量为 3，因为 0、1 和 2 是保留索引，用于*填充*、*序列开始*和*未知*。

### 调用解码函数

让我们看看最长评论的样子，如列表 9-13 所示。

```py
review = readable(train_reviews[longest_index])
print ()
print ('review:', end=' ')
# just display a slice of the full review
print (review[:50] + ' ...', br)
label = train_labels[longest_index]
idea = impression(label)
print (idea, br)
# verify length of review
print (len(train_reviews[longest_index]))
Listing 9-13
Decode the longest review
```

由于我们已经知道最长评论的索引，我们可以轻松地从*train_reviews*中检索它。显示其一部分，因为评论相当长。我们也可以轻松地从*train_labels*中检索标签。使标签可读并显示它。最后，显示最长评论的长度。

让我们看看最短评论的样子。但我们不能直接这样做。

我们必须首先找到单词的最小数量：

```py
shortest = np.amin([len(i) for i in train_reviews])
print ('shortest review:', shortest)
```

由于我们不知道哪个评论是最短的，我们使用*amin*方法来返回最小值。

我们现在可以获取训练样本中最短评论的索引：

```py
result = np.where([len(i) for i in train_reviews] == shortest)
shortest_index = result[0][0]
shortest_index
```

我们使用*where*方法来返回我们寻求的索引。由于该方法返回所有满足条件的评论，我们抓取第一个并显示其索引。

列表 9-14 显示了评论、其可读形式的标签及其长度。

```py
review = readable(train_reviews[shortest_index])
print (review[2:], br)
label = train_labels[shortest_index]
idea = impression(label)
print (idea, br)
# verify length of review
print (len(train_reviews[shortest_index]))
Listing 9-14
Display the shortest review
```

### 继续探索训练样本

返回平均评论长度：

```py
length = [len(i) for i in train_reviews]
print ('average review length:', np.mean(length))
```

显示第一个标签及其作为整数编码的评论，如列表 9-15 所示。

```py
first_label = train_labels[0]
print('label:', first_label, end=' ')
idea = impression(first_label)
print ('(' + idea + ')', br)
# display slice of first review
print (train_reviews[0][:20])
# display readable slice of first review
print (readable(train_reviews[0][:20]))
Listing 9-15
First label and its review
```

以可读形式显示第一个评论：

```py
review = readable(train_reviews[0])
print (review[2:105] + ' ...')
```

## 训练 Keras IMDb 数据

限制词汇量大小以提高性能：

```py
# limit vocabulary to 8000 most commonly used words in reviews
vocab_size = 8000
```

将文本大小裁剪到 80 个单词以提高性能：

```py
maxlen = 80
```

### 加载数据

使用有限词汇量加载数据：

```py
(x_train, y_train), (x_test, y_test) =\
tf.keras.datasets.imdb.load_data(num_words=vocab_size)
```

显示训练和测试数据的信息：

```py
print ('train and test features:')
print (len(x_train), 'train sequences')
print (len(x_test), 'test sequences', br)
print ('sequence shape before padding:')
print ('x_train shape:', x_train.shape)
print ('x_test shape:', x_test.shape)
```

### 填充样本

将训练集和测试集转换为 numpy：

```py
x_train = np.asarray(x_train)
x_test = np.asarray(x_test)
```

导入适当的库：

```py
from tensorflow.keras.preprocessing.sequence import pad_sequences
```

填充样本以确保所有序列长度相同：

```py
print('padded sequences (samples, maxlen):')
x_train = pad_sequences(x_train, maxlen=maxlen)
x_test = pad_sequences(x_test, maxlen=maxlen)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)
```

### 构建输入管道

初始化管道变量：

```py
buffer_size = 10000
batch_size = 512
```

准备 TensorFlow 消耗的训练数据：

```py
train_k = tf.data.Dataset.from_tensor_slices(
(x_train, y_train))
train_ks = train_k.shuffle(
buffer_size).batch(batch_size).prefetch(1)
```

准备 TensorFlow 消耗的测试数据：

```py
test_k = tf.data.Dataset.from_tensor_slices(
(x_test, y_test))
test_ks = test_k.batch(batch_size).prefetch(1)
```

### 构建模型

清除任何之前的模型：

```py
tf.keras.backend.clear_session()
```

创建模型：

```py
embed_size = 128
model = Sequential([
Embedding(vocab_size, embed_size, mask_zero=True,
input_shape=[None]),
GRU(128, return_sequences=True),
GRU(128),
Dense(1, activation="sigmoid")
])
```

### 编译模型

编译：

```py
model.compile(loss='binary_crossentropy', optimizer="adam",
metrics=['accuracy'])
```

### 训练模型

抑制错误信息：

```py
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
```

训练：

```py
epochs = 2
model.fit(train_ks, batch_size=BATCH_SIZE,
epochs=epochs, validation_data=(test_ks))
```

### 预测

获取预测：

```py
k_pred = model.predict(test_ks)
```

显示第一个预测：

```py
impression(k_pred[0][0])
```

显示审查的一部分：

```py
pred_first = readable(x_test[0])
pred_first[26:53]
```

显示印象：

```py
impression(y_test[0])
```
