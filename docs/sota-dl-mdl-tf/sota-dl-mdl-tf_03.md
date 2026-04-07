# 3. TensorFlow 数据集

我们通过讨论和演示它们的许多方面以及代码示例来介绍 TensorFlow 数据集。尽管 TensorFlow 数据集不是 ML 模型，但我们包括这一章，因为我们在这本书的许多章节中都使用了它们。这些数据集是由 TensorFlow 团队创建的，以提供用于实践 ML 实验的多种数据集。

当我们开始使用 TensorFlow 时，我们对 TensorFlow 数据集的机制不熟悉。我们主要使用 NumPy 数据。因此，我们不得不花相当多的时间来熟悉它们。我们相信这一章应该能帮助你轻松地导航这些数据集。如果你已经对 TensorFlow 数据集有经验，你可能不需要在本章中通过示例进行操作。

各章节的 Notebooks 位于以下 URL：

[深度学习模型](https://github.com/paperd/deep-learning-models)

## TensorFlow 数据集简介

TensorFlow 数据集（TFDSs）提供了一组可用于 TensorFlow 或其他机器学习框架（例如 Jax、Apache Spark、Accord.NET）的数据集。所有 TFDSs 都作为 tf.data.Datasets 暴露，这使得我们能够使用 tf.data API 轻松构建高性能的输入管道。它们也非常容易确定性地下载和准备 TFDSs，这使得它们成为数据集构建者。**数据集构建者**是一个利用一组方法来准备数据供 ML 模型消费的对象。

不要将 TFDSs 与 tf.data 混淆。tf.data API 允许我们构建高效的数据管道。TFDSs 是 tf.data 的高级包装器。

要获取 TFDSs 的精彩概述，请查阅

[TensorFlow 数据集概览](http://www.tensorflow.org/datasets/overview)

## 导入 TensorFlow 库

导入库并将其别名为**tf**：

```py
import tensorflow as tf
```

## GPU 硬件加速器

为了方便，我们重复说明如何启用 GPU：

1.  在右上角菜单中点击*运行时*。

1.  从下拉菜单中选择*更改运行时类型*。

1.  从*硬件加速器*下拉菜单中选择*GPU*。

1.  点击*保存*。

验证 GPU 是否处于活动状态：

```py
tf.__version__, tf.test.gpu_device_name()
```

如果显示“/device:GPU:0”，则 GPU 处于活动状态。如果显示“..”，则常规 CPU 处于活动状态。

备注

如果出现错误**NAME ‘TF’ IS NOT DEFINED**，请重新执行代码以导入 TensorFlow 库！

## 可用数据集

所有数据集构建者都是 tfds.core.DatasetBuilder 类的子类。因此，每个 TFDS 对象都定义为 tfds.core.DatasetBuilder 对象。要获取可用 TFDS 构建器的列表，请使用 tfds.list_builders()：

```py
import tensorflow_datasets as tfds
tfds.list_builders()
```

让我们看看目前有多少可用：

```py
len(tfds.list_builders())
```

哇！有很多 TFDSs（TensorFlow 数据集）可供深度学习实践！

要了解更多关于 TFDSs 的信息，请查阅

[TensorFlow 数据集目录概览](http://www.tensorflow.org/datasets/catalog/overview)

## 加载数据集

所有构建器都包含一个 *tfds.core.DatasetInfo* 对象，它包含数据集的元数据。元数据通过 *tfds.load* API 或 *tfds.core.DatasetBuilder* API 访问。tfds.load API 是 tfds.core.DatasetBuilder 的一个薄包装器。因此，它更容易使用。根据我们的研究，它是下载 TFDS 的首选方法。我们使用任何 API 都可以得到相同的结果。

使用 tfds.load 加载数据集：

```py
ds, info = tfds.load('mnist', split='train',
shuffle_files=True,
with_info=True,
try_gcs=True)
ds
```

tfds.load API 下载数据并将其保存为 TFRecord 文件。然后它加载 TFRecord 文件并创建一个 tf.data.Dataset。*TFRecord 格式* 是一种用于存储一系列二进制记录的简单格式。因此，变量 *ds* 包含一个由 MNIST 数据组成的 tf.data.Dataset。数据被打乱，返回元数据集，并从 Google Cloud Service (GCS)检索数据集。*MNIST* 数据库是一个包含手写数字的大型数据库，在机器学习领域的训练和测试中广泛使用。

tfds.load API 可用的常见参数包括

+   *split* – 分割数据（例如，‘train’，[‘train’，‘test’]，‘train[80%:]’，等等）。

+   *shuffle_files* – 控制是否在每个 epoch 之间打乱文件（TFDSs 将大数据集存储在多个较小的文件中）。

+   *as_supervised* – 如果为 True，tf.data.Dataset 具有一个二元组结构（输入，标签）。如果为 False，tf.data.Dataset 具有字典结构。

+   *data_dir* – 数据集保存的位置（默认为 ~/tensorflow_datasets/）。

+   *with_info=True* – 返回包含数据集元数据的 tfds.core.DatasetInfo。

+   *download=False* – 禁用下载。

*try_gcs* 参数在文档中未列出。它告诉加载器从 GCS 检索数据集。

## TFDS 元数据

要访问所有元数据，只需显示 *info* 对象的内容：

```py
info
```

从元数据中，我们看到 MNIST 包含 60,000 个 28 × 28 的训练特征图像和 10,000 个 28 × 28 的测试特征图像。训练集和测试集都有相应的标量标签。

## 遍历数据集

### 作为字典

默认情况下，一个 tf.data.Dataset 包含一个 tf.Tensors 的字典。让我们看看数据集中的一个示例：

```py
ds = ds.take(1)
ds
```

使用 *take()* 从 tf.data.Dataset 获取 *n* 个示例。在这种情况下，我们获取一个示例。示例的形式为 *{'image': tf.Tensor, 'label': tf.Tensor}*。简单来说，每个示例包含一个图像张量和标签张量。图像张量是一个 28 × 28 的矩阵，标签张量是一个标量。

图像数据的数据类型为 tf.uint8，是 8 位无符号整数。一个 *uint8* 数据类型包含从 0 到 255 的整数。与所有无符号数一样，值必须为非负。Uint8s 主要用于图形。作为备注，颜色（代表颜色的像素）总是非负的。

标签数据的数据类型为 tf.int64，是 64 位有符号整数。一个 *int64* 数据类型包含从负 9,223,372,036,854,775,808 到正 9,223,372,036,854,775,807 的带符号整数。

显示示例信息：

```py
for example in ds:
print ('keys:', list(example.keys()))
image = example['image']
label = example['label']
print ('shapes:', image.shape, label)
```

需要一个循环来从数据集中提取图像和标签信息。

### 作为元组

使用 *as_supervised=True*，tf.data.Dataset 包含特征和标签的元组：

```py
ds = tfds.load('mnist', split='train', as_supervised=True,
try_gcs=True)
ds = ds.take(1)
for image, label in ds:
print (image.shape, label)
```

在迭代语句中，只需包含用于存储特征和标签的变量。

### 作为 NumPy 数组

*tfds.as_numpy()* 函数将 tf.data.Dataset 转换为 NumPy 数组的可迭代对象：

```py
ds = tfds.load('mnist', split='train', as_supervised=True,
try_gcs=True)
ds = ds.take(1)
for image, label in tfds.as_numpy(ds):
print (type(image), type(label), label)
print (image.shape)
```

我们将训练数据作为特征图像元组加载。然后，我们取一个示例并将其转换为 NumPy 数组。

方便的是，我们可以将整个数据集作为 NumPy 数组加载（如果内存允许）：

```py
image_train, label_train = tfds.as_numpy(
tfds.load('mnist', split='train',
batch_size=-1, as_supervised=True,
try_gcs=True))
type(image_train), image_train.shape
```

使用 *batch_size=-1*，可以将整个数据集一次性加载到一个批次中。然后，该批次被转换为 NumPy 数组。

注意

注意，在训练之前，确保你的数据集可以放入内存，并且所有示例都具有相同的形状。

由于数据集由 NumPy 数组组成，我们可以使用常规的 Python 操作来检查它。获取示例数量：

```py
len(list(image_train))
```

如预期，训练集中有 60,000 个特征图像元组。

检查第一个示例：

```py
image_train[0].shape, label_train[0]
```

图像张量的形状为 28 × 28 × 1。这里的“1”维度表示图像是灰度的。标签是一个标量，表示图像的类别。

检查几个示例：

```py
for row in range(3):
print (image_train[row].shape, label_train[row])
```

小贴士

如果你更喜欢使用 NumPy 数组而不是 tf.Tensor 对象，如果内存允许，可以一次性将数据集加载到一个批次中。

## 可视化

我们可以方便地可视化 TFDS 对象中的图像。

### tfds.as_dataframe

一种可视化图像数据的方法是将 tf.data.Dataset 对象转换为 pandas.DataFrame 对象。加载数据集，取四个示例，并显示：

```py
ds, info = tfds.load('mnist', split='train', with_info=True,
try_gcs=True)
tfds.as_dataframe(ds.take(4), info)
```

我们使用带有元数据（*with_info=True*）的方式加载数据集，以便进行显示。

### 取示例

我们还可以可视化数据集中的示例。取四个示例，从每个张量中挤压出“1”维度，并将挤压后的张量添加到数组中。我们需要挤压出“1”维度，因为 *imshow()* 函数期望输入一个二维矩阵：

```py
import matplotlib.pyplot as plt
images = []
for example in ds.take(4):
img = tf.squeeze(example['image'])
images.append(img)
```

如列表 3-1 所示，将图像可视化。

```py
rows, cols = 2, 2
plt.figure(figsize=(10, 10))
for i in range(rows*cols):
plt.subplot(rows, cols, i + 1)
plt.imshow(images[i], cmap='bone')
plt.axis('off')
Listing 3-1
Visualize Four Images from MNIST
```

### tfds.show_examples

另一种可视化图像的方法是使用 *show_examples()*：

```py
fig = tfds.show_examples(ds, info)
```

注意

show_examples 只支持图像数据集。

几张图像整齐地显示出来。在每张图像下方是标签名称作为字符串和作为括号中的整数。例如，标签为 4 的图像下方有 4(4)，因为其标签是字符串“4”和整数（4）。

## 加载 Fashion-MNIST

将 Fashion-MNIST 作为 TFDS 对象加载。*Fashion-MNIST* 是由 Zalando 的商品图片组成的数据库，包括 60,000 个示例的训练集和 10,000 个示例的测试集。Zalando 是一家在线时尚公司，利用人工智能 (AI) 来提升客户体验。该数据库旨在作为原始 MNIST 数据集的直接替换，用于基准测试机器学习算法。由于原始 MNIST 数据集非常容易通过一个非常简单的神经网络获得出色的性能，因此正在逐步淘汰作为实践数据集。推荐使用 Fashion-MNIST，因为它更具挑战性。

加载数据集：

```py
fashion, fashion_info = tfds.load(
'fashion_mnist',
split='train',
with_info=True,
shuffle_files=True,
as_supervised=True,
try_gcs=True)
```

我们只加载训练数据。元数据在 *info* 对象中可用。我们还对数据集进行洗牌，以确保每个数据点对模型的影响是独立的，不会受到之前相同点的偏差。通过设置 *as_supervised=True*，返回的 tf.data.Dataset 有一个表示为 (input, label) 的二元结构。通过设置 *try_gcs=True*，数据集直接从 Google Cloud Service (GCS) 流式传输。

检查一个示例：

```py
for image, label in fashion.take(1):
print (image.shape, label)
```

tf.Tensor 由图像和标签组成。图像形状是 28 × 28 × 1。其中“1”维度表示图像是灰度的。标签是一个表示图像类别的标量值。

### 元数据

检查数据集：

```py
fashion
```

如预期，数据集中的图像是 28 × 28 × 1 的张量，标签是标量。

显示元数据：

```py
fashion_info
```

我们可以看到关于数据集的大量信息。其中非常重要的一点是数据的分割方式。方便的是，Fashion-MNIST 数据已经分割成训练集和测试集。因此，我们不需要手动分割数据集！

访问 tfds.features.FeatureDict：

```py
fashion_info.features
```

特征字典包含有关图像和标签的信息。此类信息通常被称为元数据。以下代码片段提供了元数据。

获取类别数量：

```py
num_classes = fashion_info.features['label'].num_classes
num_classes
```

因此，我们有十个类别标签。我们将类别数量放入变量中，以便在模型中使用。

获取类别名称：

```py
classes = fashion_info.features['label'].names
classes
```

我们可以看到每个类别标签的名称。将类别标签放入变量中有助于可视化。

获取形状信息：

```py
print (fashion_info.features.shape)
print (fashion_info.features.dtype)
print (fashion_info.features['image'].shape)
print (fashion_info.features['image'].dtype)
```

我们可以看到特征的形状和数据类型。

#### 显示分割信息

访问 tfds.core.SplitDict：

```py
fashion_info.splits
```

获取可用分割：

```py
list(fashion_info.splits.keys())
```

获取训练分割的可用信息：

```py
print (fashion_info.splits['train'].num_examples)
print (fashion_info.splits['train'].filenames)
print (fashion_info.splits['train'].num_shards)
```

### 可视化

在继续前进之前可视化数据集是个好主意！在进行任何深度学习实验之前，我们探索元数据并可视化示例，以便在进行分析之前了解我们将要处理的内容。目的是要 *了解你的数据*！

显示一些示例：

```py
fig = tfds.show_examples(fashion, fashion_info)
```

每张图像下方是标签名称作为字符串和括号中的整数。例如，外套的图像下方有 Coat(4)，因为其标签是字符串“Coat”和整数（4）。

从数据框中显示示例：

```py
tfds.as_dataframe(fashion.take(4), info)
```

取一些示例并可视化：

```py
images, labels = [], []
for image, label in fashion.take(4):
img = tf.squeeze(image)
images.append(img), labels.append(label)
```

如列表 3-2 所示，可视化图像。

```py
rows, cols = 2, 2
plt.figure(figsize=(10, 10))
for i in range(rows*cols):
plt.subplot(rows, cols, i + 1)
plt.imshow(images[i], cmap='bone')
t = classes[labels[i]] + ' (' +\
str(labels[i].numpy()) + ')'
plt.title(t)
plt.axis('off')
Listing 3-2
Visualize Four Images from Fashion-MNIST
```

## Slicing API

所有构建数据集都公开了各种数据子集，这些子集定义为分割（例如，[train, test]）。当构建一个 tf.data.Dataset 时，我们可以指定我们希望切片的分割。我们还可以检索分割的切片以及那些分割的组合。在本节中，我们提供了一些关于如何分割和切片 TFDS 的示例。如果你遇到未预先分割的数据，了解如何手动将数据集分割为训练集和测试集是一个好主意。同样，了解如何切片数据分割（例如，train，test）也是一个好主意。

### 切片说明

我们在创建的 tfds.load 对象中指定切片说明。说明可以是字符串或使用 ReadInstruction API 提供。对于简单情况，字符串更紧凑且可读，而 ReadInstruction API 提供更多选项，并且可能更容易与可变切片参数一起使用。将数据作为字符串加载意味着数据集是一个 Python 字符串。将数据作为 ReadInstruction 对象加载意味着数据集是一个 Python 对象。

注意

由于分片是并行读取的，子分片之间的顺序可能不一致。因此，先读取 test[0:100]，然后读取 test[100:200]，可能得到的示例顺序与先读取 test[:200]的顺序不同。

#### 指令作为字符串

让我们通过一些 tfds.load 示例来操作：

加载完整的训练集：

```py
fashion_train = tfds.load('fashion_mnist', split='train',
try_gcs=True)
fashion_train
```

在这种情况下，我们加载（或切片）了 Fashion-MNIST 的 train 分割。所以在这个上下文中，切片只是加载的另一种说法。

将“train”分割和“test”分割作为两个不同的数据集加载：

```py
train_ds, test_ds = tfds.load('fashion_mnist',
split=['train', 'test'],
try_gcs=True)
train_ds, test_ds
```

将“train”和“test”分割交错加载：

```py
train_test_ds = tfds.load('fashion_mnist', split='train+test',
try_gcs=True)
train_test_ds
```

从“train”分割中加载记录 100（包含）到记录 200（不包含）的切片：

```py
train_100_200_ds = tfds.load('fashion_mnist',
split='train[100:200]',
try_gcs=True)
```

在这种情况下，我们从训练集中加载一个切片。所以在这个上下文中，切片与加载有不同的含义。

加载“train”分割的前 25%的切片：

```py
train_25pct_ds = tfds.load('fashion_mnist',
split='train[:25%]',
try_gcs=True)
```

加载“train”的前 10%到“train”的最后 80%的切片：

```py
train_10_80pct_ds = tfds.load(
'fashion_mnist', try_gcs=True,
split='train[:10%]+train[-80%:]')
```

如列表 3-3 所示，执行十倍交叉验证。

```py
test_cv = tfds.load('fashion_mnist', try_gcs=True,
split=[f'train[{k}%:{k+10}%]'
for k in range(0, 100, 10)])
train_cv = tfds.load('fashion_mnist', try_gcs=True,
split=[f'train[:{k}%]+train[{k+10}%:]'
for k in range(0, 100, 10)])
Listing 3-3
Tenfold Cross-Validation
```

**交叉验证**是一种通过将原始样本划分为训练集和测试集来评估预测模型的技巧。*十倍交叉验证*是该技术的常见实现。

交叉验证过程首先通过随机打乱数据集开始。下一步是将数据集划分为*k*组。对于每个*k*组，将该组作为保留或测试数据集。将剩余的组作为训练数据集。通过在训练集上拟合模型并在测试集上评估它来继续正常操作。数据样本中的每个观察值被分配给一个单独的组，并且在整个过程中必须保持在那个组中。因此，每个样本有*1*次机会被用于保留集，并且有*k – 1*次用于训练模型。

对于我们的十倍交叉验证，以下是一些文档说明：

+   *验证数据集为 10%：

+   [0%:10%], [10%:20%], …, [90%:100%]

+   * 训练数据集是补足的 90%：

+   [10%:100%]（对应验证集为[0%:10%]）

+   [0%:10%] + [20%:100%]（验证集为[10%:20%]）

+   [0%:90%]（验证集为[90%:100%]）。

#### 使用 ReadInstruction API 的说明

我们从加载完整的训练集开始，使用 ReadInstruction API 展示了等效的说明：

```py
train_ds = tfds.load('fashion_mnist', try_gcs=True,
split=tfds.core.ReadInstruction('train'))
```

加载相同的数据，但作为一个对象而不是字符串。

将完整的“train”分割和完整的“test”分割作为两个不同的数据集加载：

```py
train_ds, test_ds = tfds.load(
'fashion_mnist', try_gcs=True,
split=[tfds.core.ReadInstruction('train'),
tfds.core.ReadInstruction('test')])
```

将完整的“train”和“test”分割交错加载：

```py
ri = tfds.core.ReadInstruction('train')\
+ tfds.core.ReadInstruction('test')
train_test_ds = tfds.load('fashion_mnist',
split=ri, try_gcs=True)
```

加载“train”分割中第 100 条（包含）到第 200 条（不包含）的切片：

```py
train_100_200_ds = tfds.load(
'fashion_mnist',
split=tfds.core.ReadInstruction(
'train', from_=100, to=200,
unit='abs'), try_gcs=True)
```

加载“train”分割前 25%的切片：

```py
train_25_pct_ds = tfds.load(
'fashion_mnist', try_gcs=True,
split=tfds.core.ReadInstruction(
'train', to=25, unit='%'))
```

加载从训练集前 10%到后 80%的切片：

```py
ri = (tfds.core.ReadInstruction('train', to=10, unit='%') +
tfds.core.ReadInstruction('train', from_=-80, unit='%'))
train_10_80pct_ds = tfds.load('fashion_mnist',
split=ri, try_gcs=True)
```

如列表 3-4 所示，执行十折交叉验证。

```py
tests = tfds.load('fashion_mnist', split=
[tfds.core.ReadInstruction('train', from_=k,
to=k+10, unit='%')
for k in range(0, 100, 10)], try_gcs=True)
trains = tfds.load('fashion_mnist', split=
[tfds.core.ReadInstruction('train', to=k, unit='%') +
tfds.core.ReadInstruction('train', from_=k+10, unit='%')
for k in range(0, 100, 10)], try_gcs=True)
Listing 3-4
Tenfold Cross-Validation with ReadInstruction
```

对于我们的十折交叉验证，以下是一些文档说明：

+   * 验证数据集将各为 10%：[0%:10%]，[10%:20%]，…，[90%:100%]。

+   * 训练数据集将各为补足的 90%：

+   [10%:100%]（对应验证集为[0%:10%]）

+   [0%:10%] + [20%:100%]（验证集为[10%:20%]）

+   [0%:90%]（验证集为[90%:100%]）

## 性能技巧

本节旨在为那些对提高 TensorFlow 性能感兴趣的人提供更多信息。我们展示了三个技巧，但还有很多。要深入了解，请查阅以下 URL：

[`www.tensorflow.org/guide/data_performance`](http://www.tensorflow.org/guide/data_performance)

[`www.tensorflow.org/datasets/performances`](https://www.tensorflow.org/datasets/performances)

### 自动缓存

默认情况下，TFDS 自动缓存满足以下约束的数据集：

+   * 总数据集大小（所有分割）已定义且小于 250 MB。

+   * shuffle_files 被禁用或只读取单个分片。

所以，除非你想更改默认设置，否则不要随意调整自动缓存。

### 基准数据集

使用*tfds.core.benchmark(ds)*对任何 tf.data.Dataset 对象进行基准测试。

我们可以在一步中加载和预处理数据集：

```py
ds = tfds.load('fashion_mnist', split='train',
try_gcs=True).batch(32).prefetch(1)
```

注意，我们只是从数据集中抓取了一个批次。

对数据集进行基准测试：

```py
tfds.core.benchmark(ds, batch_size=32)
```

确保将*batch_size*参数初始化为相同的值！

再次运行基准测试：

```py
tfds.core.benchmark(ds, batch_size=32)
```

第二次迭代基准测试由于自动缓存而变得更快！

### 重新加载 TFDS 对象

第一次加载 TFDS 对象时，特别是如果数据集很大，需要相当多的时间。下次加载 TFDS 对象时，它会更快，因为它已经在内存中！如果你想看到这个效果，创建一个新的笔记本并加载 Fashion-MNIST TFDS 对象。再次加载它，并注意它花费的时间非常少。

## 将 Fashion-MNIST 作为单个张量加载

我们不必将 TFDS 作为 TensorFlow 张量加载。如果数据集适合内存，我们可以将其作为 NumPy tf.Tensor 加载。推荐的方法是将整个数据集作为一个单独的张量（或 NumPy 数组）加载。这可以通过将*batch_size=-1*设置为将所有示例作为一个 tf.Tensor 批处理来实现。

将整个数据集作为单个 tf.Tensor 加载并转换为 NumPy 数组：

```py
(img_train, label_train), (img_test, label_test) = tfds.as_numpy(
tfds.load(
'fashion_mnist', try_gcs=True, as_supervised=True,
split=['train', 'test'], batch_size=-1))
```

注意

要像 NumPy 数组一样操作 tf.Tensor，将其作为单个张量加载为 TFDS。

显示形状：

```py
img_train.shape, label_train.shape
```

训练数据图像的形状为(60000, 28, 28, 1)，标签的形状为(60000,)。

### 准备数据以供 TensorFlow 使用

缩放特征图像并将数据转换为 tf.data.Dataset 对象：

```py
train = img_train / 255.0
test = img_test / 255.0
train_ds = tf.data.Dataset.from_tensor_slices(
(train, label_train))
test_ds = tf.data.Dataset.from_tensor_slices(
(test, label_test))
```

由于单个张量是 NumPy 张量，我们可以通过除以 255.0 来简单地缩放图像，就像处理 NumPy 数组一样。

### 构建输入管道

设置参数并构建训练和测试数据的输入管道：

```py
BATCH_SIZE = 128
SHUFFLE_SIZE = 5000
train_f = train_ds.shuffle(SHUFFLE_SIZE).batch(BATCH_SIZE)
train_fm = train_f.cache().prefetch(1)
test_f = test_ds.batch(BATCH_SIZE)
test_fm = test_f.cache().prefetch(1)
```

批量大小和洗牌大小基于试验和错误实验设置。我们建议您尝试不同的批量大小（例如，32, 64）以观察它们对学习性能的影响。洗牌大小似乎对学习性能的影响没有批量大小那么大。因此，我们建议您在实验中保持此值不变。

### 构建模型

获取模型的输入形状：

```py
img_shape = img_train.shape[1:]
img_shape
```

导入所需的库：

```py
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.losses import SparseCategoricalCrossentropy
import numpy as np
```

清除之前的模型并为可重复性生成一个种子：

```py
tf.keras.backend.clear_session()
np.random.seed(0)
tf.random.set_seed(0)
```

创建模型：

```py
model = Sequential([
Flatten(input_shape=img_shape),
Dense(128, activation='relu'),
Dropout(0.4),
Dense(num_classes, activation=None)
])
```

### 编译和训练模型

使用 SparseCategoricalCrossentropy(from_logits=True)进行编译：

```py
model.compile(optimizer='adam',
loss=SparseCategoricalCrossentropy(from_logits=True),
metrics=['accuracy'])
```

*from_logits=True* 属性通知损失函数，模型生成的输出值未经归一化。也就是说，softmax 函数尚未应用于输出层神经元以产生概率分布。除非另有说明，否则 softmax 激活会自动应用于输出值。但我们将明确表示不应用激活。

训练模型：

```py
epochs = 10
history = model.fit(train_fm, epochs=epochs,
verbose=1, validation_data=test_fm)
```

## 将豆子加载为 tf.data.Dataset

让我们加载一个不同的数据集作为 tf.data.Dataset 对象，而不是像处理 Fashion-MNIST 那样作为单个 tf.Tensor。除非您更喜欢使用 NumPy 张量，否则这是处理 TFDS 的首选方式。我们坚信，使用各种 TFDS 示例可以帮助您更好地理解如何处理此类数据。豆子数据集是一个很好的例子，因为它包含的示例并不多，而且只有三个类别。

*Beans* 是一个包含使用智能手机相机在田间拍摄的豆子图像的数据集。它由三个类别组成。其中两个是疾病类别，另一个是健康类别。描述的疾病是角斑病和豆锈。数据由乌干达国家作物资源研究学院（NaCRRI）的专家标注，并由 Makerere AI 研究实验室收集。

示例预先分为测试、训练和验证集。测试集包含 128 个示例，训练集包含 1,034 个示例。验证集包含 133 个示例。当然，如果您愿意，可以手动以不同的方式分割数据集。

加载数据集：

```py
beans, beans_info = tfds.load(
'beans', with_info=True, as_supervised=True,
try_gcs=True)
```

检查数据：

```py
beans
```

我们看到示例被分为包含 500 × 500 × 3 特征图像的测试、训练和验证集。因此，我们不需要自己分割数据。

对于深度学习实验，数据集通常分为训练集和测试集。在工业界，建议将数据分为训练集、验证集和测试集。通常，训练数据用于学习，验证数据用于调整，测试数据用于泛化。但是，我们可以使用测试数据进行调整，使用验证数据进行泛化。

将数据分为三个集合对于工业用途来说更优越，因为测试集从未被学习模型接触过。因此，它可以更自信地用于泛化。专业数据科学家擅长从验证集中调整模型。大多数在线教程只使用训练集和测试集拆分，因为重点是学习而不是在工业中的应用。

### 元数据

检查 info 对象：

```py
beans_info
```

检查形状：

```py
beans['train'], beans['test'], beans['validation']
```

检查拆分：

```py
beans_info.splits
```

从元数据中，我们可以获取类别标签和类别数量：

```py
class_labels = beans_info.features['label'].names
num_classes = beans_info.features['label'].num_classes
class_labels, num_classes
```

### 可视化

我们可以通过几种方式可视化 TFDS。

使用 show_examples 方法显示示例：

```py
fig = tfds.show_examples(beans['train'], beans_info)
```

TFDS 包含一个方法来显示一些示例。标签以字符串形式显示类别名称，并在图像下方显示数值。

我们还可以将示例显示为数据框：

```py
tfds.as_dataframe(beans['train'].take(4), info)
```

最后，我们可以手动显示示例。首先构建一个网格来显示多个示例。然后从训练集中选择图像：

```py
num = 30
images, labels = [], []
for feature, label in beans['train'].take(num):
images.append(tf.squeeze(feature.numpy()))
labels.append(label.numpy())
```

创建一个函数来显示如列表 3-5 中所示的示例网格。

```py
def display_grid(feature, target, n_rows, n_cols, cl):
plt.figure(figsize=(n_cols * 1.5, n_rows * 1.5))
for row in range(n_rows):
for col in range(n_cols):
index = n_cols * row + col
plt.subplot(n_rows, n_cols, index + 1)
plt.imshow(feature[index], cmap='twilight',
interpolation='nearest')
plt.axis('off')
t = ' ('  + str(target[index]) + ')'
plt.title(cl[target[index]] + t, fontsize=7.5)
plt.subplots_adjust(wspace=0.2, hspace=0.5)
Listing 3-5
Function That Plots a Grid of Examples
```

绘制网格：

```py
rows, cols = 5, 6
display_grid(images, labels, rows, cols, class_labels)
```

显示如列表 3-6 中所示的第一颗健康豆。

```py
for img, lbl in beans['train'].take(30):
if lbl.numpy() == 2:
plt.imshow(img)
plt.axis('off')
print (class_labels[lbl.numpy()], end=' ')
print (lbl.numpy())
break
Listing 3-6
Visualization of the First Healthy Bean
```

### 检查形状

尽管我们知道从元数据中图像都是相同的形状，但请手动检查：

```py
for i, example in enumerate(beans['train'].take(5)):
print('Image {} shape: {} label: {}'.\
format(i+1, example[0].shape, example[1]))
```

正如可视化示例是一个好主意一样，手动检查几个示例的形状也是一个好主意。我们显示五个示例。只需更改 take 方法中的数字（参数值）来改变示例的数量。

### 重新格式化图像

我们不必调整图像的大小，因为它们都是相同的形状。然而，图像相当大。为了提高训练性能，我们将图像调整到更小的尺寸。

创建一个调整和缩放图像的函数：

```py
IMAGE_RES = 224
def format_image(image, label):
image = tf.image.resize(image, (IMAGE_RES, IMAGE_RES))/255.0
return image, label
```

我们将图像调整到相同的形状，因为学习模型期望图像具有相同的大小。我们鼓励您尝试不同的图像大小，但不要使图像太大，因为较大的图像会消耗更多内存。

### 配置数据集以优化性能

使用缓冲预取和缓存来提高 I/O 性能是一种良好的实践。因此，我们在构建输入管道时应用了这两种技术。

预取与训练步骤的预处理和模型执行重叠。当模型执行训练步骤 *s* 时，输入管道正在读取步骤 *s+1* 的数据。这样做可以将步骤时间减少到训练过程和提取数据所需时间的最大值（而不是总和）。在训练期间应用 *tf.data.Dataset.prefetch* 转换来重叠数据预处理和模型执行。

*tf.data.Dataset.cache*转换将数据集缓存在内存或本地存储中。使用此转换可以节省一些操作（如文件打开和数据读取）在每次 epoch 期间执行。具体来说，tf.data.Dataset.cache 转换在第一次 epoch 将图像从磁盘加载到内存后，将图像保留在内存中。因此，数据集在训练期间不会成为瓶颈。如果数据集太大而无法放入内存，请使用此操作创建一个高效的磁盘缓存。

构建输入管道：

```py
BATCH_SIZE = 32
SHUFFLE_SIZE = 500
train_batches = beans['train'].shuffle(SHUFFLE_SIZE).\
map(format_image).batch(BATCH_SIZE).cache().prefetch(1)
validation_batches = beans['test'].\
map(format_image).batch(BATCH_SIZE).cache().prefetch(1)
```

### 构建模型

获取输入形状：

```py
for img, lbl in train_batches.take(1):
in_shape = img.shape[1:]
in_shape
```

输入形状是 TensorShape([224, 224, 3])，正如预期的那样。我们现在有了输入形状作为变量，可以在学习模型中使用。

导入内存中尚未存在的库：

```py
from tensorflow.keras.layers import Conv2D, MaxPooling2D
```

清除模型并生成种子：

```py
tf.keras.backend.clear_session()
np.random.seed(0)
tf.random.set_seed(0)
```

创建一个多层 CNN，如列表 3-7 所示。

```py
model = Sequential([
Conv2D(32, (3, 3), activation = 'relu',
input_shape=in_shape, strides=1,
kernel_regularizer='l1_l2'),
MaxPooling2D(2, 2),
Conv2D(64, (3, 3), activation='relu'),
MaxPooling2D(2, 2),
Conv2D(128, (3, 3), activation='relu'),
MaxPooling2D(2),
Conv2D(128, (3, 3), activation='relu'),
MaxPooling2D(2, 2),
Flatten(),
Dense(512, activation='relu'),
Dense(num_classes, activation='sigmoid')
])
Listing 3-7
Multilayered CNN
```

由于前馈网络在处理大图像时表现不佳，我们需要一个卷积神经网络（CNN）。该模型包括四个卷积层用于二维空间数据。**卷积**是将过滤器简单应用于输入（在我们的情况下是图像），从而产生激活。对同一过滤器重复应用于输入会产生一个称为特征图的激活图。*特征图*表示在输入（如图像）中检测到的特征的位置和强度。

每个卷积层都是一个 MaxPooling2D 层，它对二维空间数据执行最大池化操作。**最大池化**通过在每个特征轴上每个维度定义的池大小窗口中取最大值，从卷积层中下采样输入表示。窗口在每个维度上通过步长移动。

**池化**涉及选择一个池化操作（如过滤器）应用于特征图。池化操作或过滤器的大小小于特征图的大小。具体来说，它几乎总是应用 2 × 2 像素的步长为 2 像素。**步长**是在输入矩阵（或图像）上移动的像素数。当步长为 1 时，我们每次移动过滤器 1 像素。当步长为 2 时，我们每次移动过滤器 2 像素，依此类推。

### 编译并训练模型

使用 SparseCategoricalCrossentropy(from_logits=True)进行编译：

```py
loss = tf.keras.losses.\
SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer='adam',
loss=loss,
metrics=['accuracy'])
```

我们使用这个损失函数是因为我们模型的输出层激活是‘sigmoid’。我们使用这种激活是因为它与数据集配合得很好。尝试不同的激活函数。

训练模型：

```py
epochs = 10
history = model.fit(
train_batches, epochs=epochs,
verbose=1, validation_data=validation_batches)
```

### 预测

由于我们使用了测试数据来调整，因此基于*验证数据集*进行预测（因为它从未被模型看到过）。为验证集构建输入管道，以便为预测做好准备：

```py
validate = beans['validation'].\
map(format_image).batch(BATCH_SIZE).cache().prefetch(1)
```

进行预测：

```py
predictions = model.predict(validate)
```

该变量包含每个示例的预测。每个预测都是一个包含十个预测的数组。因此，我们需要额外的步骤来获取实际的预测。

获取第一个示例的实际预测：

```py
first_prediction = tf.math.argmax(predictions[0])
class_labels[first_prediction.numpy()]
```

使用 tf.math.argmax API 从预测数组中获取实际预测。

如列表 3-8 所示，获取多个预测结果。

```py
p = []
for row in range(8):
pred = tf.math.argmax(predictions[row])
p.append(pred.numpy())
print ('class:', '(' + str(pred.numpy()) + ')', end=' ')
print (class_labels[pred.numpy()])
Listing 3-8
Get Multiple Predictions
```

我们获得了八个预测。通过更改范围参数值，可以得到你想要的任意多个或更少的预测。

了解模型准确度：

```py
for i, (_, label) in enumerate(beans['validation'].take(8)):
if label.numpy() == p[i]:
print ('correct')
else:
print ('incorrect', end=' ')
print ('actual:', label.numpy(), 'predicted:', p[i])
```

对于前八个预测，我们看到了模型的表现如何。我们可以通过取整个验证集并计算平均值来获得整体准确率。

## 摘要

通过示例，我们学习了如何使用 TFDSs。我们还使用 Fashion-MNIST 和豆类数据进行了练习，以熟悉 TFDSs。
