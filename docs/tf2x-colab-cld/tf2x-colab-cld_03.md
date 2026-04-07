# 3. 与 TensorFlow 数据一起工作

我们介绍了 TensorFlow 数据集（TFDS）。我们通过代码示例讨论了 TFDS 的许多方面。我们继续使用完整的 TFDS 模型示例。

各章节的笔记本位于以下 URL：[`https://github.com/paperd/tensorflow`](https://github.com/paperd/tensorflow)。

*TFDS* 是一个包含可用于 TensorFlow 的数据集的集合。像所有 TensorFlow 可消费数据集一样，TFDS 以 tf.data.Datasets 的形式暴露，这使得我们可以创建易于使用、高性能的输入管道。

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

## TensorFlow 数据集（TFDS）

以下 URL 提供了更多关于 TFDS 的信息：

+   [`www.tensorflow.org/datasets`](http://www.tensorflow.org/datasets)

+   [`www.tensorflow.org/datasets/overview`](http://www.tensorflow.org/datasets/overview)

+   [`www.tensorflow.org/datasets/catalog/overview`](http://www.tensorflow.org/datasets/catalog/overview)

从介绍 TFDS 的第一个 URL 开始。第二个 URL 展示了如何显示所有 TFDS 的列表和额外的技术信息。第三个 URL 展示了 TFDS 的分类。

## Colab 出错

当我们长时间（数小时）运行 Google Colab 而不停歇或加载大量数据集到内存中并处理这些数据时，它可能会崩溃（或中断）。当这种情况发生时，我们知道有两种选择：

1.  重新启动所有运行时。

1.  关闭程序并从头开始重新启动。

## 可用 TFDS

让我们从显示可用的 TFDS 列表开始：

```py
import tensorflow_datasets as tfds
# See available datasets
tfds.list_builders()
```

首先导入 *tfds* 模块。使用 *list_builders()* 方法显示。

查找 tensorflow_datasets 容器中有多少 TFDS：

```py
print (str(len(tfds.list_builders())) + ' datasets')
```

哇！我们可以使用 244 个数据集（截至本文写作时）来练习 TensorFlow。

查阅以下 URL，了解关于 TFDS 的精彩教程：

[`https://colab.research.google.com/github/tensorflow/datasets/blob/master/docs/overview.ipynb`](https://colab.research.google.com/github/tensorflow/datasets/blob/master/docs/overview.ipynb)

## 加载 TFDS

我们可以用一行代码加载一个 TFDS！在 TFDS 网站上的教程中，它指出我们必须安装 *tensorflow-datasets*。但在 Google Colab 中，我们不必这样做。

小贴士

如果您在 Google Colab 以外的环境中工作，可以通过运行以下代码片段安装 tfds 模块：`!pip install tensorflow-datasets`

导入 tfds 模块：

```py
import tensorflow_datasets as tfds
```

使用 tfds.load 直接加载训练数据：

```py
# load train set
train, info = tfds.load('mnist', split="train",
with_info=True)
info
```

**tfds.load** 函数将命名数据集加载到 tf.data.Dataset 中。我们添加 *info* 元素以启用显示有关数据集的有用元数据。**元数据**是一组描述和提供其他数据信息的数据。

直接加载测试数据：

```py
# load test data
test, info = tfds.load('mnist', split="test", with_info=True)
info
```

我们从 TFDS 容器中加载了 MNIST 训练和测试数据集。**MNIST** 数据库（修改后的国家标准与技术研究院数据库）是一个包含手写数字的大型数据库，通常用于训练各种图像处理系统。该数据库也广泛用于机器学习领域的训练和测试。数据库由 60,000 个训练示例和 10,000 个测试示例组成。我们包含了 *info* 元素，它提供了有关数据集的详细信息。

注意

在机器学习的术语中，数据元素通常描述为示例或样本。单词 example 可以与 sample 互换使用。

尽管前馈神经网络通常在图像上表现不佳，但 MNIST 是一个例外，因为它经过了大量的预处理，并且图像很小。也就是说，MNIST 图像大致为相同的小尺寸，位于图像空间的中心，并且垂直排列。

## 提取有用信息

info 元素包括允许我们提取有关数据集特定信息的方法。

列表 3-1 提取有关类别的信息。

```py
# create a variable to hold a return symbol
br = '\n'
# display number of classes
num_classes = info.features['label'].num_classes
class_labels = info.features['label'].names
# display class labels
print ('number of classes:', num_classes)
print ('class labels:', class_labels)
Listing 3-1
Extracting meaningful information from a dataset
```

我们刚刚使用 *features* 方法提取了类别数量和类别标签。

## 检查 TFDS

我们有两种检查元素的方法：

1.  打印元素。

1.  使用 *element_spec* 方法打印元素。

打印元素：

```py
# display training and test set
print (train)
print (test)
```

使用 element_spec 打印元素：

```py
# display with element_spec method
print (train.element_spec)
print (test.element_spec)
```

无论哪种方式，输出都非常相似。张量形状为 (28, 28, 1)。因此，训练和测试数据由 28 `×` 28 的图像组成。1 的值表示图像以灰度显示。图像数据（特征集）由 tf.uint8 数据组成，而标签（或目标）数据由 tf.int64 数据组成。

一个 *灰度* 图像是指每个像素的值是一个单独的样本，只表示光量。也就是说，它只携带强度信息。灰度图像仅由灰色阴影组成。图像的对比度范围从最弱强度的黑色到最强强度的白色。

我们也可以用一行代码显示 TFDS 的训练示例：

```py
# Show train feature image examples
fig = tfds.show_examples(train, info)
```

*show_examples* 方法显示 tf.data.Dataset 中的样本图像。

## 特征字典

所有 TFDS 都包含将特征名称映射到张量值的 *特征字典*。默认情况下，tfds.load 返回一个 tf.Tensors 的 *字典*。一个 **tf.Tensor** 代表 TensorFlow 中的数据矩形数组。

一个典型的数据集，如 MNIST，有两个键：*image* 和 *label*。让我们使用 take(1) 检查一个 *样本*。我们输入到 *take* 函数中的数字表示我们从数据集中接收到的样本数量。

从训练数据集中取一个样本并显示其键：

```py
for sample in train.take(1):
print (list(sample.keys()))
```

我们看到了预期的两个键。image 键引用数据集中的图像。label 键引用数据集中的标签。正式的字典结构表示为 *{‘image’: tf.Tensor, ‘label’: tf.Tensor}*。

现在我们知道了键，我们可以轻松地显示第一个训练样本的特征形状和目标值：

```py
for sample in train.take(1):
print ('feature shape:', sample['image'].shape)
print ('target value: ', sample['label'].numpy())
```

第一特征样本的形状是（28, 28, 1），第一个标签的值是 4。我们使用了 *numpy()* 方法将目标张量转换为标量值。由于任何由机器学习算法消耗的数据集 *必须* 具有相同的形状，所以我们从单个样本中获取数据集的形状！

让我们从训练集中获取九个示例：

```py
n, ls = 9, []
for sample in train.take(n):
ls.append(sample['label'].numpy())
ls
```

我们看到 [4, 1, 0, 7, 8, 1, 2, 7, 1]，这与上一节中 *show_examples* 的标签相匹配。

通过使用 tfds.load 的 *as_supervised=True* 参数，我们得到一个（特征，标签）元组而不是一个字典：

```py
ds = tfds.load('mnist', split="train", as_supervised=True)
ds = ds.take(1)
for image, label in ds:
print (image.shape, br, label)
```

样本的形式是（图像，标签）。

我们也可以得到一个 *numpy* 元组（特征，标签）：

```py
ds = tfds.load('mnist', split="train", as_supervised=True)
ds = ds.take(1)
for image, label in tfds.as_numpy(ds):
print (type(image), type(label), label)
```

我们使用 *tfds.as_numpy* 将 tf.Tensor 转换为 np.array，并将 tf.data.Dataset 转换为 Generator[np.array]。

最后，我们可以得到一个 *批处理* 的 tf.Tensor：

```py
image, label = tfds.as_numpy(tfds.load(
'mnist',
split='train',
batch_size=-1,
as_supervised=True,
))
type(image), image.shape
```

通过使用 *batch_size=-1*，我们可以一次性加载整个数据集。我们看到 numpy 张量（60000, 28, 28, 1），这意味着训练数据包含 60,000 个 28 `×` 28 的灰度图像。

总结来说，*tfds.load* 默认返回一个字典，一个带有 *as_supervised=True* 的 tf.Tensor 元组，或者一个带有 *tfds.as_numpy* 的 np.array。请注意，您的数据集可以适应内存，并且所有示例都具有相同的形状。

## 构建输入管道

缩放、洗牌、批处理和预取训练数据：

```py
train_sc = train.map(lambda items:\
(tf.cast(items['image'],\
tf.float32) / 255.,\
items['label']))
train_ds = train_sc.shuffle(10000).batch(32).prefetch(1)
Use the train dataset we loaded earlier in the chapter. Although this looks complicated, we use a lambda function to divide each image by 255\. We then shuffle, batch, and prefetch.
```

**预取** 提高模型性能，因为它增加了批处理过程的效率。当我们的训练算法在一个批次上工作时，TensorFlow 正在并行地处理数据集以准备好下一个批次。

缩放、批处理和预取测试数据：

```py
test_sc = test.map(lambda items:\
(tf.cast(items['image'],\
tf.float32) / 255.,\
items['label']))
test_ds = test_sc.batch(32).prefetch(1)
```

使用本章中加载的测试集。我们不洗牌测试数据，因为它被认为是新数据。

检查张量：

```py
train_ds, test_ds
```

如预期，特征形状为（None, 28, 28, 1）。第一个维度是 *None*，表示批量大小可以是任何值。

## 构建模型

导入库：

```py
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
```

创建模型：

```py
# clear previous model
tf.keras.backend.clear_session()
model = Sequential([
Flatten(input_shape=[28, 28, 1]),
Dense(512, activation="relu"),
Dense(10, activation="softmax")
])
```

该模型有一个输入层、一个密集隐藏层和一个密集输出层。输入层将图像展平以便在下一层进行处理。隐藏层接受数据到 512 个神经元。输出层接受来自隐藏层的数据到十个神经元，这些神经元代表十个数字类别。

## 模型摘要

显示模型的摘要：

```py
model.summary()
```

第一层接受 28 `×` 28 的灰度图像。因此我们得到输出形状（None, 784）。*None* 是第一个参数，因为 TensorFlow 模型可以接受任何批量大小。我们通过将 28 乘以 28 乘以 1 得到第二个参数 *784*。因此，每张图像有 784 个像素。参数数量为 0，因为第一层不对数据进行操作。

第二层输出形状是（None, 512），因为我们有 512 个神经元。可训练参数的数量是 *401920*。我们通过将第一层的 784 个神经元乘以本层的 512 个神经元得到 401408，然后再加上本层的 512 个神经元得到 401920。

第三层输出形状是 (None, 10)，因为我们有十个类别。通过将第二层的 512 个神经元乘以 10 并加上本层的 10 个神经元，我们得到 *5130*。

## 编译和训练模型

使用优化器、损失和度量参数编译模型：

```py
model.compile(optimizer='adam',
loss='sparse_categorical_crossentropy',
metrics=['accuracy'])
```

训练模型：

```py
epochs = 3
history = model.fit(train_ds, epochs=epochs, verbose=1,
validation_data=test_ds)
```

我们对模型进行了三个 epoch 的训练。也就是说，我们三次将数据通过模型。我们用测试数据验证模型。仅用三个 epoch 就得到了相当好的结果，几乎没有过度拟合！

## 在测试数据上泛化

基于测试数据进行评估总是一个好主意：

```py
model.evaluate(test_ds)
```

## 可视化性能

将训练记录放入变量中：

```py
# get training record into a variable
history_dict = history.history
```

列表 3-2 绘制了模型的准确率和损失。

```py
import matplotlib.pyplot as plt
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
Listing 3-2
Visualize training performance
```

可视化显示，我们的模型拟合数据相当好，因为训练和测试准确率非常一致！

## DatasetBuilder (tfds.builder)

由于 tfds.load 实际上只是 DatasetBuilder 的一层便利包装，我们可以直接使用 tfds.builder 构建一个输入管道，使用 MNIST 数据集。**DatasetBuilder** 是所有数据集的抽象基类。也就是说，每个 TensorFlow 数据集都作为 DatasetBuilder 公开。

DatasetBuilder 为 TensorFlow 数据集执行多项任务：

+   使用 *DatasetBuilder.download_and_prepare* 下载数据、提取数据并将其写入标准格式的位置和方式

+   如何使用 *DatasetBuilder.as_dataset* 从磁盘加载数据

+   所有关于数据的信息，包括名称、类型、特征形状、每个训练和测试分割中的记录数以及源 URL，都可以通过 *DatasetBuilder.info* 获取

+   可以使用 *tfds.builder* 直接实例化任何 DatasetBuilder

然而，与 tfds.load 不同，我们必须手动通过名称获取 DatasetBuilder，调用 *download_and_prepare()* 和 *call as_dataset()*。*tfds.builder* 的优势在于，如果需要，它允许我们更多地控制加载过程。

列表 3-3 展示了如何使用 tfds.builder 加载 MNIST。

```py
mnist_builder = tfds.builder('mnist')
mnist_info = mnist_builder.info
mnist_builder.download_and_prepare()
datasets = mnist_builder.as_dataset()
Listing 3-3
Load MNIST with tfds.builder
```

从 *tfds.builder* 开始创建数据集。使用 *info* 方法包含信息。使用 *download_and_prepare* 方法处理数据。将处理后的数据集放入变量中。

构建训练集和测试集：

```py
mnist_train, mnist_test = datasets['train'], datasets['test']
```

使用特征字典来获取关键信息：

```py
for sample in mnist_train.take(1):
print ('feature shape:', sample['image'].shape)
print ('target value: ', sample['label'].numpy())
```

我们看到第一个特征具有形状 (28, 28, 1) 和目标值 4。

## MNIST 元数据

与 tfds.load 类似，tfds.builder 可以访问 MNIST 的元数据：

```py
mnist_info
```

我们看到了关于 MNIST 的许多有用信息。

访问特征信息：

```py
mnist_info.features
```

我们看到了有关图像和标签的有用信息。

列表 3-4 显示了类别的数量和类标签。

```py
# display number of classes
num_classes = mnist_info.features['label'].num_classes
class_labels = mnist_info.features['label'].names
# display class labels
print ('number of classes:', num_classes)
print ('class labels:', class_labels)
Listing 3-4
Number of classes and class labels
```

访问形状和数据类型：

```py
print (mnist_info.features.shape)
print (mnist_info.features.dtype)
```

访问图像信息：

```py
print (mnist_info.features['image'].shape)
print (mnist_info.features['image'].dtype)
```

访问标签信息：

```py
print (mnist_info.features['label'].shape)
print (mnist_info.features['label'].dtype)
```

训练和测试分割：

```py
print (mnist_info.splits)
```

可用的分割键：

```py
print (list(mnist_info.splits.keys()))
```

训练和测试示例的数量：

```py
print (mnist_info.splits['train'].num_examples)
print (mnist_info.splits['test'].num_examples)
```

注意

tfds.load 可以访问与 tfds.builder 相同的元数据。

## 显示示例

如本章前面所示，*tfds.show_examples* 允许我们方便地可视化图像分类数据集中的图像（和标签）。

让我们展示测试集中的示例：

```py
fig = tfds.show_examples(mnist_test, info)
```

## 准备 DatasetBuilder 数据

为 DatasetBuilder 训练和测试数据准备输入管道。

规模化、打乱、批量和预取训练数据：

```py
train_sc = mnist_train.map(lambda items:\
(tf.cast(items['image'], \
tf.float32) / 255.,\
items['label']))
train_build = train_sc.shuffle(1024).batch(128).prefetch(1)
```

规模化、批量和预取测试数据：

```py
test_sc = mnist_test.map(lambda items:\
(tf.cast(items['image'],\
tf.float32) / 255.,\
items['label']))
test_build = test_sc.batch(128).prefetch(1)
```

检查张量：

```py
train_build, test_build
```

如预期，特征形状是 (None, 28, 28, 1)。

## 构建模型

创建模型：

```py
tf.keras.backend.clear_session()
model = Sequential([
Flatten(input_shape=[28, 28, 1]),
Dense(512, activation="relu"),
Dense(10, activation="softmax")
])
```

## 编译模型

编译：

```py
model.compile(loss='sparse_categorical_crossentropy',
optimizer='adam',
metrics=['accuracy'])
```

## 训练模型

训练：

```py
model.fit(train_build, epochs=3, validation_data=test_build)
```

如预期，结果与使用 tfds.load 进行训练非常相似。

## 在测试数据上泛化

根据测试数据进行评估：

```py
model.evaluate(test_build)
```

## 加载 CIFAR-10

让我们使用 DatasetBuilder 方法 tfds.builder 来操作另一个数据集并展示一些示例。**CIFAR-10** 数据集包含 60,000 张 32 `×` 32 的彩色图像，分为十个类别，每个类别有 6,000 张图片。有 50,000 张训练图片和 10,000 张测试图片。

有十个类别：

```py
[airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck]
```

这些类别是完全互斥的。汽车和卡车之间没有重叠。类别 *automobile* 包括轿车、SUV 和其他类似物品。类别 *truck* 仅包括大型卡车。两者都不包括皮卡。

数据集被分为五个训练批次和一个测试批次，每个批次包含 10,000 张图片。测试批次包含每个类别恰好 1,000 张随机选择的图片。训练批次包含剩余的图片，以随机顺序排列，但某些训练批次可能包含比另一个类别更多的图片。在这些批次之间，每个类别包含恰好 5,000 张图片。由于 CIFAR-10 是以批次进行预处理的，我们可以使用这些信息来更好地为训练模型批量数据。

列表 3-5 加载并处理数据集以供 TensorFlow 使用。

```py
cifar10_builder = tfds.builder('cifar10')
cifar10_info = cifar10_builder.info
cifar10_builder.download_and_prepare()
cifar10_train = cifar10_builder.as_dataset(split='train')
cifar10_test = cifar10_builder.as_dataset(split='test')
Listing 3-5
Prepare the CIFAR-10 dataset for TensorFlow consumption
```

检查训练集：

```py
cifar10_train
```

我们可以看到图像张量具有形状 (32, 32, 3)。标签张量具有形状 ()。这意味着每张图像由 32 `×` 32 像素表示。*3* 这个值表示图像是彩色的。图像张量的数据类型是 tf.uint8。每个标签是一个标量值。标签张量的数据类型是 tf.int64。

TensorFlow 利用 RGB 颜色模型来生成彩色图像。**RGB 颜色模型**是一种加色模型，其中红、绿、蓝光以各种方式相加以重现广泛的颜色。该模型的名字来源于三种加色原色的首字母缩写，即红、绿、蓝。

## 检查数据集

获取数据集信息：

```py
cifar10_info
```

获取特征信息：

```py
cifar10_info.features
```

获取类别名称：

```py
cifar10_info.features['label'].names
```

获取可用的分割键：

```py
print (list(cifar10_info.splits.keys()))
```

展示训练示例：

```py
fig = tfds.show_examples(cifar10_train, info)
```

使用特征字典来显示训练标签：

```py
[sample['label'].numpy() for sample in cifar10_train.take(9)]
```

为了简化编码，我们使用了列表推导。

为了完整性，展示测试示例：

```py
fig = tfds.show_examples(cifar10_test, info)
```

## 准备输入管道

规模化、打乱、批量和预取训练数据：

```py
train_sc = cifar10_train.map(lambda items:\
(tf.cast(items['image'],\
tf.float32) / 255.,\
items['label']))
train_cd = train_sc.shuffle(1024).batch(128).prefetch(1)
```

规模化、批量和预取测试数据：

```py
test_sc = cifar10_test.map(lambda items:\
(tf.cast(items['image'],\
tf.float32) / 255.,\
items['label']))
test_cd = test_sc.batch(128).prefetch(1)
```

检查张量：

```py
train_cd, test_cd
```

## 模拟数据

创建模型：

```py
tf.keras.backend.clear_session()
model = Sequential([
Flatten(input_shape=[32, 32, 3]),
Dense(512, activation="relu"),
Dense(10, activation="softmax")
])
```

我们*必须*获取输入形状正确。对于 CIFAR-10，输入形状是 (32, 32, 3)！

检查模型：

```py
model.summary()
```

第一层接受并展平 32 `×` 32 彩色图像。因此我们得到输出形状（None, 3072）。*None* 是第一个参数，因为 TensorFlow 模型可以接受任何批量大小。我们通过将 32 乘以 32 乘以 3 得到第二个参数 *3072*。每张图像有 1,024 像素，这是将 32 乘以 32 的结果。由于每张图像都是彩色的，我们将 1,024 乘以 3 得到 3,072 个神经元。参数数量为 0，因为这是第一层。

第二层的输出形状是（None, 512），因为我们有 512 个神经元。参数数量是 *1573376*。我们通过将第一层的 3,072 个神经元乘以本层的 512 个神经元得到 1,572,864。然后我们再加上本层的 512 个神经元，得到 1,573,376。

第三层的输出形状是（None, 10），因为我们有十个类别。我们通过将 10 乘以第二层的 512 个神经元，并加上本层的 10 个神经元得到 *5130*。

编译模型：

```py
model.compile(optimizer='adam',
loss='sparse_categorical_crossentropy',
metrics=['accuracy'])
```

训练模型：

```py
epochs = 3
history = model.fit(train_cd, epochs=epochs, verbose=1,
validation_data=test_cd)
```

准确率低于 50%是不好的。我们的模型表现非常糟糕，因为前馈神经网络并不是设计来很好地处理图像数据的。

我们的目标是向您展示如何加载和建模一个 TFDS。我们将在后面的章节中介绍适用于图像数据的模型。
