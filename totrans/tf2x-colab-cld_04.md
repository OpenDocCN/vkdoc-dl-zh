# 4. 与其他数据一起工作

在上一章中，我们向您展示了如何使用 TFDS。但如果我们想使用其他类型的数据集呢？在本章中，我们将向您展示如何使用 TensorFlow 处理其他类型的数据。

各章节的笔记本位于以下 URL：[`https://github.com/paperd/tensorflow`](https://github.com/paperd/tensorflow)。

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

## 基本机制

要创建一个输入管道，我们从一个数据源开始。为了从 TensorFlow 可以处理内存中的数据构建数据集，我们可以使用 tf.data.Dataset.from_tensor_slices() 或 tf.data.Dataset.from_tensors()。*from_tensor_slices* 方法为输入张量的每一部分创建一个单独的元素。*from_tensors* 方法将输入合并并返回一个包含单个元素的数据集。我们**仅**使用 *from_tensor_slices* 方法，因为它使我们能够处理每个数据元素。

创建一个简单的 1D 张量并使其对 TensorFlow 可消费：

```py
# create a 1D tensor
ds = tf.data.Dataset.from_tensor_slices([8, 3, 0, 8, 2, 1])
ds.element_spec
```

我们从六个元素的列表创建一个 TensorSpec，并使用 from_tensor_slices 方法使其对 TensorFlow 可消费。形状是 ()，因为 from_tensor_slices 从数组的切片创建对象。

列表 4-1 展示了如何迭代数据集。

```py
# iterate and display tensor values
for elem in ds:
print(elem.numpy())
print ()
# iterate without numpy method
for elem in ds:
print(elem)
Listing 4-1
Iterate and display the from_tensor_slices dataset
```

该数据集包含六个张量。第一个循环显示张量中的每个元素作为 numpy 值。第二个循环显示原始张量。

我们也可以使用 *iter* 创建一个 Python 迭代器，并通过 *next* 方法消费其元素：

```py
it = iter(ds)
# display the first element
next(it).numpy()
```

我们看到张量中的第一个元素被显示出来。

要查看剩余的元素，只需继续运行以下代码：

```py
next(it).numpy()
```

现在，使用 tf.data.Dataset_from_tensors() 创建一个张量：

```py
# create a 1D tensor
ds = tf.data.Dataset.from_tensors([8, 3, 0, 8, 2, 1])
ds.element_spec
```

注意形状是 (6,)，这意味着单个张量由六个元素组成。

让我们迭代如列表 4-2 所示的单个张量。

```py
# iterate and display tensor values
for elem in ds:
print(elem.numpy())
print ()
# iterate without numpy method
for elem in ds:
print(elem)
Listing 4-2
Iterate and display the from_tensors dataset
```

我们有一个包含六个元素的 *单个* 张量的数据集。

## TensorFlow 数据集结构

一个数据集包含具有相同嵌套结构的元素，结构的各个组成部分可以是 tf.TypeSpec 可以表示的任何类型，包括 tf.Tensor、tf.sparse.SparseTensor、tf.RaggedTensor、tf.TensorArray 或 tf.data.Dataset。*Dataset.element_spec* 属性允许我们检查每个元素组件的类型。该属性返回一个与元素结构匹配的 *tf.TypeSpec* 对象的嵌套结构。元素可能是一个单独的组件，一个组件的元组，或者一个嵌套的组件元组。

我们可以通过以下示例更好地理解结构，如列表 4-3 所示。

```py
br = '\n'  # enter a line break in Colab
# create random uniform numbers
scope = tf.random.uniform([4, 10])
print ('shape:', scope.shape, br)
ds = tf.data.Dataset.from_tensor_slices(scope)
print (ds.element_spec, br)
# Let's look at the first element:
it = iter(ds)
# print first element
print ('first element with an iterator:', br)
print (next(it).numpy(), br)
print ('all four elements:', br)
for i, row in enumerate(ds):
print ('element ' + str(i+1))  # add 1 as index starts at 0
print (row.numpy(), br)
Listing 4-3
Data structure example
```

*scope*的形状是(4, 10)，这意味着我们有一个包含四个元素的张量，每个元素包含 0 到 1 之间随机生成的十个均匀分布的数。我们使用 from_tensor_slices 方法从 scope 生成一个 TensorFlow 可消费的数据集，并使用 element_spec 方法显示 TensorSpec，然后显示 TensorSpec 中的每个元素。

注意

除了示例和样本之外，术语*element*也用来描述数据集中的张量。

简单来说，我们创建一个包含四个元素的数据集。每个元素包含 0 到 1 之间的十个随机均匀数。我们将数据集转换为 TensorFlow 可以消费（或处理）的数据集。我们通过迭代 TensorFlow 数据集来显示每个元素。

## 读取输入数据

如果你的所有输入数据都适合内存，创建一个用于 TensorFlow 消费的数据集的最简单方法是将它转换为 tf.Tensor 对象，使用 Dataset.from_tensor_slices()。

## Colab 崩溃

如前所述，当长时间（例如，几个小时）运行 Google Colab 而没有暂停或加载大量数据集到内存中并处理这些数据时，它可能会崩溃（或崩溃）。当这种情况发生时，我们知道你有两种选择：

1.  重新启动所有运行时。

1.  关闭程序并从头开始重新启动。

要重新启动所有运行时，请点击顶部菜单中的*运行时*，从下拉菜单中选择*重启运行时*，并在提示时点击*YES*。Google Colab 推荐此选项。如果你从头开始重新启动，请先清除浏览器历史记录，然后从头开始启动 Google Colab。

## 批次大小

**批次大小**是神经网络模型在一次传递中处理的训练示例数量。不要将批次大小与 epoch 混淆！一个**epoch**是完整遍历训练数据集。因此，epoch 的数量是完整遍历训练数据集的次数。Epochs 与训练示例的处理无关！它们只是表示网络遍历的次数。简单来说，在每次通过（或 epoch）网络时，会处理训练数据集的批次。

批次的大小必须大于或等于一个，并且小于或等于训练数据集中的样本数。这很有意义，因为你不能有一个比总训练示例数更大的批次。

TensorFlow 已**优化**以运行大于一个的训练数据批次。因此，批次大小为一个是效率非常低的！因为批次大小为一是代表整个训练数据集，我们实际上并没有分批处理数据。所以除非你不想分批处理数据，否则不要使用批次大小为一个！

## Keras 数据

*tf.keras.datasets*模块提供了七个预处理好的数据集用于练习 TensorFlow。查看以下 URL 以获取有关数据集的更多信息：[`https://keras.io/api/datasets/`](https://keras.io/api/datasets/)。

让我们获取 Keras MNIST 数据集：

```py
train, test = tf.keras.datasets.mnist.load_data(path='mnist.npz')
```

训练数据和测试数据都包含 MNIST 图像和标签的元组：

```py
type(train), type(test)
```

探索训练数据的形状：

```py
print ('train data:', br)
print (train[0].shape)
print (train[1].shape)
```

探索测试数据的形状：

```py
print ('test data:', br)
print (test[0].shape)
print (test[1].shape)
```

训练数据包括 60,000 个 28 `×` 28 的特征图像和 60,000 个标签。测试数据包括 10,000 个 28 `×` 28 的特征图像和 10,000 个标签。

### 构建输入管道

将训练数据和测试数据分别分割成对应的图像和标签。对图像数据进行缩放。使用 `from_tensor_slices` 方法创建 TensorFlow 可消费的数据。

从训练数据开始：

```py
train_images, train_labels = train
train_sc = train_images / 255  # divide by 255 to scale
train_k = tf.data.Dataset.from_tensor_slices(
(train_sc, train_labels))
train_k.element_spec
```

继续使用测试数据：

```py
test_images, test_labels = test
test_sc = test_images / 255  # divide by 255 to scale
test_k = tf.data.Dataset.from_tensor_slices(
(test_sc, test_labels))
test_k
```

打乱训练数据，批量处理，并预取训练和测试数据：

```py
BATCH_SIZE = 128
SHUFFLE_BUFFER_SIZE = 1000
train_kd = train_k.shuffle(
SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE).prefetch(1)
test_kd = test_k.batch(BATCH_SIZE).prefetch(1)
```

检查张量：

```py
train_kd, test_kd
```

### 创建模型

导入库：

```py
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
```

从内存中清除之前的模型：

```py
# clear any previous models
tf.keras.backend.clear_session()
```

构建模型：

```py
model = Sequential([
Flatten(input_shape=[28, 28]),
Dense(512, activation="relu"),
Dropout(0.5),
Dense(10, activation="softmax")
])
```

该模型是一个具有密集连接层的前馈神经网络。也就是说，所有神经元都能看到数据。第一层接受 28 `×` 28 图像，并将每个图像展平成一个包含 784 个像素的 1D 数组。第二层接受数据到 512 个神经元，并使用 *relu* 激活来最小化损失。第三层使用 dropout 来减轻过拟合。第四层是输出层。它接受数据到十个神经元，因为我们的数据有十个类别标签。它使用 *softmax* 激活来减少损失。

**Dropout** 是一种用于减少神经网络过拟合的正则化技术（由谷歌专利）。该技术通过在神经网络中丢弃单元来实现。

### 模型摘要

显示模型的摘要：

```py
model.summary()
```

第一层的输出形状是 (None, 784)。*None* 是第一个参数，因为 TensorFlow 模型接受任何批量大小。我们通过将 28 乘以 28 图像像素得到第二个参数 784，因为我们想要展平的图像。我们此层没有参数，因为它接受输入形状但不处理数据。

第二层的输出形状是 (None, 512)。第二个参数是 512，因为这是为此层设置的神经元数量。参数数量为 401,920，通过将 512（此层的神经元）乘以 784（前层的神经元）并加上 512 来计算，以考虑此层的神经元。

第三层的输出形状是 (None, 512)。使用 dropout 不会影响神经元数量。因此，此层从上一层继承 (None, 512) 并没有参数。

第四层的输出形状是 (None, 10)。第二个参数是 10，以反映类别的数量。参数数量为 5,130，通过将 10（此层的神经元）乘以 512（前层的神经元）并加上 10 来计算，以考虑此层的神经元。

### 编译模型

编译：

```py
model.compile(optimizer='adam',
loss='sparse_categorical_crossentropy',
metrics=['accuracy'])
```

### 训练模型

训练：

```py
epochs = 10
history = model.fit(train_kd, epochs=epochs, verbose=1,
validation_data=test_kd)
```

由于将周期数设置为 10，我们的模型对数据集进行了 *十次* 处理。由于训练和测试准确率紧密一致，我们没有太多过拟合。

## Scikit-Learn 数据

我们还可以从 scikit-learn 库中读取数据。*Scikit-learn* 是一个用于 Python 编程语言的免费软件机器学习库。

从这个库中获取数据集：

```py
from sklearn.datasets import fetch_lfw_people
faces = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
```

*fetch_lfw_people*数据集是收集自互联网的著名人物 JPEG 图片的集合。所有详细信息都可以在官方网站上找到：[`http://vis-www.cs.umass.edu/lfw/`](http://vis-www.cs.umass.edu/lfw/)。每张图片都集中在单个面部上。典型的机器学习任务是面部验证。因此，给定一对图片，我们预测这两张图像是否为同一个人。

另一种机器学习任务是面部识别（或面部识别）。因此，给定一个未知人的面部图片，我们通过参考已识别人员的先前看到的图片库来识别该人的姓名。

### 探索数据

显示键：

```py
# get available keys from dataset
faces.keys()
```

数据集包含特征图像（在图像容器中）和目标（在目标容器中）。它还包含目标名称和数据集的描述。数据容器由每个图像的展平向量组成。

列表 4-4 显示了形状、目标名称和类别标签。

```py
image, target = faces.images, faces.target
data = faces.data
names = faces.target_names
print ('feature image tensor:', br)
print (image.shape, br)
print ('target tensor:', br)
print (target.shape, br)
print ('flattened image tensor:', br)
print (data.shape, br)
print ('target names:', br)
print (names, br)
print ('class labels:', len(names))
Listing 4-4
Information about the dataset
```

特征图像张量由 1,288 个 50×37 面部图像组成。目标张量由 1,288 个目标组成。数据形状由 1,288 个展平向量组成，每个向量有 1,850 个元素。我们通过将 50 乘以 37 得到 1,850。

列表 4-5 探讨了第一个示例。

```py
# display the first data example
i = 0
print ('first image example:', br)
print (image[i], br)
print ('first target example:', br)
print (target[i], br)
print ('name of first target:', br)
print (names[target[i]], br)
print ('first data example (flattened image):', br)
print (data[i], br)
print ('first image:', br)
import matplotlib.pyplot as plt
# display the first image in the dataset
plt.imshow(image[i], cmap="bone")
plt.title(names[target[i]])
Listing 4-5
Explore the first example
```

每个图像都由一个二维矩阵表示。第一个图像的目标值是 5，恰好是胡戈·查韦斯的图像。

### 构建输入管道

创建训练集和测试集。按列表 4-6 所示缩放特征图像。

```py
from sklearn.model_selection import train_test_split
# create train and test data
X_train, X_test, y_train, y_test = train_test_split(
image, target, test_size=0.33, random_state=0)
# scale feature image data and create TensorFlow tensors
x_train = X_train / 255.0
x_test = X_test / 255.0
# get shapes
print ('x_train shape:', end=' ')
print (x_train.shape)
print ('x_test shape:', end=' ')
print (x_test.shape)
# get sample entries
print (x_train[0])
print (X_train[0][0][0])
print (x_train[0][0][0])
Listing 4-6
Create train and test sets and scale feature images
```

*train_test_split*模块提供了一个简单的方法将数据集分为训练集和测试集。我们还可以根据需要设置训练和测试的大小。*random_state*参数提供了一种重现我们结果的方法。

我们的训练集包含 67%的数据。剩余的 33%放在测试集中。训练特征数据形状为(862, 50, 37)，测试特征数据形状为(426, 50, 37)。因此，训练特征数据由 862 个 50×37 像素的图像组成，测试特征数据由 426 个 50×37 像素的图像组成。

将图像除以 255 以缩放它们。缩放使向量运算更简单。

显示训练集中的第一像素图像。显示第一像素图像的第一像素并显示其缩放值。

继续通过将数据切割成 TensorFlow 可消费的部分：

```py
faces_train = tf.data.Dataset.from_tensor_slices(
(X_train, y_train))
faces_test = tf.data.Dataset.from_tensor_slices(
(X_test, y_test))
```

设置批量和缓冲区大小：

```py
BATCH_SIZE = 16
SHUFFLE_BUFFER_SIZE = 100
```

打乱训练数据，批量处理，并预取训练和测试数据：

```py
faces_train_ds = (faces_train
.shuffle(SHUFFLE_BUFFER_SIZE)
.batch(BATCH_SIZE).prefetch(1))
faces_test_ds = (faces_test
.batch(BATCH_SIZE).prefetch(1))
```

检查张量：

```py
faces_train_ds, faces_test_ds
```

### 构建模型

按照列表 4-7 所示构建一个简单的模型并训练数据。

```py
import numpy as np
class_labels = len(names)
# clear previous model and generate a seed
tf.keras.backend.clear_session()
np.random.seed(0)
tf.random.set_seed(0)
model = Sequential([
Flatten(input_shape=[50, 37]),
Dense(16, activation="relu"),
Dense(class_labels, activation="softmax")
])
Listing 4-7
Model for faces data
```

导入 numpy 库，以便我们可以生成一个随机种子。**随机种子**指定了计算机生成随机数序列时的**起始点**。我们设置随机种子以确保相同的起始点，以便我们可以获得可重现的结果。也就是说，每次我们训练模型时，我们都得到相同的结果。我们将种子设置为 0。当然，你可以将种子设置为任何数字，但请确保每次都使用相同的数字。

输入形状反映了特征数据的图像大小。也就是说，每张图像由 50 `×` 37 像素表示。

### 模型摘要

运行模型摘要：

```py
model.summary()
```

第一层的输出形状是 (None, 1850)。我们通过将 50 乘以 37 得到 1,850。这一层的参数数量是 0，因为这是第一层。

第二层的输出形状是 (None, 16)。在这一层我们有 16 个神经元作用于数据。参数数量是 29,616。我们通过将 16 乘以 1,850 得到 29,600，通过将 16 加到 29,600 上得到 29,616，以考虑到这一层的神经元数量。

第三层的输出形状是 (None, 7)。在这一层我们有 7 个神经元来对应 7 个类别标签。参数数量是 119。我们通过将 16 乘以 7 得到 112，通过将 7 加到 112 上得到 119，以考虑到这一层的神经元数量。None 参数存在是因为 TensorFlow 接受任何批大小。

### 编译模型

编译：

```py
model.compile(optimizer='adam',
loss='sparse_categorical_crossentropy',
metrics=['accuracy'])
```

### 训练模型

训练：

```py
history = model.fit(faces_train_ds, epochs=10,
validation_data=faces_test_ds)
```

性能不佳，因为前馈网络与图像数据配合不佳。然而，我们创建这个简单的模型来向您展示如何训练数据集。

## Numpy 数据

我们可以直接加载 numpy 数据并按如下所示进行缩放（见列表 4-8）。

```py
DATA_URL = 'https://storage.googleapis.com/\
tensorflow/tf-keras-datasets/mnist.npz'
path = tf.keras.utils.get_file('mnist.npz', DATA_URL)
with np.load(path) as data:
train_examples = data['x_train']
train_labels = data['y_train']
test_examples = data['x_test']
test_labels = data['y_test']
train_scaled = train_examples / 255.
test_scaled = test_examples / 255.
Listing 4-8
Load numpy data from a URL
```

识别一个指向 numpy 文件的 URL 路径。使用 tf.keras.utils.get_file 访问路径。使用 *np.load* 函数加载 numpy 数据。将数据分为训练集和测试集。缩放特征图像数据。

### 使用 tf.data.Dataset 加载 Numpy 数组

将训练和测试图像和标签转换为 TensorFlow 可消费的形式：

```py
train_dataset = tf.data.Dataset.from_tensor_slices(
(train_scaled, train_labels))
test_dataset = tf.data.Dataset.from_tensor_slices(
(test_scaled, test_labels))
```

### 准备训练数据

打乱训练数据，批量处理，并预取训练和测试数据：

```py
BATCH_SIZE = 128
SHUFFLE_BUFFER_SIZE = 1000
train_np = train_dataset.shuffle(
SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE).prefetch(1)
test_np = test_dataset.batch(BATCH_SIZE).prefetch(1)
```

检查张量：

```py
train_np, test_np
```

### 创建模型

清除之前的会话并设置随机种子：

```py
tf.keras.backend.clear_session()
np.random.seed(0)
tf.random.set_seed(0)
```

使用与 Keras MNIST 相同的模型：

```py
model = Sequential([
Flatten(input_shape=[28, 28]),
Dense(512, activation="relu"),
Dropout(0.5),
Dense(10, activation="softmax")
])
```

### 模型摘要

由于我们使用了相同的模型，摘要与之前相同：

```py
model.summary()
```

### 编译模型

编译：

```py
model.compile(optimizer='adam',
loss='sparse_categorical_crossentropy',
metrics=['accuracy'])
```

### 训练模型

训练：

```py
epochs = 10
history = model.fit(train_np, epochs=epochs, verbose=1,
validation_data=test_np)
```

## 从 GitHub 获取葡萄酒数据

您可以通过几个简单的步骤直接从 GitHub 获取本书的任何独立数据集：

1.  访问书籍 URL：[`github.com/paperd/tensorflow`](https://github.com/paperd/tensorflow)。

1.  定位数据集并点击它。

1.  点击 *Raw* 按钮。

1.  将 URL 复制到 Colab 并将其分配给一个变量。

1.  使用 Pandas 的 *read_csv* 方法读取数据集。

对于我们的目的，一个 *独立数据集* 是指不存储在软件环境（如 TFDS）中的数据集。让我们从 GitHub 读取一个数据集。

在我们的情况下，访问书籍 URL，点击 *chapter4*，点击 *data*，点击 *winequality-red.csv*，点击 *Raw* 按钮，复制 URL，将其粘贴到 Colab 的代码单元格中，并将其分配给一个变量。

我们已经找到了 URL 并将其分配给一个变量：

```py
url = 'https://raw.githubusercontent.com/paperd/tensorflow/\
master/chapter4/data/winequality-red.csv'
```

将数据集读取到 pandas 数据框中：

```py
import pandas as pd
wine = pd.read_csv(url, sep = ';')
```

确认数据读取正确：

```py
wine.head()
```

## CSV 数据

*CSV* 数据集是一个逗号分隔的值文件，允许数据以表格格式保存。在像 Microsoft Excel 这样的程序中打开时，它看起来像传统的电子表格，但带有 *.csv* 扩展名。CSV 文件可以与任何我们知道的电子表格程序一起使用，包括 Microsoft Excel 或 Google Sheets。

获取机器学习 CSV 数据集的好地方是 UCI 机器学习仓库。仓库的主站点位于以下 URL：[`https://archive.ics.uci.edu/ml/datasets.php`](https://archive.ics.uci.edu/ml/datasets.php)。

在机器学习文献中经常引用的 CSV 数据集是 *winequality-red.csv*。这个数据集包含葡萄牙 *Vinho Verde* 葡萄酒的红色变体。有关详细信息，请参阅以下文章：

*P. Cortez, A. Cerdeira, F. Almeida, T. Matos, and J. Reis. 通过数据挖掘物理化学特性建模葡萄酒偏好。在决策支持系统，Elsevier，2009，47(4):547–553。ISSN: 0167-9236。*

有关数据集的一般信息，请参阅以下 URL：

[`https://archive.ics.uci.edu/ml/datasets/wine+quality`](https://archive.ics.uci.edu/ml/datasets/wine+quality)

### 数据集特征

数据集由十一个独立特征变量和一个目标变量组成。特征变量包括

+   固定酸度

+   易挥发酸度

+   柠檬酸

+   残糖

+   氯化物

+   自由硫磺二氧化物

+   总硫磺二氧化物

+   密度

+   pH 值

+   硫酸盐

+   酒精

目标变量是

+   质量

目标变量（质量）可以取 0 到 10 之间的分数。0 分表示质量非常低。10 分表示质量非常高。数据集包含 1,599 个示例。

### 获取数据

知道如何直接从 UCI 机器学习仓库提取数据集是个好主意。我们现在就来做这件事。

识别数据集 URL：

```py
url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
```

建立数据集的路径：

```py
path = tf.keras.utils.get_file('winequality-red.csv', url)
path
```

从 CSV 文件创建 pandas 数据框并将其放入 Python 变量中：

```py
import pandas as pd
data = pd.read_csv(path, sep = ';')
```

查看数据框开头的记录：

```py
data.head()
```

查看数据框末尾的记录：

```py
data.tail()
```

识别数据集中出现的类标签：

```py
data.quality.unique()
```

由于质量是目标，我们使用 *unique()* 函数从数据集中提取不同的标签。数据集包含分数 3、4、5、6、7 和 8。然而，质量可以取 0 到 10 之间的分数，因此有 *十一个* 可能的类标签。

显示数据类型：

```py
data.dtypes
```

特征数据类型为 float64，目标数据类型为 int64。

显示数据集中示例的数量：

```py
len(data)
```

我们有 1,599 个示例。

### 将数据分为训练集和测试集

创建目标集：

```py
# create a copy of the dataframe
df = data.copy()
# create the target
target = df.pop('quality')
```

创建数据框的副本以保留原始数据。将目标列弹出到一个变量中，以创建目标数据集。*pop* 方法永久地从副本中删除该列。

显示数据框副本发生了什么：

```py
df.head()
```

注意，质量列不再在数据框中。

将数据框转换为 numpy 值：

```py
features = df.values
labels = target.values
```

将数据集分为训练集和测试集，并按列表 4-9 所示缩放特征数据：

```py
X_train, X_test, y_train, y_test = train_test_split(
features, labels, test_size=0.33, random_state=0)
# scale feature image data and create TensorFlow tensors
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.fit_transform(X_test)
Listing 4-9
Split data into train and test sets and scale feature data
```

特征数据不是图像。它们是标量值。因此，我们在应用机器学习技术之前，使用*StandardScaler*模块将连续特征数据转换为*均值为 0*和*标准差为 1*。

### 准备数据以供 TensorFlow 使用

将训练集和测试集切片成 tf.Data.Dataset 数据：

```py
train_wine = tf.data.Dataset.from_tensor_slices(
(X_train_std, y_train))
test_wine = tf.data.Dataset.from_tensor_slices(
(X_test_std, y_test))
```

定义用于保存换行符的变量：

```py
br = '\n'
```

创建一个函数来查看张量：

```py
def see_samples(data, num):
for feat, targ in data.take(num):
print ('Features: {}'.format(feat))
print ('Target: {}'.format(targ), br)
```

查看前三个张量：

```py
n = 3
see_samples(train_wine, n)
```

定义批量大小和缓冲区大小：

```py
BATCH_SIZE = 16
SHUFFLE_BUFFER_SIZE = 100
```

打乱训练数据，批量处理，并预取训练和测试数据：

```py
train_wine_ds = (train_wine.shuffle(
SHUFFLE_BUFFER_SIZE).
batch(BATCH_SIZE).
prefetch(1))
test_wine_ds = (test_wine.batch(
BATCH_SIZE).
prefetch(1))
```

检查张量：

```py
train_wine_ds, test_wine_ds
```

### 构建模型

清除会话并生成种子：

```py
tf.keras.backend.clear_session()
np.random.seed(0)
tf.random.set_seed(0)
```

创建模型：

```py
model = Sequential([
Dense(30, activation="relu", input_shape=[11,]),
Dense(11, activation="softmax")
])
```

输入形状是[11,]，因为数据集有十一个特征。输出形状是十一个，因为数据集有十一个类别标签。

### 模型摘要

检查模型：

```py
model.summary()
```

第一层的输出形状是(None, 30)，因为我们在这层有 30 个神经元。参数数量是 360。我们通过将 30 个神经元乘以 11 个特征得到 330。通过将这层的 30 个神经元加到 330 上得到 360。第二层的输出形状是(None, 11)，因为我们在这层有 11 个神经元来计算可能的标签数量。参数数量是 341。通过将前一层的 30 个神经元乘以这层的 11 个神经元得到 330，再将这层的 11 个神经元加到 330 上得到 341。

### 编译模型

编译：

```py
model.compile(optimizer='adam',
loss='sparse_categorical_crossentropy',
metrics=['accuracy'])
```

### 训练模型

```py
history = model.fit(train_wine_ds, epochs=10,
validation_data=test_wine_ds)
```

我们没有获得很好的性能，但我们只是想展示如何训练一个 CSV 数据集。

## 从 GitHub 获取鲍鱼数据

在下一节中，我们将处理鲍鱼数据集。我们将向您展示如何将其从 GitHub 作为替代方案加载。

我们已经找到了合适的 URL 并将其分配给一个变量：

```py
url = 'https://raw.githubusercontent.com/paperd/tensorflow/\
master/chapter4/data/abalone.data'
```

将数据集读入 pandas 数据框：

```py
# add column headings
cols = ['Sex', 'Length', 'Diameter', 'Height', 'Whole',
'Shucked', 'Viscera', 'Shell', 'Rings']
abd = pd.read_csv(url, names=cols)
```

验证数据：

```py
abd.head()
```

## 数据数据集

与*.data*扩展名的数据相比，与 CSV 数据之间的唯一区别是文件扩展名。但我们可以以相同的方式处理它。我们不是从 GitHub 加载它，而是可以从 UCI 存储库下载数据集，将其复制到 Google Drive，并从 Colab 中访问它。

## 鲍鱼数据集

鲍鱼数据集是类型 DATA。数据的一般网站是

[`archive.ics.uci.edu/ml/datasets/Abalone`](https://archive.ics.uci.edu/ml/datasets/Abalone)

要下载鲍鱼数据，请访问 URL，点击*数据文件夹*，然后点击*abalone.data*。数据集将自动下载到您的*下载*目录。由于我们使用的是 Colab 云服务，请将文件复制到您的*Google Drive*上的*Colab 笔记本*目录。最简单的方法是将文件从下载目录拖放到 Google Drive。

另一个有趣的文件是 abalone.names，它提供了数据集的详细描述。像数据文件一样，可以通过点击*abalone.names*从数据文件夹中访问。

### 数据集特征

该数据集用于根据物理测量预测鲍鱼壳的年龄。鲍鱼壳的年龄部分由切割圆锥体、染色并通过显微镜计数环数来确定。其他测量值补充了年龄预测。

特征变量包括

+   性别

+   长度

+   直径

+   高度

+   完整

+   去壳

+   内脏

+   壳

目标变量是

+   环数

目标变量环数可以取 1 到 29 之间的分数。这样的分数代表鲍鱼壳的环数。因此，*环数*是预测的值。关于这个数据集的一个有趣的观点是，我们可以将其用作连续值实验，也可以将其用作分类问题。

### 将 Google Drive 挂载到 Colab

如果我们要直接从 Google Drive 加载数据文件，我们必须将 Colab 挂载到 Google Drive：

```py
from google.colab import drive
drive.mount('/content/gdrive')
```

在执行代码片段后，点击 URL，选择一个 Google 账户，点击 *允许* 按钮，复制授权代码并将其粘贴到文本框 *输入您的授权代码:* 中，然后按键盘上的 *Enter* 键。

注意

确保您在 Google Drive 的 Colab 笔记本目录中有文件！

在 Colab 中建立路径：

```py
# establish path (be sure to copy file to Google Drive)
path = 'gdrive/My Drive/Colab Notebooks/'
abalone = path + 'abalone.data'
abalone
```

### 读取数据

由于数据集不包含列标题，我们需要在读取数据集之前定义它们：

```py
cols = ['Sex', 'Length', 'Diameter', 'Height', 'Whole',
'Shucked', 'Viscera', 'Shell', 'Rings']
ab_data = pd.read_csv(abalone, names=cols)
```

现在我们有了从 GitHub 直接加载的相同数据集。

### 探索数据

显示数据集开头的记录：

```py
ab_data.head(3)
```

显示数据集末尾的记录：

```py
ab_data.tail(3)
```

返回记录数：

```py
len(ab_data)
```

我们有 4,177 条记录。

显示数据集中使用的输出类别：

```py
# classes used
print ('classes:', br)
print (np.sort(ab_data['Rings'].unique()))
```

显示输出类别的数量：

```py
# number of classes
print ('number of classes:', len(ab_data['Rings'].unique()))
```

数据集中使用了 28 个类别。

显示类别分布：

```py
instance = ab_data['Rings'].value_counts()
instance.to_dict()
```

类别范围从 1 到 27，以及 29。每个类别代表鲍鱼壳的年龄（以年为单位）。分布非常不均匀。例如，我们有 689 个九岁的壳的实例，但只有一个一岁的壳的实例。通常，机器学习算法在处理不平衡数据时表现不佳，因为预测会偏向实例最多的类别。**不平衡数据**是分类问题中，训练集中类别分布不均的问题。然而，像欺诈检测这样的问题通常由机器学习处理，其中预期类别是不平衡的。

显示数据类型：

```py
ab_data.dtypes
```

除了性别之外，特征都是 float64 类型。目标是 int64 类型。

显示所有列的信息：

```py
ab_data.info(verbose=True)
```

没有缺失数据。

显示形状：

```py
ab_data.shape
```

我们有 4,177 个示例，每个示例有九个属性。

### 创建训练集和测试集

分割数据：

```py
train, test = train_test_split(ab_data)
print(len(train), 'train examples')
print(len(test), 'test examples')
```

我们有 3,132 个训练示例和 1,045 个测试示例。如果没有指定，默认测试大小为 25%。

### 创建特征和目标集

创建训练集和测试集副本以保留原始数据是个好主意。否则，pop 方法可能会造成永久性破坏。

创建目标：

```py
train_copy, test_copy = train.copy(), test.copy()
# create targets
train_target, test_target = train_copy.pop('Rings'),\
test_copy.pop('Rings')
```

验证目标：

```py
len(train_target), len(test_target)
```

验证训练特征数据：

```py
train_copy.head(3)
```

在处理数据时，验证内容是个好主意。

将特征数据转换为 Numpy：

```py
train_features, test_features = train_copy.values,\
test_copy.values
```

### 缩放特征

我们只能缩放连续值。由于性别特征不是连续的，因此不能缩放。所以我们将连续值切掉以进行缩放。然后我们重新创建包含性别特征和缩放连续值的训练和测试集。

显示一个样本以验证切片：

```py
train_features[0], test_features[0]
```

切片很复杂。因此，显示一个样本以确保切片按预期工作是个好主意。

创建两个训练集（一个包含性别，另一个包含连续值）：

```py
train_sex = [row[0] for row in train_features]
train_f = [row[1:] for row in train_features]
train_sex[0], train_f[0]
```

到目前为止，一切顺利！

创建两个测试集（一个包含性别，另一个包含连续值）：

```py
test_sex = [row[0] for row in test_features]
test_f = [row[1:] for row in test_features]
test_sex[0], test_f[0]
```

我们的切片与原始数据匹配。

缩放连续值：

```py
train_sc = scaler.fit_transform(train_f)
test_sc = scaler.fit_transform(test_f)
```

### 创建包含性别和缩放值的训练和测试集

现在我们已经缩放了连续值，我们需要将它们与性别特征重新组合，如列表 4-10 所示。

```py
train_ds = [np.append(train_sex[i], row)
for i, row in enumerate(train_sc)]
test_ds = [np.append(test_sex[i], row)
for i, row in enumerate(test_sc)]
train_ds[0], test_ds[0]
Listing 4-10
Recombine continuous features with the sex feature
```

重新组合训练和测试特征并显示。注意连续特征已缩放。

### 将 Numpy 特征集转换为 Pandas 数据框

为了正确构建一个 TensorFlow 可消费的非连续数据集，我们需要以 pandas 数据框形式提供特征数据：

```py
col = ['Sex', 'Length', 'Diameter', 'Height', 'Whole',
'Shucked', 'Viscera', 'Shell']
train_ab = pd.DataFrame(train_ds, columns=col)
test_ab = pd.DataFrame(test_ds, columns=col)
```

我们需要原始列名以添加到数据框中。

再次验证训练特征：

```py
train_ab.tail(3)
```

验证测试特征：

```py
test_ab.tail(3)
```

### 构建输入管道

准备训练和测试数据以供 TensorFlow 使用：

```py
train_ds = tf.data.Dataset.from_tensor_slices(
(dict(train_ab), train_target))
test_ds = tf.data.Dataset.from_tensor_slices(
(dict(test_ab), test_target))
```

注意，我们将训练和测试特征数据转换为 Python 字典。我们这样做是为了能够构建分类特征列。

打乱训练数据，分批，并预取训练和测试数据：

```py
BATCH_SIZE = 32
SHUFFLE_BUFFER_SIZE = 100
train_ads = train_ds.shuffle(
SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE).prefetch(1)
test_ads = test_ds.batch(BATCH_SIZE).prefetch(1)
train_ads, test_ads
```

注意，形状包括每个特征列名。

### 探索一个批次

由于我们将特征数据转换为字典，我们可以显示有关数据的有趣信息，如列表 4-11 所示。

```py
def see_format(data, num, feature, indx):
for feature_batch, label_batch in data.take(num):
print('Every feature:', list(feature_batch.keys()))
print('One example from a batch of ' + feature + ':',
feature_batch[feature][indx])
print('One example from a batch of targets:',
label_batch[indx])
print ('train sample:')
see_format(train_ads, 1, 'Height', 0)
print ()
print ('test sample:')
see_format(test_ads, 1, 'Sex', 0)
Listing 4-11
Display information about a sample batch
```

### 分类列

TensorFlow 的使用仅限于**数值数据**。因此，我们必须转换任何分类数据。在这种情况下，唯一的罪魁祸首是“性别”特征，因为它由“M”、“F”或“I”的字符串值表示。因此，鲍鱼壳的性别要么是男性、女性或婴儿。

由于我们不能直接将字符串输入到模型中，我们必须首先将它们映射到数值。分类词汇列特征提供了一种将字符串表示为*一维向量*的方法。这个过程称为**一维编码**，它是一种将分类变量转换为可解释格式的技术。

字符串按以下方式转换：

1.  ‘M’ => 1 0 0

1.  ‘F’ => 0 1 0

1.  ‘I’ => 0 0 1

列表 4-12 将性别特征进行了一维编码。

```py
from tensorflow import feature_column
sex_one_hot =\
feature_column.categorical_column_with_vocabulary_list(
'Sex', ['M', 'F', 'I'])
print (type(sex_one_hot))
feature_columns =\
[tf.feature_column.indicator_column(sex_one_hot)]
feature_layer = tf.keras.layers.DenseFeatures(feature_columns)
Listing 4-12
One-hot encode the sex feature
```

导入 *feature_column* 模块。对“性别”进行一维编码。创建特征列列表。我们创建一个列表，这样我们就可以在训练数据集中有多个分类特征。最后，为模型创建 *feature_layer*。

对于一个全面的示例，请参阅以下 URL：

[`www.tensorflow.org/tutorials/structured_data/feature_columns`](http://www.tensorflow.org/tutorials/structured_data/feature_columns)

### 构建模型

清除会话并生成一个种子：

```py
tf.keras.backend.clear_session()
np.random.seed(0)
tf.random.set_seed(0)
```

创建模型：

```py
model = tf.keras.Sequential([
feature_layer,
Dense(128, activation="relu"),
Dense(128, activation="relu"),
Dense(29, activation="sigmoid")
])
```

注意，第一层是*feature_layer*，它向模型提供了关于独热编码特征的信息。

### 编译模型

编译：

```py
model.compile(optimizer='adam',
loss='sparse_categorical_crossentropy',
metrics=['accuracy'])
```

### 训练模型

训练：

```py
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
model.fit(train_ads,
validation_data=test_ads,
epochs=1)
```

我们只训练了一个 epoch，因为我们知道性能会非常糟糕。我们是如何知道的呢？查看下一节以了解更多信息。

### 不平衡和不规则数据

由于以下两个原因，abalone 数据集*不是*一个好的用于预测的数据集：

1.  数据集是不平衡的。

1.  数据集是不规则的。

一个**不平衡数据集**是指类别不均衡的数据集。也就是说，类别没有相同数量的示例。这个数据集尤其不平衡，因为一些类别只有一个示例，而其他类别有数百个示例。使用不平衡数据集进行训练不会产生好的结果。所以我们不会学到很多东西。原因是预测偏向于实例更多的类别！

一个**不规则数据集**是指具有过多目标（或标签）类别但数据不足的数据集。我们应该始终检查数据集中每个标签的样本数（或示例）。样本不足的类别标签更难从中学习。

### 处理不平衡数据

我们可以通过多种方式处理不平衡数据。我们可以更改算法。一些算法可能比其他算法更有效。所以尝试多种算法。通过向少数类别或类别添加更多实例进行过采样。我们也可以通过从多数类别或类别中删除观察结果进行欠采样。最后，我们可以增加数据。

对于更详细的解释，请参阅以下 URL：

[处理不平衡数据的方法](https://towardsdatascience.com/methods-for-dealing-with-imbalanced-data-5b761be45a18)
