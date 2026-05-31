# 1. 构建 TensorFlow 输入管道

我们向您介绍 TensorFlow 输入管道，使用 tf.data API，它使您能够从简单的、可重复使用的组件构建复杂的输入管道。输入管道是任何深度学习实验的生命线，因为学习模型期望以 TensorFlow 可消费的形式接收数据。使用 tf.data.Dataset 抽象（tf.data API 的组件）创建高性能管道非常容易，因为它以简单格式表示数据集中的元素序列。

尽管数据清洗是输入管道线中的关键组成部分，但我们专注于构建清洗后的数据管道。我们希望您专注于构建 TensorFlow 可消费的管道，而不是数据清洗。数据科学家可能将超过 80%的总机器学习（ML）项目时间用于仅数据清洗。

我们从三个数据源构建输入管道。第一个数据源是从内存中加载的数据。第二个来源是外部文件。最后一个来源是云存储。

章节的笔记本位于以下 URL：

[`github.com/paperd/deep-learning-models`](https://github.com/paperd/tensorflow)

## 输入管道是什么？

**机器学习（ML）输入管道**是一种将生成机器学习模型所需的工作流程编码和自动化的方法。"ML 工作流程"是在机器学习项目期间实施的阶段。典型的阶段包括数据收集、数据预处理、构建数据集、模型训练和优化、评估以及部署到生产环境。因此，输入管道的目标是自动化与机器学习问题解决相关的工作流程（或阶段）。一旦输入管道自动化，它就可以在添加新数据到机器学习项目时重复使用。它甚至可以调整以用于类似的机器学习项目。

任何输入管道的第一步是数据预处理。在这一步中，原始数据被收集、清洗并合并到一个单一的有序框架中。"数据清洗"是识别和修复数据集中任何问题的过程。数据清洗的目标是修复任何不正确、不准确、不完整、格式错误、重复或不相关于机器学习项目目的的数据，以便清洗后的数据集是正确的、一致的、可靠的和可用的。

没有健壮和准确的数据作为训练模型的输入，项目更有可能失败。一旦数据得到适当准备，输入管道的焦点就转向编写和执行机器学习算法以获得机器学习模型。

## 为什么构建输入管道？

为了理解输入管道的重要性，查看数据科学团队在构建机器学习模型时通常会经历的典型阶段是有益的。从头开始实现机器学习模型往往非常问题导向。因此，数据科学团队专注于产生一个模型来解决单个商业问题。

### 手动工作流程

通常，团队从没有现有基础设施的**手动工作流程**开始。数据收集、数据清洗、模型训练和评估可能都写在单个笔记本中。笔记本在本地运行以生成模型，然后交给一个工程师，负责将其转换为应用程序编程接口（API）端点。**API 端点**是一种远程工具，利用机器学习在特定项目中解决特定问题。因此，工程师与训练好的模型一起工作，创建一个可以跨平台部署的 API 工具。

手动工作流程通常是临时的，当团队开始加快其迭代周期时，它开始崩溃，因为手动流程难以重复和记录。单笔记本格式的代码通常不适合协作。在手动工作流程场景中，**模型**就是产品。

### 自动化工作流程

一旦团队从偶尔更新单个模型的状态转变为在生产中拥有多个频繁更新的模型，采用管道方法变得至关重要。在这种情况下，我们不是构建和维护模型，而是开发和维护管道。因此，**管道**就是产品。

自动化管道由组件及其如何集成以生成和更新最重要的组件——模型——的蓝图组成。使用自动化工作流程，代码被拆分为更易于管理的组件，包括数据预处理、模型训练、模型评估和重新训练触发器。这些触发器被放置在那里，以便在模型需要重新训练时自动触发。

系统提供了在管道的整个上下文中以与在笔记本电脑上运行本地笔记本单元格相同的轻松和快速迭代方式执行、迭代和监控单个组件的能力。它还允许我们定义所需的输入和输出、库依赖项和监控指标。

将问题解决分解为可重复、预定义和可执行组件的能力迫使团队遵守联合（或联合）流程。反过来，这种联合流程在数据科学家和工程师之间创建了一种定义良好的语言，最终导致了一个自动化的设置，这是机器学习（ML）的持续集成（CI）等效物。CI 是一种自动化将多个贡献者的代码更改集成到单个软件项目中的实践，以便最终产品能够自动更新自身。

## 基本输入管道机制

`tf.data.Dataset` API 支持编写描述性和高效的输入管道，遵循一个常见的模式。首先，从输入数据创建一个源数据集。其次，应用数据转换来预处理数据。第三，遍历数据集并处理元素。迭代以流式方式进行，因此整个数据集不需要全部适合内存。

一旦创建了一个数据源，就可以通过在 tf.data.Dataset 对象上链式调用方法将其转换成一个新的数据集。数据集对象是一个 Python 可迭代对象，可以用 for 循环消费。

TensorFlow 数据集通常以两种不同的方式创建。我们可以从存储在内存中或一个或多个文件中的数据创建数据集。然而，如果需要，我们也可以基于一个或多个 tf.data.Dataset 对象创建一个基于数据转换的数据集。

要从内存中创建 TensorFlow 数据集，请使用 *from_tensors()* 或 *from_tensor_slices()* 方法。要从存储在文件中的数据创建 TensorFlow 数据集，请使用推荐的 TFRecord 格式，并使用 *TFRecordDataset()* 方法。

**from_tensors()** 方法将数据源中的输入张量组合起来，并返回一个包含单个元素的数据集。**from_tensor_slices()** 方法为输入张量的每一行创建一个单独元素的数据集。**输入张量**是一个表示数据源的 n 维向量或矩阵。我们专注于 *from_tensor_slices()* 方法，因为我们希望能够方便地检查和处理数据源中的每个元素。

## 高性能管道

tf.data API 通过在当前步骤完成之前提供训练的下一步数据，从而创建灵活且高效的输入管道。我们专注于构建高性能 TensorFlow 输入管道的三个最佳实践，即预取、缓存和洗牌。当我们在本章后面构建输入管道时，我们将通过示例讨论这些实践。

## Google Developers Codelabs

即使你已经完成了这本书中的示例，你也可能希望通过探索额外的教程来增加你的深度学习应用知识。*Google Developers Codelabs* 提供了强调动手编码实例的指导教程。大多数教程都会逐步引导你构建一个小型应用程序或向现有应用程序添加新功能。它们涵盖了广泛的主题，例如 Android Wear、Google Compute Engine、Project Tango 以及 iOS 上的 Google API。

要浏览 Codelabs 网站，请访问

[`codelabs.developers.google.com/`](https://codelabs.developers.google.com/)

## 在 Colab 中创建一个新的笔记本

在 Colab 环境中，创建一个新的笔记本非常容易。在浏览器中打开 Google Colab（如果尚未打开）。从弹出窗口中，点击 *New notebook*。如果已经在 Colab 环境中，请点击左上角的菜单栏下的 *Welcome to Colaboratory* 中的 *File*。从下拉菜单中选择 *New notebook*。现在就有一个代码单元格可以执行 Python 代码了！通过点击 *+ Code* 或 *+ Text* 按钮添加代码或文本单元格。要获取更多选项，请从主菜单中选择 *Insert*。

要了解 Colab 的介绍，请查阅

[`colab.research.google.com/`](https://colab.research.google.com/)

要创建你的第一段代码，请在代码单元格中添加以下内容：

```py
10 * 5
```

要执行代码，请点击左边的 *小箭头*。代码单元的输出显示了乘法的结果。

小贴士

我们建议从网站复制并粘贴代码。

## 导入 TensorFlow 库

在我们能够在 TensorFlow 中做任何事情之前，我们必须导入适当的 Python 库。将 TensorFlow 库别名为 **tf** 是一种常见的做法。因此，请在新的代码单元中执行导入操作：

```py
import tensorflow as tf
```

## GPU 硬件加速器

为了极大地加快处理速度，请使用 Google Colab 云服务提供的 GPU。Colab 提供了一个大约 12 GB RAM 的免费 Tesla K80 GPU（截至本文写作时）。在 Colab 笔记本中启用 GPU 非常简单：

1.  在右上角菜单中点击 *运行时*。

1.  从下拉菜单中选择 *更改运行时类型*。

1.  从 *硬件加速器* 下拉菜单中选择 *GPU*。

1.  点击 *保存*。

注意

必须在每个笔记本中启用 GPU。但只需启用一次。

验证 GPU 是否处于活动状态：

```py
tf.__version__, tf.test.gpu_device_name()
```

如果显示 ‘/device:GPU:0’，则 GPU 正在运行。如果显示 ‘’，则常规 CPU 正在运行。

小贴士

如果出现错误 **NAME ‘TF’ IS NOT DEFINED’，重新执行代码以导入 TensorFlow 库！由于某种原因，我们有时必须在 Colab 中重新执行 TensorFlow 库导入。我们不知道这是为什么。

Colab 是一个与 TensorFlow 一起工作的出色工具。然而，它确实有其局限性。Colab 应用动态资源分配。为了能够免费提供计算资源，Colab 会动态调整使用限制和硬件可用性。因此，Colab 中的可用资源会随时间变化，以适应需求波动。简而言之，这意味着 Colab 可能并不总是可用！一个解决方案是转为使用 Colab Pro，只需支付小额月费。截至本文写作时，费用为 $9.99/月。

小贴士

对于严肃的 TensorFlow 用户，我们建议转为使用 Colab Pro。它不是免费的，但价格相当低廉。根据我们的经验，它比免费版本更强大，并且更易于获得。

## 创建 TensorFlow 数据集

创建一个包含三个张量，每个张量有六个元素的数据集：

```py
data = [[8, 5, 7, 3, 9, 1],
[0, 3, 1, 8, 5, 7],
[9, 9, 9, 0, 0, 7]]
dataset = tf.data.Dataset.from_tensor_slices(data)
dataset
```

创建数据集。使用 from_tensor_slices() 方法将其转换为 tf.data.Dataset 对象。数据集的形状为 (6,), 这意味着每一行包含六个标量值。

小贴士

我们强烈建议在各自的代码单元中测试小块代码，以减少调试时间和精力。

## 消费数据集

遍历数据集以显示张量信息：

```py
for i, row in enumerate(dataset):
print ('row ' + str(i), ':', end=' ')
print (row.numpy())
```

由于 tf.data.Dataset 对象是通过 from_tensor_slices() 创建的，它是一个 Python 可迭代对象，可以用 for 循环来消费。在使用 TensorFlow 数据集时，使用 *numpy()* 方法显式地将每个张量转换为 NumPy 数组。

或者，我们可以使用 *take()* 方法遍历 TensorFlow 数据集：

```py
for i, e in enumerate(dataset.take(3)):
print ('row ' + str(i), ':', end=' ')
print (e.numpy())
```

我们在 take() 方法中添加 *3* 作为参数，以获取三个示例。

另一个选项是创建一个 Python 迭代器：

```py
i = 0
it = iter(dataset)
print ('row ' + str(i), ':', end=' ')
print (next(it).numpy())
i += 1
print ('row ' + str(i), ':', end=' ')
print (next(it).numpy())
i += 1
print ('row ' + str(i), ':', end=' ')
print (next(it).numpy())
```

初始化一个计数器变量。使用*iter()*方法创建一个迭代器。使用*next()*方法消费迭代器并显示结果。

## 数据集结构

tf.data.Dataset 的*element_spec*属性允许检查数据集。该属性返回一个与元素结构匹配的 tf.TypeSpec 对象的嵌套结构。嵌套结构可能是一个组件，一个组件的元组，或者一个组件的嵌套元组。

检查数据集：

```py
dataset.element_spec
```

显示形状和数据类型。

或者，我们也可以直接显示 tf.data.Dataset 对象：

```py
dataset
```

## 从内存中创建数据集

如果所有输入数据都适合内存，创建 TensorFlow 数据集的最简单方法是将它转换为 tf.Tensor 对象，使用 from_tensor_slices()方法。现在，我们将构建一个管道。我们首先加载一个干净的数据库。然后，我们继续缩放特征数据图像。缩放（或特征缩放）是一种用于归一化数据集独立变量或特征范围的方法。缩放很重要，因为机器学习模型如果组成每个图像的像素尺寸更小，则往往表现更好。我们使用代码和可视化来检查数据。接下来，我们配置管道以获得性能。最后，我们创建一个模型，训练模型并评估模型。

### 加载数据并检查：

要构建输入管道，我们需要一个数据集。由于重点是构建 TensorFlow 可消费的管道，我们使用清洗过的数据集。

在内存中加载训练和测试数据：

```py
train, test = tf.keras.datasets.fashion_mnist.load_data()
```

将 Fashion-MNIST 数据下载到训练集和测试集中。我们使用训练数据来训练模型。我们使用测试数据来评估模型。Fashion-MNIST 是 Zalondo 文章图像的数据集。它包含 60,000 个训练样本和 10,000 个测试样本。该数据集旨在作为原始 MNIST 数据集的直接替换，用于基准测试机器学习算法。

检查：

```py
type(train[0]), type(train[1])
```

训练集和测试集是元组，其中第一个元组元素包含特征图像，第二个包含相应的标签。这两个数据集都是 NumPy 数组。

将图像和标签加载到变量中：

```py
train_img, train_lbl = train
test_img, test_lbl = test
```

通过将图像和标签从各自的数据库中分离出来，我们可以更轻松地按需处理图像和标签。

验证形状：

```py
print ('train:', train_img.shape, train_lbl.shape)
print ('test:', test_img.shape, test_lbl.shape)
```

训练数据由 60,000 个 28 × 28 的特征图像和 60,000 个标签组成。测试数据由 10,000 个 28 × 28 的特征图像和 10,000 个标签组成。

### 缩放并创建 tf.data.Dataset

缩放数据以进行高效处理并创建训练集和测试集：

```py
train_image = train_img / 255.0
test_image = test_img / 255.0
train_ds = tf.data.Dataset.from_tensor_slices(
(train_image, train_lbl))
test_ds = tf.data.Dataset.from_tensor_slices(
(test_image, test_lbl))
```

使用*from_tensor_slices()*从 NumPy 数组中获取切片，形成 tf.data.Dataset 对象。特征图像像素值通常是介于 0 到 255 之间的整数。为了缩放，将特征图像除以 255 以获得介于 0 到 1 之间的像素值。

缩放图像是一个关键的前处理步骤，因为深度学习模型在较小的图像上训练得更快。此外，许多深度学习模型架构要求图像尺寸相同。但是原始图像的尺寸往往不同。

检查训练和测试张量：

```py
train_ds, test_ds
```

两个数据集都是 *TensorSliceDataset* 对象，这意味着它们是迭代器。一个 **迭代器** 是一个包含可计数示例的对象，可以使用 *next()* 方法遍历。

显示训练集中的第一个标签：

```py
next(train_ds.as_numpy_iterator())[1]
```

训练集中的每个示例都包含一个图像矩阵及其对应的标签。*next()* 方法返回一个元组，其中第一个图像矩阵和其标签分别位于元组的第 0 和第 1 个位置。

显示训练集中的十个标签：

```py
next(train_ds.batch(10).as_numpy_iterator())[1]
```

*batch()* 方法从一个数据集中取 *n* 个示例。

显示训练集中的所有 60,000 个标签：

```py
labels = next(train_ds.batch(60_000).as_numpy_iterator())[1]
labels, len(labels)
```

显示训练集中的第一张图像：

```py
next(train_ds.as_numpy_iterator())[0]
```

验证第一张图像是一个 28 × 28 的矩阵：

```py
arrays = len(next(train_ds.as_numpy_iterator())[0])
pixels = len(next(train_ds.as_numpy_iterator())[0][0])
arrays, pixels
```

要在 Python 中找到矩阵的维度，高度（或行）是 *len(matrix)*，宽度（或列）是 *len(matrix[0])*。

### 验证缩放

显示训练集中的预缩放张量：

```py
train_img[0][3]
```

显示缩放后的相同张量：

```py
train_image[0][3]
```

哇！像素被缩放到 0 到 1 之间。

### 检查张量形状

检查形状：

```py
for img, lbl in train_ds.take(5):
print ('image shape:', img.shape, end=' ')
print ('label:', lbl.numpy())
```

Fashion-MNIST 图像大小相同。所以我们不需要调整它们的大小！

### 检查张量

检查训练和测试张量：

```py
train_ds, test_ds
```

一切正常。

### 保留形状

为模型使用分配一个变量给特征图像形状：

```py
for img, _ in train_ds.take(1):
img.shape
img_shape = img.shape
img_shape
```

### 可视化

可视化训练集中的元素：

```py
import matplotlib.pyplot as plt
for feature, label in train_ds.take(1):
plt.imshow(feature, cmap='ocean')
plt.axis('off')
plt.grid(b=None)
```

虽然 Fashion-MNIST 图像是灰度的，但我们可以使用 matplotlib 库中预定义的颜色图用颜色使它们生动起来。**颜色图** 是一个颜色数组，用于将像素数据映射到实际颜色值。

查阅以下网址以获取有关 matplotlib 颜色图的详细信息：

[matplotlib 颜色图教程](https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html)

### 定义类别标签

从我们与 Fashion-MNIST 合作的经验中，我们知道相应的标签：

```py
class_labels = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress',
'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag',
'Ankle boot']
```

### 将数值标签转换为类别标签

在我们刚刚加载的 tf.data.Dataset 中，标签是数值的，但我们可以使用我们刚刚创建的 *class_labels* 列显示相应的类别名称：

```py
for _, label in train_ds.take(1):
print ('numerical label:', label.numpy())
print ('string label:', class_labels[label.numpy()])
```

取一个示例并显示标签的数值和字符串值。

### 从数据集中创建示例的绘图

从训练集中取一些图像和标签：

```py
num = 30
images, labels = [], []
for feature, label in train_ds.take(num):
images.append(tf.squeeze(feature.numpy()))
labels.append(label.numpy())
```

创建一个函数来显示如图 1-1 所示的示例网格。

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
plt.title(cl[target[index]], fontsize=12)
plt.subplots_adjust(wspace=0.2, hspace=0.5)
Listing 1-1
Function to Display a Grid of Examples
```

调用函数：

```py
rows, cols = 5, 6
display_grid(images, labels, rows, cols, class_labels)
```

总是检查数据集是否如我们所期望的那样是一个好主意。

### 构建可消费的输入管道

我们称之为 *可消费输入管道*，因为实际的管道是在数据实际获取时开始的。我们使用这个术语来强调将训练和测试数据集转换为高效张量以供 TensorFlow 模型消费的重要性。我们看到一些例子将这部分称为构建输入管道，但输入管道涵盖了从原始数据到通用模型的整个工作流程。在后面的章节中，我们将省略“可消费”这个词。

#### 配置数据集以优化性能

使用缓冲预取和缓存来提高 I/O 性能。打乱数据以提高模型性能。

**预取**是 tf.data API 中的一个功能，它在训练过程中重叠数据预处理和模型执行，从而减少了模型的总体训练时间。要执行此操作，将*tf.Dataset.prefetch*转换添加到输入管道中。

将*tf.data.Dataset.cache*转换添加到管道中，以在第一次周期从磁盘加载图像后将其保留在内存中，这确保了数据集在训练期间不会成为瓶颈。因此，缓存可以节省在每个周期执行的操作（例如，文件打开、数据读取）。

**打乱**数据旨在减少方差（确保模型保持泛化性）和减少过拟合。一个明显的打乱案例是当数据按类别（或目标）排序时。我们打乱以确保训练、测试和验证集代表数据的整体分布。要执行此操作，将*tf.Dataset.shuffle*转换添加到管道中。

训练始终在训练数据和标签的批次上执行。这样做有助于算法收敛。**批次**是在一次迭代中使用整个数据集来计算梯度。**小批量**是在一次迭代中使用数据集的子集来计算梯度。要执行此操作，将*tf.Dataset.batch*转换添加到管道中。

*批维度*通常是数据张量的第一个维度。因此，形状为[100, 192, 192, 3]的张量包含 100 张 192×192 像素的图像，每个像素有三个值（RGB）在每个批次中。**RGB 颜色模型**是一种加色模型，其中红、绿、蓝光以各种方式相加以重现广泛的颜色。

构建可消费的输入管道：

```py
BATCH_SIZE = 128
SHUFFLE_SIZE = 5000
train_f = train_ds.shuffle(SHUFFLE_SIZE).batch(BATCH_SIZE)
train_fm = train_f.cache().prefetch(1)
test_f = test_ds.batch(BATCH_SIZE)
test_fm = test_f.cache().prefetch(1)
```

打乱训练数据。打乱随机化训练数据，确保在每次训练周期中每个数据元素与其他数据元素独立。学习模型在接触到独立采样的数据时往往表现最佳。

批量、缓存和预取训练和测试数据。添加*cache()*转换可以提高性能，因为数据只在第一次周期中读取和写入一次，而不是在每个周期中。添加*prefetch(1)*转换是一个好主意，因为它增加了批处理过程的效率。也就是说，当我们的训练算法在一个批次上工作时，TensorFlow 正在并行处理数据集，以便准备好下一个批次。因此，这种转换可以显著提高训练性能。

与其他 tf.data.Dataset 方法一样，预取操作在输入数据集的元素上执行。它没有示例与批次的概念。因此，使用 examples.prefetch(2)预取两个示例，使用 examples.batch(20).prefetch(2)预取每个批次包含 20 个示例的两个批次。

测试（或验证）集用于展示训练好的模型在训练过程中未见过的示例上的表现如何。因此，它是否被洗牌是无关紧要的。

我们根据试错实验设置批量大小和洗牌大小。你可以通过调整批量和洗牌大小来实验。

检查张量：

```py
train_fm, test_fm
```

### 构建模型

导入必要的库：

```py
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.losses import SparseCategoricalCrossentropy
import numpy as np
```

清除之前的模型并为结果的重复性生成一个种子：

```py
tf.keras.backend.clear_session()
np.random.seed(0)
tf.random.set_seed(0)
```

我们使用零作为种子值，但任何数字都可以替换。

小贴士

清除之前的模型并不会将当前模型重置到其初始状态。要重置模型，只需重新构建模型的输入管道！

创建模型：

```py
model = Sequential([
Flatten(input_shape=img_shape),
Dense(128, activation='relu'),
Dropout(0.4),
Dense(10, activation=None)
])
```

神经网络的基本构建块是层。层从输入的数据中提取表示。希望这些表示对于当前问题是有意义的。大多数深度学习都是由简单的层连接而成的。大多数层，如*密集层*，在训练过程中学习参数。

这个网络中的第一层是一个*Flatten*层，它将图像的格式从二维数组（28x28 像素）转换为了一维数组（28x28=784 像素）。将这个层想象成将图像中的像素行展开并排列起来。这个层没有需要学习的参数，因为它只重新格式化数据。

在像素被展平后，网络由一系列两个密集层组成。密集层是完全连接的神经网络层，这意味着一个层中的所有神经元都与下一层的所有神经元相连。

第一个密集层有 128 个节点（或神经元）。我们在第一个密集层之后添加一个*Dropout*层来减少过拟合。第二个（也是最后一个）层返回一个长度为 10 的 logits 数组。**Logits**是在应用激活函数之前神经层输出的结果。每个节点包含一个分数，表示当前图像属于十个类别中的哪一个。

**Dropout**是一种正则化方法，它近似于并行训练具有不同架构的大量神经网络。在训练过程中，一些层输出会被随机忽略或“丢弃”，这使得层看起来像并且被当作具有不同节点数量和与前一层连接性的层来处理。实际上，在训练过程中对层的每次更新都是使用配置层的一个不同“视图”来执行的。

检查模型：

```py
model.summary()
```

### 编译和训练模型

使用*SparseCategoricalCrossentropy*损失函数编译模型。当类别互斥时，稀疏分类交叉熵表现良好。也就是说，每个样本恰好属于一个类别。使用稀疏分类交叉熵的优点是它节省了内存和计算时间，因为它使用单个整数来表示一个类别而不是整个向量。

*from_logits=True* 属性通知损失函数，模型生成的输出值未经归一化。也就是说，softmax 函数*尚未*应用于它们以产生概率分布。

编译：

```py
model.compile(optimizer='adam',
loss=SparseCategoricalCrossentropy(from_logits=True),
metrics=['accuracy'])
```

训练模型：

```py
epochs = 10
history = model.fit(train_fm, epochs=epochs,
verbose=1, validation_data=test_fm)
```

模型正在使用十个周期进行训练。**周期**的数量是学习算法遍历整个训练数据集的次数。训练数据和测试数据的损失和准确度被显示出来。训练损失和准确度基于模型在训练期间学到的内容。测试损失和准确度基于模型尚未学习的新数据。因此，测试准确度越接近训练准确度，模型的泛化能力越强。当然，我们希望测试准确度高，测试损失低。

## 将 TensorFlow 数据集加载为 NumPy

上一节基于 Keras 数据集对 Fashion-MNIST 数据进行了建模。然而，我们可以将数据作为 TensorFlow 数据集（TFDS）加载，并将其转换为 NumPy 数组以进行非常简单的处理。我们将在后面的章节中详细介绍 TFDS。

对于这个实验，我们加载 MNIST 数据集而不是 Fashion-MNIST。我们这样做是因为我们在后面的章节中多次使用 Fashion-MNIST。所以我们只想让你接触另一个数据集进行练习。一旦数据被加载并转换为 NumPy，输入管道阶段与上一节相同。

将训练集作为一个批次创建为 NumPy 数组：

```py
import tensorflow_datasets as tfds
image_train, label_train = tfds.as_numpy(
tfds.load(
'mnist', split='train',
batch_size=-1, as_supervised=True,
try_gcs=True))
type(image_train), image_train.shape
```

使用 *batch_size=-1*，整个数据集被加载为一个单独的批次。*tfds.load()* 函数加载数据集。*tfds.as_numpy()* 函数将数据集转换为 NumPy 数组。

训练集包含 60,000 个 28 × 28 的图像。*1* 维度表示数据是灰度的。**灰度**图像是只包含灰色阴影的图像。也就是说，图像只包含亮度（或亮度）信息，没有颜色信息。

创建相应的测试集：

```py
image_test, label_test = tfds.as_numpy(
tfds.load(
'mnist', split='test',
batch_size=-1, as_supervised=True,
try_gcs=True))
type(image_test), image_test.shape
```

### 检查形状和像素强度

获取训练形状：

```py
image_train.shape, label_train.shape
```

获取测试形状：

```py
image_test.shape, label_test.shape
```

创建一个函数来查找像素强度值如清单 1-2 所示的第一个像素向量。

```py
def find_intensity(m):
for i, vector in enumerate(m):
for j, pixels in enumerate(vector):
if pixels > 0:
print (vector)
return i, j
Listing 1-2
Function to Find Pixel Intensity
```

调用函数：

```py
M = image_train[0]
indx = find_intensity(M)
image_train[0][indx[0]][indx[1]]
```

非零值是像素强度。

显示第一个强度大于零的像素：

```py
image_train[0][indx[0]][indx[1]]
```

### 缩放

由于 NumPy 数组值是浮点数，因此将它们除以 255 以缩放图像像素：

```py
train_sc = image_train / 255.0
test_sc = image_test / 255.0
```

验证缩放是否成功：

```py
image_train[0][indx[0]][indx[1]], train_sc[0][indx[0]][indx[1]]
```

### 准备数据以供 TensorFlow 使用

将 NumPy 数组切割成 TensorFlow 数据集：

```py
train_mnds = tf.data.Dataset.from_tensor_slices(
(image_train, label_train))
test_mnds = tf.data.Dataset.from_tensor_slices(
(image_test, label_test))
```

检查：

```py
train_mnds, test_mnds
```

### 构建可消费的输入管道

初始化参数，打乱训练数据，并批量和预取训练和测试数据：

```py
BATCH_SIZE = 100
SHUFFLE_SIZE = 10000
train_mnist = train_mnds.shuffle(SHUFFLE_SIZE).\
batch(BATCH_SIZE).prefetch(1)
test_mnist = train_mnds.batch(BATCH_SIZE).prefetch(1)
```

检查张量：

```py
train_mnist, test_mnist
```

### 构建模型

之前，我们导入了所需的库。由于它们已经在内存中，我们不需要再次导入它们（假设我们使用的是同一个笔记本）。

获取张量形状：

```py
np_shape = image_test.shape[1:]
np_shape
```

清除之前的模型并为结果的重复性生成一个种子：

```py
np.random.seed(0)
tf.random.set_seed(0)
tf.keras.backend.clear_session()
```

创建模型：

```py
model = Sequential([
Flatten(input_shape=np_shape),
Dense(512, activation='relu'),
Dense(10, activation='softmax')
])
```

### 编译并训练模型

使用稀疏分类交叉熵进行编译。注意，我们没有设置*from_logits=True*，因为我们使用模型输出层的*softmax*激活函数来生成从 logits 的概率分布。**softmax**激活函数作用于向量，增加最大分量与其他分量的差异，并将向量归一化，使其总和为 1，以便它可以被解释为概率向量。它在分类器的最后一步中使用：

```py
model.compile(optimizer='adam',
loss='sparse_categorical_crossentropy',
metrics=['accuracy'])
```

训练模型：

```py
epochs = 3
history = model.fit(train_mnist, epochs=epochs, verbose=1,
validation_data=test_mnist)
```

我们只训练了三个 epoch，因为 MNIST 很容易训练。

## 从文件创建数据集

使用 Keras 实用工具从文件创建 TensorFlow 数据集。我们使用 Keras 实用工具，因为它极大地简化了花朵数据集的处理。*flowers dataset*是公开的，包含数千张花朵照片，分布在五个类别中。就像 Fashion-MNIST 和 MNIST 示例一样，我们按照类似的流程阶段构建输入管道。

### 下载并检查数据集

导入用于可视化的库：

```py
import PIL.Image
```

数据集包含五个子目录中的数千张花朵照片，每个类别一张花朵照片。目录结构如下：

```py
flowers_photos/
daisy/
dandelion/
roses/
sunflowers/
tulips/
```

使用 tf.keras.utils.get_file 实用工具下载数据：

```py
import pathlib
url1 = 'https://storage.googleapis.com/download.tensorflow.org/'
url2 = 'example_images/flower_photos.tgz'
dataset_url = url1 + url2
data_dir = tf.keras.utils.get_file(origin=dataset_url,
fname='flower_photos',
untar=True)
data_dir = pathlib.Path(data_dir)
```

*tf.keras.utils.get_file*实用工具如果缓存中没有，会从 URL 下载文件。*pathlib.Path*函数提供了文件的具体路径。

计算在 data_dir 中下载并可用的花朵照片数量：

```py
image_count = len(list(data_dir.glob('*/*.jpg')))
print (image_count)
```

有 3670 个花朵图像文件。

*data_dir*路径指向包含不同类型花朵的目录。让我们看看这些目录：

```py
dirs = [item.name for item in data_dir.glob('*')\
if item.name != 'LICENSE.txt']
dirs
```

每个目录都包含该类型花朵的图像。

访问一些文件：

```py
files = tf.data.Dataset.list_files(str(data_dir/'*/*'))
fn = []
for f in files.take(4):
print(f.numpy()), fn.append(str(f.numpy()))
```

显示每个文件的标签：

```py
from pathlib import Path
label = []
for i in range(4):
parts = Path(fn[i]).parts
label.append(parts[5])
print (parts[5])
```

每个目录都包含该类型花朵的图像。以下是*daisy*目录中的第一朵花：

```py
daisy = list(data_dir.glob('daisy/*'))
parts = Path(daisy[0]).parts
print (parts[5])
PIL.Image.open(str(daisy[0]))
```

显示向日葵图像的数量：

```py
len(daisy)
```

让我们显示*roses*目录中的几幅图像。创建一个列表来保存玫瑰：

```py
roses = list(data_dir.glob('roses/*'))
```

从一些文件中获取标签：

```py
label = []
for i in range(4):
tup = Path(str(roses[i])).parts
label.append(tup[5])
```

显示如列表 1-3 中所示的一些玫瑰：

```py
rows, cols = 2, 2
plt.figure(figsize=(10, 10))
for i in range(rows*cols):
plt.subplot(rows, cols, i + 1)
pix = np.array(PIL.Image.open(str(roses[i])))
plt.imshow(pix)
plt.title(label[i])
plt.axis('off')
Listing 1-3
Plot Rose Images
```

注意到图像大小并不相同！

### 使用 tf.keras.preprocessing 实用工具解析数据

*tf.keras.preprocessing.image_dataset_from_directory*实用工具为从磁盘加载和解析图像提供了极大的便利！我们在“*创建训练集和测试集*”子节中展示了该实用工具的便利性。

#### 设置参数

设置批量大小、图像高度和图像宽度：

```py
BATCH_SIZE = 32
img_height = 180
img_width = 180
```

我们最初将批量大小设置为 32，因为它对于许多我们工作的数据集来说通常是一个很好的大小。我们将图像高度和宽度设置为 180，因为我们得到了良好的结果，模型训练得非常快。请随意尝试这些参数。

我们的检查显示图像大小不同。由于 TensorFlow 模型期望图像大小相同，我们必须调整它们的大小。

#### 创建训练集和测试集

The *tf.keras.preprocessing.image_dataset_from_directory* 工具从目录中的图像文件生成一个 tf.data.Dataset。这个工具非常有用，因为它允许我们方便地分割、设置种子、调整大小和批处理数据。让我们将数据分割成 81%的训练集和 19%的测试集。我们根据多次实验设置了这个分割。当然，你可以根据自己的实验调整大小。*validation_split* 和 *subset* 参数的组合决定了训练和测试的分割。

为训练数据留出 81%：

```py
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
data_dir,
validation_split=0.19,
subset='training',
seed=0,
image_size=(img_height, img_width),
batch_size=BATCH_SIZE)
```

为测试数据留出 19%：

```py
test_ds = tf.keras.preprocessing.image_dataset_from_directory(
data_dir,
validation_split=0.19,
subset='validation',
seed=0,
image_size=(img_height, img_width),
batch_size=BATCH_SIZE)
```

#### 检查张量

检查：

```py
train_ds, test_ds
```

从训练集中取第一个批次并保留形状：

```py
for img, lbl in train_ds.take(1):
print (img.shape, lbl.shape)
flower_shape, just_img = img.shape[1:],\
img.shape[1:3]
```

我们取第一批数据来帮助我们检查数据集中的一个批次。我们保留批次的形状和批次大小以供模型使用。批次大小为 32，图像被调整大小为 180 × 180 × 3。*3*这个值表示图像有三个通道，这意味着它们是 RGB（彩色）。标签的形状为(32,)，对应于 32 个标量图像。

#### 获取类别名称

我们已经从目录名称中识别了类别。但现在我们可以使用 *class_names* 方法访问它们：

```py
class_names = train_ds.class_names
class_names
```

#### 显示示例

从训练集中取一个批次并绘制一些图像，如列表 1-4 所示。

```py
plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
for i in range(9):
ax = plt.subplot(3, 3, i + 1)
plt.imshow(images[i].numpy().astype('uint8'))
plt.title(class_names[labels[i]])
plt.axis('off')
Listing 1-4
Plot Flower Images from the First Training Batch
```

#### 数据缩放

如本章前面所述，一个像素由 256 个值表示。因此 RGB 通道值在[0, 255]范围内。由于神经网络在小值上表现更好，数据通常被缩放到[0, 1]范围内。

创建一个缩放图像的函数：

```py
def format_image(image, label):
image = tf.image.resize(image, just_img) / 255.0
return image, label
```

当我们配置输入管道时使用该函数。

#### 配置数据集以优化性能

使用缓冲预取从磁盘获取数据以减轻 I/O 问题。缓存数据以在从磁盘加载后保持图像在内存中。缓存可以节省在每个 epoch 执行文件打开和数据读取等操作。

#### 构建输入管道

缩放、打乱训练集，并缓存和预取训练和测试集：

```py
SHUFFLE_SIZE = 100
train_fds = train_ds.map(format_image).\
shuffle(SHUFFLE_SIZE).cache().prefetch(1)
test_fds = test_ds.map(format_image).\
cache().prefetch(1)
```

注意

由于训练数据和测试数据已经被工具批处理，因此在构建输入管道时*不要*进行批处理！

#### 构建模型

由于我们处理的是大尺寸彩色图像，因此我们需要构建一个卷积神经网络（CNN）模型以获得可观的性能，因为花卉图像是彩色的，并且具有更高的像素计数。

我们需要额外的库来构建 CNN：

```py
from tensorflow.keras.layers import Conv2D, MaxPooling2D
```

获取用于模型的类别数量：

```py
num_classes = len(class_names)
num_classes
```

清除任何之前的模型并生成一个随机种子：

```py
tf.keras.backend.clear_session()
np.random.seed(0)
tf.random.set_seed(0)
```

创建一个多层 CNN，如列表 1-5 所示。

```py
flower_model = tf.keras.Sequential([
Conv2D(32, 3, activation='relu',
input_shape=flower_shape),
MaxPooling2D(),
Conv2D(32, 3, activation='relu'),
MaxPooling2D(),
Conv2D(32, 3, activation='relu'),
MaxPooling2D(),
Flatten(),
Dense(128, activation='relu'),
Dense(num_classes, activation='softmax')
])
Listing 1-5
A Multilayer CNN
```

第一层缩放数据。第二层包含 32 个神经元，具有 3 × 3 的卷积核（或过滤器）。激活为*relu*。第三层使用最大池化来通过仅保留最大值来减少层的空间大小。因此，池化层在不丢失重要特征或模式的情况下减少了图像的维度。接下来的四层重复了第二和第三层的相同模式。Flatten 层将池化数据转换为单列，因为 Dense 层期望数据以这种形式。最终的 Dense 层使分类和预测成为可能。

#### 编译和训练模型

使用 SparseCategoricalCrossentropy()进行编译：

```py
flower_model.compile(
optimizer='adam',
loss=tf.losses.SparseCategoricalCrossentropy(),
metrics=['accuracy'])
```

由于*softmax*应用于输出，我们**不**设置*from_logits=True*。

训练模型：

```py
history = flower_model.fit(
train_fds,
validation_data=test_fds,
epochs=5)
```

模型过拟合，因为验证准确率低于训练准确率。但我们没有尝试调整模型。在下一章中，我们将探讨一种强大的技术来减轻过拟合。

### 从 Google Cloud Storage 获取 Flowers

我们展示了使用内存和文件中的数据进行的输入管道化。我们也可以从云存储中管道化数据。Flowers 数据托管在 Google Cloud Storage (GCS)上的公共存储桶中。因此，我们可以从 GCS 中获取 flower 文件。我们可以将花朵作为 JPEG 文件或 TFRecord 文件读取。对于数据建模，我们使用 TFRecord 文件。为了获得最佳性能，我们一次读取多个 TFRecord 文件。**TFRecord 格式**是一种简单的格式，用于存储一系列二进制记录。TFRecord 文件包含一系列记录，这些记录只能顺序读取。

#### 将 Flowers 作为 JPEG 文件读取并执行简单处理

根据 GCS 模式读取 JPEG 文件：

```py
GCS_PATTERN = 'gs://flowers-public/*/*.jpg'
filenames = tf.io.gfile.glob(GCS_PATTERN)
```

GCS_PATTERN 是一个*glob pattern*，支持“*”和“？”通配符。**Globs**（也称为 glob patterns）是模式，可以将通配符模式扩展为与给定模式匹配的路径名列表。

获取 JPEG 图像的数量：

```py
num_images = len(filenames)
print ('Pattern matches {} images.'.format(num_images))
```

从 GCS_PATTERN 创建文件名数据集并查看其内容：

```py
filenames_ds = tf.data.Dataset.list_files(GCS_PATTERN)
for filename in filenames_ds.take(5):
print (filename.numpy().decode('utf-8'))
```

我们需要以(image, label)元组的形式提供数据，以便独立处理图像和标签。因此，创建一个函数以返回包含(image, label)元组的数据集，如列表 1-6 所示。

```py
def decode_jpeg_and_label(filename):
bits = tf.io.read_file(filename)
image = tf.image.decode_jpeg(bits)
label = tf.strings.split(
tf.expand_dims(filename, axis=-1), sep='/')
label = label.values[-2]
return image, label
Listing 1-6
Function That Returns a Dataset of (image, label) Tuples
```

将函数映射到每个文件名以创建一个包含(image, label)元组的数据集：

```py
ds = filenames_ds.map(decode_jpeg_and_label)
```

查看：

```py
for image, label in ds.take(5):
print (image.numpy().shape,
label.numpy().decode('utf-8'))
```

显示图像：

```py
for img, lbl  in ds.take(1):
plt.axis('off')
plt.title(lbl.numpy().decode('utf-8'))
fig = plt.imshow(img)
```

虽然我们不使用此数据集进行训练，但让我们看看如何将文本标签转换为编码标签，如列表 1-7 所示。

```py
for img, lbl  in ds.take(1):
label = lbl.numpy().decode('utf-8')
matches = tf.stack([tf.equal(label, s)\
for s in class_names], axis=-1)
one_hot = tf.cast(matches, tf.float32)
print (matches.numpy(), one_hot.numpy())
new_label = tf.math.argmax(one_hot)
new_label.numpy()
Listing 1-7
Convert Text Labels to Encoded Labels
```

取一个标签。将其与类名列表进行比较，以找到其在列表中的位置。创建一个 one-hot 向量。将 one-hot 向量转换为标签张量。我们不使用此数据集进行训练，因为这不是从 GCS 建模复杂数据的方式。但它是一种简单的方法来加载和检查数据。

### 读取和处理 Flowers 作为 TFRecord 文件

从 GCS 建模复杂数据最佳方式是作为 TFRecord 文件。TFRecord 文件将数据存储为一系列二进制字符串。二进制字符串非常高效。

#### 读取 TFRecord 文件

根据 GCS 模式读取 TFRecord 文件：

```py
piece1 = 'gs://flowers-public/'
piece2 = 'tfrecords-jpeg-192x192-2/*.tfrec'
TFR_GCS_PATTERN = piece1 + piece2
tfr_filenames = tf.io.gfile.glob(TFR_GCS_PATTERN)
```

获取桶的数量：

```py
num_images = len(tfr_filenames)
print ('Pattern matches {} image buckets.'.format(num_images))
```

我们抓取了 16 个桶。由于有 3670 个花朵文件，15 个桶包含 230 张图片（15 × 230 = 3,450），最后一个桶包含 220 张图片。将 3,450 加上 220 得到 3,670。

显示一个文件：

```py
filenames_tfrds = tf.data.Dataset.list_files(TFR_GCS_PATTERN)
for filename in filenames_tfrds.take(1):
print (filename.numpy())
```

#### 设置训练参数

设置图像调整大小、管道化和训练轮数参数：

```py
IMAGE_SIZE = [192, 192]
AUTO = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 64
SHUFFLE_SIZE = 100
EPOCHS = 5
```

使用 *AUTOTUNE* 来提示 tf.data 运行时，它会在运行时动态调整值。

注意

AUTOTUNE 是实验性的，这意味着该操作可能在将来发生变化。

设置验证分割和类别标签：

```py
VALIDATION_SPLIT = 0.19
CLASSES = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']
```

创建数据分割、验证步骤和每轮步骤，如列表 1-8 所示。

```py
split = int(len(tfr_filenames) * VALIDATION_SPLIT)
training_filenames = tfr_filenames[split:]
validation_filenames = tfr_filenames[:split]
print ('Splitting dataset into {} training files and {}'
' validation files'.format(
len(tfr_filenames), len(training_filenames),
len(validation_filenames)), end = ' ')
print ('with a batch size of {}.'.format(BATCH_SIZE))
validation_steps = int(3670 // len(tfr_filenames) *\
len(validation_filenames)) // BATCH_SIZE
steps_per_epoch = int(3670 // len(tfr_filenames) *\
len(training_filenames)) // BATCH_SIZE
print ('There are {} batches per training epoch and {} '\
'batches per validation run.'\
.format(BATCH_SIZE, steps_per_epoch, validation_steps))
Listing 1-8
Create Training Splits and Steps
```

#### 创建加载和处理 TFRecord 文件的函数

创建一个函数来解析 TFRecord 文件，如列表 1-9 所示。

```py
def read_tfrecord(example):
features = {
'image': tf.io.FixedLenFeature([], tf.string),
'class': tf.io.FixedLenFeature([], tf.int64)
}
example = tf.io.parse_single_example(example, features)
image = tf.image.decode_jpeg(example['image'], channels=3)
image = tf.cast(image, tf.float32) / 255.0
image = tf.reshape(image, [*IMAGE_SIZE, 3])
class_label = example['class']
return image, class_label
Listing 1-9
Function to Parse a TFRecord
```

该函数接受一个 TFRecord 文件。一个字典包含 TFRecords 中常见的数据类型。tf.string 数据类型将图像转换为字节字符串（字节列表）。tf.int64 将类别标签转换为 64 位整数标量值。TFRecord 文件被解析为 (image, label) 元组。图像元素，一个 JPEG 编码的图像，被解码为一个 uint8 图像张量。图像张量被缩放到 [0, 1] 范围以加快训练速度。然后将其重塑为模型消费的标准大小。类别标签元素被转换为标量。

创建一个函数来将 TFRecord 文件加载为 tf.data.Dataset，如列表 1-10 所示。

```py
def load_dataset(filenames):
option_no_order = tf.data.Options()
option_no_order.experimental_deterministic = False
dataset = tf.data.TFRecordDataset(
filenames, num_parallel_reads=AUTO)
dataset = dataset.with_options(option_no_order)
dataset = dataset.map(read_tfrecord, num_parallel_calls=AUTO)
return dataset
Listing 1-10
Function to Create a tf.data.Dataset from TFRecord Files
```

该函数接受 TFRecord 文件。为了最佳性能，代码包括从多个 TFRecord 文件同时读取的功能。选项设置允许顺序改变优化。因此，*n* 个文件并行读取，数据顺序被忽略，以读取速度为优先。

创建一个函数，从 TFRecord 文件构建输入管道，如列表 1-11 所示。

```py
def get_batched_dataset(filenames, train=False):
dataset = load_dataset(filenames)
dataset = dataset.cache()
if train:
dataset = dataset.repeat()
dataset = dataset.shuffle(SHUFFLE_SIZE)
dataset = dataset.batch(BATCH_SIZE)
dataset = dataset.prefetch(AUTO)
return dataset
Listing 1-11
Function to Build an Input Pipeline from TFRecord Files
```

该函数接受 TFRecord 文件并调用 *load_dataset* 函数。函数继续通过缓存、重复、打乱、批处理和预取数据集来构建输入管道。重复和打乱仅映射到训练数据，以遵循 Keras 数据集的最佳实践。

#### 创建训练和测试集

实例化数据集：

```py
training_dataset = get_batched_dataset(
training_filenames, train=True)
validation_dataset = get_batched_dataset(
validation_filenames, train=False)
training_dataset, validation_dataset
```

显示一张图片并保留模型形状：

```py
for img, lbl in training_dataset.take(1):
plt.axis('off')
plt.title(CLASSES[lbl[0].numpy()])
fig = plt.imshow(img[0])
tfr_flower_shape = img.shape[1:]
```

#### 模型数据

清除和设置种子：

```py
tf.keras.backend.clear_session()
np.random.seed(0)
tf.random.set_seed(0)
```

按照列表 1-12 创建模型。

```py
tfr_model = Sequential([
Conv2D(32, (3, 3), activation = 'relu',
input_shape=tfr_flower_shape),
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
Listing 1-12
Create the Model
```

检查：

```py
tfr_model.summary()
```

编译：

```py
loss = tf.keras.losses.SparseCategoricalCrossentropy(
from_logits=True)
tfr_model.compile(optimizer='adam',
loss=loss,
metrics=['accuracy'])
```

训练：

```py
history = tfr_model.fit(training_dataset, epochs=EPOCHS,
verbose=1, steps_per_epoch=steps_per_epoch,
validation_steps=validation_steps,
validation_data=validation_dataset)
```

## 摘要

我们从三种类型的数据构建了 ML 输入管道示例。第一次实验是从加载到内存中的数据构建管道。然后我们从外部文件构建了一个管道。最后的实验是从云存储构建了一个管道。
