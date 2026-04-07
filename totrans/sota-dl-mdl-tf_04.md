# 4. 使用 TensorFlow 数据集进行深度学习

在上一章中，我们展示了如何处理 TFDS 对象。在本章中，我们将通过两个端到端深度学习实验来处理大型和复杂的 TFDS 对象。Fashion-MNIST 和豆类数据集具有简单的图像。

第一个实验使用包含猫和狗复杂图像的大型数据集。第二个实验使用包含人类手部复杂图像的数据集。人类手部数据集的大小不如第一个实验中的数据集大。两个实验都包括数据增强以提高性能。目标是帮助你完成更大和更复杂的端到端学习实验。

各章节的笔记本位于以下 URL：

[`github.com/paperd/deep-learning-models`](https://github.com/paperd/tensorflow)

## 进行 cats_vs_dogs 实验

我们使用*cats_vs_dogs*数据集演示了一个端到端的示例。该数据集包含 23,262 个猫和狗的大图像示例。本实验的目的是为你提供训练大型和复杂 TFDS 的经验。我们鼓励你处理多个数据集，以获得更多深度学习的经验。数据是关键！与各种数据集一起工作可以提供不同的体验，因为每个数据集都是独特的。

注意

处理大型图像文件需要大量的 RAM。因此，你可能需要投资更多内存或订阅 Google 的 Colab Pro 服务，以防你的电脑崩溃。在我们的 PC 上，我们偶尔会遇到这个数据集导致 RAM 崩溃的情况。但自从我们升级到 Colab Pro 以来，我们没有遇到过任何问题。

## 导入 TensorFlow 库

导入库并将其别名为**tf**：

```py
import tensorflow as tf
```

将 TensorFlow 库别名为 tf 是常见做法。

## GPU 硬件加速器

为了方便起见，我们提供了在 Colab 笔记本中启用 GPU 的步骤：

1.  点击左上角的*运行*菜单。

1.  从下拉菜单中选择*更改运行时类型*。

1.  从*硬件加速器*下拉菜单中选择*GPU*。

1.  点击*保存*。

验证 GPU 是否激活：

```py
tf.__version__, tf.test.gpu_device_name()
```

如果显示“/device:GPU:0”，则 GPU 已激活。如果显示“..”，则常规 CPU 已激活。

注意

如果你遇到错误**“名称‘TF’未定义”**，请重新执行代码以导入 TensorFlow 库！

## 开始实验

当处理新的数据集时，探索其元数据是一个好主意。然后我们可以做出明智的决定来分割它。

### 加载 TFDS 对象

使用简单的参数加载数据集以获取其元数据：

```py
import tensorflow_datasets as tfds
data, info = tfds.load(name='cats_vs_dogs', with_info=True,
try_gcs=True)
```

由于数据集包含超过 20,000 张大型图像，因此加载数据集需要更长的时间。但一旦数据集被加载到内存中，重新加载数据集就非常快。损坏的图像会自动跳过。

### 元数据

显示 info 对象的内容：

```py
info
```

这个对象包含了很多信息。但我们只需要从中访问其中的一些信息。尽管您可能不需要对象中包含的大量信息，但显示它并阅读可用的信息并无害处。

获取类别标签和类别数量：

```py
class_labels = info.features['label'].names
num_classes = info.features['label'].num_classes
class_labels, num_classes
```

从元数据中，我们看到数据集**没有**预先分割。因此，我们必须自己分割数据。我们选择将 80%的数据分割为训练集，10%的数据分割为验证集，10%的数据分割为测试集。如果您愿意，可以选择不同的分割比例。

显示分割信息：

```py
num_train_img = info.splits['train[0%:80%]'].num_examples
num_validation_img = info.splits['train[80%:90%]'].num_examples
num_test_img = info.splits['train[90%:100%]'].num_examples
print ('train images:', num_train_img)
print ('validation images:', num_validation_img)
print ('test images:', num_test_img)
```

使用我们的分割方案（80:10:10），我们应该有 18,610 个训练示例，2,326 个验证示例和 2,326 个测试示例。将数据分割成训练、验证和测试集的动机是从训练数据中学习，从验证数据中调整，从测试数据中泛化。

注意

我们在本节中实际上并没有分割数据。我们只是用我们的分割方案向您展示每个分割中会有多少个示例。

虽然将数据仅分割为训练集和测试集是很常见的，但添加一个额外的分割来保留模型从未接触过的数据是有利的。有了三个分割，训练数据用于学习，验证数据用于调整和微调，测试数据用于评估泛化能力。只有训练集和测试集时，我们必须从测试集进行调整和泛化。由于模型接触了测试集，它并不是完全的新数据！

验证分割百分比：

```py
train_num = num_train_img /23262
validation_num = num_validation_img /23262
test_num = num_test_img /23262
'{0:.0%}'.format(train_num), '{0:.0%}'.format(validation_num),\
'{0:.0%}'.format(test_num)
```

由于数据集包含 23,262 张图像，我们通过除以该数字来验证每个集合中的图像数量。

### 分割数据

我们可以将数据加载到一个容器中，并手动执行所需的分割。或者，我们可以在分割参数中包含分割信息，如下所示：

```py
(training_set, validation_set, test_set), info = tfds.load(
'cats_vs_dogs', with_info=True,
split=['train[:80%]', 'train[80%:90%]',
'train[90%:]'], shuffle_files=True,
as_supervised=True, try_gcs=True)
```

代码在单步中将 80%的数据加载到训练集中，10%的数据加载到验证集中，10%的数据加载到测试集中。之前，我们演示了如何单步加载训练集和测试集。

手动验证每个分割中的示例数量：

```py
len(list(training_set)), len(list(validation_set)),\
len(list(test_set))
```

列表操作需要一些时间，因为每个张量示例都需要被处理成列表。所以我们**不**推荐对非常大的数据集执行此操作！

### 可视化

让我们可视化训练集中的某些示例，看看图像和标签的外观。我们可以以三种方式可视化。

使用 show_examples 显示示例：

```py
fig = tfds.show_examples(training_set, info)
```

以数据框的形式显示示例：

```py
tfds.as_dataframe(training_set.take(4), info)
```

为了获得最大控制，手动显示示例。首先，取一些示例：

```py
images, labels = [], []
for img, lbl in training_set.take(4):
img = tf.squeeze(img)
images.append(img), labels.append(lbl)
```

然后按照列表 4-1 所示可视化示例：

```py
import matplotlib.pyplot as plt
rows, cols = 2, 2
plt.figure(figsize=(10, 10))
for i in range(rows*cols):
plt.subplot(rows, cols, i + 1)
plt.imshow(images[i], cmap='bone')
t = class_labels[labels[i]] + ' (' +\
str(labels[i].numpy()) + ')'
plt.title(t)
plt.axis('off')
Listing 4-1
Visualize Examples
```

现在，我们已经了解了数据集中示例的外观。

### 检查示例

按照列表 4-2 所示取一些示例并将它们转换为 NumPy 数组：

```py
features, labels = [], []
for img, lbl in training_set.take(4):
img = tfds.as_numpy(img)
lbl = tfds.as_numpy(lbl)
features.append(img)
labels.append(lbl)
Listing 4-2
Add NumPy Examples to Lists
```

在可视化示例时，使用 NumPy 数组更容易。

如列表 4-3 所示可视化：

```py
rows, cols = 2, 2
plt.figure(figsize=(10, 10))
for i in range(rows*cols):
c = class_labels[labels[i]]
s = str(features[i].shape)
title = c + ' ' + s
plt.subplot(rows, cols, i + 1)
plt.title(title)
plt.imshow(features[i], cmap='binary')
plt.axis('off')
Listing 4-3
Visualize Inspected Examples
```

### 重新格式化图像

创建一个调整和缩放图像的函数：

```py
def format_image(image, label):
image = tf.image.resize(image, (150, 150))/255.0
return image, label
```

使用较小的图像训练模型速度更快。因此，将图像调整大小为 150 × 150 像素。调整图像大小以提高训练性能。

注意

当将图像调整到更小的尺寸时，会丢失一些信息。但 tf.image.resize 很好地保留了尽可能多的信息。我们将图像调整到 150 × 150 以保留大部分信息，同时提高学习速度。尝试不同的尺寸以查看对性能的影响。

函数在下一节中用于简化输入管道的构建。

### 构建输入管道

通过准备学习模型的训练、验证和测试集来构建输入管道。

设置批量和打乱参数：

```py
BATCH_SIZE = 200
SHUFFLE_SIZE = 500
```

小贴士

尝试不同的批量和打乱大小，以查看对学习性能的影响。

转换数据以实现最佳性能：

```py
train_batches = training_set.shuffle(SHUFFLE_SIZE).\
map(format_image).batch(BATCH_SIZE).cache().prefetch(1)
validation_batches = validation_set.\
map(format_image).batch(BATCH_SIZE).cache().prefetch(1)
test_batches = test_set.\
map(format_image).batch(BATCH_SIZE).cache().prefetch(1)
```

通过批处理，可以减少训练时间。通过打乱，通常可以提高准确性。缓存有助于更好地管理内存，而预取应该减少训练时间。

检查张量：

```py
train_batches, validation_batches, test_batches
```

### 可视化和检查批次中的示例

取第一个训练批次：

```py
for img, lbl in train_batches.take(1):
print (img.shape)
```

检查批次中的第一个示例：

```py
img[0].shape, class_labels[lbl[0].numpy()]
```

示例包含一个 150 × 150 × 3 的“狗”或“猫”图像。由于随机化效应，我们无法知道是哪一种。

将四个示例提取到列表中：

```py
images, labels = [], []
for i in range(4):
tf.squeeze(img[i])
images.append(img[i]), labels.append(lbl[i])
```

挤压掉*1*维，因为*imshow()*函数期望一个 2D 矩阵。

如列表 4-4 所示可视化图像。

```py
rows, cols = 2, 2
plt.figure(figsize=(10, 10))
for i in range(rows*cols):
plt.subplot(rows, cols, i + 1)
plt.imshow(images[i], cmap='bone')
t = class_labels[labels[i]] + ' (' +\
str(labels[i].numpy()) + ') ' +\
str(images[i].shape)
plt.title(t)
plt.axis('off')
Listing 4-4
Visualize Batched Examples
```

### 构建模型

获取输入形状：

```py
for img, lbl in train_batches.take(1):
in_shape = img.shape[1:]
in_shape
```

导入库：

```py
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D,\
Dense, Flatten, Dropout
```

清除和设置随机种子：

```py
import numpy as np
tf.keras.backend.clear_session()
np.random.seed(0)
tf.random.set_seed(0)
```

创建一个函数来构建如列表 4-5 所示的模型。

```py
def build_model():
model = \
Sequential([
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
Dense(num_classes, activation='sigmoid')])
return model
Listing 4-5
Multilayered CNN Model
```

我们创建了一个函数来保存这个实验中的模型。我们这样做是为了向您展示如何使用函数来完成这项工作。

模型开始于四个 Conv2D 层和四个相关的 MaxPooling2D 层。一个*Conv2D*层根据输入和 4D 滤波器张量计算 2D 卷积。在我们的情况下，张量的维度是(200, 150, 150, 3)。输入张量是 200 个 150 × 150 彩色图像的批次。一个*MaxPooling2D*层对 2D 空间数据进行最大池化操作。因此，每个最大池化操作都会减少从每个 Conv2D 层接收到的 2D 空间数据的维度。我们使用 2 × 2 的池化大小，因为我们发现这个大小有很好的性能。

*卷积层*将滤波器应用于原始图像或其他特征图。最重要的参数是核的数量和大小。*池化层*执行最大池化或平均池化以减少网络的维度。我们在每个卷积层之后使用池化层来减少卷积层产生的维度。最大池化取滤波器区域内的最大值，而平均池化取滤波器区域内的平均值。

第一个 Conv2D 层接受核的数量、核大小、输入形状、步长和正则化。我们使用 3 × 3 的核大小，因为我们发现这个尺寸的性能很好。我们添加了 l1 和 l2 正则化来减少过拟合。*核* 是一个用于从图像中提取特征的过滤器。具体来说，核是一个在输入数据上移动的矩阵，它与输入数据的子区域执行点积，并将输出作为点积矩阵（或特征图）。*特征图* 捕获将过滤器（或核）应用于输入图像的结果。

可视化特定输入图像的特征图的原因是试图了解我们的 CNN 检测到的特征。我们应用几个卷积层到数据上，因为我们希望每一层产生的特征图能帮助我们检测更清晰的特征。

剩余的 Conv2D 层增加核大小以提高模型性能。在每个 Conv2D 层之后是一个 MaxPooling2D，以减少卷积贡献的维度。每个 Conv2D 层使用修正线性单元（ReLU）激活，这有助于解决与 sigmoid 分布相关的梯度消失问题。

Flatten 层为卷积层的输出准备输入到 Dense 层。Dense 层需要一个单个长特征向量作为输入。包括两个完全连接的 Dense 层用于输出分类。

创建模型：

```py
cat_dog_model = build_model()
```

检查模型：

```py
cat_dog_model.summary()
```

关于将正则化应用于学习模型的优秀资源，请参阅

[如何使用 Keras 中的 L1、L2 和弹性网正则化](http://www.machinecurve.com/index.php/2020/01/23/how-to-use-l1-l2-and-elastic-net-regularization-with-keras/)

### 编译和训练模型

编译：

```py
loss = tf.keras.losses.SparseCategoricalCrossentropy(
from_logits=True)
cat_dog_model.compile(optimizer='adam',
loss=loss,
metrics=['accuracy'])
```

由于模型输出 logits，设置 *from_logits=True*。模型输出 logits 是因为我们使用了模型输出层的 sigmoid 激活函数。

训练：

```py
epochs = 10
history = cat_dog_model.fit(
train_batches, epochs=epochs, verbose=1,
validation_data=validation_batches)
```

模型从训练集学习，并在验证集上进行验证。模型从未接触过测试集。

### 评估模型的泛化性

使用测试批次执行 evaluate 方法：

```py
metrics = cat_dog_model.evaluate(test_batches)
```

*evaluate()* 方法返回损失和准确度值。我们使用 *test_batches* 来评估泛化性，因为模型从未见过这个数据集。**泛化**是学习模型适应新数据的能力，这些新数据来自与创建模型相同的分布。

### 可视化性能

创建一个如图 4-6 所示的可视化函数。

```py
def viz(hd):
acc = hd['accuracy']
val_acc = hd['val_accuracy']
loss = hd['loss']
val_loss = hd['val_loss']
plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')
plt.subplot(1, 2, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
Listing 4-6
Plot Model Performance
```

调用函数：

```py
viz(history.history)
```

模型过拟合，因为测试准确度并没有与训练准确度非常接近。

## 预处理层增强

为了减轻过拟合并提高模型性能，我们尝试数据增强。应用随机水平翻转、随机旋转和随机缩放，如图 4-7 所示。

```py
from tensorflow.keras import layers
data_augmentation = tf.keras.Sequential(
[
layers.experimental.preprocessing.RandomFlip('horizontal'),
layers.experimental.preprocessing.RandomRotation(0.1),
layers.experimental.preprocessing.RandomZoom(0.1),
]
)
Listing 4-7
Keras Preprocessing Layers for Transformation
```

我们选择使用这些层是基于试错实验。请随意尝试其他层。

注意

这种转换操作是实验性的，这意味着它可能在将来发生变化。

如列表 4-8 所示显示增强图像。

```py
plt.figure(figsize=(10, 10))
for images, _ in train_batches.take(1):
for i in range(9):
augmented_images = data_augmentation(images)
ax = plt.subplot(3, 3, i + 1)
plt.imshow(augmented_images[0])
plt.axis('off')
Listing 4-8
Display an Augmented Image
```

可视化显示了多次应用于同一图像的数据增强。将索引从 0（在*augmented_images[0]*中）更改为 1 到 199（批大小 200），以在*plt.imshow*代码语句中查看不同的图像。

### 构建模型

通过清除之前的会话、生成一个种子并创建实验模型来构建模型。

清除和设置种子：

```py
tf.keras.backend.clear_session()
np.random.seed(0)
tf.random.set_seed(0)
```

如列表 4-9 所示创建模型。

```py
cat_dog_layers = Sequential([
data_augmentation,
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
Listing 4-9
Model with Augmentation Layer
```

注意，第一层是我们创建的增强！因此，模型从第一层获取增强，并从该点开始像其他卷积模型一样继续。

### 编译和训练模型

编译：

```py
cat_dog_layers.compile(
optimizer='adam',
loss=tf.keras.losses.SparseCategoricalCrossentropy(
from_logits=True),
metrics=['accuracy'])
```

训练：

```py
epochs = 10
history = cat_dog_layers.fit(
train_batches, epochs=epochs, verbose=1,
validation_data=validation_batches)
```

### 评估模型的泛化能力

评估：

```py
metrics = cat_dog_layers.evaluate(test_batches)
```

数据增强不能保证更好的性能。它只是一种可以应用于此目的的技术。此外，我们只运行了这个模型十轮。因此，可能需要更多轮次才能看到更好的性能。

### 可视化性能

可视化训练性能：

```py
viz(history.history)
```

过度拟合已经有所缓解。尝试运行更多轮次的实验以查看会发生什么。

## 在图像上应用数据增强

通过直接对图像执行转换来应用数据增强。

创建函数以增强图像，如列表 4-10 所示。

```py
def random_crop(image):
shape = tf.shape(image)
min_dim = tf.reduce_min([shape[0], shape[1]]) * 90 // 100
return tf.image.random_crop(image, [min_dim, min_dim, 3])
def preprocess(image, label):
cropped_image = random_crop(image)
cropped_image = tf.image.random_flip_left_right(cropped_image)
resized_image = tf.image.resize(cropped_image, [150, 150])
final_image = tf.keras.applications.xception.preprocess_input(
resized_image)
return final_image, label
Listing 4-10
Functions That Apply Augmentations on Images
```

*random_crop*函数随机裁剪图像。*preprocess*函数调用 random_crop，随机左右翻转图像，并调整大小。*tf.keras.applications.xception.preprocess_input*实用工具通过编码一批图像来预处理张量（或 NumPy 数组）。通过研究和实验，我们发现这些增强提高了性能并缓解了过度拟合。尝试不同的增强并查看会发生什么。

### 构建输入管道

导入*partial*包：

```py
from functools import partial
```

partial 包允许我们创建部分函数。部分函数允许我们固定函数的一定数量的参数并生成一个新的函数。

设置批量和洗牌大小参数：

```py
BATCH_SIZE = 200
SHUFFLE_SIZE = 500
```

小贴士

尝试不同的批量和洗牌大小，以查看学习性能如何受到影响。

构建管道：

```py
train_shuffle = training_set.shuffle(1000)
train_batches = train_shuffle.map(partial(preprocess)).\
batch(BATCH_SIZE).prefetch(1)
validation_batches = validation_set.map(preprocess).\
batch(BATCH_SIZE).prefetch(1)
test_batches = test_set.map(preprocess).\
batch(BATCH_SIZE).prefetch(1)
```

在预处理函数上设置部分参数可以固定其参数，这样我们就可以仅对训练集中的图像应用转换！我们不需要增强其他数据集，因为模型只从训练集中学习。

### 显示增强图像

如列表 4-11 所示，这是增强对图像的影响。

```py
plt.figure(figsize=(10, 10))
for images, _ in train_batches.take(1):
for i in range(9):
augmented_images = data_augmentation(images)
Images = np.clip(augmented_images, 0, 1)
ax = plt.subplot(3, 3, i + 1)
plt.imshow(Images[0])
plt.axis('off')
Listing 4-11
Visualize Augmentations on an Image
```

我们显示批次的第一个图像上的增强。尝试不同的索引号以查看批次中的任何 200 个图像的增强。请确保选择介于 0 和 199 之间的索引，因为批大小是 200！

### 构建模型

清除和设置种子：

```py
tf.keras.backend.clear_session()
np.random.seed(0)
tf.random.set_seed(0)
```

创建模型：

```py
cat_dog_images = build_model()
```

### 编译和训练模型

编译：

```py
cat_dog_images.compile(
optimizer='adam',
loss=tf.keras.losses.SparseCategoricalCrossentropy(
from_logits=True),
metrics=['accuracy'])
```

训练：

```py
epochs = 5
history = cat_dog_images.fit(
train_batches, epochs=epochs,
verbose=1, validation_data=validation_batches)
```

我们只训练了几个轮次，因为应用到的图像增强增加了训练时间。尝试不同的轮次数量，看看会发生什么。

### 评估模型的泛化能力

评估：

```py
cat_dog_images.evaluate(test_batches)
```

我们使用测试集，因为它从未被模型看到。

注意

准确率可能看起来较低，但实际上并非如此。原因是我们可能需要运行模型更多的轮次。这就是为什么我们鼓励你尝试不同的轮次数量。

### 可视化性能

可视化训练性能：

```py
viz(history.history)
```

验证准确率轨迹要好得多！也就是说，训练和测试值更接近。

### 预测

为了完成实验，让我们看看模型从测试集预测的效果如何。我们可以在整个测试数据集或数据集的一个批次上进行预测。

首先对整个测试数据集进行预测，并显示第一个示例的预测数组：

```py
predictions =cat_dog_images.predict(test_batches)
list(predictions[0])
```

在 *test_batches* 上进行预测，因为它从未被模型看到。*predict()* 方法返回预测的 NumPy 数组。对于分类实验，预测数组中的每个元素代表一个类别标签。由于我们的实验中有两个类别（猫和狗），每个预测数组包含两个元素。猫的预测是数组中的第一个值，狗的预测是第二个值。每个值是预测的概率。概率的大小表示元素是预测类别的可能性。因此，预测数组中两个元素之间较高的概率代表预测。例如，预测数组 [0.33, 0.778] 表示预测是一只狗，因为它有更高的概率。

返回第一个示例的预测：

```py
first_prediction = tf.math.argmax(predictions[0])
class_labels[first_prediction.numpy()]
```

*tf.math.argmax()* API 返回张量轴上的最大值的索引。简单来说，它返回预测数组中值最高的索引。

我们也可以对单个批次进行预测：

```py
first_batch = test_batches.take(1)
predict_batch = cat_dog_images.predict(first_batch)
```

对第一个批次取第一个批次并对其进行预测。

获取批次中的第一个示例的预测：

```py
first_batch_prediction = tf.math.argmax(predict_batch[0])
class_labels[first_batch_prediction.numpy()]
```

获取第一个实际标签：

```py
for image, label in first_batch:
print ('batch size:', image.shape[0])
class_labels[label[0].numpy()]
```

从第一个批次获取实际的图像和标签。显示第一个实际图像的形状及其对应的标签。如果预测与实际标签匹配，则预测正确！

如列表 4-12 所示，手动检查预测准确度。

```py
cnt = 0
for i in range(image.shape[0]):
pred = tf.math.argmax(predict_batch[i]).numpy()
actual = label[i].numpy()
if actual == pred:
cnt += 1
acc = cnt / image.shape[0]
'{percent:.2%}'.format(percent=acc) + ' accuracy'
Listing 4-12
Check Prediction Accuracy
```

代码计算测试数据的预测准确率，并以百分比的形式显示。

### 可视化预测

从测试集中获取第一个批次的图像和标签：

```py
for img, lbl in test_batches.take(1):
print (img.shape)
```

每个批次包含 200 张 150 × 150 的彩色图像。

检查第一批次的第一张图像：

```py
img[0].shape, class_labels[lbl[0].numpy()]
```

可视化图像：

```py
Image= np.clip(img[0], 0, 1)
fig = plt.imshow(Image)
fig = plt.axis('off')
```

我们使用 `tf.keras.applications.xception.preprocess_input` 工具来预处理图像（参见列表 4-10 中的预处理函数）。该工具实例化 Xception 架构以实现最佳性能。然而，它不会缩放图像。因此，我们必须将图像像素转换为介于 0 和 1 之间，以便适当地显示。

处理一些示例：

```py
images, labels = [], []
for i in range(20):
tf.squeeze(img[i])
images.append(img[i]), labels.append(lbl[i])
```

创建一个函数来显示一组图像和标签。该函数确定预测是正确还是错误。正确预测的标题会正常显示，但错误预测的标题会以红色文本显示。

创建如列表 4-13 所示的函数。

```py
def display_test(feature, target, num_images,
n_rows, n_cols, cl, p):
for i in range(num_images):
plt.subplot(n_rows, 2*n_cols, 2*i+1)
Image= np.clip(feature[i], 0, 1)
plt.imshow(Image)
pred = cl[tf.math.argmax(p[i]).numpy()]
actual = cl[target[i]]
title_obj = plt.title(actual + ' (' +\
pred + ') ')
if pred == actual:
title_obj
else:
plt.getp(title_obj, 'text')
plt.setp(title_obj, color='r')
plt.tight_layout()
plt.axis('off')
Listing 4-13
Function to Display Predictions Against Actual Labels
```

调用函数：

```py
num_rows, num_cols = 5, 4
num_images = num_rows*num_cols
plt.figure(figsize=(20, 20))
display_test(images, labels, num_images, num_rows,
num_cols, class_labels, predictions)
```

图像带有标题显示。每个标题显示实际标签和括号中的预测。如果预测错误，标题将以红色显示。

备注

尽管我们应用的机器学习模型可能对初学者来说看起来很复杂，但它们实际上非常简单。随着你对学习模型的经验积累，你将获得构建更复杂模型或迁移到迁移学习模型的信心。我们在本书中介绍了迁移学习模型。

## 一个关于 rock_paper_scissors 的实验

尽管我们刚刚完成了一个大型且复杂数据集的实验，但我们相信，使用多个数据集进行深度学习对于成为一名合格的从业者至关重要。在我们的研讨会上，我们发现当参与者能够将所学知识应用于另一个情境时，他们的学习速度会更快。在这种情况下，是一个新的数据集。实验是在 rock_paper_scissors 数据集上进行的，该数据集包含 2,892 张玩剪刀石头布游戏的图片。

### 配置 TensorBoard

**TensorBoard** 是一个在机器学习工作流程中提供测量和可视化的工具。它跟踪损失、准确度、模型图可视化以及将嵌入投影到低维空间等指标。我们在这个实验中介绍 TensorBoard，以便提供访问另一个可视化工具的机会。

对于一个很好的教程，请参阅

[TensorBoard 入门教程](http://www.tensorflow.org/tensorboard/get_started)

加载 TensorBoard 笔记本扩展：

```py
%load_ext tensorboard
```

导入我们在实验中使用的必需库：

```py
import datetime
```

清除之前运行的日志：

```py
!rm -rf ./logs/
```

### 加载数据

加载训练和测试集：

```py
(train_digits, test_digits), rps_info = tfds.load(
'rock_paper_scissors', with_info=True,
data_dir='tmp', as_supervised=True,
split=['train', 'test'])
```

### 检查数据

我们认为探索一个新的数据集以了解其配置是一个好主意。因此，我们获取元数据对象，可视化一些示例，并获取训练和测试示例的数量、示例形状和标签名称。我们需要了解所有这些信息才能进行实验。

获取元数据：

```py
rps_info
```

可视化：

```py
fig = tfds.show_examples(train_digits, rps_info)
```

获取示例和标签数量：

```py
train_examples = rps_info.splits['train'].num_examples
test_examples = rps_info.splits['test'].num_examples
num_labels = rps_info.features['label'].num_classes
train_examples, test_examples, num_labels
```

获取形状：

```py
rps_info.features['image'].shape,\
rps_info.features['label'].shape
```

获取标签名称：

```py
label_name = rps_info.features['label'].int2str
for lbl in range(num_labels):
print (label_name(lbl), end=' ')
```

检查图像形状：

```py
for image, label in train_digits.take(5):
print (image.shape)
```

### 预处理数据

下一步是为学习模型预处理数据。我们首先获取当前图像大小的一半值。我们稍后会使用这个值来调整图像大小。我们选择这个大小是为了减少训练时间。通过选择这个大小，我们发现模型仍然表现良好。我们鼓励你尝试不同的图像大小，看看学习效果如何。我们还对图像进行缩放和调整大小。记住，学习模型在较小的像素尺寸上表现更好。缩放是减少图像大小而不影响学习效果的最佳方式。

创建一个减半图像大小的形状：

```py
new_pixels = rps_info.features['image'].shape[0] // 2
new_pixels
```

创建一个预处理函数：

```py
def format_digits(image, label):
image = tf.cast(image, tf.float32) / 255.
image = tf.image.resize(image, [new_pixels, new_pixels])
return image, label
```

预处理训练和测试集：

```py
train_original = train_digits.map(format_digits)
test_original = test_digits.map(format_digits)
```

探索一个示例：

```py
for image, label in train_original.take(1):
finger_img_shape = image.shape
print (image.shape, image[0][0].numpy(), label.numpy())
```

我们总是在预处理后至少探索一个示例，看看它的样子。

### 可视化处理后的数据

创建一个如列表 4-14 所示的可视化函数。

```py
def preview_dataset(dataset):
plt.figure(figsize = (12, 12))
plot_index = 0
for image, label in dataset.take(12):
plot_index += 1
plt.subplot(3, 4, plot_index)
plt.axis('Off')
label = label_name(label.numpy())
plt.title(label)
plt.imshow(image.numpy())
Listing 4-14
Visualization Function
```

调用函数：

```py
preview_dataset(train_original)
```

### 增强训练数据

我们包括数据增强来提高学习效果。我们根据实验选择了增强方法。我们将不同的增强方法分别放入单独的函数中，以增加灵活性。我们使用所有增强方法进行实验，但你也可以使用一个或多个函数进行新的实验。你甚至可以跳过增强步骤，看看模型在没有增强的训练数据上的学习效果如何。

增强是一种艺术。我们还没有找到任何关于如何系统地增强给定数据集的深度学习应用研究。深度学习算法已经存在很多年了，但它们的实际应用却相对较晚。一个重要原因是计算能力远远达不到现在的水平。而且现在它变得更便宜了！另一个原因是 TensorFlow 作为开源产品只有几年时间。

#### 创建数据增强函数

现在我们创建函数来翻转、着色、旋转、反转和缩放图像。

创建一个随机翻转图像的函数：

```py
def flip(image: tf.Tensor) -> tf.Tensor:
image = tf.image.random_flip_left_right(image)
image = tf.image.random_flip_up_down(image)
return image
```

注意

使用*tf.Tensor* API 将图像转换为张量。

创建一个函数来随机增强颜色，如列表 4-15 所示。

```py
def paint(image: tf.Tensor) -> tf.Tensor:
image = tf.image.random_hue(image, max_delta=0.2)
image = tf.image.random_saturation(image, lower=0.7, upper=1.3)
image = tf.image.random_brightness(image, 0.05)
image = tf.image.random_contrast(image, lower=0.8, upper=1)
image = tf.clip_by_value(image, clip_value_min=0,
clip_value_max=1)
return image
Listing 4-15
Randomly Augment Image Color
```

创建一个随机旋转图像的函数：

```py
def rotate(image: tf.Tensor) -> tf.Tensor:
return tf.image.rot90(
image,
tf.random.uniform(
shape=[], minval=0,
maxval=4, dtype=tf.int32))
```

创建一个随机反转图像的函数：

```py
def invert(image: tf.Tensor) -> tf.Tensor:
random = tf.random.uniform(shape=[], minval=0, maxval=1)
if random > 0.5:
image = tf.math.multiply(image, -1)
image = tf.math.add(image, 1)
return image
```

创建一个如列表 4-16 所示的缩放图像的函数：

```py
def zoom(
image: tf.Tensor, min_zoom=0.8, max_zoom=1.0) -> tf.Tensor:
image_width, image_height, image_colors = image.shape
crop_size = (image_width, image_height)
scales = list(np.arange(min_zoom, max_zoom, 0.01))
boxes = np.zeros((len(scales), 4))
for i, scale in enumerate(scales):
x1 = y1 = 0.5 - (0.5 * scale)
x2 = y2 = 0.5 + (0.5 * scale)
boxes[i] = [x1, y1, x2, y2]
def random_crop(img):
crops = tf.image.crop_and_resize(
[img], boxes=boxes,
box_indices=np.zeros(len(scales)),
crop_size=crop_size)
return crops[tf.random.uniform(
shape=[], minval=0, maxval=len(scales), dtype=tf.int32)]
choice = tf.random.uniform(
shape=[], minval=0., maxval=1., dtype=tf.float32)
return tf.cond(
choice < 0.5, lambda: image, lambda: random_crop(image))
Listing 4-16
Zoom Augmentation
```

缩放函数生成从 1%到 20%的裁剪设置。然后创建边界框来容纳裁剪的图像。返回的裁剪图像被调整大小以保持训练时大小一致。裁剪只进行 50%的时间。

创建一个增强函数：

```py
def augment_data(image, label):
image = flip(image)
image = paint(image)
image = rotate(image)
image = invert(image)
image = zoom(image)
return image, label
```

为了进行实验，通过注释掉函数来移除一个或多个增强方法。

### 增强训练数据

将增强映射到训练数据：

```py
train_augmented = train_original.map(augment_data)
```

可视化训练示例增强：

```py
preview_dataset(train_augmented)
```

可视化原始测试集：

```py
preview_dataset(test_original)
```

### 构建输入管道

完成管道：

```py
BATCH_SIZE = 32
train_fingers = train_augmented.shuffle(train_examples).cache().\
batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)
test_fingers = test_original.batch(BATCH_SIZE)
```

Prefetch API 允许在模型训练时异步获取批次。

### 创建模型

清除和设置随机种子：

```py
tf.keras.backend.clear_session()
np.random.seed(0)
tf.random.set_seed(0)
```

验证图像形状：

```py
finger_img_shape
```

按照列表 4-17 所示构建模型。

```py
finger_model = Sequential([
Conv2D(64, 3, activation='relu',
input_shape=finger_img_shape,
kernel_regularizer='l1_l2'),
MaxPooling2D(2, 2),
Conv2D(64, 3, activation='relu'),
MaxPooling2D(2, 2),
Conv2D(128, 3, activation='relu'),
MaxPooling2D(2, 2),
Conv2D(128, 3, activation='relu'),
MaxPooling2D(2, 2),
Flatten(),
Dense(512, activation='relu'),
Dense(num_labels, activation='softmax')])
Listing 4-17
Create the Model
```

检查模型：

```py
tf.keras.utils.plot_model(
finger_model,
show_shapes=True,
show_layer_names=True)
```

我们介绍另一个工具来检查模型，作为替代方案。我们试图向您展示尽可能多的选项。但我们不想让您感到不知所措。所以我们在实验的上下文中加入一些新内容。我们发现，在某个上下文中学习像 TensorFlow 这样复杂的软件要快得多。

### 编译和训练

编译：

```py
optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001)
finger_model.compile(
optimizer=optimizer,
loss=tf.keras.losses.sparse_categorical_crossentropy,
metrics=['accuracy'])
```

建立训练参数：

```py
steps_per_epoch = train_examples // BATCH_SIZE
validation_steps = test_examples // BATCH_SIZE
print('steps_per_epoch:', steps_per_epoch)
print('validation_steps:', validation_steps)
```

删除日志和检查点以清除之前的模型会话：

```py
!rm -rf tmp/checkpoints
!rm -rf logs
```

准备一个 TensorBoard 回调：

```py
log_dir = 'logs/fit/' +\
datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
tensorboard_callback = tf.keras.callbacks.TensorBoard(
log_dir=log_dir, histogram_freq=1)
```

训练：

```py
training_history = finger_model.fit(
train_fingers.repeat(),
validation_data=test_fingers.repeat(),
epochs=10,
steps_per_epoch=steps_per_epoch,
validation_steps=validation_steps,
callbacks=[tensorboard_callback])
```

### 可视化性能

创建一个如列表 4-18 所示的函数。

```py
def viz_history(training_history):
loss = training_history.history['loss']
val_loss = training_history.history['val_loss']
accuracy = training_history.history['accuracy']
val_accuracy = training_history.history['val_accuracy']
plt.figure(figsize=(14, 4))
plt.subplot(1, 2, 1)
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.plot(loss, label='Training set')
plt.plot(val_loss, label='Test set', linestyle='--')
plt.legend()
plt.grid(linestyle='--', linewidth=1, alpha=0.5)
plt.subplot(1, 2, 2)
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.plot(accuracy, label='Training set')
plt.plot(val_accuracy, label='Test set', linestyle='--')
plt.legend()
plt.grid(linestyle='--', linewidth=1, alpha=0.5)
plt.show()
Listing 4-18
Visualization Function
```

调用函数：

```py
viz_history(training_history)
```

启动 TensorBoard：

```py
%tensorboard --logdir logs/fit
```

再次，我们向您展示如何在实验的上下文中使用 TensorBoard 来加速您的学习。我们并不经常使用 TensorBoard，因为我们只需几行代码就能看到模型性能，就像我们之前所做的那样。但我们认为让您了解这个产品是个好主意。您可能更喜欢我们可视化学习性能的方式。

### 关闭 TensorBoard 服务器

使用全局正则表达式打印（grep）命令来获取正在运行的 TensorBoard 进程详细信息：

```py
!ps -ef | grep tensorboard
```

输出看起来像这样：

```py
root       10757    4202 26 19:13 ?        00:00:04 python3 /usr/local/bin/tensorboard --logdir logs/fit
root       10794    4202  0 19:13 ?        00:00:00 /bin/bash -c ps -ef | grep tensorboard
root       10796   10794  0 19:13 ?        00:00:00 grep tensorboard
```

第一个数字 10757 是 TensorFlow 的当前进程标识符（pid）。

使用 pid 杀死进程：

```py
!kill 10757
```

注意

每次激活 TensorBoard 时，进程 ID（pid）都会改变。
