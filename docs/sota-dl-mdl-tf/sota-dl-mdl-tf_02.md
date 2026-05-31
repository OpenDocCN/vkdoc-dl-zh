# 2. 使用数据增强增加数据集的多样性

我们指导您创建增强数据实验，通过应用随机（但真实）的变换来增加训练集的多样性。数据增强对于小型数据集非常有用，因为深度学习模型需要大量数据才能表现良好。

各章节的笔记本位于以下 URL：

[`github.com/paperd/deep-learning-models`](https://github.com/paperd/deep-learning-models)

## 数据增强

通常情况下，更多的数据会增加模型性能。那么，如果我们只有少量图像训练数据且无法收集更多数据时，我们该怎么办？一种流行的方法是数据增强。

**数据增强**是增加现有训练集的大小和多样性，而不需要手动收集任何新数据的过程。这个过程通过使用随机变换（产生看起来可信的图像）来增强现有示例，从而生成额外的训练数据。有了增强的训练数据，学习模型可以接触到数据的更多方面，这有助于它更好地泛化。通常，当训练数据复杂且示例较少时，需要数据增强。

数据增强是通过执行一系列随机预处理变换来实现，例如水平翻转、垂直翻转、倾斜、裁剪、剪切、放大缩小和旋转，对现有数据进行处理。总的来说，增强数据能够模拟出多种细微不同的数据点，而不是仅仅复制相同的数据。增强图像的细微差异应该（希望）足够帮助训练出一个更鲁棒的模式。

数据增强也可以减轻过拟合。过拟合通常发生在训练示例数量较少的情况下。通过从现有示例生成额外的训练数据，可以减轻过拟合。

过拟合发生在模型学习训练数据中的细节和噪声到一定程度，以至于它对模型在新数据上的性能产生负面影响。因此，训练数据中的噪声或随机波动被模型拾取并作为概念学习。小型数据集不包含足够的多样性（或随机性）来减轻学习噪声或随机波动。

深度学习的目标是调整学习模型的参数，使其能够有效地将特定的输入（例如，图像）映射到某些输出（例如，标签）。本质上，我们试图找到模型损失最小化的最佳点，这发生在参数以正确的方式调整时。数据增强是一种有效的调整机制，可以帮助实现深度学习的目标，尤其是在小型数据集的情况下！

对于数据增强的出色介绍，请阅读

[`www.tensorflow.org/tutorials/images/data_augmentation`](http://www.tensorflow.org/tutorials/images/data_augmentation)

## 导入 TensorFlow 库

将库导入并别名为**tf**：

```py
import tensorflow as tf
```

将 TensorFlow 库别名为 tf 是常见的做法。

## GPU 硬件加速器

记住从第一章中，你可以通过使用 Google Colab 云服务提供的 GPU 来大大加快处理速度。为了避免你翻回第一章，我们重复了启用 GPU 的说明：

1.  在右上角菜单中点击*运行时*。

1.  从下拉菜单中选择*更改运行时类型*。

1.  从*硬件加速器*下拉菜单中选择*GPU*。

1.  点击*保存*。

验证 GPU 是否处于活动状态：

```py
tf.__version__, tf.test.gpu_device_name()
```

如果显示“/device:GPU:0”，则 GPU 处于活动状态。如果显示“..”，则常规 CPU 处于活动状态。

注意

如果你遇到错误**“名称‘TF’未定义”**，请重新执行代码以导入 TensorFlow 库！

## 使用 Keras API 进行增强

在上一章中，我们在对花朵数据集建模时遇到了过拟合问题。那一章的问题在于验证准确率相对于训练准确率较低。让我们看看我们是否可以使用 Keras API 进行数据增强来减轻这个问题。

### 获取数据

首先，你将想要再次获取数据。通过执行以下代码来实现，该代码检索与第一章中使用相同的花朵数据：

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

### 分割数据

*tf.keras.preprocessing.image_dataset_from_directory*实用工具从目录中读取数据。该实用工具还能够分割、设置种子、调整大小和分批数据。

设置参数，并且作为一个起点，尝试以下三个值：

```py
BATCH_SIZE = 32
img_height = 180
img_width = 180
```

批量大小设置为 32。我们设置了 32 的批量大小，但你可以自由地尝试不同的尺寸。由于图像形状不同，我们必须调整它们以供模型使用。我们选择了 180 × 180，但你可以自由地尝试不同的图像大小。但请确保高度和宽度相同。

注意

设置批量和图像大小不是一门科学。这项任务需要实验。根据我们的研究，批量大小的良好起点是 32 或 64。任何小于 32 的值都会减慢学习速度。任何大于 64 的值都会增加计算成本。理想的情况是将一批数据完全放入内存中。由于计算机内存的存储容量是 2 的幂，建议将批大小设置为 2 的幂。较小的图像大小可以加快学习速度，但会保留较少的原始图像。因此，模型性能会受到影响。较大的图像大小可以保留更多的原始图像。因此，模型性能会得到提升，但会牺牲计算资源。

#### 将磁盘上的图像加载到训练集和测试集中

现在创建训练集和测试集。将数据分割成这两种类型是常见的做法。从训练集开始，创建它如下：

```py
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
data_dir,
validation_split=0.19,
subset='training',
seed=0,
image_size=(img_height, img_width),
batch_size=BATCH_SIZE)
```

将*validation_split=0.19*和*subset='training'*设置为将 81%的数据放入训练集。将*图像大小*设置为已确定的尺寸以调整图像大小。同时设置种子和批值。

创建测试集：

```py
test_ds = tf.keras.preprocessing.image_dataset_from_directory(
data_dir,
validation_split=0.19,
subset='validation',
seed=0,
image_size=(img_height, img_width),
batch_size=BATCH_SIZE)
```

设置 *validation_split=0.19* 和 *subset='validation'* 以将 19%的数据放入测试集。设置 *image size* 为我们设定的图像大小。同时设置种子和批次值。

### 检查训练集

以一个示例批次为例，显示图像形状和标签：

```py
for images, labels in train_ds.take(1):
print ('image shape:', images.shape)
print ('labels:', labels.numpy())
print ('number of labels in a batch:', len(labels))
```

如预期，第一个批次中有 32 张 180 × 180 × 3 的图像。*3* 的值表示图像是 RGB 颜色。我们还有 32 个相应的标签。我们有 32 张图像和 32 个相应的标签，因为我们设置了批次大小为 32。

### 获取类别数量

使用实用工具的 *class_names* 方法显示类别标签的数量：

```py
class_names = train_ds.class_names
num = len(class_names)
num
```

### 创建缩放函数

创建一个用于缩放特征图像的函数：

```py
def scale(image, label):
image = tf.cast(image, tf.float32)
image /= 255.0
return image, label
```

### 构建输入管道

缩放训练和测试数据。学习模型在较小的图像上训练速度更快。一个两倍大小的输入图像需要网络从四倍数量的像素中学习——而这额外的训练时间会累积。仅对训练数据进行洗牌。缓存和预取训练和测试数据以改进模型性能：

```py
SHUFFLE_SIZE = 100
train_fds = train_ds.map(scale).shuffle(SHUFFLE_SIZE).\
cache().prefetch(1)
test_fds = test_ds.map(scale).cache().prefetch(1)
```

注意

由于训练和测试数据已经被工具批处理，构建输入管道时*不要*进行批处理！

### 带预处理层的数据增强

使用实验性 Keras 预处理层应用数据增强。*Keras 预处理层 API* 允许开发者构建 Keras 本地的输入处理管道。因此，TensorFlow 模型可以接受原始图像或原始结构化数据作为输入，并自行处理特征归一化和特征值索引。简单来说，Keras API 使得输入、预处理和模型原始数据变得更加容易。

我们首先导入所需的库并创建一个简单的增强模型。我们可以像其他层一样在模型中包含预处理层。

让我们通过随机水平翻转、旋转、缩放和平移来增强图像，如列表 2-1 所示。

```py
from tensorflow.keras.layers.experimental.preprocessing\
import RandomFlip
from tensorflow.keras.layers.experimental.preprocessing\
import RandomRotation
from tensorflow.keras.layers.experimental.preprocessing\
import RandomZoom
from tensorflow.keras.layers.experimental.preprocessing\
import RandomTranslation
data_augmentation = tf.keras.Sequential(
[
RandomFlip('horizontal'),
RandomRotation(0.1),
RandomZoom(0.1),
RandomTranslation(height_factor=0.2, width_factor=0.2)
]
)
Listing 2-1
Data Augmentation with Several Preprocessing Layers
```

注意

Keras 预处理层是实验性的，这意味着操作可能会在未来发生变化。

我们将图像左右翻转，随机以 0.1（或 10%）的因子旋转，随机以 0.1（或 10%）的因子缩放，并分别以 0.2（或 20%）的因子随机平移高度和宽度。层的选择和参数值设置是通过试错实验完成的。所以请随意尝试其他增强方式。

更多关于 Keras 实验性预处理层的信息，请参阅

[`www.tensorflow.org/api_docs/python/tf/keras/layers/experimental/preprocessing`](https://www.tensorflow.org/api_docs/python/tf/keras/layers/experimental/preprocessing)

### 显示增强图像

这里展示了将数据增强应用于同一图像多次的情况，如列表 2-2 所示。

```py
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 10))
for images, _ in train_fds.take(1):
for i in range(9):
augmented_images = data_augmentation(images)
ax = plt.subplot(3, 3, i + 1)
plt.imshow(augmented_images[0])
plt.axis('off')
Listing 2-2
Visualize an Augmented Image
```

我们显示第一个图像（索引为 0）的增强效果。更改索引以查看其他被增强的图像。但请保持索引值在 0 到 31 之间，以考虑到 32 个图像的批次大小。

### 创建模型

清除之前的模型并生成一个种子以实现可重复性：

```py
import numpy as np
tf.keras.backend.clear_session()
np.random.seed(0)
tf.random.set_seed(0)
```

导入数据建模所需的库：

```py
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.layers import Conv2D, MaxPooling2D
```

按照图 2-3 所示构建多层 CNN。

```py
model = tf.keras.Sequential([
data_augmentation,
Conv2D(32, 3, activation='relu'),
MaxPooling2D(),
Conv2D(32, 3, activation='relu'),
MaxPooling2D(),
Conv2D(32, 3, activation='relu'),
MaxPooling2D(),
Flatten(),
Dense(128, activation='relu'),
Dropout(0.5),
Dense(num)
])
Listing 2-3
Multilayered CNN
```

注意，第一层就是我们构建的数据增强！

### 编译和训练模型

使用 SparseCategoricalCrossentropy(from_logits=True) 编译：

```py
model.compile(
optimizer='adam',
loss=SparseCategoricalCrossentropy(from_logits=True),
metrics=['accuracy'])
```

我们在损失函数中设置 *from_logits=True*，因为我们不使用 softmax 激活来归一化神经元的输入到模型的输出层。Softmax 通过取每个输出的指数并按这些指数的总和归一化每个数字，将 logits 转换为概率，使得整个输出向量的和为 1。所以所有概率的总和应该为 1。**Logits**是多类分类神经网络最后线性层的数值输出。由于我们不使用 softmax，我们通知编译器输出 logits。

训练模型：

```py
history = model.fit(
train_fds,
validation_data=test_fds,
epochs=10)
```

与我们在前一章中没有增强时遇到的情况相比，过度拟合得到了缓解。也就是说，测试准确率与训练准确率更加接近。

### 可视化性能

让我们通过图示来观察数据增强的影响，如图 2-4 所示。

```py
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(10)
plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')
plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
Listing 2-4
Visualize Training Performance with Augmentation
```

注意，对于训练了十个周期的模型，过度拟合在很大程度上得到了缓解。因此，数据增强提高了性能。

## 在图像上应用增强

在上一节中，我们向神经网络添加了 Keras 预处理层以进行增强。另一种方法是直接在图像上应用数据增强变换，然后将它们馈送到没有 Keras 预处理层的神经网络。

在本节中，我们首先演示了可以在图像上执行的各种变换。我们从上一节中创建的预处理张量中抓取一批数据，并演示每种变换技术。在演示了如何在图像上应用各种变换后，我们继续使用这些技术中的几个对图像进行操作，并使用这些增强图像训练模型。

让我们从抓取我们刚刚创建的训练集中的一个示例图像开始：

```py
for batch_images, _ in train_fds.take(1):
print ('image shape:', batch_images.shape)
```

由于我们已经在上一节中有了可用的预处理的张量，我们从流水线训练集中抓取了这些图像的第一批次。现在，我们有一个包含在 *batch_images* 中的 32 张图像的批次，我们可以用这些图像进行操作。我们首先向您展示如何处理批次中的图像。

### 设置索引变量

由于批次大小为 32，将索引值设置为 0 到 31 之间的值：

```py
indx = 0
indx
```

我们将索引设置为 0 以获取批次中的第一张图像。请随意更改索引值。但请注意，索引值必须在 0 到 31 之间！

### 设置一个图像

设置索引以从批次中抓取第一张图像。更改索引值（介于 0 到 31 之间）以显示不同的图像。

我们的图像：

```py
our_image = batch_images[indx]
```

### 展示一个示例

可视化第一张图像：

```py
plt.imshow(our_image)
plt.axis('off')
plt.grid(b=None)
```

### 创建显示图像的函数

创建一个可视化函数来显示原始图像和修改后的图像，如图 2-5 所示。

```py
def show(original_img, trans_img):
f = plt.figure(figsize=(6, 6))
f.add_subplot(1,2,1)
plt.imshow(original_img)
plt.axis('off')
f.add_subplot(1,2,2)
plt.imshow(trans_img)
plt.axis('off')
plt.show(block=True)
Listing 2-5
Function to Visualize Original Image and Modified Image
```

创建一个可视化函数来显示图像的几种变换，如列表 2-6 所示。

```py
def show_images(img, indx, trans, p1=None, p2=None, b=False):
plt.figure(figsize=(10, 10))
for i in range(9):
ax = plt.subplot(3, 3, i + 1)
if not b:
new_img = trans(img[indx])
elif p2==None:
new_img = trans(img[indx], p1)
new_img = np.clip(new_img, 0, 1)
else:
new_img = trans(img[indx], p1, p2)
new_img = np.clip(new_img, 0, 1)
plt.imshow(new_img)
plt.axis('off')
Listing 2-6
Visualize Transformations of an Image
```

### 裁剪图像

要裁剪图像，请移除或调整其外部边缘。

裁剪图像：

```py
new_image = tf.image.random_crop(our_image, [120, 120, 3])
new_image.shape
```

此操作从图像中均匀选择的偏移量中切出一个 *形状大小* 部分。在这种情况下，新图像的大小是 120 × 120。

显示原始图像和修改后的图像：

```py
show(our_image, new_image)
```

由于变换是随机的，显示多个图像：

```py
show_images(batch_images, indx, tf.image.random_crop,
[120, 120, 3], b=True)
```

我们使用此可视化显示了随机裁剪。请注意，由于 API 随机生成裁剪，每个图像的裁剪方式略有不同。

中心裁剪图像：

```py
new_image = tf.image.central_crop(our_image, 0.5)
print (new_image.shape)
show(our_image, new_image)
```

此操作将图像大小减半并中心裁剪。也就是说，它移除了背景噪声，并确保剩余的图像像素居中。

裁剪到边界框：

```py
new_image = tf.image.crop_to_bounding_box(
our_image, 10, 10, 120, 120)
print (new_image.shape)
show(our_image, new_image)
```

此操作将图像裁剪到指定的边界框。

**边界框**是像素图像中的一个区域，由两个经度和两个纬度定义。每个纬度是介于 –90.0 和 90.0 之间的十进制值。每个经度是介于 –180.0 和 180.0 之间的十进制值。

### 随机翻转图像左右

由于变换是随机的，图像不总是左右翻转：

```py
show_images(batch_images, indx, tf.image.random_flip_left_right)
```

### 随机翻转图像上下

由于变换是随机的，图像不总是上下翻转：

```py
show_images(batch_images, indx, tf.image.random_flip_up_down)
```

### 将图像上下翻转

图像总是上下翻转：

```py
show(our_image, tf.image.flip_up_down(our_image))
```

### 旋转图像 90 度

通过设置 *k=1* 旋转图像 90°：

```py
show(our_image, tf.image.rot90(our_image, k=1))
```

*k* 参数是图像旋转 90 度的次数。

通过设置 *k=2* 旋转图像 180°：

```py
show(our_image, tf.image.rot90(our_image, k=2))
```

通过设置 k=3 旋转图像 270°：

```py
show(our_image, tf.image.rot90(our_image, k=3))
```

如果我们设置 k=4，我们就回到了起点，因为这将图像旋转 360°！

### 调整伽马

**伽马编码**通过利用人类对光和颜色的非线性感知方式来优化编码图像时位的使用。在常见的照明条件下（既不是漆黑一片也不是刺眼明亮），人类对亮度的感知（或亮度）遵循一个近似幂函数，对较暗色调之间的相对差异比较亮色调之间的相对差异更敏感。简单来说，伽马编码可以被视为图像的强度。

使用伽马编码调整图像：

```py
new_image = tf.image.adjust_gamma(
our_image, gamma=0.75, gain=1.5)
new_image = np.clip(new_image, 0, 1)
show(our_image, new_image)
```

调整伽马以调整亮度。伽马小于 1 时，图像更亮。伽马大于 1 时，图像更暗。强度由 *增益* 参数控制。通过实验 *伽马* 和 *增益* 参数来观察它们对变换图像的影响。

注意

将伽马和增益值视为可以调整以修改图像外观的旋钮。

### 调整对比度

**对比度**在图像处理中是指使物体可区分的亮度（或颜色）差异。它取决于物体及其在同一视野内其他物体的颜色和亮度差异。

固定对比度：

```py
new_image = tf.image.adjust_contrast(
our_image, contrast_factor=1.8)
new_image = np.clip(new_image, 0, 1)
show(our_image, new_image)
```

调整 *contrast_factor* 参数以增加或减少亮度。通过实验 *contrast_factor* 参数来观察它对变换图像的影响。

随机对比度：

```py
show_images(batch_images, indx, tf.image.random_contrast,
0.75, 2.9, b=True)
```

调整上下限以改变随机对比度边界。

### 调整亮度

固定亮度：

```py
new_image = tf.image.adjust_brightness(our_image, .25)
new_image = np.clip(new_image, 0, 1)
show(our_image, new_image)
```

调整 delta 值以增加图像像素的值。因此，delta 为 0.25 会给图像增加 25% 的亮度。

随机亮度：

```py
show_images(batch_images, indx, tf.image.random_brightness,
0.25, b=True)
```

### 调整饱和度

**图像处理中的饱和度**是图像中存在的颜色的深度或强度。图像越饱和，看起来就越多彩和生动。较低的饱和度会使图像看起来压抑或暗淡。

固定饱和度：

```py
show(our_image, tf.image.adjust_saturation(our_image, 3.0))
```

调整饱和度因子以增加饱和度。因此，饱和度因子为 3.0 会将图像的饱和度增加三倍。

随机饱和度：

```py
show_images(batch_images, indx, tf.image.random_saturation,
0.3, 3.5, b=True)
```

调整上下限以改变随机饱和度边界。

### 色调

**色调**是 RGB 颜色的主要指示。它是告诉我们物体是红色、绿色还是蓝色的值。相比之下，饱和度是感知的强度。因此，饱和度是物体看起来有多色彩，而色调是实际的颜色。

固定色调：

```py
show(our_image, tf.image.adjust_hue(our_image, 0.2))
```

随机色调：

```py
show_images(batch_images, indx, tf.image.random_hue, 0.2, b=True)
```

调整 0.2 参数值以查看其对转换图像的影响。

要查看所有可能的数据增强转换，请查阅

[TensorFlow API 文档](http://www.tensorflow.org/api_docs/python/tf/image/)

对于一个可以直接在图像上应用增强的一般网站，请查阅

[TensorFlow 在 GPU 上的图像增强](https://towardsdatascience.com/tensorflow-image-augmentation-on-gpu-bf0eaac4c967)

### 直接在图像上应用转换

现在我们已经讨论了几个潜在的转换，让我们看看我们是否可以提高训练性能。我们只应用了几个转换，效果相当不错，但你可以尝试你想要的任何数量。

#### 创建一个增强函数

创建一个函数，随机翻转图像从左到右，并像清单 2-7 中所示对图像应用饱和度操作。

```py
def augment(image, label):
img = tf.image.random_flip_left_right(image)
final_image = tf.image.random_saturation(img, 0, 2)
return (final_image, label)
Listing 2-7
Augmentation Function
```

由于我们找到了很多转换选项和很少的指导，因此很难找到正确的组合。我们尝试了许多变体，但前面的简单一个对我们来说效果最好。我们鼓励你尝试不同的转换，看看你是否能提高学习效果。

#### 显示增强图像

如清单 2-8 所示，当 *augment* 函数多次应用于同一图像时，会发生以下情况。

```py
plt.figure(figsize=(10, 10))
for i in range(9):
image, _ = augment(our_image, labels[0])
ax = plt.subplot(3, 3, i + 1)
plt.imshow(image)
plt.axis('off')
Listing 2-8
Plot Transformations for an Image
```

#### 构建输入管道

构建训练和测试数据的管道：

```py
SHUFFLE_SIZE = 100
train1 = train_ds.map(scale, num_parallel_calls=4)
train2 = train1.map(augment, num_parallel_calls=4)
train_da = train2.shuffle(SHUFFLE_SIZE).cache().prefetch(1)
test_da = test_ds.map(scale).cache().prefetch(1)
```

注意，我们只将增强函数映射到训练数据。我们添加了 *num_parallel* 参数以在训练期间提高性能。

注意

仅增强训练图像！训练数据被呈现给模型，以帮助模型变得更加通用和健壮。测试数据作为新数据呈现，以帮助评估模型。

#### 创建模型

清除所有之前的模型并为可重复性生成一个种子：

```py
tf.keras.backend.clear_session()
np.random.seed(0)
tf.random.set_seed(0)
```

创建一个如清单 2-9 所示的多层 CNN。

```py
model = tf.keras.Sequential([
Conv2D(32, 3, activation='relu',
input_shape=[180, 180, 3]),
MaxPooling2D(),
Conv2D(32, 3, activation='relu'),
MaxPooling2D(),
Conv2D(32, 3, activation='relu'),
MaxPooling2D(),
Flatten(),
Dense(128, activation='relu'),
Dropout(0.5),
Dense(num)
])
Listing 2-9
Multilayer CNN
```

注意，该模型没有像上一节中那样的 Keras 预处理层。由于我们是直接增强图像，因此不需要添加 Keras 层。

#### 编译和训练模型

使用 SparseCategoricalCrossentropy(from_logits=True)进行编译：

```py
model.compile(
optimizer='adam',
loss=SparseCategoricalCrossentropy(from_logits=True),
metrics=['accuracy'])
```

训练模型：

```py
history = model.fit(
train_da,
validation_data=test_da,
epochs=5)
```

哇！我们减轻了过拟合！我们使用了五个周期，但你可以替换你想要的任何周期数。更多的周期需要更多的内存。由于我们的数据集不是很大，内存实际上并不是一个问题。但是，对于大型数据集，如果设置周期数过高，内存可能会成为一个问题。

## 使用 ImageGenerator 进行数据增强

到目前为止，我们已经介绍了两种增强技术。第一种技术向模型添加了 Keras 预处理层。第二种技术直接对图像进行变换，然后将它们输入到模型中。两种技术都构建了 tf.data 管道以供模型使用。

另一个选择是使用 ImageGenerator 类。*ImageGenerator 类*使从磁盘加载图像并以各种方式增强它们变得非常容易。我们可以平移、旋转、缩放、水平或垂直翻转、剪切或对图像应用变换函数。虽然 ImageGenerator 对于简单的项目非常方便，但构建 tf.data 管道更有利于复杂项目，因为它可以从任何来源（而不仅仅是本地磁盘）并行读取图像，并以任何方式操作数据集。此外，基于 tf.image 操作的预处理函数可以在 tf.data 管道和部署到生产的模型中使用。

我们不提倡一种方法优于另一种方法。这取决于手头的任务。ImageGenerator 类非常易于使用。因此，它应该适用于项目的初期阶段。tf.data 管道针对并行处理进行了优化，因此对于需要大量计算资源的持续项目来说，它是一个不错的选择。

关于这个主题的宝贵资源，请参阅

[`www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator`](https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator)

### 处理花卉数据

该过程比之前的技术更简单。通过利用字典，直接从 flowers 目录加载数据到训练和测试分割。第一个字典为 ImageGenerator 类中的 ImageDataGenerator 方法的缩放和分割信息提供信息。第二个字典为方法提供目标和批量信息。

导入适当的库：

```py
from tensorflow.keras.preprocessing.image\
import ImageDataGenerator
```

创建一个字典以缩放和分割数据：

```py
datagen_kwargs = dict(rescale=1./255, validation_split=.19)
```

数据分为 81%用于训练，19%用于测试。

创建一个字典以调整图像大小、设置批量大小和插值：

```py
BATCH_SIZE = 32
IMAGE_SIZE = (180, 180)
dataflow_kwargs = dict(target_size=IMAGE_SIZE,
batch_size=BATCH_SIZE,
interpolation='bilinear')
```

### 创建数据集

现在目录设置正确后，我们使用 Keras API 创建训练集和测试集。测试集首先基于第一个字典的构建创建。

创建测试集：

```py
valid_datagen = tf.keras.preprocessing.image.\
ImageDataGenerator(**datagen_kwargs)
valid_generator = valid_datagen.flow_from_directory(
data_dir, subset='validation', shuffle=False,
**dataflow_kwargs)
```

创建训练集：

```py
train_datagen = valid_datagen
train_generator = train_datagen.flow_from_directory(
data_dir, subset='training', shuffle=True,
**dataflow_kwargs)
```

注意，我们只对训练数据进行打乱。

检查张量：

```py
valid_datagen, train_datagen
```

两个张量都是 ImageDataGenerator 对象。因此，张量已准备好进行训练，因为该类负责所有预处理！

### 创建模型

清除之前的模型并生成一个种子：

```py
tf.keras.backend.clear_session()
np.random.seed(0)
tf.random.set_seed(0)
```

创建一个多层 CNN，如列表 2-10 所示。

```py
model = Sequential([
Conv2D(filters=32, kernel_size=(5, 5), activation = 'relu',
input_shape=(180, 180, 3)),
MaxPooling2D(2),
Conv2D(64, (5, 5), activation='relu'),
MaxPooling2D(2),
Flatten(),
Dense(64, activation='relu'),
Dense(5, activation='softmax')
])
Listing 2-10
Multilayer CNN for ImageDataGenerator Objects
```

我们本可以使用与上一节相同的模型，但我们使用了一个稍微不同的模型来观察会发生什么。我们使用了一个更少的卷积层，并将 softmax 应用于模型输出。

### 编译和训练模型

编译模型：

```py
model.compile(optimizer='adam',
loss='categorical_crossentropy',
metrics=['accuracy'])
```

由于模型输出层的神经元应用了 *softmax* 激活，我们在损失度量中 **不** 使用 *from_logits=True* 进行编译。Softmax 为多类问题中的每个类别分配小数概率，这些概率必须加起来为 1.0。添加这个额外的约束可以帮助训练比其他方式更快地收敛。

训练模型：

```py
history = model.fit(train_generator, batch_size=BATCH_SIZE,
epochs=5, validation_data=valid_generator,
verbose=1)
```

在没有数据增强的情况下，仅仅经过几个 epoch 后，过拟合就开始真正地发生了。

### 增强训练数据

让我们看看使用 ImageDataGenerator 在训练数据上创建数据增强是否可以提高性能：

```py
aug_train_datagen = tf.keras.preprocessing.\
image.ImageDataGenerator(
rotation_range=40, horizontal_flip=True,
width_shift_range=0.2, height_shift_range=0.2,
shear_range=0.2, zoom_range=0.2,
**datagen_kwargs)
```

旋转和翻转图像。以及平移、剪切和缩放图像。由于只有训练数据用于学习，我们不会增强验证（或测试）数据。在这个例子中，我们包含了多个参数值。我们进行了相当多的实验来得到这些值。我们强烈建议您进行自己的实验。

创建带有图像转换的训练集：

```py
aug_train_generator = aug_train_datagen.flow_from_directory(
data_dir, subset='training', shuffle=True, **dataflow_kwargs)
```

检查：

```py
aug_train_generator
```

张量是一个 *DirectoryIterator* 对象。

### 重新编译和训练模型

由于我们只是增强了训练数据，我们必须重新训练模型。因此，首先清除之前的模型会话并生成一个种子以实现可重复性。然后继续重新编译模型。

清除模型并生成一个种子：

```py
tf.keras.backend.clear_session()
np.random.seed(0)
tf.random.set_seed(0)
```

重新编译模型：

```py
model.compile(optimizer='adam',
loss='categorical_crossentropy',
metrics=['accuracy'])
```

我们重新编译模型，因为我们正在使用新的数据集进行训练。

训练模型：

```py
history = model.fit(aug_train_generator, batch_size=BATCH_SIZE,
epochs=5, validation_data=valid_generator,
verbose=1)
```

模型表现出更少的过拟合！

### 检查数据

ImageDataGenerator 数据与 tf.data 不同。因此，让我们探索它。首先处理原始训练集的一个批次和增强集的一个批次。尽管使用该类很简单，但要探索数据需要做更多的工作。

从原始训练集中获取一个批次，如列表 2-11 所示。

```py
data_list = []
batch_index, end_index = 0, 1
while batch_index <= train_generator.batch_index:
if batch_index < end_index:
data = train_generator.next()
data_list.append(data[0])
batch_index = batch_index + 1
else: break
original = np.asarray(data_list)
Listing 2-11
Process a Batch from the Original Training Set
```

要遍历一个 ImageDataGenerator 对象，使用数据集名称和起始索引值（分配给变量）作为方法。由于我们只想获取一个批次，将结束索引值设置为 1。使用 *next()* 方法获取第一个元素，将其添加到列表中并增加批次索引。在第一次迭代后中断循环。最后，将列表转换为 NumPy 数组。

检查形状：

```py
original[0].shape
```

如预期，形状是 (32, 180, 180, 3)。因此，每个批次包含 32 个 180 × 180 × 3 的彩色图像。

验证我们获取了一个批次：

```py
print ('We grabbed', len(original), 'batch.')
```

从增强后的训练集中获取一个批次，如列表 2-12 所示。

```py
data_list = []
batch_index, end_index = 0, 1
while batch_index <= aug_train_generator.batch_index:
if batch_index < end_index:
data = aug_train_generator.next()
data_list.append(data[0])
batch_index = batch_index + 1
else: break
augmented = np.asarray(data_list)
Listing 2-12
Process a Batch from the Augmented Training Set
```

检查形状：

```py
augmented[0].shape
```

如预期，形状是 (32, 180, 180, 3)。因此，每个批次包含 32 张 180 × 180 × 3 的彩色图像。

验证我们是否抓取了一个批次：

```py
print ('We grabbed', len(augmented), 'batch.')
```

### 可视化

从原始训练集中抓取第一张图像：

```py
train_image = original[0][0]
```

可视化图像：

```py
plt.imshow(train_image)
plt.axis('off')
plt.grid(b=None)
```

我们看到一张正常的鲜花图像。

将原始训练集中的图像可视化，如图 2-13 所示。

```py
plt.figure(figsize=(10, 10))
for images in original:
for i in range(9):
ax = plt.subplot(3, 3, i + 1)
plt.imshow(images[i])
plt.axis('off')
Listing 2-13
Visualize Several Original Training Images
```

我们看到几张正常的鲜花图像。

从增强后的训练集中抓取第一张图像：

```py
aug_train_image = augmented[0][0]
```

可视化增强后的图像：

```py
plt.imshow(aug_train_image)
plt.axis('off')
plt.grid(b=None)
```

我们看到一张增强后的图像。

将增强后的训练集中的图像可视化，如图 2-14 所示。

```py
plt.figure(figsize=(10, 10))
for images in augmented:
for i in range(9):
ax = plt.subplot(3, 3, i + 1)
plt.imshow(images[i])
plt.axis('off')
Listing 2-14
Visualize Several Augmented Training Images
```

我们看到几张增强后的鲜花图像。由于我们希望为模型提供新的数据，增强数据看起来与原始数据不同。如果我们给模型提供原始数据的副本，性能不会得到提升。但增强数据为模型提供了原始数据。

## 摘要

我们展示了使用三种不同技术进行数据增强。Keras 技术实现起来相当简单，但不如直接在图像上应用增强那样灵活。ImageGenerator 技术实现起来最简单，但仅限于较小的项目。
