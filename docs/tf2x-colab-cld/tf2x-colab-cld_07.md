# 7. 卷积神经网络

使用前馈神经网络，我们在 MNIST 和 Fashion-MNIST 数据集上实现了良好的训练性能。但是，这些数据集中的图像简单且位于包含它们的输入空间中心。也就是说，它们位于包含它们的像素矩阵中心。**输入空间**是模型的所有可能输入。

前馈神经网络在识别模式方面非常出色。因此，如果图像在它们的输入空间中占据相同的位置，前馈网络可以快速有效地识别图像模式。而且，如果图像在像素数量方面简单，模式更容易出现。但是，如果图像在它们的输入空间中不占据相同的位置，前馈网络在识别模式方面有很大困难，因此表现得很糟糕！因此，我们需要一个不同的模型来处理这些类型的图像。

我们可以使用卷积神经网络训练复杂且偏离中心的图像，并获得良好的结果。一个**卷积神经网络**（CNN 或 ConvNet）是一类深度神经网络，最常应用于视觉图像分析。CNN 受到其与生物过程的相似性的启发，即神经元之间的连接类似于人类视觉皮层的组织。

CNN 的工作方式与前馈网络不同，因为它将数据视为空间性的。神经元不是与前一层的每个神经元连接，而是只与它们附近的神经元连接，并且所有这些神经元都有相同的权重。连接的简化意味着网络保持了数据集的空间特性。

假设图像是孩子的面部轮廓。CNN 不会认为孩子的眼睛在整张图像中重复出现。由于它所执行的过滤过程，CNN 能够高效地在图像中定位孩子的眼睛。

CNN 通过为每张图像中的各种元素分配重要性来处理图像数据集，这使得它能够区分不同的图像。重要性是通过可学习的权重和偏差进行校准的。与其它分类算法相比，预处理要低得多，因为 CNN 有能力在训练过程中学习如何调整其滤波器。

CNN 的核心构建块是卷积层。一个**卷积层**包含一系列滤波器，通过提取特征并将它们转换为特征图供网络中的下一层使用，从而将输入图像进行转换。这种转换是通过与一组具有小感受野的可学习滤波器（或卷积核）进行卷积来实现的。

**卷积**通过使用输入数据的小方块学习图像特征来保留像素之间的关系。**卷积核**是用于输入图像像素子集的滤波器。因此，卷积核是学习图像特征的小方块输入数据之一。**感受野**是在特定时间点卷积核操作的图像部分。CNN 的**特征图**捕获了将卷积核应用于输入图像的结果。因此，单个神经元仅在感受野（或视觉场）的受限区域内对刺激做出反应。

呼吸！简单来说，卷积核是一个比要卷积的图像高度和宽度都小的矩阵。在训练过程中，核在整个输入图像的高度和宽度上滑动，并在图像的每个空间位置计算核和图像的点积。这些计算生成了特征图作为输出。所以整个图像都是由卷积核卷积的！这种卷积是 CNN 效率的关键，因为过滤过程允许它在训练过程中调整滤波器参数。

我们首先讨论 CNN 架构。我们从一些示例图像开始，帮助你理解我们正在处理的数据类型。然后我们构建一个完整的 CNN 实验。我们使用著名的 *cifar10* 数据集。这个数据集包含 60,000 张图像，旨在让深度学习爱好者创建和测试深度学习模型。我们展示了如何加载数据，构建输入管道，并建模数据。我们还展示了如何进行预测。

各章节的笔记本位于以下 URL：[`https://github.com/paperd/tensorflow`](https://github.com/paperd/tensorflow)。

启用 GPU（如果尚未启用）：

1.  在右上角菜单中点击**运行时**。

1.  从下拉菜单中选择**更改运行时类型**。

1.  从**硬件加速器**下拉菜单中选择**GPU**。

1.  点击**保存**。

测试 GPU 是否激活：

```py
import tensorflow as tf
# display tf version and test if GPU is active
tf.__version__, tf.test.gpu_device_name()
```

导入 *tensorflow* 库。如果显示‘/device:GPU:0’，则 GPU 已激活。如果显示‘..’，则常规 CPU 已激活。

## CNN 架构

与前馈神经网络一样，CNN 由多个层组成。然而，卷积层和池化层使其独特。像其他神经网络一样，它也有一个 ReLU（修正线性单元）层和一个全连接层。任何神经网络中的 ReLU 层都充当激活函数，确保数据通过网络的每一层时保持非线性。如果没有 ReLU 激活，输入到每一层的数据将失去我们希望其保持的维度。也就是说，当数据通过网络时，我们将失去原始数据的完整性。全连接层允许 CNN 对数据进行分类。

如前所述，卷积神经网络（CNN）最重要的构建块是卷积层。第一层的神经元与输入图像中的每个像素相连，但仅限于它们的感觉域，即仅限于它们附近的像素。卷积层通过在图像像素数组上放置一个滤波器（或卷积核）来工作。滤波过程创建了一个*卷积特征图*，这是卷积层的输出。

*特征图*是通过将输入特征投影到隐藏单元以形成新特征并馈送到下一层而创建的。*隐藏单元*对应于输入体积中单个特定 x/y 偏移处的单个滤波器的输出。简单地说，隐藏单元是输出体积中特定 x,y,z 坐标的值。

一旦我们有了卷积特征图，我们就转向池化层。*池化层*对特定特征图进行子采样。子采样缩小了输入图像的大小，以减少计算负载、内存使用和参数数量。减少网络需要处理的参数数量也限制了过拟合的风险。池化层的输出是一个*池化特征图*。

我们可以通过两种方式对特征图进行池化。*最大池化*取特定卷积特征图的最大输入。*平均池化*取特定卷积特征图的平均输入。

创建池化特征图的过程导致特征提取，这使得网络能够构建图像数据的图像。有了图像数据的图像，网络进入全连接层进行分类。正如我们在前馈网络中所做的那样，我们展平数据以便全连接层消费，因为它只能处理线性数据。

从我们的讨论中可以看出，从概念上讲，CNN 相当复杂。但在 TensorFlow 中实现 CNN 相当直接。每个输入图像通常表示为一个形状为*高度*、*宽度*和*通道数*的 3D 张量。在分类 3D 彩色图像时，我们通过 CNN 图像数据的三通道，即*红色*、*绿色*和*蓝色*。彩色图像通常被称为*RGB*图像。一个批次（例如，小批量）表示为一个形状为*批次大小*、*高度*、*宽度*和*通道数*的 4D 张量。

## 加载样本图像

scikit-learn 的*load_sample_image*方法允许我们使用两个彩色图像进行练习 - *china.jpg*和*flower.jpg*。该方法加载单个样本图像的 numpy 数组，并将其作为高度、宽度和颜色组成的 3D numpy 数组返回。

加载图像：

```py
from sklearn.datasets import load_sample_image
china, flower = load_sample_image('china.jpg'),\
load_sample_image('flower.jpg')
china.shape, flower.shape
```

中国和花卉图像都表示为具有三个通道的 427 `×` 640 像素矩阵，以考虑 RGB 颜色。

## 显示图像

列表 7-1 展示了图像。

```py
import matplotlib.pyplot as plt
# function to plot RGB images
def plot_color_image(image):
plt.imshow(image, interpolation="nearest")
plt.axis("off")
plot_color_image(china)
plt.show()
plot_color_image(flower)
Listing 7-1
Display china and flower images
```

## 缩放图像

缩放图像可以提高训练性能。由于每个图像像素由 0 到 255 的字节表示，我们将每个图像除以 255 以进行缩放。

列表 7-2 缩放图片。

```py
import numpy as np
# slice off a few pixels prior to scaling
br = '\n'
print ('pixels as loaded:', br)
print ('china pixels:', end = '  ')
print (np.around(china[0][0]))
print ('flower pixels:', end = ' ')
print (np.around(flower[0][0]), br)
# scale images
china_sc, flower_sc = china / 255., flower / 255.
# slice off some pixels to verify that scaling worked
print ('pixels scaled:', br)
print ('china pixels:', end = '  ')
print (np.around(china_sc[0][0], decimals=3))
print ('flower pixels:', end = ' ')
print (np.around(flower_sc[0][0], decimals=3))
Listing 7-2
Scale images
```

缩放成功，因为像素强度介于 0 和 1 之间。

## 显示缩放后的图片

绘制缩放后的图片：

```py
plot_color_image(china_sc)
plt.show()
plot_color_image(flower_sc)
```

缩放不会影响图片，这是有道理的，因为缩放按比例修改像素强度。也就是说，每个像素值按比例转换为介于 0 和 1 之间的数值。

## 获取更多图片

让我们再获取一些图片。要获取本书的图片，只需遵循以下简单步骤：

1.  前往本书的 GitHub URL：[`https://github.com/paperd/tensorflow`](https://github.com/paperd/tensorflow)。

1.  定位您想要下载的图片并点击它。

1.  点击 *下载* 按钮。

1.  在图片内部任意位置右键点击。

1.  点击 *另存为…*。

1.  在您的计算机上保存图片。

1.  将图片拖放到您的 Google Drive *Colab 笔记本* 文件夹中。

1.  如有必要，重复步骤 1–7 以处理多张图片。

对于本课，前往本书的 URL，点击 *chapter7*，点击 *images*，点击 *fish.jpg*，点击 *下载* 按钮，在图片内部右键点击，并点击 *另存为…* 以将其保存到您的计算机上。将图片拖放到您的 Google Drive *Colab 笔记本* 文件夹中。为 *happy_moon.jpg* 图片重复相同的步骤。

![img/501128_1_En_7_Chapter/501128_1_En_7_Figa_HTML.jpg](img/501128_1_En_7_Figa_HTML.jpg)![img/501128_1_En_7_Chapter/501128_1_En_7_Figb_HTML.jpg](img/501128_1_En_7_Figb_HTML.jpg)

## 挂载 Google Drive

将 Colab 挂载到 Google Drive：

```py
from google.colab import drive
drive.mount('/content/drive')
```

点击 URL，选择一个 Google 账户，点击 *允许*，复制授权代码并将其粘贴到 Colab 中的文本框 *输入您的授权代码:*，然后按键盘上的 *Enter* 键。

## 将图片复制到 Google Drive

在执行本节中的代码之前，请确保您在 Google Drive 的 *Colab 笔记本* 目录中已有 *fish.jpg* 和 *happy_moon.jpg* 图片！

检查您的 Google Drive 账户以验证正确的路径。我们已将图片保存到 Colab 笔记本目录中，这是推荐的。如果您将它们保存在其他地方，您必须相应地更改路径。

列表 7-3 将图片加载到 Colab 环境中，缩放图片并显示它们。

```py
# be sure to copy images to the directory on Google Drive
from PIL import Image
import numpy as np
# create paths to images
fish_path = 'drive/My Drive/Colab Notebooks/fish.jpg'
moon_path = 'drive/My Drive/Colab Notebooks/happy_moon.jpg'
# create images
fish, moon  = Image.open(fish_path), Image.open(moon_path)
# convert images to numpy arrays and scale
fish_np, moon_np = np.array(fish), np.array(moon)
fish_sc, moon_sc = fish_np / 255., moon_np / 255.
# display images
plot_color_image(fish_sc)
plt.show()
plot_color_image(moon_sc)
Listing 7-3
Load, scale, and display images
```

确认新图片已正确缩放：

```py
# slice off some pixels and display
print ('fish pixels:', end = ' ')
print (np.around(fish_sc[0][0], decimals=3), br)
print ('moon pixels:', end = ' ')
print (np.around(moon_sc[0][0], decimals=3))
```

到目前为止一切顺利。

## 检查图片形状

对于机器学习应用，图片必须是 *相同* 的形状。

让我们探索图片形状：

```py
print ('original shapes:')
display (china_sc.shape, flower_sc.shape)
print (), print ('new shapes:')
display (fish_sc.shape, moon_sc.shape)
```

哎呀！形状不相同！我们该怎么办？

## 调整图片大小

让我们将鱼和月亮图片调整大小以使形状相等：

```py
fish_rs = np.array(tf.image.resize(
fish_sc, [427, 640]))
moon_rs = np.array(tf.image.resize(
moon_sc, [427, 640]))
fish_rs.shape, moon_rs.shape
```

现在，所有四张图片的大小为 (427, 640, 3)。

绘制调整大小后的图片：

```py
plot_color_image(fish_rs)
plt.show()
plot_color_image(moon_rs)
```

成功！我们已经将新图片的大小调整为与原始图片相匹配。

## 创建图片批次

创建包含所有四张图片的批次：

```py
new_images = np.array([china_sc, flower_sc,
fish_rs, moon_rs])
new_images.shape
```

现在，我们有一批 *四* 张 427 `×` 640 的彩色图片。RGB 颜色由 3 个维度表示。

## 创建过滤器

让我们创建两个简单的 7 `×` 7 过滤器。我们希望我们的第一个过滤器中间有一条垂直白线，而第二个过滤器中间有一条水平白线。*过滤器* 用于在卷积过程中从图像中提取特征。通常，过滤器被称为 *卷积核*。

创建过滤器：

```py
# assign some variables
batch_size, height, width, channels = new_images.shape
# create 2 filters
ck = np.zeros(shape=(7, 7, channels, 2), dtype=np.float32)
ck.shape
```

*zeros* 方法返回一个给定形状和类型且填充为零的数组。由于变量 *ck* 被填充为零，其所有像素都是黑色。记住，像素图像值是介于 0（黑色）到 255（白色）之间的整数。

因此，*ck* 是一个包含 *两个* 7 `×` 7 卷积核且具有三个通道的 4D 张量。过滤器必须具有三个通道，以匹配我们创建的图像批次的彩色图像。

添加一条垂直白线和一条水平白线：

```py
ck[:, 3, :, 0] = 1  # add vertical line
ck[3, :, :, 1] = 1  # add horizontal line
```

代码将选定像素的强度改变以获得一条垂直的白线和一条水平的白线。

## 绘制卷积核

列表 7-4 绘制了我们刚刚创建的两个卷积核。

```py
# function to plot filters
def plot_image(image):
plt.imshow(image, cmap="gray", interpolation="nearest")
plt.axis("off")
print ('vertical convolutional kernels:')
plot_image(ck[:, :, 0, 0])
plt.show()
print ('horizontal convolutional kernels:')
plot_image(ck[:, :, 0, 1])
Listing 7-4
Convolutional kernel plots
```

我们看到垂直和水平白线（或卷积核）处于正确的位置。因此，我们已经成功创建了两个简单的卷积核。

## 应用 2D 卷积层

将 2D 卷积层应用于图像批次：

```py
# apply a 2D convolutional layer
outputs = tf.nn.conv2d(new_images, ck, strides=1,
padding='SAME')
```

*tf.nn.conv2d* 方法计算给定 4D 输入和卷积核张量的 2D 卷积。我们将步长设置为 *1*。**步长** 是我们在训练过程中将卷积核在输入矩阵上移动的像素数。步长为 1 时，我们一次移动一个像素。我们将填充设置为 *SAME*。**填充** 是在 CNN 处理图像时添加到图像中的像素数。例如，如果填充设置为零，则添加的每个像素值都将为零。将填充设置为 SAME 意味着我们将使用零填充。

在应用卷积层之后，变量 *outputs* 包含基于我们图像的特征图。由于每个卷积核创建一个特征图（并且我们有两个卷积核），每个图像都有两个特征图。

## 可视化特征图

如列表 7-5 所示，可视化我们刚刚创建的特征图。

```py
rows = 4  # one row for each image
columns = 2  # two feature maps for each image
cnt = 1
fig = plt.figure(figsize=(8, 8))
for i, img in enumerate(outputs):
for j in (0, 1):
fig.add_subplot(rows, columns, cnt)
plt.imshow(outputs[i, :, :, j], cmap="gray")
plt.axis('off')
cnt += 1
plt.show()
Listing 7-5
Feature maps plot
```

由于我们有两个卷积核和四个图像，卷积层产生了八个特征图。只需将 2 乘以 4！因此，每个图像有两个特征图。哇！通过应用单个卷积层，我们能够使用两个简单的卷积核提取出我们图像批次的优秀复制品。

## 带有可训练过滤器的 CNN

我们刚刚 *手动* 定义了两个卷积核。但在一个真实的 CNN 中，我们通常将卷积核定义为可训练变量，以便神经网络可以学习到最佳的卷积核。

创建一个简单的模型，让网络决定最佳的卷积核：

```py
conv = tf.keras.layers.Conv2D(filters=32, kernel_size=3,
strides=1, padding="SAME",
activation='relu')
```

我们创建了一个带有 32 个卷积核的*Conv2D*层。每个卷积核是一个由*kernel_size*指定的 3 `×` 3 张量。我们在水平和垂直方向上使用步长*1*。填充为*SAME*。最后，我们对输出应用*relu*激活函数。

卷积层有相当多的超参数，包括过滤器数量（或卷积核）、卷积核的高度和宽度、步长、填充类型和激活类型。为了获得最佳性能，我们可以调整超参数。但调整是一个高级主题，我们认为它不适合入门书籍。相反，我们提供了一些基本示例，您可以练习以培养实际技能。

## 构建一个 CNN

虽然 CNN 是一个顺序神经网络，但它与前馈顺序神经网络在两个重要方面有所不同。首先，它有一个不是完全连接的卷积基。其次，它有一个池化层，用于减少每个卷积层创建的特征图样本大小。我们仍然使用全连接层进行分类。

我们通过加载彩色图像数据集开始这个实验。然后我们准备数据以供 TensorFlow 使用。然后我们构建并测试一个 CNN 模型。我们使用的数据集是*cifar10*。我们之前使用前馈模型建模了这个数据集，但我们的结果很糟糕。因此，我们想向您展示 CNN 在复杂彩色图像上的表现有多好。

### 加载数据

加载 cifar10 的推荐方式是作为 TFDS：

```py
# import TFDS library
import tensorflow_datasets as tfds
```

加载训练集和测试集：

```py
train, info = tfds.load('cifar10', split="train",
with_info=True, shuffle_files=True)
test = tfds.load('cifar10', split="test")
```

由于我们已经有训练集的*info*信息，因此不需要再次为测试集使用它。

验证张量：

```py
train.element_spec, test.element_spec
```

训练集和测试集的形状为 32 `×` 32 `×` 3。因此，每张图像是一个 32 `×` 32 的三通道图像。*3*维度通知模型图像是 RGB 颜色。

### 显示数据集信息

*info*对象显示有关数据的信息：

```py
info
```

我们可以看到特征图像和标签的名称、描述、主页以及特征图像和标签的形状和数据类型。我们还看到该数据集有 60,000 张图像，训练集和测试集分别为 50,000 和 10,000。

### 提取类别标签

从*info*对象中提取一些有用的信息：

```py
br = '\n'
num_classes = info.features['label'].num_classes
class_labels = info.features['label'].names
print ('number of classes:', num_classes, br)
print ('class labels:', class_labels)
```

### 显示样本

*show_examples*方法显示几个示例：

```py
fig = tfds.show_examples(train, info)
```

### 构建一个用于显示样本的自定义函数

列表 7-6 是一个显示样本的函数。

```py
import matplotlib.pyplot as plt, numpy as np
def display_samples(data, num, cmap):
for example in data.take(num):
image, label = example['image'], example['label']
print ('Label:', class_labels[label.numpy()], end=', ')
print ('Index:', label.numpy())
plt.imshow(image.numpy()[:, :, 0].astype(np.float32),
cmap=plt.get_cmap(cmap))
plt.show()
Listing 7-6
Function to display samples
```

该函数检索图像和标签名称。然后显示带有其标签名称和索引的图像。索引是作为数字的类别标签。

调用该函数：

```py
# choose colormap by changing 'indx'
cmap = ['coolwarm', 'viridis', 'plasma',
'seismic', 'twilight', 'Spectral']
indx, samples = 5, 3
display_samples(train, samples, cmap[indx])
```

通过调整*indx*在 0 到 5 之间改变颜色。通过调整*samples*改变显示的样本数量。查看以下 URL 了解更多关于 colormaps 的信息：[`https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html`](https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html)。

### 构建一个用于显示示例网格的自定义函数

首先从训练集中选取 30 个示例，如列表 7-7 所示。

```py
num = 30
images, labels = [], []
for example in train.take(num):
image, label = example['image'], example['label']
images.append(tf.squeeze(image.numpy()))
labels.append(label.numpy())
Listing 7-7
Processed examples from the train set
```

为了启用图像绘制，我们从图像矩阵中移除（或压缩）3 个维度。

按照列表 7-8 所示构建函数。

```py
def display_grid(feature, target, n_rows, n_cols, cl):
plt.figure(figsize=(n_cols * 1.5, n_rows * 1.5))
for row in range(n_rows):
for col in range(n_cols):
index = n_cols * row + col
plt.subplot(n_rows, n_cols, index + 1)
plt.imshow(feature[index], cmap="binary",
interpolation='nearest')
plt.axis('off')
plt.title(cl[target[index]], fontsize=12)
plt.subplots_adjust(wspace=0.2, hspace=0.5)
Listing 7-8
Function to display a grid of examples
```

调用函数：

```py
rows, cols = 5, 6
display_grid(images, labels, rows, cols, class_labels)
```

哇哦！

### 确定元数据

利用 info 对象确定元数据：

```py
print ('Number of training examples:', end=' ')
print (info.splits['train'].num_examples)
print ('Number of test examples:', end=' ')
print (info.splits['test'].num_examples)
```

### 构建输入管道

按照列表 7-9 所示构建输入管道。

```py
BATCH_SIZE = 128
SHUFFLE_SIZE = 5000
train_1 = train.shuffle(SHUFFLE_SIZE).batch(BATCH_SIZE)
train_2 = train_1.map(lambda items: (
tf.cast(items['image'], tf.float32) / 255., items['label']))
train_cf = train_2.cache().prefetch(1)
test_1 = test.batch(BATCH_SIZE)
test_2 = test_1.map(lambda items: (
tf.cast(items['image'], tf.float32) / 255., items['label']))
test_cf = test_2.cache().prefetch(1)
Listing 7-9
Build the input pipeline
```

我们通过打乱训练数据、分批、缩放、缓存和预取来构建输入管道。我们通过 lambda 函数映射来缩放图像。添加*cache*方法可以增加 TFDS 上的性能，因为数据只读一次而不是在每个 epoch 期间读取。添加*prefetch*方法是一个好主意，因为它增加了批处理过程的效率。也就是说，当我们的训练算法在一个批次上工作时，TensorFlow 并行地在数据集上工作，以准备好下一个批次。因此，预取可以显著提高训练性能。

确认训练集和测试集已正确创建：

```py
train_cf.element_spec, test_cf.element_spec
```

### 创建模型

从一个相对稳健的 CNN 模型开始，因为这是从复杂彩色图像中获得良好性能的唯一方法。不要被层数吓倒！记住，CNN 有一个卷积基础和全连接网络。因此，我们可以将 CNN 视为两部分。首先，我们构建卷积基础，包括一个或多个卷积层和池化层。池化层包括在内，以从卷积层输出的特征图中下采样，以减少计算开销。接下来，我们构建一个用于分类的全连接层。

按照以下步骤创建模型：

1.  导入库。

1.  清除之前的模型。

1.  创建模型。

导入库：

```py
# import libraries
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D,\
Dense, Flatten, Dropout
```

清除之前的模型并设置随机种子：

```py
# clear previous models and generate a seed
tf.keras.backend.clear_session()
np.random.seed(0)
tf.random.set_seed(0)
```

按照列表 7-10 所示创建模型。

```py
# build the model
model = Sequential([
Conv2D(32, (3, 3), activation = 'relu', padding="same",
input_shape=[32, 32, 3], strides=1),
MaxPooling2D(2),
Conv2D(64, (3, 3), activation="relu", padding="same"),
MaxPooling2D(2),
Conv2D(64, (3, 3), activation="relu", padding="same"),
Flatten(),
Dense(64, activation="relu"),
Dropout(0.5),
Dense(10, activation="softmax")
])
Listing 7-10
Build the model
```

第一层是卷积基础，它使用 32 个卷积核和 3 `×` 3 的核大小。我们使用*relu*激活、*same*填充和*1*步长。我们还设置形状为 32 `×` 32 `×` 3 以匹配 32 `×` 32 像素的图像。由于图像是彩色的，我们在末尾包含 3 个值。接下来，我们包含一个大小为 2 的最大池化层（因此它将每个空间维度除以 2），以从第一个卷积层中下采样特征图。然后我们重复相同的结构两次，但将卷积核的数量增加到 64。在池化层之后加倍卷积核的数量是一种常见的做法。

我们继续使用全连接网络，因为它将输入展平，因为密集网络期望每个实例的特征为 1D 数组。我们需要添加全连接层以实现对十个标签的分类。我们继续使用 64 个神经元的密集层。我们添加 dropout 以减少过拟合。最终的密集层接受十个输入以匹配标签数量。它使用*softmax*激活。

### 模型摘要

检查模型：

```py
model.summary()
```

**参数**是训练期间可学习的权重数量。卷积层是 CNN 开始学习的地方。但计算 CNN 的参数比前馈网络更复杂。

第一层是一个有 32 个神经元的卷积层，作用于数据。过滤器大小是 3 `×` 3。因此，我们有一个 3 `×` 3 `×` 32 的过滤器，因为我们的输入有 32 个维度（或神经元），总共 288 个。将 288 `×` 3 相乘以考虑 3D RGB 彩色图像，总共 864 个。在此层添加 32 个神经元，总共得到 896 个参数。

池化层没有可学习的参数。因此，我们有 0 个参数。

第二个卷积层有 64 个神经元作用于数据。过滤器大小是 3 `×` 3。因此，我们有一个 3 `×` 3 `×` 64 的过滤器，因为我们有 64 个维度，总共 576 个。将前一个卷积层的 32 个神经元乘以 576，总共得到 18,432 个。在此层添加 64 个神经元，总共得到 18,496 个参数。

第三个卷积层有 64 个神经元作用于数据。过滤器大小是 3 `×` 3。因此，我们有一个 3 `×` 3 `×` 64 的过滤器，因为我们有 64 个维度，总共 576 个。将前一个卷积层的 64 个神经元乘以 576，总共得到 36,864 个。在此层添加 64 个神经元，总共得到 36,928 个参数。

完全连接的密集层计算如前所述。我们通过将此层的 4096 个神经元乘以前层的 64 个神经元得到 262,144。在此层添加 64 个神经元，总共得到 262,208 个参数。

输出层有 650 个参数，通过将前一层 64 个神经元乘以 10 并在此层添加 10 个神经元来计算。哇！

### 模型层

检查模型层：

```py
model.layers
```

### 编译模型

通过实验，我们发现*Nadam*优化器表现最佳：

```py
model.compile(optimizer='nadam',
loss='sparse_categorical_crossentropy',
metrics=['accuracy'])
```

### 训练模型

训练模型 10 个周期：

```py
epochs = 10
history = model.fit(train_cf, epochs=epochs,
verbose=1, validation_data=test_cf)
```

虽然我们的模型不是最先进的，但比使用前馈网络要好得多。

### 在测试数据上归纳

归纳：

```py
print('Test accuracy:', end=' ')
test_loss, test_acc = model.evaluate(test_cf, verbose=2)
```

### 可视化训练性能

列表 7-11 可视化训练。

```py
plt.plot(history.history['accuracy'], label="accuracy")
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
plt.show()
plt.plot(history.history['loss'], label="loss")
plt.plot(history.history['val_loss'], label = 'val_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.ylim([0.5, 1.0])
plt.legend(loc='lower right')
plt.show()
Listing 7-11
Visualization of training performance
```

### 预测测试图像的标签

在*test_cf*数据上预测：

```py
predictions = np.argmax(model.predict(test_cf), axis=-1)
```

将*predict*方法与*argmax*方法包装起来，以直接获取预测标签而不是生成概率数组。

通过类别编号获取预测：

```py
# predictions by class number
predictions
```

通过标签获取预测：

```py
# predictions by class label
np.array(class_labels)[predictions]
```

获取前五个预测：

```py
# 5 predictions
pred_5 = predictions[:5]
pred_5
```

将标签数字转换为标签名称：

```py
pred_labels = np.array(class_labels)[pred_5]
pred_labels
```

如列表 7-12 所示，获取前五个实际标签。

```py
# take the first batch of images
ls = []
for _, label in test_cf.take(1):
ls.append(label.numpy())
# slice first five from batch
actuals = ls[0][0:5]
# convert to labels
actuals = [class_labels[row] for row in actuals]
actuals
Listing 7-12
First five actual labels
```

获取第一批图像。由于我们设置了批大小为 128，我们得到前 128 个图像。从批中切出前五个图像。转换为标签名称。

将*pred_labels*与*actual_labels*比较，以了解预测性能。

### 构建预测图

首先，从测试集中取出 20 个样本，如列表 7-13 所示。

```py
num = 20
images, labels = [], []
for example in test.take(num):
image, label = example['image'], example['label']
images.append(tf.squeeze(image.numpy()))
labels.append(label.numpy())
Listing 7-13
Take samples from the test set
```

### 构建自定义函数

构建一个函数来显示结果，如列表 7-14 所示。

```py
def display_test(feature, target, num_images,
n_rows, n_cols, cl, p):
for i in range(num_images):
plt.subplot(n_rows, 2*n_cols, 2*i+1)
plt.imshow(feature[i])
title_obj = plt.title(cl[target[i]] + ' (' +\
cl[p[i]] + ') ')
if cl[target[i]] == cl[p[i]]:
title_obj
else:
plt.getp(title_obj, 'text')
plt.setp(title_obj, color="r")
plt.tight_layout()
plt.show()
Listing 7-14
Function to display results
```

调用函数：

```py
num_rows, num_cols = 5, 4
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
display_test(images, labels, num_images, num_rows,
num_cols, class_labels, predictions)
```

红色标题表示错误分类。

## 使用 Keras 数据构建 CNN

虽然推荐以 TFDS 的形式加载数据，但 Keras 在工业界非常受欢迎。所以让我们从*keras.datasets*构建一个模型。

加载训练和测试数据：

```py
train_k, test_k = tf.keras.datasets.cifar10.load_data()
```

验证数据形状：

```py
print ('train data:', br)
print (train_k[0].shape)
print (train_k[1].shape, br)
print ('test data:', br)
print (test_k[0].shape)
print (test_k[1].shape)
```

创建类标签名称：

```py
class_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer',
'dog', 'frog', 'horse', 'ship', 'truck']
```

### 创建用于存储训练和测试样本的变量

创建如列表 7-15 所示的存储训练和测试数据的变量。

```py
# create simple variables from train tuple
train_images = train_k[0]
train_labels = train_k[1]
# display first train label
print ('1st train label:', class_labels[train_labels[0][0]])
# create simple variables from test tuple
test_images = test_k[0]
test_labels = test_k[1]
# display first test label
print ('1st test label: ', class_labels[test_labels[0][0]])
Listing 7-15
Create variables to hold train and test data
```

### 显示样本图像

总是显示一些图像是个好主意。在这种情况下，我们显示了训练数据集中的 30 个图像。可视化使我们能够验证图像和标签是否对应。也就是说，青蛙图像被标记为青蛙，依此类推。

列表 7-16 显示训练集中的样本图像。

```py
n_rows = 5
n_cols = 6
plt.figure(figsize=(n_cols * 1.5, n_rows * 1.5))
for row in range(n_rows):
for col in range(n_cols):
index = n_cols * row + col
plt.subplot(n_rows, n_cols, index + 1)
plt.imshow(train_images[index], cmap="binary",
interpolation='nearest')
plt.axis('off')
plt.title(class_labels[int(train_labels[index])],
fontsize=12)
plt.subplots_adjust(wspace=0.2, hspace=0.5)
Listing 7-16
Display sample images
```

### 创建输入管道

通过缩放图像并将它们切割成 TensorFlow 可消费的部分来构建输入管道。继续进行（在适当的情况下）洗牌、分批和预取。

列表 7-17 创建输入管道。

```py
# scale images
train_img_sc = train_images / 255\.  # divide by 255 to scale
train_lbls = train_labels.astype(np.int32)
test_img_sc = test_images/255\.  # divide by 255 to scale
test_lbls = test_labels.astype(np.int32)
# slice data
train_ks = tf.data.Dataset.from_tensor_slices(
(train_img_sc, train_lbls))
test_ks = tf.data.Dataset.from_tensor_slices(
(test_img_sc, test_lbls))
# shuffle, batch, and prefetch
BATCH_SIZE = 128
SHUFFLE_BUFFER_SIZE = 5000
train_ds = train_ks.shuffle(
SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE).prefetch(1)
test_ds = test_ks.batch(BATCH_SIZE).prefetch(1)
Listing 7-17
Build the input pipeline
```

检查张量：

```py
train_ds, test_ds
```

### 创建模型

列表 7-18 创建模型。

```py
# import libraries
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D,\
Dense, Flatten, Dropout
# clear previous models and generate a seed
tf.keras.backend.clear_session()
np.random.seed(0)
tf.random.set_seed(0)
# build the model
model = Sequential([
Conv2D(32, 3, activation = 'relu', padding="same",
input_shape=[32, 32, 3]),
MaxPooling2D(2),
Conv2D(64, 3, activation="relu", padding="same"),
MaxPooling2D(2),
Conv2D(64, 3, activation="relu", padding="same"),
Flatten(),
Dense(64, activation="relu"),
Dropout(0.5),
Dense(10, activation="softmax")
])
Listing 7-18
Create the model
```

### 编译和训练

列表 7-19 编译并训练模型。

```py
# compile
model.compile(optimizer='nadam',
loss='sparse_categorical_crossentropy',
metrics=['accuracy'])
# train
epochs = 10
history = model.fit(train_ds, epochs=epochs,
verbose=1, validation_data=test_ds)
Listing 7-19
Compile and train the model
```

### 预测

从模型中进行预测：

```py
pred_ks = np.argmax(model.predict(test_images), axis=-1)
```

### 可视化结果

列表 7-20 可视化训练性能结果。

```py
# plot the first X (num_rows * num_cols) test images
# (true and predicted labels)
num_rows = 5
num_cols = 4
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
ax = plt.subplot(num_rows, 2*num_cols, 2*i+1)
plt.imshow(test_images[i])
title = class_labels[int(test_labels[i])] +
' (' +\class_labels[pred_ks[i]] + ') '
plt.title(title)
if class_labels[int(test_labels[i])] !=\
class_labels[pred_ks[i]]:
ax.set_title(title, style="italic", color="red")
plt.axis('off')
plt.tight_layout()
Listing 7-20
Visualize training performance results
```

## 结语

在过去几年中，许多改进基本 CNN 架构的方法已经开发出来，这些方法大大提高了预测性能。尽管我们在这节课中没有涵盖这些进展，但我们相信我们已经通过 CNN 提供了基本的基础，以帮助您轻松地使用这些最新的进展，甚至未来可能出现的许多进展。
