# 12. 快速风格迁移

**神经风格迁移（NST）**是一种计算机视觉技术，它将两张图像——内容图像和风格参考图像——混合在一起，使得生成的输出图像保留了内容图像的核心元素，但看起来像是用风格参考图像的风格绘制的。NST 网络的输出图像被称为混合体。**混合体**是一种视觉艺术、文学、戏剧或音乐作品，它模仿一位或多位其他艺术家的风格（或人物）。与讽刺作品不同，混合体赞美而不是嘲笑它所模仿的作品。

在 NST 的许多应用中，内容图像是一张照片，风格参考图像是一幅画。但这不是必需的。NST 的奇妙之处在于我们可以将著名画家的作品混合，使得输出图像看起来像是用参考图像的风格绘制的。因此，我们可以训练 NST 网络自动创建新的艺术表现！

注意

在数据科学术语中，网络和模型是可以互换的（至少在处理神经网络模型时是这样）。

该技术通过优化输出图像以匹配内容图像的内容统计和风格参考图像的风格统计来实现。统计信息是通过卷积网络从图像中提取的。想法是训练网络以匹配基本输入图像与内容和风格参考图像。NST 通过反向传播最小化内容和风格参考距离（或损失），从而创建一个与内容图像的内容相匹配且具有风格参考图像风格的图像。

为了将内容和风格表示提取成混合体，神经网络包括中间层。中间层代表特征图，随着我们深入网络，这些特征图的阶数越来越高，从而定义了从各自图像中提取的内容和风格表示。在训练过程中，网络试图将基本输入图像与中间层中相应的风格和内容目标表示相匹配。

为了使网络执行图像分类，它必须理解图像。因此，它以原始图像作为输入像素，并通过将原始图像像素转换为对图像中存在的特征复杂理解的一系列变换来构建内部表示。网络的*中间层*执行允许模型从图像的原始输入像素中提取有意义的特征的变换。因此，中间层能够精确地描述输入图像的内容和风格。

尽管取得了惊人的成果，但 NST 由于将其视为一个需要数百次甚至数千次迭代来对单张图像进行风格迁移的优化问题，因此运行速度较慢！为了解决这种低效性，深度学习研究人员开发了一种称为快速（神经网络）风格迁移的技术。**快速风格迁移**使用深度神经网络，但训练一个独立的模型，通过单次前馈传递来转换图像！因此，训练好的快速风格迁移模型只需通过网络进行一次（或一个周期）迭代即可对任何图像进行风格化，而不是数百或数千次。

## 为什么风格迁移很重要？

并非每个人都是天生的艺术家。但随着风格迁移等技术的最新进展，几乎任何人都可以享受到创作和分享艺术杰作带来的乐趣。

风格迁移的变革力量使得艺术家可以轻松地将自己的创意美学赋予他人。没有天生艺术能力的人也能创造出与原创杰作并驾齐驱的新颖和创新的艺术风格。因此，风格迁移赋予了人们培养自己创造力的能力！创意的转换可能导致新的艺术表现，这些表现可能在其他情况下永远不会被创造出来。

## 任意神经网络艺术风格化

尽管快速风格迁移网络速度快，但它们仅限于预先选定的少数几种风格，因为必须为每种风格图像训练一个独立的神经网络。*任意神经网络艺术风格化*（ANAS）通过使用风格网络和转换网络来减轻这种限制。*风格网络*学习如何将图像分解成一个代表其风格的 100 维向量（或风格向量）。*转换网络*学习如何从风格向量和原始内容图像中生成最终的风格化图像。

注意

风格向量也被称为风格瓶颈向量。

ANAS 网络是快速风格网络的最新版本。ANAS 被认为比 NST 和快速风格迁移更好，因为它们能够实现实时任意风格迁移。因此，它们比 NST 更快，比快速风格迁移更灵活，因为它们可以自动适应任意新的风格。

ANAS 增加了一个*自适应实例归一化*（AdaIN）层，该层将内容特征的平均值和方差与风格特征的平均值和方差对齐。ANAS 在不限制预定义风格集的情况下，实现了与现有最快方法相当的速度。ANAS 还允许灵活的用户控制，如内容-风格权衡、风格插值以及仅通过单次前馈神经网络传递进行颜色和空间控制！

章节的笔记本位于以下网址：

[`github.com/paperd/deep-learning-models`](https://github.com/paperd/deep-learning-models)

由于 ANAS 是 NST 最快和最好的实现，我们通过端到端代码实验来展示它。我们还包括一个使用 TensorFlow Lite 模块中的预训练迁移模型的第二个 ANAS 实验。

**TensorFlow Lite** 是一套工具，帮助开发者将 TensorFlow 模型在移动、嵌入式和物联网 (IoT) 设备上运行。它允许在设备上以低延迟和小二进制大小进行机器学习推理。**IoT** 设备是连接了传感器并能通过互联网将数据从一个对象传输到另一个对象或传输给人们的设备。*低延迟* 描述的是一个优化以处理非常高的数据消息量并具有最小延迟（或延迟）的计算机网络。此类网络旨在支持需要近乎实时访问快速变化数据的操作。

通过导入主 TensorFlow 库并实例化 GPU 来开始设置 Colab 生态系统。

## 导入 TensorFlow 库

导入库并将其别名为 **tf**：

```py
import tensorflow as tf
```

将 TensorFlow 库别名为 tf 是常见做法。

## GPU 硬件加速器

为了方便，我们包括在 Colab 笔记本中启用 GPU 的步骤：

1.  在右上角菜单中点击 *运行时*。

1.  从下拉菜单中选择 *更改运行时类型*。

1.  从 *硬件加速器* 下拉菜单中选择 *GPU*。

1.  点击 *保存*。

验证 GPU 是否处于活动状态：

```py
tf.__version__, tf.test.gpu_device_name()
```

如果显示 ‘/device:GPU:0’，则 GPU 处于活动状态。如果显示 ‘..’，则常规 CPU 处于活动状态。

注意

如果出现错误 **NAME** **‘****TF****’** **IS NOT DEFINED**，请重新执行代码以导入 TensorFlow 库！

## ANAS 实验

我们使用 *arbitrary-image-stylization-v1-256* 网络进行实验。该网络是一个用于快速任意图像风格转换的预训练模型。网络不需要调整图像大小，但更喜欢风格参考图像约为 256 像素，因为它是在 256 × 256 像素图像上训练的。但内容图像可以是任何大小。

注意

我们发现，尺寸大于或小于 256 × 256 像素的风格参考图像并不提供非常有说服力的拼贴。因此，我们强烈建议将风格参考图像调整到首选尺寸。

### 导入必需的库

导入：

```py
from matplotlib import gridspec
import matplotlib.pylab as plt
import numpy as np
import tensorflow_hub as hub
from PIL import Image
```

*gridspec* 模块指定了子图放置的网格几何形状。设置网格的行数和列数是必需的。*TensorFlow Hub* 是一个预训练机器学习模型的存储库，只需几行代码即可在任何地方调整和部署。*Image* 模块用于表示 Python 图像库 (PIL) 图像。*PIL* 模块提供了一些工厂函数，包括从文件加载图像和创建新图像的函数。

### 从 Google Drive 获取图像

我们从 Google Drive 获取实验图像。其他选项包括（但不限于）从本地驱动器上传图像或从维基百科公共领域下载图像。

注意

确保所有图像都在 Google Drive 上的 *Colab Notebooks* 目录中。

挂载 Google Drive：

```py
from google.colab import drive
drive.mount('/content/gdrive')
```

点击 URL。选择一个 Google Gmail 账户。点击 *允许* 按钮。将授权代码复制并粘贴到文本框中，然后按键盘上的 *Enter* 键。

加载并显示风格参考图像：

```py
img_path = 'gdrive/My Drive/Colab Notebooks/images/serene.jpeg'
style = Image.open(img_path)
plt.axis('off')
_ = plt.imshow(style)
```

图像可通过 Apress 网站或我的 GitHub 获取。

获取图像类型：

```py
type(style)
```

该图像是一个 PIL 图像。

获取风格参考图像的形状：

```py
w, h = style.size
w, h
```

*size* 方法返回 PIL 图像的形状。

加载并显示内容图像：

```py
img_path = 'gdrive/My Drive/Colab Notebooks/images/'\
'humming_bird.jpeg'
content  = Image.open(img_path)
plt.axis('off')
_ = plt.imshow(content)
```

获取图像类型：

```py
type(content)
```

获取内容图像的形状：

```py
w, h = content.size
w, h
```

### 预处理图像

由于两个图像都是 PIL 图像，我们将它们转换为张量，以便它们可以被预训练模型消费。风格参考图像的推荐大小是 256 × 256，因为这是我们用于此实验的预训练风格迁移网络所期望的大小。内容图像可以是任何大小。

将风格参考图像转换为 NumPy 数组并缩放：

```py
style_array = tf.keras.preprocessing.image.img_to_array(
style) / 255.
style_array.shape
```

将风格参考图像调整到风格迁移网络所期望的大小：

```py
style_img = tf.image.resize(style_array, (256, 256))
style_img.shape
```

哇！风格参考图像已准备好被风格迁移网络消费。

将内容图像转换为 NumPy 数组并缩放：

```py
content_img = tf.keras.preprocessing.image.img_to_array(
content) / 255.
content_img.shape
```

由于内容图像可以是任何大小，它现在已准备好被风格迁移网络消费。

### 显示处理后的图像

创建一个显示函数：

```py
def display_one(img):
plt.imshow(img)
plt.axis('off')
plt.show()
```

显示处理后的风格参考图像：

```py
display_one(style_img)
```

显示处理后的内容图像：

```py
display_one(content_img)
```

### 准备图像批次

虽然处理后的内容和风格参考图像可以被风格迁移网络消费，但它期望每个图像作为一个单独的批次。因此，每个批次必须是一个形状为 *[batch_size, image_height, image_width, 3]* 的 4D 张量。由于内容和风格参考图像目前是形状为 *[image_height, image_width, 3]* 的 3D 张量，我们添加一个“1”的批处理维度以满足要求。网络还要求图像为 TensorFlow 张量。因此，我们将它们转换为 TensorFlow 张量。

输入和输出图像的值预期在范围 [0, 1] 内。我们已经通过缩放两个图像来满足这一要求。内容和风格参考图像的形状不必匹配。所以在这一点上我们没问题。

注意

拼贴画（输出图像）的形状是从内容图像的形状中适配的。

为两个图像添加批处理维度：

```py
style_image = np.expand_dims(style_img, axis=0)
content_image = np.expand_dims(content_img, axis=0)
style_image.shape, content_image.shape
```

将 NumPy 图像转换为 TensorFlow 张量：

```py
style_tensor = tf.convert_to_tensor(style_image)
content_tensor = tf.convert_to_tensor(content_image)
```

### 加载模型

如前所述，我们使用 *arbitrary-image-stylization-v1-256* 网络来创建拼贴画。早期的 NST 模型受限于预先选择的少数几种风格，因为必须为每种风格图像训练一个单独的神经网络。任意风格迁移通过包括风格网络和转换网络来减轻这种限制。

*风格网络*学习如何将图像分解成一个 100 维向量（或风格瓶颈向量），它代表了其风格。*转换网络*学习如何从风格瓶颈向量和原始内容图像生成最终的风格化图像。任意风格网络（截至本文写作时）是当前最先进的，因为它们比原始 NST 网络更快，比快速风格迁移网络更灵活。

加载预训练的 ANAS 网络：

```py
p1 = 'https://tfhub.dev/google/magenta/'
p2 = 'arbitrary-image-stylization-v1-256/2'
URL = p1 + p2
hub_handle = URL
hub_module = hub.load(hub_handle)
```

从预训练的 ANAS 网络构建 hub 模块。

### 提供模型：

展示任意风格迁移：

```py
outputs = hub_module(content_tensor, style_tensor)
pastiche = outputs[0]
pastiche.shape
```

hub 模块的签名接受处理后的内容图像和处理后的风格参考图像，通过学习如何混合两个张量来创建 Pastiche。*签名*是训练模型所需的程序语法。

使用 GPU 进行训练速度快！但使用 CPU 训练 hub 模块签名需要一些时间。

### 探索 Pastiche

探索图像形状：

```py
pastiche_numpy = tf.squeeze(pastiche).numpy()
pastiche_numpy.shape, content_img.shape
```

将 Pastiche 转换为 NumPy 以方便探索。形状与内容图像不完全相同，但非常接近。

从风格化图像张量中探索一个切片：

```py
pastiche_numpy[0][0]
```

Pastiche 具有与内容和风格参考图像相同的像素特征。

提取矩阵组件：

```py
m = pastiche_numpy
r, c, channels = m.shape[0], m.shape[1], m.shape[2]
r, c, channels
```

Pastiche 数组是一个由 184 行、280 列和 3 个通道组成的 3D 矩阵。

获取矩阵中的像素数：

```py
pixels = r * c
pixels
```

每个 RGB 通道有 51,520 个像素。**RGB**指的是三种光（红色、绿色和蓝色）的组合，可以混合成不同的颜色。将红、绿、蓝光混合是生产电视、计算机显示器和智能手机等屏幕上彩色图像的标准方法。*RGB 颜色模型*是一个加色模型，因为三种颜色的光束（光光谱波长与光光谱波长）相加，以创建最终的颜色光谱。

检查 RGB 通道像素是否缩放：

```py
red = m[m[:, :, 0] < 1, 0] < 1
green = m[m[:, :, 1] < 1, 0] < 1
blue = m[m[:, :, 2] < 1, 0] < 1
print (len(red), len(green), len(blue))
print (red, green, blue)
```

算法检查每个通道中的像素是否小于 1。我们显示每个算法修改后的通道的长度，以查看它们是否包含预期的像素数量。到目前为止，一切检查正常。

由于没有显示所有真值表值，我们需要额外一步来验证缩放是否按预期工作：

```py
all(red), all(green), all(blue)
```

*all()*函数如果可迭代中的所有项目都为真，则返回 True；否则，返回 False。所以所有像素都缩放了。一个可迭代的对象是一个可以迭代的对象。Python 列表就是一个可迭代的例子。

但我们可以检查所有矩阵像素是否在*一步*中缩放：

```py
truth = np.where((m < 1), True, False)
truth.all()
```

备注

我们展示了多步骤过程，以一瞥风格图像的内部机制。

### 可视化

创建一个如图 12-1 所示的可视化函数。

```py
def show_n(images, titles=('',)):
n = len(images)
image_sizes = [image.shape[1] for image in images]
w = (image_sizes[0] * 6) // 320
plt.figure(figsize=(w  * n, w))
gs = gridspec.GridSpec(1, n, width_ratios=image_sizes)
for i in range(n):
plt.subplot(gs[i])
plt.imshow(images[i][0], aspect='equal')
plt.axis('off')
plt.title(titles[i] if len(titles) > i else '')
plt.show()
Listing 12-1
Visualization Function
```

可视化原始内容、风格参考和风格化图像：

```py
show_n([content_image, style_image, pastiche],
titles=['Original content image', 'Style image',
'Pastiche'])
```

创建一个用于可视化新风格图像的函数：

```py
def display_pastiche(img, size):
plt.figure(figsize = size)
plt.imshow(tf.squeeze(img))
plt.axis('off')
plt.show()
```

可视化：

```py
f_size = (10, 15)
display_pastiche(pastiche, f_size)
```

### 多图像图像风格化

创建风格参考和内容图像的列表。从每个列表中选择一个来创建一个 Pastiche。

#### 获取图像：

创建一个从 Google Drive 获取图像的函数：

```py
def get_image(img_path):
return Image.open(img_path)
```

获取列表 12-2 中的风格图像。

```py
d = 'gdrive/My Drive/Colab Notebooks/images/dali.jpg'
dali = get_image(d)
v = 'gdrive/My Drive/Colab Notebooks/images/van-gogh.jpg'
van_gogh = get_image(v)
m = 'gdrive/My Drive/Colab Notebooks/images/modern.jpg'
modern = get_image(m)
e = 'gdrive/My Drive/Colab Notebooks/images/escher.jpeg'
escher = get_image(e)
pic = 'gdrive/My Drive/Colab Notebooks/images/picasso.jpg'
picasso = get_image(pic)
p = 'gdrive/My Drive/Colab Notebooks/images/pollock.jpg'
pollock = get_image(p)
mon = 'gdrive/My Drive/Colab Notebooks/images/monet.jpg'
monet = get_image(mon)
Listing 12-2
Get Style Reference Images
```

显示风格参考图像形状：

```py
dali.size, van_gogh.size, modern.size, escher.size,\
picasso.size, pollock.size, monet.size
```

获取内容图像：

```py
t = 'gdrive/My Drive/Colab Notebooks/images/teddy.jpeg'
teddy = get_image(t)
e = 'gdrive/My Drive/Colab Notebooks/images/einstein.jpg'
einstein = get_image(e)
g = 'gdrive/My Drive/Colab Notebooks/images/gem.jpeg'
gem = get_image(g)
```

显示内容图像形状：

```py
teddy.size, einstein.size, gem.size
```

#### 处理图像

创建一个如列表 12-3 所示的预处理函数。

```py
def preprocess(img, style=True):
img_array = tf.keras.preprocessing.image.img_to_array(
img) / 255.
if style:
img_array = tf.image.resize(img_array, (256, 256))
return\
tf.convert_to_tensor(np.expand_dims(img_array, axis=0))
Listing 12-3
Preprocessing Function
```

函数将 PIL 图像转换为 NumPy 数组。然后它进行缩放、调整大小，并将数组转换为具有适当维度的 TensorFlow 张量。

按照列表 12-4 中的说明处理风格参考图像。

```py
dali_style = preprocess(dali)
van_gogh_style = preprocess(van_gogh)
modern_style = preprocess(modern)
escher_style = preprocess(escher)
picasso_style = preprocess(picasso)
pollock_style = preprocess(pollock)
monet_style = preprocess(monet)
dali_style.shape, van_gogh_style.shape, modern_style.shape,\
escher_style.shape, picasso_style.shape, pollock_style.shape,\
monet_style.shape
Listing 12-4
Process the Style Reference Images
```

处理内容图像：

```py
einstein_content = preprocess(einstein, False)
teddy_content = preprocess(teddy, False)
gem_content = preprocess(gem, False)
einstein_content.shape, teddy_content.shape, gem_content.shape
```

#### 可视化处理后的图像

将风格参考图像和内容图像放在列表中：

```py
styles = [dali_style, van_gogh_style, modern_style,
escher_style, picasso_style, pollock_style,
monet_style]
contents = [einstein_content, teddy_content, gem_content]
```

创建一个函数来显示列表 12-5 中的张量。

```py
def display_tensors(imgs, r, c):
_, axs = plt.subplots(r, c, figsize=(12, 12))
axs = axs.flatten()
for img, ax in zip(imgs, axs):
ax.imshow(tf.squeeze(img))
ax.axis('off')
plt.show()
Listing 12-5
Function to Display Tensors
```

显示风格张量：

```py
rows, cols = 1, 7
display_tensors(styles, rows, cols)
```

显示内容张量：

```py
rows, cols = 1, 3
display_tensors(contents, rows, cols)
```

#### 创建参考字典

创建一个表示风格张量的字典：

```py
style_names = {'dali' : styles[0],
'van_gogh' : styles[1],
'modern' : styles[2],
'escher' : styles[3],
'picasso' : styles[4],
'pollock' : styles[5],
'monet' : styles[6]}
```

创建一个表示内容张量的字典：

```py
content_names = {'einstein' : contents[0],
'teddy' : contents[1],
'gem' : contents[2]}
```

#### 创建拼贴

创建一个创建拼贴的函数：

```py
def create(c, s):
content_im = content_names[c]
style_im = style_names[s]
outputs = hub_module(content_im, style_im)
return content_im, style_im, outputs[0]
```

创建拼贴：

```py
content, style = 'einstein', 'dali'
content_im, style_im, sim = create(content, style)
f_size = (7, 9)
display_pastiche(sim, f_size)
```

小贴士

通过实验内容参考图像来创建您自己的拼贴。当然，您也可以从您自己的图像中创建拼贴！

创建一个用于可视化的列表：

```py
imgs = [content_im, style_im, sim]
```

可视化内容、风格和拼贴：

```py
display_tensors(imgs, 1, 3)
```

尝试使用西奥多·罗斯福：

```py
content, style = 'teddy', 'picasso'
content_im, style_im, sim = create(content, style)
f_size = (8, 10)
display_pastiche(sim, f_size)
```

可视化内容、风格和拼贴：

```py
imgs = [content_im, style_im, sim]
display_tensors(imgs, 1, 3)
```

尝试使用宝石：

```py
content, style = 'gem', 'escher'
content_im, style_im, sim = create(content, style)
f_size = (8, 10)
display_pastiche(sim, f_size)
```

可视化内容、风格和拼贴：

```py
imgs = [content_im, style_im, sim]
display_tensors(imgs, 1, 3)
```

## TensorFlow Lite 实验

*TensorFlow Lite* 是一个开源深度学习框架，用于在设备上运行 TensorFlow 模型。**设备门户** (ODPs) 允许手机用户轻松浏览、购买和使用移动内容和服务。ODP 平台使运营商能够在大量服务中提供一致且品牌化的设备体验。TensorFlow Lite 为实验和部署深度学习实验提供了 ODP 平台。

要在您的设备上开始使用 TensorFlow Lite，请参阅

[www.tensorflow.org/lite/examples](http://www.tensorflow.org/lite/examples)

对于一个优秀的 TensorFlow Lite 风格迁移示例，请参阅

[www.tensorflow.org/lite/examples/style_transfer/overview](http://www.tensorflow.org/lite/examples/style_transfer/overview)

但 TensorFlow Lite 不必在设备上部署。它可以在 PC 上运行。我们出于三个原因在 PC 上的 Colab 笔记本中运行此实验。首先，我们知道我们的 PC 有足够的 RAM 来运行完整的 TensorFlow 实验。因此，我们不会在运行 TensorFlow Lite 实验时遇到任何问题！拥有实际的键盘和大屏幕也很不错。其次，我们不知道你使用的是哪种类型的设备（例如，Android、iOS 等）。第三，TensorFlow Lite 模块已在 Colab 中预先安装！

如果您想在设备上开发，我们强烈推荐 TensorFlow Lite，因为它针对各种设备进行了优化，包括手机、嵌入式 Linux 设备和微控制器。因此，TensorFlow Lite 比 TensorFlow 在设备上具有更好的性能和更小的二进制文件大小。

### 预训练 TensorFlow Lite 模型的架构

内容图像与我们已经在第一个实验中处理过的图像完全相同。与第一个实验一样，风格图像在输入到风格转换模型之前被转换（或瓶颈化）为*100 维度的风格瓶颈向量*。但与第一个实验不同，我们手动将风格图像转换为模型可消费的形式。

TensorFlow Lite 艺术风格迁移模型由两个子模型组成——风格预测模型和风格转换模型。*风格预测模型*是一个基于预训练的 MobileNet-v2 神经网络，它接受一个输入风格参考图像并将其转换为 100 维度的风格瓶颈向量。*风格转换模型*是一个神经网络，它将风格瓶颈向量应用于内容图像以创建一幅拼贴画。

### 裁剪图像

从图像中移除不需要的噪声。

导入显示所需的库：

```py
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (12,12)
mpl.rcParams['axes.grid'] = False
```

mpl 参数设置笔记本中所有图像的显示大小。

获取内容图像：

```py
content_cd = content_names['einstein']
content_cd.shape
```

风格转换模型期望的内容图像大小为 384 × 384：

```py
dim = [384, 384]
content_lte = tf.image.resize(content_cd, dim)
content_lte.shape
```

中心裁剪内容图像：

```py
content_lite = tf.image.resize_with_crop_or_pad(
content_lte, dim[0], dim[1])
content_lite.shape
```

裁剪是图像最基本的数据增强过程之一。其想法是从图像的边缘移除不需要或不相关的噪声，改变其长宽比或改善其整体构图。

获取风格图像：

```py
style_lte = style_names['modern']
style_lte.shape
```

中心裁剪风格参考图像：

```py
dim = [256, 256]
style_lite = tf.image.resize_with_crop_or_pad(
style_lte, dim[0], dim[1])
style_lite.shape
```

#### 显示裁剪图像

创建一个如清单 12-6 所示的显示函数。

```py
def imshow(image, title=None):
if len(image.shape) > 3:
image = tf.squeeze(image, axis=0)
plt.axis('off')
plt.imshow(image)
if title:
plt.title(title)
Listing 12-6
Display Function for Cropped Images
```

显示裁剪图像：

```py
plt.subplot(1, 2, 1)
imshow(content_lite, 'Content Image')
plt.subplot(1, 2, 2)
imshow(style_lite, 'Style Image')
```

### 装饰图像

我们将裁剪的风格图像输入到风格预测模型中，以创建风格瓶颈向量。然后，我们将裁剪的内容图像和新建的风格瓶颈向量输入到风格转换模型中，以创建一幅拼贴画。

#### 创建风格预测模型

创建一个如清单 12-7 所示的风格预测模型函数。

```py
def run_style_predict(processed_style_image):
# load the model
interpreter = tf.lite.Interpreter(
model_path=style_predict_path)
# set model input
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
interpreter.set_tensor(
input_details[0]["index"], processed_style_image)
# calculate style bottleneck
interpreter.invoke()
style_bottleneck = interpreter.tensor(
interpreter.get_output_details()[0]["index"])()
return style_bottleneck
Listing 12-7
Function to Create the Style Prediction Model
```

该函数创建一个风格预测模型，用于在裁剪的风格图像（*style_lite*）上运行风格预测。该函数接受一个处理过的风格参考图像。然后，它将风格预测路径加载到解释器对象（*tf.lite.Interpreter*）中。解释器是一个基于 TensorFlow Lite 预训练的 MobileNet-v2 神经网络。解释器使用 style_predict_path 创建一个 tflite 预测风格。然后设置解释器模型的输入。该函数继续通过计算风格瓶颈。该函数通过使用瓶颈计算和 tflite 预测风格将风格图像转换为 100 维度的风格瓶颈向量。简单来说，该函数接受一个风格参考图像并将其转换为 100 维度的风格瓶颈向量。

设置风格预测路径：

```py
tflite_predict = 'style_predict.tflite'
p1 = 'https://tfhub.dev/google/lite-model/magenta/'
p2 = 'arbitrary-image-stylization-v1-256/'
p3 = 'int8/prediction/1?lite-format=tflite'
URL = p1 + p2 + p3
style_predict_path = tf.keras.utils.get_file(
tflite_predict, URL)
```

将裁剪的风格图像转换为 100 维度的风格瓶颈向量：

```py
style_bottleneck = run_style_predict(style_lite)
print('style bottleneck vector shape:',
style_bottleneck.shape)
```

#### 创建风格转换模型

创建一个如清单 12-8 所示的风格转换模型函数。

```py
def run_style_transform(
style_bottleneck, processed_content_image):
# load the model
interpreter = tf.lite.Interpreter(
model_path=style_transform_path)
# set model input
input_details = interpreter.get_input_details()
interpreter.allocate_tensors()
# set content and style bottleneck
interpreter.set_tensor(
input_details[0]["index"], processed_content_image)
interpreter.set_tensor(
input_details[1]["index"], style_bottleneck)
interpreter.invoke()
# return the transformed content image
return interpreter.tensor(
interpreter.get_output_details()[0]["index"])()
Listing 12-8
Function to Create the Style Transform Model
```

函数创建一个风格变换模型，该模型将风格瓶颈向量应用于内容图像以创建拼贴。该函数接受风格瓶颈向量和处理过的内容图像。然后它将风格变换路径加载到解释器对象（tf.lite.Interpreter）中。解释器使用 style_transform_path 创建 tflite 变换风格。然后设置解释器模型的输入。函数继续设置处理过的内容图像和风格瓶颈向量。函数通过返回变换后的内容图像结束。

设置风格变换路径：

```py
tflite_transform= 'style_transform.tflite'
p1 = 'https://tfhub.dev/google/lite-model/magenta/'
p2 = 'arbitrary-image-stylization-v1-256/'
p3 = 'int8/transfer/1?lite-format=tflite'
URL = p1 + p2 + p3
style_transform_path = tf.keras.utils.get_file(
tflite_transform, URL)
```

### 创建拼贴

使用风格瓶颈风格化内容图像：

```py
stylized_image = run_style_transform(
style_bottleneck, content_lite)
```

确保风格化的图像（或拼贴）的大小符合 TensorFlow Lite 预训练模型的要求：

```py
pastiche = tf.image.resize(stylized_image, [384, 384])
pastiche.shape
```

可视化拼贴：

```py
imshow(pastiche, 'Pastiche')
```

### 风格混合

通过将内容图像的风格混合到风格化输出中，我们使拼贴看起来更像内容图像。

#### 准备内容图像

将内容图像重塑为风格预测模型可消费的形状：

```py
dim = [256, 256]
content_blend = tf.image.resize_with_crop_or_pad(
content_lite, dim[0], dim[1])
content_blend.shape
```

将重塑的内容图像转换为 100 维的风格瓶颈向量：

```py
style_bottleneck_content = run_style_predict(
content_blend)
style_bottleneck_content.shape
```

内容图像现在是一个 100 维的风格瓶颈向量。

#### 混合风格瓶颈向量

将风格瓶颈向量与内容-风格瓶颈向量（*style_bottleneck_content*）混合。

定义内容混合比例（介于 0 和 1 之间）：

```py
content_blending_ratio = 0.5
```

将内容图像混合到拼贴中的范围是从 0%到 100%。为了从内容图像中提取无风格，分配 0%。零百分比意味着混合拼贴与拼贴相同。为了从内容图像中提取所有风格，分配 100%。我们将混合比例设置为 50%以从内容图像中提取一半的风格。

从风格瓶颈和内容-风格瓶颈向量中获取混合的风格瓶颈向量：

```py
style_bottleneck_blended =\
content_blending_ratio * style_bottleneck_content +\
(1 - content_blending_ratio) * style_bottleneck
```

使用风格瓶颈风格化内容图像：

```py
stylized_image_blended = run_style_transform(
style_bottleneck_blended, content_lite)
```

可视化拼贴：

```py
imshow(stylized_image_blended, 'Blended Stylized Image')
```

### 保存拼贴

将拼贴保存到本地驱动器。

创建一个函数，将张量转换为 PIL 图像，如清单 12-9 所示。

```py
def tensor_to_image(tensor):
tensor = tensor * 255
tensor = np.array(tensor, dtype=np.uint8)
if np.ndim(tensor) > 3:
assert tensor.shape[0] == 1
tensor = tensor[0]
return Image.fromarray(tensor)
Listing 12-9
Function to Convert a Tensor to a PIL Image
```

函数将缩放的张量上采样并转换为 NumPy 数组。然后它移除“1”维度并返回一个 PIL 图像。

将文件保存到本地驱动器：

```py
from google.colab import files
fn = 'patiche.jpg'
tensor_to_image(stylized_image_blended).save(fn)
files.download(fn)
```

从 google.colab 库中导入*files*模块。在 PIL 图像上调用函数和保存方法。使用 files 模块的下载方法将 PIL 图像下载到本地驱动器。

## 摘要

第一个实验演示了使用端到端代码实验的 ANAS 实现 NST。第二个实验演示了使用从 TensorFlow Lite 模块预训练的迁移模型的 ANAS 实现 NST。
