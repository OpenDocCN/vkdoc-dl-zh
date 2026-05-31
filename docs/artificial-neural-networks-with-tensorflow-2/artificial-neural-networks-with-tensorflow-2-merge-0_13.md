# 12. 风格迁移

## 引言

你是否曾希望自己能像毕加索或印度著名画家 M.F. 侯赛因那样作画？看起来神经网络已经让你美梦成真。在本章中，你将学习一种利用神经网络，以著名艺术家的风格（或者更确切地说，以你自己选择的风格）来创作你拍摄的照片的技术。这种技术被称为神经风格迁移，由 Leon A. Gatys 的著名论文《一种艺术风格的神经算法》所阐述。虽然这篇论文很值得一读，但理解本章内容并不需要论文中给出的所有细节。

神经风格迁移是一种优化技术，它将一幅图像的内容与另一幅图像的风格融合在一起。你在前一章中学习使用的 TensorFlow Hub 包含一个用于风格迁移的预训练模型。首先，我将向你展示如何使用这个模型快速入门风格迁移学习。接着是一个动手实践的例子，它将教你如何从两幅不同的图像中提取内容和风格，然后对内容图像进行变换，以创建另一幅风格化图像。

为了让你快速了解将要实现的目标，请看表 12-1。

**表 12-1** 内容图像、风格图像和风格化图像

![../images/495303_1_En_12_Chapter/495303_1_En_12_Figa_HTML.gif](img/495303_1_En_12_Figa_HTML.gif)

左侧第一幅图像是内容图像，中间的图像是风格图像，右侧的图像是风格化图像。请注意，中间图像的风格是如何应用于左侧图像的内容，从而生成一幅新的风格化图像的。

风格迁移背后的理论并不复杂，我将在本章后面的自定义风格迁移程序中介绍它。

那么，让我们开始快速风格迁移吧。

## 快速风格迁移

TF Hub 提供了一个用于快速风格迁移的预训练模型。该模块的名称为 `"arbitrary-image-stylization-v1-256/2"`。与使用神经网络进行艺术风格迁移的原始工作相比，它执行的是快速艺术风格迁移。该模型可以处理任意绘画风格。该模型基于 Golnaz 等人在其著名论文《探索实时、任意神经艺术风格化网络的结构》([`https://arxiv.org/abs/1705.06830`](https://arxiv.org/abs/1705.06830)) 中提出的技术。所提出的模型结合了艺术风格神经算法的灵活性和快速风格迁移网络的速度。这使得使用任何内容/风格图像对进行实时风格化成为可能。

你将使用这个托管在 TensorFlow Hub 上的预训练模型，来快速了解风格迁移的效果。

### 创建项目

创建一个新的 Colab 项目，并将其重命名为 `TFHubStyleTransfer`。导入所需的库。

```python
import tensorflow as tf
import re
import urllib
import numpy as np
import matplotlib.pyplot as plt
import PIL.Image
import tensorflow_hub as hub
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from matplotlib import gridspec
from IPython import display
from PIL import Image
```

## 下载图像

该项目在任何时候都需要两张图像——内容图像和风格图像。内容图像将被修改，以适应风格图像中给出的风格。为了便于你测试，我在本书的仓库中上传了一些图像。图像的 URL 格式如下：

```
https://raw.githubusercontent.com/Apress/artificial-neural-networks-with-tensorflow-2/main/ch12/ferns.jpg
```

编写一个函数来提取文件名、下载文件并设置图像的新路径。函数定义很简单，如清单 12-1 所示。

```python
def download_image_from_URL(imageURL):
    imageName = re.search('[a-z0-9\-]+\.(jpe?g|png|gif|bmp|JPG)', imageURL, re.IGNORECASE)
    imageName = imageName.group(0)
    urllib.request.urlretrieve(imageURL, imageName)
    imagePath = "./" + imageName
    return imagePath
```

**清单 12-1** 创建图像 URL 的函数

调用此函数以创建目标图像的路径。

```python
#### 这是你想要变换的图像的路径。
target_url = "https://raw.githubusercontent.com/Apress/artificial-neural-networks-with-tensorflow-2/main/ch12/ferns.jpg"
target_path = download_image_from_URL(target_url)
```

同样地，下载用于风格化的图像并设置其路径。

```python
### 这是风格图像的路径。
style_url = "https://raw.githubusercontent.com/Apress/artificial-neural-networks-with-tensorflow-2/main/ch12/on-the-road.jpg"
style_path = download_image_from_URL(style_url)
```

在 Colab 笔记本中看到的用户界面如图 12-1 所示。

![../images/495303_1_En_12_Chapter/495303_1_En_12_Fig1_HTML.jpg](img/495303_1_En_12_Fig1_HTML.jpg)

**图 12-1** 用于选择图像文件的 Colab 界面

从单元格右侧的下拉列表中选择所需的目标图像和风格图像。

你可以使用以下代码片段，通过 `matplotlib imshow` 函数显示所选的两张图像：

```python
content = Image.open(target_path)
style = Image.open(style_path)
plt.figure(figsize=(10, 10))
plt.subplot(1, 2, 1)
plt.imshow(content)
plt.title('Content Image')
plt.subplot(1, 2, 2)
plt.imshow(style)
plt.title('Style Image')
plt.tight_layout()
plt.show()
```

输出如图 12-2 所示。

![../images/495303_1_En_12_Chapter/495303_1_En_12_Fig2_HTML.jpg](img/495303_1_En_12_Fig2_HTML.jpg)

**图 12-2** 内容图像和风格图像

因此，我们有两张尺寸不同的图像。

### 准备模型输入的图像

`tfhub` 中执行图像变换的模块要求图像采用特定格式才能获得良好效果。首先，我们使用以下函数将风格图像转换为张量：

```python
def image_to_tensor_style(path_to_img):
    img = tf.io.read_file(path_to_img)
    img = tf.image.decode_image(img, channels=3, dtype=tf.float32)
    img = tf.image.resize(img, [256, 256])
    img = img[tf.newaxis, :]
    return img
```

该函数通过调用 `read_file` 方法读取图像数据。它通过调用 `decode_image` 将图像解码为三个 RGB 通道。图像被调整为 `256x256`，因为预训练模型使用此特定尺寸进行风格化。最后，图像数据以形状为 `(1, 256, 256, 3)` 的张量形式返回给调用者。我们为图像添加了一个新维度，稍后在处理一批图像时会用到。

同样，我们编写一个函数将目标图像转换为张量，如下所示：

```python
def image_to_tensor_target(path_to_img, image_size):
    img = tf.io.read_file(path_to_img)
    img = tf.image.decode_image(img, channels=3, dtype=tf.float32)
    img = tf.image.resize(img, [image_size, image_size], preserve_aspect_ratio=True)
    img = img[tf.newaxis, :]
    return img
```

该函数除了图像路径外，还接受一个额外的参数，即用户自定义的图像尺寸。大图像在处理过程中会占用大量内存，因此我添加了一个参数，以便你可以在保持图像宽高比的同时缩小图像尺寸。请注意图像缩放方法调用中的 `preserve_aspect_ratio` 参数。当你之前加载的目标图像被缩放到 400 时，输出张量的形状将为 `(1, 1200, 1600, 3)`。注意，宽高比得以保留。宽高比是图像宽度与高度的比值。如果你将宽度设置为 400，高度将根据此宽高比按比例增加或减少。

现在，你将通过调用之前定义的两个方法将两张图像转换为张量：

```python
output_image_size = 400
target_image = image_to_tensor_target(target_path, output_image_size)
style_image = image_to_tensor_style(style_path)
```

## 执行风格迁移

要将新样式应用于目标图像，你需要从 `tfhub` 加载预训练模块。使用以下语句完成此操作：

```python
hub_module = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')
```

模块名称为 `arbitrary-image-stylization-v1-256/2`。加载模块后，只需将两个张量作为输入传递给模块即可执行转换，如下所示：

```python
outputs = hub_module(tf.constant(target_image), tf.constant(style_image))
stylized_image = outputs[0]
```

`outputs` 是一个形状为 `(1, 300, 400, 3)` 的张量。这是转换后图像的数据。请注意尺寸 300x400，它是原始图像尺寸 1600x1200 的缩小版本。

## 显示输出

要显示图像，我们需要使用以下代码将张量转换为图像格式：

```python
tensor = stylized_image * 256
tensor = np.array(tensor, dtype=np.uint8)
tensor = tensor[0]
PIL.Image.fromarray(tensor)
```

我们需要将图像数据乘以 256 的缩放比例，因为 `stylized_image` 包含的数据范围是 0 到 1。输出图像如图 12-3 所示。

![../images/495303_1_En_12_Chapter/495303_1_En_12_Fig3_HTML.jpg](img/495303_1_En_12_Fig3_HTML.jpg)

**图 12-3** 风格化图像

如果你想显示缩小的图像，如图 12-4 所示，只需按如下方式调用 `imshow` 方法：

![../images/495303_1_En_12_Chapter/495303_1_En_12_Fig4_HTML.jpg](img/495303_1_En_12_Fig4_HTML.jpg)

**图 12-4** 缩小的风格化图像

```python
plt.imshow(tensor)
```

## 更多结果

我对项目中加载的图像进行了更多转换。结果如表 12-2 所示。

**表 12-2** 不同图像的模型推理结果

- ![../images/495303_1_En_12_Chapter/495303_1_En_12_Figb_HTML.gif](img/495303_1_En_12_Figb_HTML.gif)

## 完整源代码

`TFHubStyleTransfer` 的完整源代码如代码清单 12-2 所示。

```python
import tensorflow as tf
import re
import urllib
import numpy as np
import matplotlib.pyplot as plt
import PIL.Image
import tensorflow_hub as hub
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from matplotlib import gridspec
from IPython import display
from PIL import Image

def download_image_from_URL(imageURL):
    imageName = re.search('[a-z0-9\-]+\.(jpe?g|png|gif|bmp|JPG)', imageURL, re.IGNORECASE)
    imageName = imageName.group(0)
    urllib.request.urlretrieve(imageURL, imageName)
    imagePath = "./" + imageName
    return imagePath

### 这是你想要转换的图像路径。
target_url = "https://raw.githubusercontent.com/Apress/artificial-neural-networks-with-tensorflow-2/main/ch12/ferns.jpg"
target_path = download_image_from_URL(target_url)

### 这是风格图像的路径。
style_url = "https://raw.githubusercontent.com/Apress/artificial-neural-networks-with-tensorflow-2/main/ch12/on-the-road.jpg"
style_path = download_image_from_URL(style_url)

content = Image.open(target_path)
style = Image.open(style_path)

plt.figure(figsize=(10, 10))
plt.subplot(1, 2, 1)
plt.imshow(content)
plt.title('内容图像')
plt.subplot(1, 2, 2)
plt.imshow(style)
plt.title('风格图像')
plt.tight_layout()
plt.show()

def image_to_tensor_style(path_to_img):
    img = tf.io.read_file(path_to_img)
    img = tf.image.decode_image(img, channels=3, dtype=tf.float32)
    img = tf.image.resize(img, [256, 256])
    img = img[tf.newaxis, :]
    return img

def image_to_tensor_target(path_to_img, image_size):
    img = tf.io.read_file(path_to_img)
    img = tf.image.decode_image(img, channels=3, dtype=tf.float32)
    img = tf.image.resize(img, [image_size, image_size], preserve_aspect_ratio=True)
    img = img[tf.newaxis, :]
    return img

output_image_size = 400
target_image = image_to_tensor_target(target_path, output_image_size)
style_image = image_to_tensor_style(style_path)

hub_module = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

outputs = hub_module(tf.constant(target_image), tf.constant(style_image))
stylized_image = outputs[0]

tensor = stylized_image * 256
tensor = np.array(tensor, dtype=np.uint8)
tensor = tensor[0]
PIL.Image.fromarray(tensor)

plt.imshow(tensor)
```

**代码清单 12-2** `TFHubStyleTransfer` 完整源代码

## 动手实践

学习了如何进行快速风格迁移后，现在是时候了解这些项目背后的技术了。风格迁移的原理是提取一张图像（通常是著名画作）的风格，并将其应用于你选择的图像内容上。因此，有两个输入图像，即内容图像和风格图像。新生成的图像通常称为风格化图像。生成的图像包含与内容图像相同的内容，但获得了与*风格*图像相似的风格。正如你所理解的，这显然不是通过简单叠加图像来实现的。因此，我们的程序必须能够分别区分给定图像的内容和风格。这时我们将使用 `VGG16` 预训练网络模型来提取这些信息，并构建我们自己的网络，基于这些输入创建风格化图像。诸如 Prisma 和 Lucid 等 Android 应用就是进行此类风格迁移的。虽然你不会被教授如何开发类似的 Android 应用，但这个项目将教会你此类应用的内部原理。

让我们首先看看 `VGG16` 架构，以了解如何从图像中提取内容和风格。

## VGG16 架构

# 使用 VGG16 进行风格迁移

`Gatys 等人 (2015)` 提出了风格迁移背后的核心思想。其核心思想是，用于图像分类的预训练 CNN（卷积神经网络）能够编码图像的感知和语义信息。目前世界上有许多这样的预训练 CNN 可供使用。我们将使用 `VGG16` 来提取图像的特征，然后分别处理其内容和风格。原始论文使用了来自 `Simonyan 和 Zisserman (2015)` 的 19 层 VGG 网络模型。`VGG16` 模型架构如图 12-5 所示。

![../images/495303_1_En_12_Chapter/495303_1_En_12_Fig5_HTML.jpg](img/495303_1_En_12_Fig5_HTML.jpg)

**图 12-5** VGG16 架构（图片来源：[researchgate.net](https://www.researchgate.net)）

由于我们不是在进行图像分类，而只对特征提取感兴趣，因此我们不需要 VGG 网络的全连接层或最终的 softmax 分类器。我们只需要模型的一部分。那么，我们如何只提取模型的特定部分呢？幸运的是，这对我们来说是一项非常简单的任务，因为 Keras 提供了一个预训练的 `VGG16` 模型，你可以从中分离出各层。Keras 还提供了许多其他模型，包括后来的 `VGG19`。要移除顶部的全连接层，你需要在提取模型层时将 `include_top` 变量的值设置为 `False`。

### 创建项目

创建一个新的 Colab 项目，并将其重命名为 `CustomStyleTransfer`。安装以下两个包：

```
!pip install keras==2.3.1
!pip install tensorflow==2.1.0
```

**注意：** 在发布时发现，本项目使用的预训练 `VGG16` 模型在上述指定的 Keras 和 TensorFlow 版本下运行，并且截至撰写本文时，尚不支持更新的版本。

导入所需的库。

```
import tensorflow as tf
import re
import urllib
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from matplotlib import pyplot as plt
from IPython import display
from PIL import Image
import numpy as np
from tensorflow.keras.applications import vgg16
from tensorflow.keras import backend as K
from keras import backend as K
from scipy.optimize import fmin_l_bfgs_b
```

## 下载图像

与之前的项目一样，你需要编写一个下载函数并调用它来下载本项目所需的两张图像。代码如代码清单 12-3 所示。

```
def download_image_from_URL(imageURL):
    imageName = re.search('[a-z0-9\-]+\.(jpe?g|png|gif|bmp|JPG)', imageURL, re.IGNORECASE)
    imageName = imageName.group(0)
    urllib.request.urlretrieve(imageURL, imageName)
    imagePath = "./" + imageName
    return imagePath

### 这是你想要转换的目标图像的路径。
target_url = "https://raw.githubusercontent.com/Apress/artificial-neural-networks-with-tensorflow-2/main/ch12/blank-sign.jpg"
target_path = download_image_from_URL(target_url)

### 这是风格图像的路径。
style_url = "https://raw.githubusercontent.com/Apress/artificial-neural-networks-with-tensorflow-2/main/ch12/road.jpg"
```

**代码清单 12-3** 下载图像的函数

我们将目标图像缩放为高度 400 像素。为了保持宽高比，我们按如下方式重新计算宽度：

```
width, height = load_img(target_path).size
img_height = 400
img_width = int(width * img_height / height)
```

## 显示图像

为了显示这两张图像，我们使用与上一个项目类似的代码。代码如代码清单 12-4 所示。

```
content = Image.open(target_path)
style = Image.open(style_path)

plt.figure(figsize=(10, 10))
plt.subplot(1, 2, 1)
plt.imshow(content)
plt.title('内容图像')
plt.subplot(1, 2, 2)
plt.imshow(style)
plt.title('风格图像')
plt.tight_layout()
plt.show()
```

**代码清单 12-4** 显示内容和风格图像

输出如图 12-6 所示。

![../images/495303_1_En_12_Chapter/495303_1_En_12_Fig6_HTML.jpg](img/495303_1_En_12_Fig6_HTML.jpg)

**图 12-6** 内容和风格图像

## 预处理图像

如前所述，你将使用 `VGG16` 模型来提取图像中的特征。我们需要根据 VGG 训练过程来处理图像数据。幸运的是，Keras 不仅为 `VGG16` 提供了这种预处理，还为许多其他流行模型（如 ResNet、Inception、DenseNet 等）提供了预处理。该库提供了一个名为 `preprocess_input` 的函数，该函数接受一个编码了一批图像的张量或 numpy 数组作为输入，并返回一个预处理后的 numpy 数组或类型为 `float32` 的 `tf.tensor`。该方法将图像从 RGB 转换为 BGR，并对每个通道进行零中心化。请注意，VGG 网络是在每个通道均值为 `[103.939, 116.779, 123.68]` 且通道顺序为 BGR（蓝/绿/红）的图像上训练的。代码清单 12-5 中的代码对给定图像执行此预处理。

```
def preprocess_image(image_path):
    img = load_img(image_path, target_size=(img_height, img_width))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = tf.keras.applications.vgg16.preprocess_input(img)
    return img
```

**代码清单 12-5** 为 VGG16 网络预处理图像

如果你希望查看输出，我们需要进行反向预处理。此外，我们必须将所有值裁剪到 0–255 范围内。我们在以下函数定义中执行此操作：

```
def deprocess_image(x):
    # 通过均值像素移除零中心化
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    # 'BGR' -> 'RGB'
    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype('uint8')
    return x
```

现在，你将基于 `VGG16` 模型构建模型。

### 模型构建

为了构建模型，我们将图像张量数据输入 `VGG16`，并提取特征图、内容表示和风格表示。模型将加载预训练的 ImageNet 权重。模型构建代码如下所示：

```
target = K.constant(preprocess_image(target_path))
style = K.constant(preprocess_image(style_path))

### 这个占位符将包含我们生成的图像
combination_image = K.placeholder((1, img_height, img_width, 3))

### 我们将三张图像合并成一个批次
input_tensor = K.concatenate([target, style, combination_image], axis=0)

### 使用我们的三张图像批次作为输入来构建 VGG16 网络。
model = vgg16.VGG16(input_tensor=input_tensor, weights='imagenet', include_top=False)
```

在这段代码中，我们首先通过调用之前定义的 `preprocess` 方法来构建内容和风格图像的输入张量。我们为目标图像创建一个占位符，然后通过调用 `concatenate` 方法为三张图像创建一个张量。然后将其作为参数传递给 `VGG16` 方法，以提取我们所需的模型。

现在，你可以查看模型摘要。

```
model.summary()
```

摘要输出如图 12-7 所示。

![../images/495303_1_En_12_Chapter/495303_1_En_12_Fig7_HTML.jpg](img/495303_1_En_12_Fig7_HTML.jpg)

**图 12-7** 模型摘要

## 内容损失

我们将在每个期望的层计算内容损失，并将它们相加。在每次迭代中，我们将输入图像馈送到模型。模型将正确计算所有内容损失，并且由于我们使用的是即时执行模式，所有梯度也将被计算。内容损失表示随机生成的噪声图像 (G) 与内容图像 (C) 的相似程度。内容损失的计算方式如下。

假设我们选择预训练网络（VGG 网络）中的一个隐藏层 (L) 来计算损失。令 P 和 F 分别代表原始图像和生成的图像。令 `F[l]` 和 `P[l]` 为层 L 中各自图像的特征表示。那么，内容损失定义如下：

![$$ {L}_{content}\left(\overrightarrow{P},\overrightarrow{X},l\right)=\frac{1}{2}\sum \limits_{ij}{\left({F}_{ij}^l-{P}_{ij}^l\right)}² $$](img/495303_1_En_12_Chapter_TeX_Equa.png)

我们将内容损失的公式编码如下：

```
def content_loss(base, combination):
    return K.sum(K.square(combination - base))
```

## 风格损失

为了计算风格损失，我们首先需要计算 Gram 矩阵。Gram 矩阵是一个额外的预处理步骤，用于找出不同通道之间的相关性，这些相关性随后将用于衡量风格本身。

我们定义 Gram 矩阵如下：

```
def gram_matrix(x):
    features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
    gram = K.dot(features, K.transpose(features))
    return gram
```

风格损失会计算风格图像和生成图像的 Gram 矩阵，然后将代价返回给调用者。该代价是风格图像的 Gram 矩阵与生成图像的 Gram 矩阵之差的平方。用数学公式表示如下：

![$$ {L}_{GM}\left(S,G,l\right)=\frac{1}{4{N}_l²{M}_l²}\sum \limits_{ij}{\left( GM\left[l\right]{(S)}_{ij}- GM\left[l\right]{(G)}_{ij}\right)}² $$](images/495303_1_En_12_Chapter/495303_1_En_12_Chapter_TeX_Equb.png)

风格损失函数的定义如下：

```
def style_loss(style, combination):
    S = gram_matrix(style)
    C = gram_matrix(combination)
    channels = 3
    size = img_height * img_width
    return K.sum(K.square(S - C)) / (4. * (channels ** 2) * (size ** 2))
```

## 总变差损失

为了对输出进行正则化以实现平滑，我们定义了相邻像素的总变差损失，如下所示：

```
def total_variation_loss(x):
    a = K.square(x[:, :img_height - 1, :img_width - 1, :] - x[:, 1:, :img_width - 1, :])
    b = K.square(x[:, :img_height - 1, :img_width - 1, :] - x[:, :img_height - 1, 1:, :])
    return K.sum(K.pow(a + b, 1.25))
```

## 计算内容和风格的损失

我们首先选择 VGG16 的内容层和风格层。我使用了 Johnson 等人（2016）定义的层，而不是 Gatys 等人（2015）建议的层，因为这样能产生更好的最终结果。

首先，我们将所有层映射到一个字典中。

```
### 将层名称映射到激活张量的字典
outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])
```

我们提取内容层：

```
### 用于内容损失的层名称
content_layer = 'block5_conv2'
```

我们提取风格层：

```
### 用于风格损失的层名称列表
style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']
```

我们定义几个权重变量，用于计算损失分量的加权平均值。可以将它们视为风格层和内容层的超参数，决定这些层在最终模型中的权重。

```
total_variation_weight = 1e-4
style_weight = 10.
content_weight = 0.025
```

我们通过添加所有分量来计算总损失。

```
### 通过将所有分量添加到一个 `loss` 变量中来定义损失
loss = K.variable(0.)
layer_features = outputs_dict[content_layer]
target_features = layer_features[0, :, :, :]
combination_features = layer_features[2, :, :, :]
loss = loss + content_weight * content_loss(target_features, combination_features)
for layer_name in style_layers:
    layer_features = outputs_dict[layer_name]
    style_reference_features = layer_features[1, :, :, :]
    combination_features = layer_features[2, :, :, :]
    sl = style_loss(style_reference_features, combination_features)
    loss += (style_weight / len(style_layers)) * sl
loss += total_variation_weight * total_variation_loss(combination_image)
```

## 评估器类

最后，我们将定义一个名为 `Evaluator` 的类，用于在一次传递中计算损失和梯度。

```
grads = K.gradients(loss, combination_image)[0]
### 用于获取当前损失值和当前梯度值的函数
fetch_loss_and_grads = K.function([combination_image], [loss, grads])

class Evaluator(object):
    def __init__(self):
        self.loss_value = None
        self.grads_values = None

    def loss(self, x):
        assert self.loss_value is None
        x = x.reshape((1, img_height, img_width, 3))
        outs = fetch_loss_and_grads([x])
        loss_value = outs[0]
        grad_values = outs[1].flatten().astype('float64')
        self.loss_value = loss_value
        self.grad_values = grad_values
        return self.loss_value

    def grads(self, x):
        assert self.loss_value is not None
        grad_values = np.copy(self.grad_values)
        self.loss_value = None
        self.grad_values = None
        return grad_values

evaluator = Evaluator()
```

## 生成输出图像

现在我们已经准备好了所有实用函数，是时候生成风格化图像了。我们从一组随机像素（一张随机图像）开始，并使用 L-BFGS（有限内存 Broyden-Fletcher-Goldfarb-Shanno）算法进行优化。该算法使用二阶导数来最小化或最大化函数，并且比标准梯度下降法快得多。训练循环如下所示：

```
iterations = 50
x = preprocess_image(target_path)
x = x.flatten()
for i in range(1, iterations):
    x, min_val, info = fmin_l_bfgs_b(evaluator.loss, x, fprime=evaluator.grads, maxfun=10)
    print('Iteration %0d, loss: %0.02f' % (i, min_val))
img = x.copy().reshape((img_height, img_width, 3))
img = deprocess_image(img)
```

训练结束时，我们将最终输出图像复制到一个变量中，并重新处理它，使其准备好显示。

## 显示图像

我们现在使用以下代码显示所有三张图像：

```
plt.figure(figsize=(50, 50))
plt.subplot(3,3,1)
plt.imshow(load_img(target_path, target_size=(img_height, img_width)))
plt.subplot(3,3,2)
plt.imshow(load_img(style_path, target_size=(img_height, img_width)))
plt.subplot(3,3,3)
plt.imshow(img)
plt.show()
```

输出图像如图 12-8 所示。

![../images/495303_1_En_12_Chapter/495303_1_En_12_Fig8_HTML.jpg](img/495303_1_En_12_Fig8_HTML.jpg)

**图 12-8** 内容图像、风格图像和风格化图像

## 完整源代码

`CustomStyleTransfer` 的完整源代码见代码清单 12-6。

```python
!pip install keras==2.3.1
!pip install tensorflow==2.1.0
import tensorflow as tf
import re
import urllib
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from matplotlib import pyplot as plt
from IPython import display
from PIL import Image
import numpy as np
from tensorflow.keras.applications import vgg16
from tensorflow.keras import backend as K
from keras import backend as K
from scipy.optimize import fmin_l_bfgs_b

def download_image_from_URL(imageURL):
    imageName = re.search('[a-z0-9\-]+\.(jpe?g|png|gif|bmp|JPG)', imageURL, re.IGNORECASE)
    imageName = imageName.group(0)
    urllib.request.urlretrieve(imageURL, imageName)
    imagePath = "./" + imageName
    return imagePath

### 这是你想要转换的图片路径。
target_url = "https://raw.githubusercontent.com/Apress/artificial-neural-networks-with-tensorflow-2/main/ch12/blank-sign.jpg"
target_path = download_image_from_URL(target_url)

### 这是风格图片的路径。
style_url = "https://raw.githubusercontent.com/Apress/artificial-neural-networks-with-tensorflow-2/main/ch12/road.jpg"
style_path = download_image_from_URL(style_url)

### 生成图片的尺寸。
width, height = load_img(target_path).size
img_height = 400
img_width = int(width * img_height / height)

content = Image.open(target_path)
style = Image.open(style_path)

plt.figure(figsize=(10, 10))
plt.subplot(1, 2, 1)
plt.imshow(content)
plt.title('内容图片')
plt.subplot(1, 2, 2)
plt.imshow(style)
plt.title('风格图片')
plt.tight_layout()
plt.show()

### 根据 VGG16 的要求预处理数据
def preprocess_image(image_path):
    img = load_img(image_path, target_size=(img_height, img_width))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = tf.keras.applications.vgg16.preprocess_input(img)
    return img

def deprocess_image(x):
    # 通过均值像素去除零中心化
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    # 'BGR' -> 'RGB'
    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype('uint8')
    return x

target = K.constant(preprocess_image(target_path))
style = K.constant(preprocess_image(style_path))

### 这个占位符将包含我们生成的图片
combination_image = K.placeholder((1, img_height, img_width, 3))

### 我们将三张图片合并成一个批次
input_tensor = K.concatenate([target, style, combination_image], axis=0)

### 以包含三张图片的批次作为输入，构建 VGG16 网络。
model = vgg16.VGG16(input_tensor=input_tensor, weights='imagenet', include_top=False)
model.summary()

### 计算生成图片的内容损失
def content_loss(base, combination):
    return K.sum(K.square(combination - base))

def gram_matrix(x):
    features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
    gram = K.dot(features, K.transpose(features))
    return gram

def style_loss(style, combination):
    S = gram_matrix(style)
    C = gram_matrix(combination)
    channels = 3
    size = img_height * img_width
    return K.sum(K.square(S - C)) / (4. * (channels ** 2) * (size ** 2))

def total_variation_loss(x):
    a = K.square(x[:, :img_height - 1, :img_width - 1, :] - x[:, 1:, :img_width - 1, :])
    b = K.square(x[:, :img_height - 1, :img_width - 1, :] - x[:, :img_height - 1, 1:, :])
    return K.sum(K.pow(a + b, 1.25))

### 将层名称映射到激活张量的字典
outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])

### 用于内容损失的层名称
content_layer = 'block5_conv2'

### 用于风格损失的层名称；
style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']

### 损失分量加权平均中的权重
total_variation_weight = 1e-4
style_weight = 10.
content_weight = 0.025

### 通过将所有分量添加到 `loss` 变量来定义损失
loss = K.variable(0.)
layer_features = outputs_dict[content_layer]
target_features = layer_features[0, :, :, :]
combination_features = layer_features[2, :, :, :]
loss = loss + content_weight * content_loss(target_features, combination_features)

for layer_name in style_layers:
    layer_features = outputs_dict[layer_name]
    style_reference_features = layer_features[1, :, :, :]
    combination_features = layer_features[2, :, :, :]
    sl = style_loss(style_reference_features, combination_features)
    loss += (style_weight / len(style_layers)) * sl

loss += total_variation_weight * total_variation_loss(combination_image)

grads = K.gradients(loss, combination_image)[0]

### 用于获取当前损失值和当前梯度值的函数
fetch_loss_and_grads = K.function([combination_image], [loss, grads])

class Evaluator(object):
    def __init__(self):
        self.loss_value = None
        self.grads_values = None

    def loss(self, x):
        assert self.loss_value is None
        x = x.reshape((1, img_height, img_width, 3))
        outs = fetch_loss_and_grads([x])
        loss_value = outs[0]
        grad_values = outs[1].flatten().astype('float64')
        self.loss_value = loss_value
        self.grad_values = grad_values
        return self.loss_value

    def grads(self, x):
        assert self.loss_value is not None
        grad_values = np.copy(self.grad_values)
        self.loss_value = None
        self.grad_values = None
        return grad_values

evaluator = Evaluator()

iterations = 50
x = preprocess_image(target_path)
x = x.flatten()

for i in range(1, iterations):
    x, min_val, info = fmin_l_bfgs_b(evaluator.loss, x, fprime=evaluator.grads, maxfun=10)
    print('迭代次数 %0d，损失值: %0.02f' % (i, min_val))

img = x.copy().reshape((img_height, img_width, 3))
img = deprocess_image(img)

plt.figure(figsize=(50, 50))
plt.subplot(3,3,1)
plt.imshow(load_img(target_path, target_size=(img_height, img_width)))
plt.subplot(3,3,2)
plt.imshow(load_img(style_path, target_size=(img_height, img_width)))
plt.subplot(3,3,3)
plt.imshow(img)
plt.show()

列表 12-6
CustomStyleTransfer 完整源代码
```

# 总结

在本章中，你学习了神经网络中的另一项重要技术——神经风格迁移。该技术允许你将所选图像的内容转换为另一幅图像的风格。你学会了通过两种不同方式实现风格迁移。第一种方法是使用`tfhub`中提供的预训练模型来执行快速的艺术风格迁移。使用这种方法进行风格迁移速度快且效果出色。第二种方法是构建你自己的网络，从核心层面实现风格迁移。我们使用了用于图像分类的`VGG16`预训练模型来提取图像的内容和风格。接着，你学会了创建一个网络，该网络通过多次迭代学习如何将风格应用于给定的内容。这种方法允许你自行进行风格迁移实验。

在下一章中，你将学习如何使用生成对抗网络（GAN）生成图像。

