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



