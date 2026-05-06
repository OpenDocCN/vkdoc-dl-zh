# 4. 迁移学习

在上一章中，你开发了一个二值图像分类器。使用 60,000 张图像，训练模型花费了一些时间。我们达到的准确率约为 80% 到 90%。如果你想要更高的准确率，则需要更多图像进行训练。事实上，深度学习网络在拥有更多数据点时学习效果更好。ImageNet（`https://devopedia.org/imagenet`）是首个在规模上达到如此程度的数据库，它包含 14,197,122 张图像，分为 21,841 个子类别，这些子类别又进一步分为 27 个子树。为了对 ImageNet 中的图像进行分类，人们开发了许多机器学习模型；这些模型主要是作为研究和竞赛的一种方式而开发的。2017 年，其中一个模型实现了低至 2.3% 的错误率。其底层网络非常复杂。考虑到这样一个复杂网络的可训练参数数量，想象一下训练该模型所需的资源和时间。

现在的问题是，我们能否重用这些网络所学到的知识，并利用它们的知识为我们自身谋利？而这正是本章的全部内容。其背后的技术称为迁移学习。这就像人类将他们的学识传授给年轻一代。在本章中，你将学习如何将其他网络的学习成果转化为自身优势，并在它们工作的基础上开发自己的新模型。

简而言之，本章涵盖以下内容：

-   知识迁移是什么意思？
-   什么是 TensorFlow Hub？
-   有哪些可用的预训练模型？
-   如何使用预训练模型？
-   如何使用迁移学习技术创建犬种分类器？

那么，让我们从这个问题开始：什么是知识迁移？

## 知识迁移

在过去的几十年里，开发人员一直使用二进制库来重用自己或他人共享的代码。在机器学习中，我们能否通过重用他人开发的模型来促进自身发展，从而应用相同的概念？这并不像听起来那么简单。在软件库中，共享的只是代码。在机器学习中，需要共享的远不止代码。一个训练好的模型的四个重要组成部分如下：

-   算法
-   数据
-   训练
-   专业知识

首先，是某人开发的用于训练神经网络的算法。这是知识迁移的代码部分。第二是数据。通常，模型借助大量数据点进行学习。第三是训练——训练一个模型需要巨大的处理能力和时间。训练网络时会消耗大量资源。最后，是领域专家的知识和专业技能，它们会内在地嵌入到训练好的模型中。那么，我们如何将这四个方面迁移到一个新模型中呢？为了促进这一点，TensorFlow 的研究人员受到启发，开发了 TensorFlow Hub。

### TensorFlow Hub

TensorFlow Hub 是 TensorFlow 提供的一个平台，用于发布预训练模型。其他用户可以从中发现并重用这些模型中的部分机器学习模块。那么，什么是模块？模块的可视化表示如图 4-1 所示。

![../images/495303_1_En_4_Chapter/495303_1_En_4_Fig1_HTML.jpg](img/495303_1_En_4_Fig1_HTML.jpg)

图 4-1 模块图

模块本质上是 TensorFlow 图的一个自包含片段，连同其训练权重。该图可以在其他地方重用，以执行类似的任务。开发者可以向该图添加更多层来创建自己的模型。然后，可以使用较小的数据集训练新模型，从而加快训练速度。从某种意义上说，这也会产生泛化效果，使得泛化模块可以在多个模型中重用。你可以将模型想象成软件工程中的最终可执行文件（二进制文件），而模块则像是用于创建可执行文件的通用库。

模块是可组合的，如图 4-2 所示。

![../images/495303_1_En_4_Chapter/495303_1_En_4_Fig2_HTML.jpg](img/495303_1_En_4_Fig2_HTML.jpg)

图 4-2 TensorFlow Hub 架构

你可以在预训练模块之上添加自己的层来创建自己的网络。然后，可以训练整个网络，包括嵌入的预训练模块。

模块是可重用且可重新训练的，只需向函数调用传递一个参数即可。当我说它们是可重新训练时，意味着你可以像对待普通神经网络一样通过它们进行反向传播。请注意，如果你进行重新训练，请确保使用较低的学习率；否则，现有的权重可能会失控，产生完全意想不到的结果。

当你创建自己的模型，并且觉得该模型可能对社区有用时，你可以将其提交给 Google，以便纳入 TensorFlow Hub。Hub 中目前有许多第三方模块。Google 警告你要谨慎使用第三方模块，尤其是在你不信任其来源的情况下。

那么，目前有哪些模块可供你使用呢？



### 预训练模块

TensorFlow Hub 中提供了由谷歌及其渠道合作伙伴开发的多个预训练模块。这些模块分为三类：图像、文本和视频。其中“图像”领域类别如图 4-3 所示。

![../images/495303_1_En_4_Chapter/495303_1_En_4_Fig3_HTML.jpg](img/495303_1_En_4_Fig3_HTML.jpg)

图 4-3 TensorFlow 中的预定义模型

图像领域中的部分模块包括：

-   使用 MobileNet/Inception ResNet V2/NASNet-A 进行 ImageNet 分类
-   使用 MobileNet/Inception/ResNet/PNASnet 提取图像特征向量

上述模块支持 TF2。此外，还有一份详尽的模块列表，这些模块由谷歌及其他合作伙伴开发，运行于 TF1 环境下，并正在等待迁移至 TF2。以下是一些你可能感兴趣的 TF1 类别下的模块：

-   Progressive GAN
-   快速任意图像风格迁移
-   执行随机裁剪、小角度旋转和颜色失真的图像增强模块
-   预测任意照片大致地理位置的 PlaNet
-   Google landmarks - 深度局部特征 (DELF)

在文本类别下，你将找到如下模块：

-   ELMO
-   BERT
-   通用句子编码器
-   基于英文维基百科语料库训练的令牌级文本嵌入

在视频类别下，截至撰写本文时，尚无适用于 TF2 的模型。在 TF1 环境下，有谷歌和 DeepMind 创建的少量模型。

再次强调，TF1 类别下的模块总数非常庞大。你可以访问 TensorFlow Hub 网站 ([`https://tfhub.dev/`](https://tfhub.dev/)) 查看所有可用模块的完整列表。

现在，你已经了解了有哪些资源可供使用。接下来的问题是，如何使用这样的预训练模块。

### 使用模块

要在你的程序代码中使用预训练模块，请从 `tfdev` 网站选择所需的模块。查找“模型格式”；在“保存的模型”下，你将找到该模块的 URL，如图 4-4 的截图所示。

![../images/495303_1_En_4_Chapter/495303_1_En_4_Fig4_HTML.jpg](img/495303_1_En_4_Fig4_HTML.jpg)

图 4-4 模块的 URL

复制该 URL，并按如下所示在你的程序代码中使用它：

```
module_url = "https://tfhub.dev/google/imagenet/mobilenet_v2_100_160/feature_vector/4"
my_model = hub.KerasLayer(module_url)
```

然后，你可以向 `my_model` 添加更多层。使用 `fit` 方法训练你的新模型，就像训练其他模型一样。训练完成后，用它进行预测。

了解了预训练模块的能力之后，是时候动手尝试了。

### ImageNet 分类器

在本项目中，你将使用谷歌提供的名为 `mobilenet_v2` 的 ImageNet 分类器。该分类器提供 1001 种不同的分类。该模型基于超过一百万张图像进行训练。它使用的初始学习率为 0.045，每个 epoch 的学习率衰减率为 0.98。它使用了 16 个 GPU 异步工作器，批量大小为 96。你可以想象创建此模型所涉及的资源数量。你将利用迁移学习，将这种知识迁移到在此之上构建你自己的图像分类器。你将在本章接下来的两个项目中完成这项工作。第一个项目将教你如何使用 MobileNet 模型对你自己的图像进行分类。第二个项目将向你展示如何扩展它以添加更多你自己的分类。

首先，我将向你展示如何使用 ImageNet 分类器对你自己的图像进行分类。本质上，在本项目中，你将按原样使用 MobileNet 分类器，而不向其添加任何层。你只需学习如何加载预训练模型并将其应用于你自己的图像。本章后面的项目（犬种分类器）将深入探讨许多细节，届时你将向 ImageNet 模块添加你自己的分类层，并进行大量研究，以了解迁移学习带来的好处。

那么，首先在 Colab 中创建一个名为 `ImageNetClassifier` 的项目。

### 设置项目

使用以下代码片段设置 TensorFlow 并导入所需的包：

```
import tensorflow as tf
#### 其他导入
import tensorflow_hub as hub
import numpy as np
import matplotlib.pyplot as plt
```

请注意，你需要导入 `tensorflow_hub`。另外两个导入 `numpy` 和 `matplotlib` 的作用与往常一样；你在之前的项目中已经使用过它们。

### 分类器 URL

本项目将使用预训练的 MobileNet 分类器。在 TensorFlow Hub 上找到此分类器。在 `tfhub.dev` 网站上搜索的截图如图 4-5 所示。

![../images/495303_1_En_4_Chapter/495303_1_En_4_Fig5_HTML.jpg](img/495303_1_En_4_Fig5_HTML.jpg)

图 4-5 TensorFlow Hub 上的 MobileNet 模型

点击“复制 URL”按钮复制模型的 URL。你将获得以下 URL：

```
https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/4
```

请注意，你获得的版本可能比此处显示的更新。

在你的项目代码中声明两个变量，如下所示：

```
classifier_url = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/4"
IMAGE_SHAPE = (224, 224)
```

根据模型描述，该模型的输入需要 224x224 像素的图像。模型描述可在用户指南中找到，你可以通过点击图 4-5 中显示的“我们的用户指南”链接来阅读。

## 创建模型

接下来，你使用 Sequential API 创建模型，如下所示：

```
#### 创建一个模型
classifier = tf.keras.Sequential([
hub.KerasLayer(classifier_url, input_shape = IMAGE_SHAPE+(3,))
])
```

`hub.KerasLayer` 返回添加到我们顺序模型中的模块。该模块中包含的所有层现在都可供我们的网络使用。我们不会向此模型添加任何我们自己的层。因此，我们的模型定义现已完成。由于这是一个完全训练好的模型，我们无需在其上调用 `fit` 方法进行进一步训练。你只需调用其 `predict` 方法来推断任何给定的输入图像。在调用 `predict` 之前，让我们准备一些图像作为模型的输入。



### 准备图像

我在本书的下载站点上上传了几张图片。这些图片的 URL 已列在下方。请将以下代码添加到你的项目中：

```
#### 设置图片下载的 URL
image_url1 = "https://raw.githubusercontent.com/Apress/artificial-neural-networks-with-tensorflow-2/main/Ch04/bulck_cart.jpg"
image_url2 = "https://raw.githubusercontent.com/Apress/artificial-neural-networks-with-tensorflow-2/main/Ch04/flower.jpg"
image_url3 = "https://raw.githubusercontent.com/Apress/artificial-neural-networks-with-tensorflow-2/main/Ch04/swordweapon.jpg"
image_url4 = "https://raw.githubusercontent.com/Apress/artificial-neural-networks-with-tensorflow-2/main/Ch04/tiger.jpg"
image_url5 = "https://raw.githubusercontent.com/Apress/artificial-neural-networks-with-tensorflow-2/main/Ch04/tree.jpg"
```

你将使用以下代码将这些图片下载到你的驱动器中：

```
#### 下载图片
!pip install wget
import wget
wget.download(image_url1,'image1.jpg')
wget.download(image_url2,'image2.jpg')
wget.download(image_url3,'image3.jpg')
wget.download(image_url4,'image4.jpg')
wget.download(image_url5,'image5.jpg')
```

我们使用 `wget` 来下载这五个图片文件。它们会被存储在你的驱动器的 `/content/` 文件夹中，文件名分别为 `image1`、`image2`，以此类推。

由于这些图片尺寸不一，你需要在将它们加载到内存后，将其大小调整为 224x224。同时，你还需要将数据重新缩放到 0 到 1 的范围内，以便进行更好的机器学习。所有这些操作都在以下代码中完成：

```
#### 加载图片并调整为模型所需的 224x224 尺寸
import PIL.Image as Image
image1 = tf.keras.utils.get_file("/content/image1.jpg", image_url1)
image1 = Image.open(image1).resize(IMAGE_SHAPE)
#### 加载图片并调整为模型所需的 224x224 尺寸
import PIL.Image as Image
image1 = tf.keras.utils.get_file("/content/image1.jpg", image_url1)
image1 = Image.open(image1).resize(IMAGE_SHAPE)
#### 缩放数组
image1 = np.array(image1)/255.0
image2 = tf.keras.utils.get_file("/content/image2.jpg", image_url2)
image2 = Image.open(image2).resize(IMAGE_SHAPE)
image2 = np.array(image2)/255.0
image3 = tf.keras.utils.get_file("/content/image3.jpg", image_url3)
image3 = Image.open(image3).resize(IMAGE_SHAPE)
image3 = np.array(image3)/255.0
image4 = tf.keras.utils.get_file("/content/image4.jpg", image_url4)
image4 = Image.open(image4).resize(IMAGE_SHAPE)
image4 = np.array(image4)/255.0
image5 = tf.keras.utils.get_file("/content/image5.jpg", image_url5)
image5 = Image.open(image5).resize(IMAGE_SHAPE)
image5 = np.array(image5)/255.0
```

现在，你已经准备好对图片进行推理了。那么，让我们从第一张图片开始。你通过调用模型的 `predict` 方法来推理图片：

```
result = classifier.predict(image1[np.newaxis, ...])
```

概率预测结果现在存储在 `result` 张量中。使用 `shape` 方法打印结果的形状：

```
result.shape
(1, 1001)
```

你可以看到 `result` 数组中有 1001 个值，表明输出中有 1001 个类别。概率预测结果按升序排列。因此，选取最后一个值作为我们模型做出的最高预测。

```
predicted_class = np.argmax(result[0], axis=-1)
predicted_class
```

如你所见，预测的类别是 293。这个整数对我们来说毫无意义。幸运的是，Google 提供了这些整数到标签的映射。

### 加载标签映射

标签名称可在 Google 站点的 `ImageNetLabels.txt` 文件中找到，我们使用以下代码将其加载到程序中：

```
labels_path = tf.keras.utils.get_file('ImageNetLabels.txt','https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt')
imagenet_labels = np.array(open(labels_path).read().splitlines())
```

使用以下两个打印语句打印标签名称和数组长度：

```
print (imagenet_labels)
print ("Number of labels: " , len(imagenet_labels))
['background' 'tench' 'goldfish' ... 'bolete' 'ear' 'toilet tissue']
Number of labels:  1001
```

接下来两行显示的输出列出了数组开头和结尾的几个名称。它还告诉你共有 1001 个名称，每个名称对应我们模型的一个输出类别。

### 显示预测结果

为了显示预测结果，我们使用以下代码绘制图片和预测的类别名称：

```
plt.imshow(image1)
plt.axis('off')
predicted_class_name = imagenet_labels[predicted_class]
_ = plt.title("Prediction: " + predicted_class_name.title())
```

结果如图 4-6 所示。

![../images/495303_1_En_4_Chapter/495303_1_En_4_Fig6_HTML.jpg](img/495303_1_En_4_Fig6_HTML.jpg)

图 4-6

带有预测标题的测试图片

如你所见，我们的模型正确预测了图片，表明它是一只老虎。现在，通过编写与 `image1` 类似的代码，对其他四张图片进行预测。

为此，你可以使用此处展示的 `predict_display_image` 函数：

```
def predict_display_image(imagex):
result = classifier.predict(imagex[np.newaxis, ...])
predicted_class = np.argmax(result[0], axis=-1)
plt.imshow(imagex)
plt.axis('off')
predicted_class_name = imagenet_labels[predicted_class]
_ = plt.title("Prediction: " + predicted_class_name.title())
```

该函数接收一个预处理后的图片作为参数，对其进行推理，并连同图片本身一起打印预测结果。

你将按如下方式对剩余的四张图片调用此函数：

```
#### 对指定图片进行预测并打印结果
predict_display_image(image2)
predict_display_image(image3)
predict_display_image(image4)
predict_display_image(image5)
```

四个预测结果如图 4-7 所示。

![../images/495303_1_En_4_Chapter/495303_1_En_4_Fig7_HTML.jpg](img/495303_1_En_4_Fig7_HTML.jpg)

图 4-7

其他四张图片的预测结果

在讨论图 4-7 的输出之前，让我们先看看 ImageNet 分类器所分类的类别列表。

### 列出所有类别

要查看 ImageNet 预测的所有类别列表，请使用以下代码：

```
for i in range (len(imagenet_labels)):
print (imagenet_labels[i])
```

输出中的前几项如下所示：

```
background
tench
goldfish
great white shark
tiger shark
hammerhead
electric ray
stingray
cock
Hen
```

现在，我将讨论图 4-7 中显示的结果。

### 结果讨论

输出中的第一张图片被解释为手推车，这是正确的。第二张图片被解释为开信刀，其外观与剑相似。请注意，实际图片是一把剑。开信刀在外观上与剑相似。因此，这个推理结果可以认为是合理的。第三张图片是一棵树，被解释为花盆。最后一张是一束玫瑰，被解释为粉饼——完全错误。那么，我们从这次讨论中能得出什么结论呢？为了区分泰迪熊和玫瑰花束，ImageNet 需要更多图片进行进一步训练。这显然超出了我们的能力范围。然而，我们将能够扩展 ImageNet，添加我们自己的分类。这正是我将在下一个主题中展示的内容。



