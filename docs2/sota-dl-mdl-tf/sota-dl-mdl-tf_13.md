# 13. 目标检测

**目标检测**是一种自动化的计算机视觉技术，用于在数字照片或视频中定位对象的实例。具体来说，目标检测在静态图像或视频数据中围绕一个或多个有效目标绘制边界框。一个**有效目标**是正在调查的图像或视频数据中感兴趣的对象。有效目标（或目标）应在任务开始时已知。

## 自然场景中的目标检测

对于我们来说，在自然场景中检测对象是轻而易举的，但对于计算机算法来说却极具挑战性。作为人类，我们看一个场景并识别感兴趣的对象，无需思考。我们在腹侧视觉流中处理视觉数据，这是一个大脑中帮助识别对象的区域层次结构。我们识别不同类型和大小的对象，并借助视觉皮层中响应简单形状（如线条和曲线）的细胞对它们进行分类。

使用目标检测，数据科学家试图模仿人类用计算机算法所做的事情。但在计算机算法甚至开始检测对象之前，必须将自然场景捕获为数字图像（或数字视频）。一个**数字图像**由图像元素（或像素）组成。一个**像素**是数字图形中的基本逻辑单元。简单来说，像素是一个微小的彩色方块。

像素在二维（2D）网格中均匀组合，形成一个完整的数字图像、视频、文本或计算机显示上的任何可见元素。每个像素都有一个特定的数值，这个数值告诉算法其颜色。这个数值代表像素强度值。像素强度值代表灰度或彩色图像。

在灰度图像中，每个像素只代表一种颜色的强度。因此，灰度图像可以用一个单一的二维矩阵（或网格）表示。在标准的 RGB 系统中，彩色图像有三个通道（红色、绿色和蓝色）。因此，彩色图像可以用三个二维矩阵表示。一个矩阵代表像素中红色的强度。一个代表像素中绿色的强度。还有一个代表像素中蓝色的强度。

每个矩阵被视为一个颜色通道。一个**颜色通道**是一组值（每个像素一个）的组合，共同指定图像的一个方面或维度。因此，RGB 颜色包含三个颜色通道——红色、绿色和蓝色。每个颜色通道从 0（最不饱和）到 255（最饱和）表示。因此，通过组合红色、绿色和蓝色的像素强度，RGB 颜色空间可以表示 16,777,216（256³）种不同的颜色。

一旦将数字图像捕获为灰度（2D 矩阵）或彩色（三个颜色通道的 2D 矩阵）图像，它就会经过处理以供模型使用。只有在这种情况下，计算机算法才能检查每个像素值的二维网格，以开始识别模式。

## 检测与分类

图像分类将输入图像呈现给神经网络，以便它能够学习与图像相关的单个类别标签。网络还可以学习与类别标签相关的概率。学习到的类别标签描述了整个图像的内容，或者至少是图像中最主要和最明显的部分。因此，图像分类网络能够学习如何正确地标记一张猫的图片。

目标检测将输入图像呈现给神经网络，以便它能够学习图像在图片对象（或场景）中的确切位置。为了在场景中定位输入图像，目标检测算法为图片中的每个对象创建一个边界框列表。边界框作为输入图像的(x, y)坐标位置创建。算法还识别与每个边界框和类别标签相关的类别标签以及与每个边界框和类别标签相关的概率/置信度分数。

简而言之，图像分类涉及将一张图像输入网络，并输出一个类别标签。目标检测涉及将一张图像输入网络，并输出多个边界框及其相关的类别标签。

目标检测通常用于精确地计数、定位和标记场景中的对象。神经网络是目标检测的最先进方法。卷积神经网络常用于自动学习对象的固有特征，通过将其与背景区分开来，在图像中识别它们。

与图像分类涉及将类别标签分配给图像不同，目标定位涉及在图像场景中围绕一个或多个对象绘制边界框。因此，分类将图像分为不同的类别，而目标定位则识别图像场景中的有效目标。目标检测比定位更具挑战性，因为它结合了分类和目标定位，以学习如何在图像中围绕每个感兴趣的对象（或有效目标）绘制边界框并为其分配类别标签。目标定位和目标检测之间的区别很微妙。目标定位旨在定位图像场景中的主要（或最明显的）对象，而目标检测则试图找出所有对象及其边界。

想象一下一张图片中包含两只猫和一个人。目标检测网络能够定位实体实例并分类图像中发现的实体类型。因此，目标检测网络可以在这张图片中定位两只猫和一个人，并正确地分类它们。

对象检测通常与图像识别混淆。那么它们有什么不同？图像识别为图像分配一个标签。一只狗的图片收到标签*dog*。两只狗的图片仍然收到标签*dog*。然而，对象检测却在每个狗周围画一个边界框，并将该框标记为*dog*。模型预测每个对象在图像中的位置及其应应用的标签。因此，对象检测比识别提供了更多关于图像的信息。

## 边界框

**边界框**是一个想象中的矩形，用作对象检测的参考点，并为该对象创建一个碰撞框。边界框的面积（通常简称为*bbox*）由两个经度和两个纬度定义，其中纬度是介于-90.0 和 90.0 之间的十进制数，经度是介于-180.0 和 180.0 之间的十进制数。数据标注员通过定义其 x 和 y 坐标来绘制 bbox 矩形，以在每个图像（场景）中勾勒出感兴趣的对象。

击中框（或碰撞框）是在视频游戏中常用的一种不可见形状，用于实时碰撞检测。因此，它是一种边界框。它通常是一个矩形（在 2D 游戏中）或长方体（在 3D 中），它附着并跟随一个可见对象上的一个点（例如模型或精灵）。

## 基本结构

深度学习对象检测模型通常有两个部分。一个*编码器*接受图像作为输入，并通过一系列块和层运行它，这些块和层学会提取用于定位和标记对象的统计特征。编码器的输出随后传递给一个*解码器*，该解码器预测每个对象的边界框和标签。

最简单的解码器是纯回归器。回归器是指任何用于预测响应变量的回归模型中的变量。回归器也被称为解释变量、自变量、操作变量、预测变量或特征。构建回归模型的整体目的是理解回归器的变化如何导致响应变量（或回归和）的变化。因此，回归器是一个特征，而回归和是一个响应变量（或目标）。

**回归**是一种监督式机器学习技术，用于预测连续值。回归算法的目标是在数据之间绘制最佳拟合线或曲线。回归器连接到编码器的输出，并直接预测每个边界框的位置和大小。

回归器模型的输出是对象在图像中的 x，y 坐标对及其范围（或强度）。但回归器有限制，因为我们需要提前指定框的数量。如果一个图像中有两只狗，但回归器模型被设计为检测单个对象，其中一只狗将不会被标记。但如果我们提前知道每个图像中需要预测的对象数量，纯回归器模型可能是一个不错的选择。

回归器方法的扩展是区域提议网络。区域提议网络中的解码器提议图像中可能存在对象的区域。然后，将这些区域的像素输入到分类子网络中，以确定标签（或拒绝提议）。然后，网络将包含这些区域的像素通过分类网络。这种方法的好处是更准确、更灵活的模型，可以提出任意数量的可能包含边界框的区域。但增加的准确性是以计算效率为代价的。

单次射击检测器（SSD）寻求一个中间地带。而不是使用子网络来提出区域，SSD 依赖于一组预定的区域。在输入图像上放置一个锚点网格。在每一个锚点上，多个形状和大小的框作为区域。对于每个锚点上的每个框，模型输出一个预测，以确定该区域内是否存在对象，以及修改框的位置和大小以使其更精确地适应对象。因为每个锚点有多个框，且锚点可能彼此靠近，所以 SSD 会产生许多潜在的检测，这些检测可能重叠。因此，必须对 SSD 输出进行后处理，通过剪枝不足的检测来筛选出最佳的一个。SDD 最流行的后处理技术是非最大值抑制。*非最大值抑制*用于选择对象的最合适的边界框。

物体检测器输出每个对象的定位和标签，但我们如何衡量性能？物体定位最常用的指标是交并比（IOU）。给定两个边界框，IOU 计算交集的面积，并将其除以并集的面积。值范围从 0（无交互）到 1（完全重叠）。可以使用简单的正确百分比指标来衡量标签。

各章节的笔记本位于以下 URL：

[`github.com/paperd/deep-learning-models`](https://github.com/paperd/deep-learning-models)

我们通过一个端到端的代码实验来演示物体检测。通过导入主 TensorFlow 库并实例化 GPU 来开始设置 Colab 生态系统。

## 导入 TensorFlow 库

导入库并将其别名为**tf**：

```py
import tensorflow as tf
```

将 TensorFlow 库别名为 tf 是常见做法。

## GPU 硬件加速器

作为便利，我们提供了在 Colab 笔记本中启用 GPU 的步骤：

1.  在右上角菜单中点击*运行时*。

1.  从下拉菜单中选择*更改运行时类型*。

1.  从*硬件加速器*下拉菜单中选择*GPU*。

1.  点击*保存*。

验证 GPU 是否激活：

```py
tf.__version__, tf.test.gpu_device_name()
```

如果显示“/device:GPU:0”，则 GPU 已激活。如果显示“..”，则常规 CPU 已激活。

备注

如果出现错误**“TF”未定义**，请重新执行代码以导入 TensorFlow 库！

## 物体检测实验

我们从 Google Drive 和 Wikimedia Commons 中抓取图像用于实验。我们使用预训练的 *faster_rcnn/openimages_v4/inception_resnet_v2* 对象检测模块进行图像对象检测。该对象检测模型在 Open Images V4（版本 4）上训练，使用 ImageNet 预训练的 Inception ResNet V2（版本 2）作为图像特征提取器。该模块内部执行非极大值抑制。输出的最大检测数量为 100。因此，它在一个场景中最多检测 100 个对象。对于 600 个可检测类别，输出检测结果。*可检测类别* 是指那些能够（或适合）放入边界框中的类别。

*Open Images* 是一个包含大约九百万丰富标注图像的数据集，具有图像级标签、对象边界框、对象分割、视觉关系和局部叙述。图片种类繁多，通常包含复杂场景和多个对象（平均每个场景有 8.4 个对象）。

*Open Images V4* 的训练集包含在 1.74 百万张图片上的 600 个对象类别，共计 1460 万个边界框，这使得它成为目前最大的具有对象位置标注的数据集（截至本文写作之时）。这些边界框主要由专业标注员手动绘制，以确保准确性和一致性。图片种类繁多，通常包含复杂场景和多个对象（平均每张图片有 8.4 个对象）。此外，数据集还标注了涵盖数千个类别的图像级标签。

注意

建议在 GPU 上运行此模块以获得可接受的推理时间。

### 导入必需的库

启用对 TF-hub 模块的访问：

```py
import tensorflow_hub as hub
```

访问绘图模块：

```py
import matplotlib.pyplot as plt
```

访问文件处理和内存中数据处理模块：

```py
import tempfile
from six.moves.urllib.request import urlopen
from six import BytesIO
```

这个 *tempfile* 模块用于创建临时文件和目录。*urlopen* 模块是 Python 的统一资源定位符（URL）处理模块。它用于获取 URL。URL 是对网络资源的引用，它指定了其在计算机网络上的位置以及检索它的机制。*BytesIO* 模块用于在内存中操作字节数据。

从 PIL 库访问模块：

```py
from PIL import Image
from PIL import ImageColor
from PIL import ImageDraw
from PIL import ImageFont
from PIL import ImageOps
```

*Image* 模块允许将图像上传到内存。其余模块用于图像处理。

导入 NumPy：

```py
import numpy as np
```

### 为实验创建函数

第一个函数显示一张图像：

```py
def display_image(image):
fig = plt.figure(figsize=(20, 15))
plt.grid(False)
plt.imshow(image)
plt.axis('off')
```

我们提供了许多不同的显示函数，以提供显示图像的替代方式。为了实践，创建你自己的显示函数（或函数），并用于这次实验。

第二个函数在场景中的图像周围绘制一个边界框，如图 13-1 所示。

```py
def draw_bounding_box_on_image(
image, ymin, xmin, ymax, xmax,
color, font, thickness=4, display_str_list=()):
"""Adds a bounding box to an image."""
draw = ImageDraw.Draw(image)
im_width, im_height = image.size
(left, right, top, bottom) = (
xmin * im_width, xmax * im_width,
ymin * im_height, ymax * im_height)
draw.line([(left, top), (left, bottom),
(right, bottom), (right, top),
(left, top)],
width=thickness, fill=color)
display_str_heights = [font.getsize(ds)[1]
for ds in display_str_list]
total_display_str_height = (
1 + 2 * 0.05) * sum(display_str_heights)
if top > total_display_str_height:
text_bottom = top
else:
text_bottom = top + total_display_str_height
for display_str in display_str_list[::-1]:
text_width, text_height = font.getsize(display_str)
margin = np.ceil(0.05 * text_height)
draw.rectangle(
[(left, text_bottom - text_height - 2 * margin),
(left + text_width, text_bottom)], fill=color)
draw.text(
(left + margin, text_bottom - text_height - margin),
display_str, fill='black', font=font)
text_bottom -= text_height - 2 * margin
Listing 13-1
Function to Draw Bounding Boxes
```

函数接受一个处理后的图像、x 和 y 的最小和最大坐标、颜色、字体、粗细和显示字符串列表。处理后的图像由 *draw_boxes* 函数发送，该函数将在下面介绍。x 和 y 坐标提供了图像边界框的边界。颜色、字体、粗细和字符串列表由 *draw_boxes* 函数提供。函数的其余部分向图像添加一个边界框。

首先创建一个 ImageDraw 对象并将其分配给变量 *draw*。*ImageDraw* 模块用于创建新图像、注释或修复现有图像，并为网络使用动态生成图形。将图像的 x 和 y 坐标分配给变量，并将边界线分配给 ImageDraw 对象。

函数继续通过将显示字符串的列表分配给变量 *display_str_heights*。显示字符串用于为场景中每个图像的边界框标记。如果添加到边界框顶部的显示字符串的总高度超过图像的顶部，我们需要将字符串堆叠在边界框下方而不是上方。原因是确保识别每个边界框中图像的字符串可读。*total_display_str_height* 变量确保每个显示字符串在显示时有合理的顶部和底部边距。您可以尝试调整此值，但当前的设置已经相当好了。接下来的逻辑检查顶部和底部边距，以确保图像在边界框容器内良好适配。

函数的其余部分通过循环（for 循环）反转显示字符串的列表，以便从底部到顶部显示。*draw.rectangle* 方法在每个图像周围绘制边界框。*draw.text* 方法用适当的标签标记每个边界框。当循环完成后，创建一个包含所有边界框的图像。

第三个函数接受一个图像、边界框、类别标签、分数、最大边界框数量和最小分数。除了图像之外的所有参数都是由预训练模型生成的。使用接受的参数值，它将带有格式化分数和标签名称的标记边界框叠加到图像上，如列表 13-2 所示。

```py
def draw_boxes(
image, boxes, class_names, scores,
max_boxes=10, min_score=0.1):
# Overlay labeled boxes on an image with formatted scores and label names.
colors = list(ImageColor.colormap.values())
one = '/usr/share/fonts/truetype/liberation/'
two =  'LiberationSansNarrow-Regular.ttf'
font_url = one + two
try:
font = ImageFont.truetype(font_url, 25)
except IOError:
print('Font not found, using default font.')
font = ImageFont.load_default()
for i in range(min(boxes.shape[0], max_boxes)):
if scores[i] >= min_score:
ymin, xmin, ymax, xmax = tuple(boxes[i])
display_str = '{}: {}%'.format(
class_names[i].decode('ascii'),
int(100 * scores[i]))
color = colors[hash(class_names[i]) % len(colors)]
image_pil = Image.fromarray(
np.uint8(image)).convert('RGB')
draw_bounding_box_on_image(
image_pil, ymin, xmin, ymax, xmax,
color, font, display_str_list=[display_str])
np.copyto(image, np.array(image_pil))
return image
Listing 13-2
Container Function for Drawing Bounding Boxes
```

draw_boxes 函数是一个容器，因为它接受来自预训练模型的图像和字典，调用 *draw_bounding_box_on_image*，并返回一个在检测到的对象周围绘制了边界框的图像。

虽然函数看起来很复杂，但实际上非常直接。它创建了图像中固有的颜色。我们需要颜色，因为我们需要从场景中重新创建检测到的对象，以便在它们周围绘制边界框。然后它创建了用于标记每个边界框的字体。为了练习，你可以更改字体。然后它遍历所有对象，从预训练模型输出的字典中检索场景中的 x 和 y 坐标、标签、分数、图像像素和颜色，以提供给 *draw_bounding_box_on_image* 函数。此函数为每个对象绘制边界框，并返回带有边界框图像的场景。

注意

为了避免混淆，请将原始图像想象成一个场景。预训练模型学习如何识别场景中的图像对象。它输出一系列字典，包含它检测到的对象信息和场景中边界框的坐标。然后我们根据字典信息绘制边界框。所以当我们谈论图像对象时，我们指的是从场景中检测到的对象。

### 加载预训练的对象检测模型

加载对象检测模块并将其应用于下载的图像：

```py
p1 = 'https://tfhub.dev/google/faster_rcnn/'
p2 = 'openimages_v4/inception_resnet_v2/1'
URL = p1 + p2
module_handle = URL
obj_detect = hub.load(module_handle).signatures['default']
```

预训练模型是在 Open Images V4 上训练的对象检测模型，使用 ImageNet 预训练的 Inception ResNet V2 作为图像特征提取器。

Open Images 非常庞大！它包含 600 个类别中的 15,851,536 个边界框，350 个类别中的 2,785,498 个实例分割，1,466 个关系中的 3,284,280 个关系注释，675,155 个局部叙述，以及 19,957 个类别中的 59,919,574 个图像级标签。它还包括一个可选扩展，包含 478,000 个具有 6,000 多个类别的众包图像。

该模块在内部执行非最大值抑制。输出的最大检测数量为 100。检测输出针对 600 个可检测类别。建议在 GPU 上运行此模块以获得可接受的推理时间。

该模型接受一个可变大小的三通道图像。它输出多个字典，包括 detection_boxes（边界框坐标）、detection_class_entities（检测类别名称）、detection_class_names（可读类别名称）、detection_class_labels（作为张量的标签）和 detection_scores（检测分数）。检测分数表示模型在标记对象时的置信度。

注意

我们为好奇者提供了 Open Images 的详细信息。预训练模型处理工作负载。我们只是使用其输出创建可视化。

### 从 Google Drive 加载图像

将 Google Drive 挂载到 Colab 笔记本：

```py
from google.colab import drive
drive.mount('/content/gdrive')
```

确保图像位于 Google Drive 中的 *适当目录*（即 Colab 笔记本）！

访问和显示图像：

```py
img_path = 'gdrive/My Drive/Colab Notebooks/images/cats_dogs.jpg'
pil_image = Image.open(img_path)
display_image(pil_image)
```

将 JPEG 图像转换为 Python 图像库 (PIL) 图像并显示。PIL 是一个支持打开、操作和保存多种不同图像文件格式的库。它也被称为 Pillow 库。

检查图像大小：

```py
pil_image.size
```

### 准备图像

为图像文件生成一个临时路径：

```py
_, filename = tempfile.mkstemp(suffix='.jpg')
filename
```

准备图像以进行处理并将其保存到临时文件路径：

```py
pil_image_rgb = pil_image.convert('RGB')
pil_image_rgb.save(filename, format='JPEG', quality=90)
print('Image downloaded to %s.' % filename)
display_image(pil_image)
```

### 在图像上运行对象检测

创建一个函数以加载图像：

```py
def load_img(path):
img = tf.io.read_file(path)
img = tf.image.decode_jpeg(img, channels=3)
return img
```

该函数加载图像并为其准备预训练模型。

创建一个函数以运行如清单 13-3 所示的对象检测。

```py
def run_detector(detector, path):
img = load_img(path)
converted_img  = tf.image.convert_image_dtype(
img, tf.float32)[tf.newaxis, ...]
result = detector(converted_img)
result = {key:value.numpy()
for key,value in result.items()}
print("Found %d objects." %\
len(result["detection_scores"]))
image_with_boxes = draw_boxes(
img.numpy(), result["detection_boxes"],
result["detection_class_entities"],
result["detection_scores"])
display_image(image_with_boxes)
Listing 13-3
Object Detection Function
```

该函数接受加载的预训练模型签名和图像（场景）的路径。它加载图像并将其转换为 NumPy 数组以供模型使用。在 NumPy 图像上运行模型签名。从预训练模型输出的字典中检索字典键/值对。创建一个带有检测对象边界框的场景并显示它。

运行检测器：

```py
run_detector(obj_detect, filename)
```

如我们所知，预训练模型限制在检测 100 个对象。但信息具有误导性，因为它说无论将什么场景输入到模型中，都会找到 100 个对象。

检测完美无缺，但不要过于兴奋。模型足够强大，可以检测简单场景中的图像。场景之所以简单，是因为背景没有干扰，每只狗/猫都是单独呈现的。

让我们再试一次：

```py
img_path = 'gdrive/My Drive/Colab Notebooks/images/butterfly.jpg'
pil_image = Image.open(img_path)
display_image(pil_image)
```

处理图像：

```py
_, filename = tempfile.mkstemp(suffix='.jpg')
pil_image_rgb = pil_image.convert('RGB')
pil_image_rgb.save(filename, format='JPEG', quality=90)
print('Image downloaded to %s.' % filename)
```

运行检测器：

```py
run_detector(obj_detect, filename)
```

检测完美无缺，但图像很简单。

### 从复杂场景中检测图像

让我们尝试在更复杂的图像上进行检测。我们已经从维基媒体共享资源找到了一些图像，但您可以通过遵循几个简单的步骤找到自己的图像：

1.  前往以下 URL：[`https://commons.wikimedia.org/wiki/Main_Page`](https://commons.wikimedia.org/wiki/Main_Page)

1.  点击图像链接。

1.  点击一张图像。

1.  右键点击图像。

1.  从下拉菜单中选择“复制链接地址”。

1.  将链接地址粘贴到代码单元格中。

1.  将链接地址用单引号或双引号括起来。

1.  赋值给一个变量。

#### 创建一个下载函数

创建一个函数以下载、处理并将图像保存到临时文件路径，如清单 13-4 所示。

```py
def download_and_resize_image(
url, new_width=256, new_height=256,
display=False):
_, filename = tempfile.mkstemp(suffix='.jpg')
response = urlopen(url)
image_data = response.read()
image_data = BytesIO(image_data)
pil_image = Image.open(image_data)
pil_image = ImageOps.fit(
pil_image, (new_width, new_height),
Image.ANTIALIAS)
pil_image_rgb = pil_image.convert('RGB')
pil_image_rgb.save(
filename, format='JPEG', quality=90)
print('Image downloaded to %s.' % filename)
if display:
display_image(pil_image)
return filename
Listing 13-4
Download and Preprocess Function
```

该函数为图像文件生成一个临时路径。然后从提供的 URL 读取图像文件。函数继续将图像文件转换为 PIL 图像。然后 PIL 图像被调整大小，转换为 RGB，并保存到临时文件路径。函数最后通过返回 PIL 图像的文件名结束。

#### 加载图像场景

从维基媒体共享资源 URL 加载场景：

```py
p1 = 'https://upload.wikimedia.org/wikipedia/commons/7/79/'
p2 = 'At_taverna_under_the_church%2C_Ano_Potamia%2C_Naxos%'
p3 = '2C_190574.jpg'
URL = p1 + p2 + p3
downloaded_image_path = download_and_resize_image(
URL, 1280, 856, True)
```

图像场景的来源位于

[`https://commons.wikimedia.org/wiki/File:At_taverna_under_the_church,_Ano_Potamia,_Naxos,_190574.jpg`](https://commons.wikimedia.org/wiki/File:At_taverna_under_the_church,_Ano_Potamia,_Naxos,_190574.jpg)

#### 检测

运行对象检测

```py
run_detector(obj_detect, downloaded_image_path)
```

在更复杂的场景中，检测并不完美。但它确实进行了一些正确的检测。

#### 在更多场景中进行检测

按照清单 13-5 所示拼接一些路径。

```py
p1 = 'https://upload.wikimedia.org/wikipedia/commons/4/45/'
p2 = 'Green_Dragon_Tavern_%2836196%29.jpg'
tavern = p1 + p2
p1 = 'https://upload.wikimedia.org/wikipedia/commons/3/31/'
p2 = 'Circus_Circus_Hotel-Casino_sign.jpg'
casino = p1 + p2
p1 = 'https://upload.wikimedia.org/wikipedia/commons/9/91/'
p2 = 'Leon_hot_air_balloon_festival_2010.jpg'
balloon = p1 + p2
p1 = 'https://upload.wikimedia.org/wikipedia/commons/d/d8/'
p2 = '2012_Festival_of_Sail_-_7943922284.jpg'
sail = p1 + p2
p1 = 'https://upload.wikimedia.org/wikipedia/commons/a/ab/'
p2 = '17_mai_2018.jpg'
flag = p1 + p2
p1 = 'https://upload.wikimedia.org/wikipedia/commons/4/43/'
p2 = 'Fruit_baskets.jpg'
basket= p1 + p2
p1 = 'https://upload.wikimedia.org/wikipedia/commons/c/c7/'
p2 = 'Fruit_stands%2C_Rue_de_Seine%2C_Paris_22_May_2014.jpg'
stand= p1 + p2
p1 = 'https://upload.wikimedia.org/wikipedia/commons/9/95/'
p2 = 'Wine_tasting_%40_brown_brothers.jpg'
wine = p1 + p2
Listing 13-5
Image Paths
```

创建一个函数以检测场景中的图像：

```py
def detect_img(image_url):
image_path = download_and_resize_image(image_url, 640, 480)
run_detector(obj_detect, image_path)
```

在场景之一上运行对象检测

```py
detect_img(wine)
```

因此，场景的复杂性限制了检测的准确性。

再试一次：

```py
detect_img(sail)
```

#### 找到来源

我们在研究过程中有时会遇到维基百科 Commons 图片。但此类图片的来源从未（至少在我们经验中）被包括。如果我们想以任何方式使用图片，我们必须找到其来源以查看是否允许。

#### 查找维基百科 Commons 图片的来源

我们可以通过几个步骤找到图片的来源：

1.  将 *commons* 替换为 *upload.*

1.  将 *wikipedia* 替换为 *wiki.*

1.  将 *commons/(number)/(number)* 替换为 *File:*

1.  将 *%(number)* 翻译为其 *HTML 编码的等效值*。

要找到 HTML 编码的等效值，请查阅

[HTML 编码参考](https://krypted.com/utilities/html-encoding-reference/)

注意

我们不能保证这个过程适用于每张图片，但它适用于我们使用的那些图片。

让我们尝试对酒馆图片进行此过程：

[绿龙酒馆图片](https://upload.wikimedia.org/wikipedia/commons/4/45/Green_Dragon_Tavern_%252836196%2529.jpg)

将 *commons* 替换为 *upload*:

[绿龙酒馆图片](https://commons.wikimedia.org/wikipedia/commons/4/45/Green_Dragon_Tavern_%2836196%29.jpg)

将 *wikipedia* 替换为 *wiki*:

[绿龙酒馆图片](https://commons.wikimedia.org/wiki/commons/4/45/Green_Dragon_Tavern_%2836196%29.jpg)

将 *commons/(number)/(number)* 替换为 *File*:

[绿龙酒馆图片](https://commons.wikimedia.org/wiki/File:Green_Dragon_Tavern_%252836196%2529.jpg)

翻译：

[绿龙酒馆图片](https://commons.wikimedia.org/wiki/File:Green_Dragon_Tavern_(36196).jpg)

%28 和 %29 代码在 HTML 编码参考中转换为左右括号。

赌场图片更容易，因为我们不需要进行任何翻译：

[马戏团酒店-赌场标志图片](https://commons.wikimedia.org/wiki/File:Circus_Circus_Hotel-Casino_sign.jpg)

这里是剩余图片的来源：

[莱昂热气球节 2010 年图片](https://commons.wikimedia.org/wiki/File:Leon_hot_air_balloon_festival_2010.jpg)

[2012 年帆船节图片](https://commons.wikimedia.org/wiki/File:2012_Festival_of_Sail_-_7943922284.jpg)

[2018 年 5 月 17 日图片](https://commons.wikimedia.org/wiki/File:17_mai_2018.jpg)

[水果篮图片](https://commons.wikimedia.org/wiki/File:Fruit_baskets.jpg)

[塞纳河畔水果摊，巴黎，2014 年 5 月 22 日图片](https://commons.wikimedia.org/wiki/File:Fruit_stands,_Rue_de_Seine,_Paris_22_May_2014.jpg)

[葡萄酒品鉴 @ 布朗兄弟](https://commons.wikimedia.org/wiki/File:Wine_tasting_%2540_brown_brothers.jpg)

## 摘要

我们在几个图像场景上使用了一个强大的预训练图像检测模型来展示目标检测。该模型在简单场景上工作得非常好，但在复杂场景上则不那么理想。随着深度学习中的目标检测技术不断进步，我们相信未来的模型将大大提高检测能力。
