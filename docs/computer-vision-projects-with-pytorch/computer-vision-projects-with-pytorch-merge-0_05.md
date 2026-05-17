# 4. 构建图像分割模型

我们周围的图像具有不同的纹理、图案、形状和大小。它们携带了海量信息，人眼和大脑可以轻松理解这些信息，但计算机却较难理解。图像分割是一类问题，我们试图训练计算机理解图像，以便它们能够分离不同的物体并对相似的物体进行分组。这可以表现为相似的像素强度，或相似的纹理和形状。

目前已经开发出许多算法并用于图像分割。就像目标检测分离物体一样，图像分割从不太相似的物体中识别出更相似的物体。如果我们考虑诸如 `k-means` 等基本聚类方法中使用的概念，我们就知道数据点是如何在相似数据附近对齐的。

例如，假设一个碗里放着两种苹果和两种橙子。如果我们观察特征，可以从对可食用数据点进行分类开始。当我们考虑纹理或颜色时，可以将数据点分成两组。当我们把成本作为另一个特征加入时，可以将数据点分成四个簇。同样，这个序列可以帮助我们找出数据点中的不同点或相似点，并按簇进行分组。这在生物医学领域和自动驾驶汽车中非常有用。这仍然是一个活跃的研究领域，并且大多与目标检测框架结合使用。我们将介绍基础知识，然后考虑一些示例。让我们开始吧。

# 图像分割

分割的主观性取决于我们所处理的领域类型。分割主要分为两种：语义分割和实例分割。在进行语义分割时，来自相似物体的像素被视为同一类别，但物体内部没有区分。想象一个实时场景：当高速公路的图像中有多辆汽车时，分割会将所有汽车归为一组，并将这些组与路边或风景区分开来。

让我们看一个例子。图 4-1a 展示了一条有汽车的高速公路。除了多辆汽车，高速公路旁还有草地和一些树木。

![](img/520381_1_En_4_Fig1_HTML.jpg)

一张长路段的照片，前景有一辆皮卡，后方还有另外三辆汽车。道路两旁长满了草地和树木。

图 4-1a

原始输入图像

现在，考虑从原始输入中提取一个像素块，并将其传递给一个能够对输入中的物体进行分类的卷积神经网络（见图 4-1b）。这将给我们一个输出，比如图中属于汽车的像素块。接下来，我们尝试将中心像素映射到汽车，并像这样遍历整个图像。这将为我们提供图像中的语义分割分离。它将把汽车与树木和道路分开。这里需要注意的重要一点是，所有汽车都属于同一类别。

![](img/520381_1_En_4_Fig2_HTML.jpg)

一张长路段的照片，前景有一辆皮卡，后方还有另外三辆汽车。道路两旁长满了草地和树木。在皮卡的后端有一个蓝色轮廓的黑色方块，旁边有一个指向右侧的箭头，上面写着“像素块”。

图 4-1b

从输入中提取像素块

解决这类问题的另一种方法是运行一个没有下采样的卷积神经网络分类器，并用它来对每个像素进行分类，从而将相似的物体聚类在一起。

这些方法在需要区分不同类别时效果很好。例如，假设有多辆汽车，我们想对每辆汽车单独分类。在这种情况下，就需要用到实例分割，其中每个像素都被映射到特定的类别，并通过用适当的类别标记像素来分离物体。语义分割的概念可以追溯到图像处理中使用的非学习技术，而实例分割则是一个相对较新的概念。

当我们开始研究实例分割时，出现的基本方法之一几乎就是 R-CNN 方法的翻版。但我们预测的是分割区域，而不是区域。

请看图 4-2 中的流程图。图像被传递到一个分割提议网络，该网络生成图像的分割区域。一方面，这些分割区域可以形成一个边界框，并传递给边界框卷积神经网络以生成特征。另一方面，分割区域被接收并应用背景掩码变换。它只取图像的均值，并将物体的背景转换为黑色。一旦分割区域被掩码，它就会被传递给区域卷积神经网络以获取另一组特征。

![](img/520381_1_En_4_Fig4_HTML.jpg)

一张照片，显示一位男士和一位女士在人行道上，背景是建筑物。男士和女士的身体分别被紫色和黄色的轮廓覆盖。两个人物都被绿色框框住。框的上方写着“行人”字样。

图 4-3

自定义数据上的输出

![](img/520381_1_En_4_Fig3_HTML.png)

一个流程图从左到右描述如下：一张照片，一个右箭头，一个标有“提议生成”的框，通过箭头连接到“特征提取”（其中包含一张照片和一个标有“CNN 边界框”的框）以及“背景掩码”（其中包含一张照片和一个标有“CNN 区域”的框）。右侧是“区域分类”和“区域细化”。

图 4-2

实例分割流程

这里我们得到的是边界框图像和网络提取的区域特征的组合。这些特征将被合并，然后根据它们包含的物体实例进行进一步分类。在此之后，还有一个额外的步骤，即细化分割区域。

这些仅仅是实验性的分割技术设置。其他方法上的改进也已经实现，包括级联网络（类似于 Faster R-CNN）、超列等。

语义分割和实例分割之间存在多个差异。

语义分割：

*   对所有像素进行分类。
*   使用全卷积模型。
*   在各种方法中使用下采样，然后使用可学习的上采样技术来重建图像。
*   如果使用类似 ResNet 的架构，则会使用跳跃连接。

实例分割：

*   不仅对每个像素进行分类，还检测实例。
*   其过程几乎遵循目标检测架构。

## PyTorch 的预训练支持

PyTorch 的发展速度远超其他任何框架。它拥有大量的模块和类。由于它接近 Python，因此更容易适应这个框架。在深度学习领域，普遍倾向于选择 PyTorch 框架，并利用其丰富的资源来产生有影响力的变革。

与目标检测一样，分割也属于架构中计算量较大的部分。从头开始训练这些模型并不总是容易或理想的。在 CPU 上训练时间相当长，即使 GPU 能有所帮助，提升也有限。由于所有这些训练过程的限制，我们可能会选择迁移学习技术。这有助于我们利用已经提取到的丰富信息来开展工作。这些模型在多样化的数据集上训练，并且已经泛化，能够处理我们遇到的大多数问题中的变化。让我们深入了解 `torch` 库中一些出色的模型。

### 语义分割

- **全卷积神经网络。** 如论文所述，全卷积网络在语义分割任务上进行端到端训练。它是一个卷积神经网络模块，后接逐像素预测。
- **使用空洞卷积进行语义分割（DeepLabV3）。** 在该架构中，通过改变感受野，使用并行的空洞卷积堆叠来捕获多尺度上下文信息。
- **轻量级缩减版空洞空间金字塔池化（LR-ASPP）。** 这是 `MobileNetV3` 的进阶版本，借助 NAS（神经架构搜索）技术创建而成。

我们将使用一个预训练模型来评估我们的图像。该模型参数众多，因此在 CPU 或低配置基础设施上运行推理会较慢。如果使用 Colab，我们可以切换到 GPU 作为基础设施支持来运行代码。

让我们从配置所需的基本 `import` 开始。该模型是预训练的，并放置在 Torchvision 中。我们也将导入该模型。

```python
import numpy as np
import torch
import matplotlib.pyplot as plt

## torchvision related imports
import torchvision.transforms.functional as F
from torchvision.io import read_image
from torchvision.utils import draw_bounding_boxes
from torchvision.utils import make_grid

## models and transforms
from torchvision.transforms.functional import convert_image_dtype
from torchvision.models.segmentation import fcn_resnet50
```

到目前为止，我们导入了 Torch 和 Torchvision 相关的函数。我们需要构建所有可在整个代码中重复使用的工具函数。这是重构代码并消除不必要重复的有效方法。在本例中，我们需要显示图像，因此可以使用一个图像可视化工具。

```python

## utilities for multiple images
def img_show(images):
if not isinstance(images, list):

## generalise cast images to list
images = [images]
fig, axis = plt.subplots(ncols=len(images), squeeze=False)
for i, image in enumerate(images):
image = image.detach() # detached from current DAG, no gradient
image = F.to_pil_image(image)
axis[0, i].imshow(np.asarray(image))
axis[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
```

这段代码接受多张图像或单张图像。它会检查对象是否为列表，如果不是，则将其转换为列表。根据图像的可迭代对象分配坐标轴。图像会从计算图中分离，并且不会为这些变量计算梯度。

在工具函数之后，让我们获取一张示例图像，并将其配置为用于分割过程。

```python

## get an image on which segmentation needs to be done
img1 = read_image("/content/semantic_example_highway.jpg")
box_car = torch.tensor([ [170, 70, 220, 120]], dtype=torch.float) ## (xmin,ymin,xmax,ymax)
colors = ["blue"]
check_box = draw_bounding_boxes(img1, box_car, colors=colors, width=2)
img_show(check_box)

## batch for images
batch_imgs = torch.stack([img1])
batch_torch = convert_image_dtype(batch_imgs, dtype=torch.float)
```

需要上传图像并将其放置在可访问的位置。图像中包含多辆汽车，目前我们用一个边界框（`X[min]`、`Y[min]`、`X[max]` 和 `Y[max]`）标记其中一辆。这些值需要调整，以便全卷积网络能够理解物体的存在。最终，图像批次在堆叠输入模型之前会被转换为张量。

现在，让我们加载模型并准备进行评估。

```python
model = fcn_resnet50(pretrained=True, progress=False)

## switching on eval mode
model = model.eval()

# standard normalizing based on train config
normalized_batch_torch = F.normalize(batch_torch, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
result = model(normalized_batch_torch)['out']
```

如前所述，`fcn_resnet50` 从仓库中下载，并且已经过训练。模型被设置为评估模式。之前步骤中创建的批次，现在根据训练好的模型配置进行归一化。

现在是时候将我们的图像输入模型了。

```python
classes = [
'__background__', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
'person', 'pottedplant', 'sheep', 'sofa', 'train', ’tvmonitor’
]
class_to_idx = {cls: idx for (idx, cls) in enumerate(classes)}
normalized_out_masks = torch.nn.functional.softmax(result, dim=1)
car_mask = [
normalized_out_masks[img_idx, class_to_idx[cls]]
for img_idx in range(batch_torch.shape[0])
for cls in ('car', 'pottedplant','bus')
]
img_show(car_mask)
```

我们定义了包含所有可能类别的列表，并将批次图像结果通过 softmax 层。然后绘制掩码。这个示例展示了我们之前讨论的语义分割的处理流程。它说明了如何获取任意数据并根据模型进行准备。我们加载一个模型，在其上运行推理以获取掩码。

### 实例分割

我们之前处理语义分割是为了生成物体的掩码，然后将它们叠加到原始图像上。但是，实例分割呢？现在我们将了解一些可用于生成掩码的预训练模型。

用于检测和掩码的模型：

- **Faster R-CNN。** 这项研究引入了一个区域提议网络，该网络能同时预测物体边界框和与边界框对应的物体性核心分数。它解决了早期论文中存在的瓶颈问题。
- **Mask R-CNN。** 该过程扩展了 Faster R-CNN，并在图像上执行物体检测和生成掩码。
- **RetinaNet。** 这篇论文在准确性和速度方面对两阶段检测器进行了一些惊人的改进。它利用*焦点损失*这一新概念来处理所有这些问题。
- **单次检测器（SSD）。** 该论文解释了如何为默认边界框生成物体性分数，并根据物体对其进行优化。

这些模型主要基于 COCO 数据集进行训练，能够处理预测任务。

对于 Faster R-CNN：

```python
x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
faster_rcnn_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
faster_rcnn_model.eval()
result = faster_rcnn_model (x)
```

对于 MobileNet：

```python
x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
mobilenet_model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(pretrained=True)
mobilenet_model.eval()
result = mobilenet_model(x)
```

对于 RetinaNet：

```python
x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
retinanet_model = torchvision.models.detection.retinanet_resnet50_fpn(pretrained=True)
retinanet_model.eval()
result = retinanet_model(x)
```

对于单次检测器（SSD）：

```python
x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
ssd_model = torchvision.models.detection.ssd300_vgg16(pretrained=True)
ssd_model.eval()
result = ssd_model(x)
```

对于所有这些实例，我们都是从 PyTorch 仓库中提取模型，并使用它来运行推理。

# 微调模型

如果我们处理的领域包含研究人员已标注并训练好的类别，使用预训练模型即可获得预测结果。若领域高度相关但缺少精确类别，则可能出现预测偏差。这正是我们为项目进行训练和分类的最重要原因之一。

在本节中，我们将深入解析代码，详细说明微调现有模型以增强其预测能力、适配我们目标所需的步骤。如前所述，图像分割具备超越单纯类别识别的额外能力。

我们将使用的数据集是开源数据集，位于 [`https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip`](https://www.cis.upenn.edu/%257Ejshi/ped_html/PennFudanPed.zip)。

该数据集包含行人数据，我们将对其进行微调。多种应用场景将分割结果作为决策依据。在自动驾驶汽车需要瞬间决定行驶方向时，识别行人是一个重要的应用场景。因此，这些模型也需要具备足够高的准确率。

让我们先了解流程所需的基本导入。

项目配置是任何项目中最具决定性的环节之一。在本项目中，我们可以使用 Jupyter notebook 或 Colab notebook 进行训练。

首先，我们使用 `wget` 命令从源地址下载数据集。

```

## 提取交通数据
!wget https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip .
!unzip PennFudanPed.zip
```

感叹号（`!`）帮助 Colab 单元格识别这些命令为 shell 脚本。下载完成后，`unzip` 命令解压压缩包。需要注意的是，这些命令基于 Linux，且假设后端操作系统也是 Linux。

将数据集下载到系统后，我们可以运行项目所需的基本 `import` 语句。

```

## 基本导入
import os
import numpy as np

## torch 导入
import torch
import torch.utils.data
from torch.utils.data import Dataset

## torchvision 导入
import torchvision
import torchvision.transforms as T
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

## 图像工具
from PIL import Image
import matplotlib.pyplot as plt

## 代码工具
import random
import cv2
```

请注意，我们正在导入 Torch 和 Torchvision 相关的包。同时，我们也在获取 MaskRCNN 和 FasterRCNN 的预训练模型。现在，让我们导入 PyTorch 的基本训练框架。这将帮助我们扩展函数，避免重写复杂的代码。

```

## 克隆 PyTorch 仓库以建立与原始训练完全相同的目录结构
!git clone https://github.com/pytorch/vision.git
%cd vision
!git checkout v0.3.0
!cp references/detection/engine.py ../
!cp references/detection/transforms.py ../
!cp references/detection/utils.py ../
!cp references/detection/coco_utils.py ../
!cp references/detection/coco_eval.py ../
```

一旦基本框架准备就绪，我们将复制将要使用的重要 Python 脚本，例如 `engine`、`transforms`、`utils`、`coco_utils` 和 `coco_eval`。

完成这些导入并确认文件位于我们运行代码的同一基础设施上后，我们可以再运行一些 `import` 语句。

```

## 从 PyTorch 仓库导入
import utils
import transforms as T
from engine import train_one_epoch, evaluate
```

这些导入基于 PyTorch 训练框架和脚本中的代码。完成此步骤后，让我们看看创建微调模型所需的自定义数据集类。

```
class CustomDataset(Dataset):
def __init__(self, dir_path, transforms=None):

## 初始化对象属性
self.transforms = transforms
self.dir_path = dir_path

## 从 dir_path

## 添加来自 PedMasks 目录的掩码列表
self.mask_list = list(sorted(os.listdir(os.path.join(dir_path, "PedMasks"))))

## 添加来自目录列表的实际图像列表
self.image_list = list(sorted(os.listdir(os.path.join(dir_path, "PNGImages"))))
def __getitem__(self, idx):

# 获取图像和掩码
img_path = os.path.join(self.dir_path, "PNGImages", self.image_list[idx])
mask_path = os.path.join(self.dir_path, "PedMasks", self.mask_list[idx])
image_obj = Image.open(img_path).convert("RGB")
mask_obj = Image.open(mask_path)
mask_obj = np.array(mask_obj)
obj_ids = np.unique(mask_obj)

# 背景具有第一个 ID，因此排除它
obj_ids = obj_ids[1:]

# 将掩码分割为二进制
masks_obj = mask_obj == obj_ids[:, None, None]

# 边界框
num_objs = len(obj_ids)
bboxes = []
for i in range(num_objs):
pos = np.where(masks_obj[i])
xmax = np.max(pos[1])
xmin = np.min(pos[1])
ymax = np.max(pos[0])
ymin = np.min(pos[0])
bboxes.append([xmin, ymin, xmax, ymax])
image_id = torch.tensor([idx])
masks_obj = torch.as_tensor(masks_obj, dtype=torch.uint8)
bboxes = torch.as_tensor(bboxes, dtype=torch.float32)
labels = torch.ones((num_objs,), dtype=torch.int64)
area = (bboxes[:, 3] - bboxes[:, 1]) * (bboxes[:, 2] - bboxes[:, 0])
iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
target = {}
target["image_id"] = image_id
target["masks"] = masks_obj
target["boxes"] = bboxes
target["labels"] = labels
target["area"] = area
target["iscrowd"] = iscrowd
if self.transforms is not None:
image_obj, target = self.transforms(image_obj, target)
return image_obj, target
def __len__(self):
return len(self.image_list)
```

创建自定义数据集是将新数据集整合到训练流程中的标准技术。代码中需要关注的重点如下：

- 我们正在扩展 PyTorch 的 `Dataset` 类。
- 我们为类定义了三个重要函数——`initialization`、`get_item` 和 `len`。
- 我们初始化了变换，这些变换在测试、验证和训练阶段可以不同。
- 我们定义了边界框。
- 我们定义了目标。

完成数据集创建后，我们必须根据新数据修改模型。让我们看看相应的代码。

```
def modify_model(classes_num):

# 从 PyTorch 仓库加载已在 COCO 上训练好的模型
maskrcnn_model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

# 识别输入特征数量
in_features = maskrcnn_model.roi_heads.box_predictor.cls_score.in_features

# 修改头部
maskrcnn_model.roi_heads.box_predictor = FastRCNNPredictor(in_features, classes_num)
in_features_mask = maskrcnn_model.roi_heads.mask_predictor.conv5_mask.in_channels
hidden_layer = 256
maskrcnn_model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
hidden_layer,
num_classes)
return maskrcnn_model
```

这些步骤改变了模型的头部配置。之后，我们将尝试对数据进行变换。这将为训练准备好数据。

```python
def get_transform_data(train):
    transforms = []
    # 将 PIL 图像转换为 PyTorch 模型所需的张量
    transforms.append(T.ToTensor())
    if train:
        # 基础图像增强技术
        ## 可根据实验需要添加更多方法
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

# 获取交通数据进行转换
train_dataset = CustomDataset('/content/PennFudanPed', get_transform_data(train=True))
test_dataset = CustomDataset('/content/PennFudanPed', get_transform_data(train=False))

# 训练集与测试集划分
torch.manual_seed(1)
indices = torch.randperm(len(train_dataset)).tolist()
train_dataset = torch.utils.data.Subset(train_dataset, indices[:-50])
test_dataset = torch.utils.data.Subset(test_dataset, indices[-50:])

# 定义训练和验证数据加载器
train_data_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=2, shuffle=True, num_workers=4,
    collate_fn=utils.collate_fn)
test_data_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=1, shuffle=False, num_workers=4,
    collate_fn=utils.collate_fn)
```

在转换数据中需要关注的重点包括：

- 将数据转换为张量，以便在 PyTorch 的有向无环图（DAG）中使用。
- 对于训练阶段，我们使用转换或增强技术。而对于测试和验证等其他阶段，则不使用任何增强技术。
- 我们通过之前阶段构建的自定义数据集类来创建训练集和测试集。
- 定义好自定义数据后，我们将利用它创建可迭代对象，这能直接辅助我们进行训练。
- 该可迭代对象也被称为*数据加载器*。

创建数据加载器后，即可进入训练环节。我们需要定义训练所需的设备，并设置优化器和学习率调度器。

```python
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# 由于处理的是行人和背景，类别数量为 2
num_classes = 2
final_model = modify_model(num_classes)

# 将模型移至 GPU（若可用）或 CPU
final_model.to(device)

## 获取 SGD 优化器
params = [p for p in final_model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params,
                            lr=0.005,
                            momentum=0.9,
                            weight_decay=0.0005)

# 设置步进学习率
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                               step_size=2,
                                               gamma=0.1)
```

代码中需要关注的关键点如下：

- 模型所在的设备必须与训练数据所在的设备保持一致。模型与数据之间不应存在跨基础设施的迁移。如果数据量过大无法放入单个 GPU，则数据批次应与模型位于同一系统中。
- 设置优化器和学习率调度器。
- 需要注意的是，在训练复杂网络时，固定学习率无法帮助我们更快或更高效地训练。学习率本身是训练过程中最重要的超参数之一，我们需要谨慎处理。

现在我们已经设置了模型参数，接下来运行几个周期来检查训练过程。

```python

# 设置训练周期数
num_epochs = 5
for epoch in range(num_epochs):
    ## 直接使用 PyTorch 辅助函数中的 train_one_epoch
    # 熟悉微调框架
    train_one_epoch(final_model, optimizer, train_data_loader, device, epoch, print_freq=10)
    # 更新权重和学习率
    lr_scheduler.step()
    # 通过权重变化获取评估结果
    evaluate(final_model, test_data_loader, device=device)
```

为简单起见，我们仅运行了五个周期，但为了获得更好的结果，建议延长训练时间。训练过程中最重要的环节之一是检查生成的日志。日志应能让我们清晰了解数据如何通过模型运行，以及训练过程如何建立。让我们快速浏览一下各周期生成的日志。

```
Epoch: [0]  [ 0/60]  eta: 0:02:17  lr: 0.000090  loss: 2.7890 (2.7890)  loss_classifier: 0.7472 (0.7472)  loss_box_reg: 0.3405 (0.3405)  loss_mask: 1.6637 (1.6637)  loss_objectness: 0.0351 (0.0351)  loss_rpn_box_reg: 0.0025 (0.0025)  time: 2.2894  data: 0.4357  max mem: 2161
Epoch: [0]  [10/60]  eta: 0:01:26  lr: 0.000936  loss: 1.3992 (1.7301)  loss_classifier: 0.5175 (0.4831)  loss_box_reg: 0.2951 (0.2971)  loss_mask: 0.7160 (0.9201)  loss_objectness: 0.0279 (0.0249)  loss_rpn_box_reg: 0.0045 (0.0048)  time: 1.7208  data: 0.0469  max mem: 3316
Epoch: [0]  [20/60]  eta: 0:01:05  lr: 0.001783  loss: 1.0006 (1.2323)  loss_classifier: 0.2196 (0.3358)  loss_box_reg: 0.2905 (0.2854)  loss_mask: 0.3228 (0.5877)  loss_objectness: 0.0172 (0.0188)  loss_rpn_box_reg: 0.0042 (0.0045)  time: 1.6055  data: 0.0096  max mem: 3316
Epoch: [0]  [30/60]  eta: 0:00:49  lr: 0.002629  loss: 0.5668 (1.0164)  loss_classifier: 0.0936 (0.2558)  loss_box_reg: 0.2643 (0.2860)  loss_mask: 0.1797 (0.4540)  loss_objectness: 0.0056 (0.0156)  loss_rpn_box_reg: 0.0045 (0.0050)  time: 1.6322  data: 0.0108  max mem: 3316
Epoch: [0]  [40/60]  eta: 0:00:33  lr: 0.003476  loss: 0.4461 (0.8835)  loss_classifier: 0.0639 (0.2070)  loss_box_reg: 0.2200 (0.2681)  loss_mask: 0.1693 (0.3904)  loss_objectness: 0.0028 (0.0126)  loss_rpn_box_reg: 0.0057 (0.0054)  time: 1.6640  data: 0.0107  max mem: 3316
Epoch: [0]  [50/60]  eta: 0:00:16  lr: 0.004323  loss: 0.3779 (0.7842)  loss_classifier: 0.0396 (0.1749)  loss_box_reg: 0.1619 (0.2452)  loss_mask: 0.1670 (0.3476)  loss_objectness: 0.0014 (0.0107)  loss_rpn_box_reg: 0.0051 (0.0058)  time: 1.5650  data: 0.0107  max mem: 3316
Epoch: [0]  [59/60]  eta: 0:00:01  lr: 0.005000  loss: 0.3066 (0.7143)  loss_classifier: 0.0329 (0.1549)  loss_box_reg: 0.1074 (0.2265)  loss_mask: 0.1508 (0.3172)  loss_objectness: 0.0022 (0.0097)  loss_rpn_box_reg: 0.0052 (0.0059)  time: 1.5627  data: 0.0109  max mem: 3316
Epoch: [0] Total time: 0:01:37 (1.6202 s / it)
creating index...
index created!
Test:  [ 0/50]  eta: 0:00:27  model_time: 0.3958 (0.3958)  evaluator_time: 0.0052 (0.0052)  time: 0.5474  data: 0.1449  max mem: 3316
Test:  [49/50]  eta: 0:00:00  model_time: 0.3451 (0.3489)  evaluator_time: 0.0061 (0.0110)  time: 0.3666  data: 0.0055  max mem: 3316
Test: Total time: 0:00:18 (0.3715 s / it)
Averaged stats: model_time: 0.3451 (0.3489)  evaluator_time: 0.0061 (0.0110)
Accumulating evaluation results...
DONE (t=0.01s).
Accumulating evaluation results...
DONE (t=0.01s).
IoU metric: bbox
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.690
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.976
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.863
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.363
Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.708
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.311
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.747
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.747
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.637
Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.755
IoU metric: segm
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.722
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.976
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.886
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.448
Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.740
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.325
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.760
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.761
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.675
Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.767
```

在检查任何训练网络的日志时，需要注意以下几个要点：

- 由于我们采用了逐步学习率变化策略，这在日志输出中很容易观察到。
- 每个批次的分类器损失和目标性损失都可以被记录下来。
- 其他需要关注的重要指标是平均精度和平均召回率。
- 预计完成时间（ETA）和内存分配有助于我们估算更大规模模型评估所需的计算量。

既然我们已经训练好了模型，可以选择使用 `torch save` 命令来保存模型。我们也可以使用“保存字典”选项，这比仅仅保存序列化形式具有更多优势。当保存字典时，我们可以在需要时修改字典内容，但以序列化形式存储时则无法做到这一点。序列化形式会存储目录路径和模型参数，且很难更改或修改。

```

## 保存模型的完整版本

## 也可以选择保存状态字典版本
torch.save(final_model, 'mask-rcnn-fine_tuned.pt')
```

既然我们已经有了训练好的模型，就可以开始进行推理了。首先需要切换到 `eval` 模式，该模式会将模型设置为评估模式，此时不会计算任何梯度。

```

# PyTorch 帮助将模型设置为评估模式
final_model.eval()
CLASSES = ['__background__', 'pedestrian']
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
final_model.to(device)
```

这也有助于我们描述将要用于推理的模型。

```python
MaskRCNN(
(transform): GeneralizedRCNNTransform(
Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
Resize(min_size=(800,), max_size=1333, mode='bilinear')
)
(backbone): BackboneWithFPN(
(body): IntermediateLayerGetter(
(conv1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
(bn1): FrozenBatchNorm2d(64, eps=0.0)
(relu): ReLU(inplace=True)
(maxpool): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
(layer1): Sequential(
(0): Bottleneck(
(conv1): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
(bn1): FrozenBatchNorm2d(64, eps=0.0)
(conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
(bn2): FrozenBatchNorm2d(64, eps=0.0)
(conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
(bn3): FrozenBatchNorm2d(256, eps=0.0)
(relu): ReLU(inplace=True)
(downsample): Sequential(
(0): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
(1): FrozenBatchNorm2d(256, eps=0.0)
)
)
(1): Bottleneck(
(conv1): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
(bn1): FrozenBatchNorm2d(64, eps=0.0)
(conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
(bn2): FrozenBatchNorm2d(64, eps=0.0)
(conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
(bn3): FrozenBatchNorm2d(256, eps=0.0)
(relu): ReLU(inplace=True)
)
(2): Bottleneck(
(conv1): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
(bn1): FrozenBatchNorm2d(64, eps=0.0)
(conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
(bn2): FrozenBatchNorm2d(64, eps=0.0)
(conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
(bn3): FrozenBatchNorm2d(256, eps=0.0)
(relu): ReLU(inplace=True)
)
)
(layer2): Sequential(
(0): Bottleneck(
(conv1): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
(bn1): FrozenBatchNorm2d(128, eps=0.0)
(conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
(bn2): FrozenBatchNorm2d(128, eps=0.0)
(conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
(bn3): FrozenBatchNorm2d(512, eps=0.0)
(relu): ReLU(inplace=True)
(downsample): Sequential(
(0): Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
(1): FrozenBatchNorm2d(512, eps=0.0)
)
)
(1): Bottleneck(
(conv1): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
(bn1): FrozenBatchNorm2d(128, eps=0.0)
(conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
(bn2): FrozenBatchNorm2d(128, eps=0.0)
(conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
(bn3): FrozenBatchNorm2d(512, eps=0.0)
(relu): ReLU(inplace=True)
)
(2): Bottleneck(
(conv1): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
(bn1): FrozenBatchNorm2d(128, eps=0.0)
(conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
(bn2): FrozenBatchNorm2d(128, eps=0.0)
(conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
(bn3): FrozenBatchNorm2d(512, eps=0.0)
(relu): ReLU(inplace=True)
)
(3): Bottleneck(
(conv1): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
(bn1): FrozenBatchNorm2d(128, eps=0.0)
(conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
(bn2): FrozenBatchNorm2d(128, eps=0.0)
(conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
(bn3): FrozenBatchNorm2d(512, eps=0.0)
(relu): ReLU(inplace=True)
)
)
(layer3): Sequential(
(0): Bottleneck(
(conv1): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
(bn1): FrozenBatchNorm2d(256, eps=0.0)
(conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
(bn2): FrozenBatchNorm2d(256, eps=0.0)
(conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
(bn3): FrozenBatchNorm2d(1024, eps=0.0)
(relu): ReLU(inplace=True)
(downsample): Sequential(
(0): Conv2d(512, 1024, kernel_size=(1, 1), stride=(2, 2), bias=False)
(1): FrozenBatchNorm2d(1024, eps=0.0)
)
)
(1): Bottleneck(
(conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
(bn1): FrozenBatchNorm2d(256, eps=0.0)
(conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
(bn2): FrozenBatchNorm2d(256, eps=0.0)
(conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
(bn3): FrozenBatchNorm2d(1024, eps=0.0)
(relu): ReLU(inplace=True)
)
(2): Bottleneck(
(conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
(bn1): FrozenBatchNorm2d(256, eps=0.0)
(conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
(bn2): FrozenBatchNorm2d(256, eps=0.0)
(conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
(bn3): FrozenBatchNorm2d(1024, eps=0.0)
(relu): ReLU(inplace=True)
)
(3): Bottleneck(
(conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
(bn1): FrozenBatchNorm2d(256, eps=0.0)
(conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
(bn2): FrozenBatchNorm2d(256, eps=0.0)
(conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
(bn3): FrozenBatchNorm2d(1024, eps=0.0)
(relu): ReLU(inplace=True)
)
(4): Bottleneck(
(conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
(bn1): FrozenBatchNorm2d(256, eps=0.0)
(conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
(bn2): FrozenBatchNorm2d(256, eps=0.0)
(conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
(bn3): FrozenBatchNorm2d(1024, eps=0.0)
(relu): ReLU(inplace=True)
)
(5): Bottleneck(
(conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
(bn1): FrozenBatchNorm2d(256, eps=0.0)
(conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
(bn2): FrozenBatchNorm2d(256, eps=0.0)
(conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
(bn3): FrozenBatchNorm2d(1024, eps=0.0)
(relu): ReLU(inplace=True)
)
)
(layer4): Sequential(
(0): Bottleneck(
(conv1): Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
(bn1): FrozenBatchNorm2d(512, eps=0.0)
(conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
(bn2): FrozenBatchNorm2d(512, eps=0.0)
(conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
(bn3): FrozenBatchNorm2d(2048, eps=0.0)
(relu): ReLU(inplace=True)
(downsample): Sequential(
(0): Conv2d(1024, 2048, kernel_size=(1, 1), stride=(2, 2), bias=False)
(1): FrozenBatchNorm2d(2048, eps=0.0)
)
)
(1): Bottleneck(
(conv1): Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
(bn1): FrozenBatchNorm2d(512, eps=0.0)
(conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
(bn2): FrozenBatchNorm2d(512, eps=0.0)
(conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
(bn3): FrozenBatchNorm2d(2048, eps=0.0)
(relu): ReLU(inplace=True)
)
(2): Bottleneck(
(conv1): Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
(bn1): FrozenBatchNorm2d(512, eps=0.0)
(conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
(bn2): FrozenBatchNorm2d(512, eps=0.0)
(conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
(bn3): FrozenBatchNorm2d(2048, eps=0.0)
(relu): ReLU(inplace=True)
)
)
)
(fpn): FeaturePyramidNetwork(
(inner_blocks): ModuleList(
(0): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
(1): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1))
(2): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1))
(3): Conv2d(2048, 256, kernel_size=(1, 1), stride=(1, 1))
)
(layer_blocks): ModuleList(
(0): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
(1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
(2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
(3): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
)
(extra_blocks): LastLevelMaxPool()
)
)
(rpn): RegionProposalNetwork(
(anchor_generator): AnchorGenerator()
(head): RPNHead(
(conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
(cls_logits): Conv2d(256, 3, kernel_size=(1, 1), stride=(1, 1))
(bbox_pred): Conv2d(256, 12, kernel_size=(1, 1), stride=(1, 1))
)
)
(roi_heads): RoIHeads(
(box_roi_pool): MultiScaleRoIAlign(featmap_names=['0', '1', '2', '3'], output_size=(7, 7), sampling_ratio=2)
(box_head): TwoMLPHead(
(fc6): Linear(in_features=12544, out_features=1024, bias=True)
(fc7): Linear(in_features=1024, out_features=1024, bias=True)
)
(box_predictor): FastRCNNPredictor(
(cls_score): Linear(in_features=1024, out_features=2, bias=True)
(bbox_pred): Linear(in_features=1024, out_features=8, bias=True)
)
(mask_roi_pool): MultiScaleRoIAlign(featmap_names=['0', '1', '2', '3'], output_size=(14, 14), sampling_ratio=2)
(mask_head): MaskRCNNHeads(
(mask_fcn1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
(relu1): ReLU(inplace=True)
(mask_fcn2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
(relu2): ReLU(inplace=True)
(mask_fcn3): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
(relu3): ReLU(inplace=True)
(mask_fcn4): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
(relu4): ReLU(inplace=True)
)
(mask_predictor): MaskRCNNPredictor(
(conv5_mask): ConvTranspose2d(256, 256, kernel_size=(2, 2), stride=(2, 2))
(relu): ReLU(inplace=True)
(mask_fcn_logits): Conv2d(256, 2, kernel_size=(1, 1), stride=(1, 1))
)
)
)
```

该模型定义有助于我们确立当前使用的架构以及可围绕其进行的修改。完成这一步后，我们再编写几行代码来显示掩码。

```
def get_mask_color(mask_conf):

## helper function to generate mask
colour_option = [[0, 250, 0],[0, 0, 250],[250, 0, 0],[0, 250, 250],[250, 250, 0],[250, 0, 250],[75, 65, 170],[230, 75, 180],[235, 130, 40],[60, 140, 240],[40, 180, 180]]
blue = np.zeros_like(mask_conf).astype(np.uint8)
green = np.zeros_like(mask_conf).astype(np.uint8)
red = np.zeros_like(mask_conf).astype(np.uint8)
red[mask_conf == 1], green[mask_conf == 1], blue[mask_conf == 1] = colour_option[random.randrange(0,10)]
mask_color = np.stack([red, green, blue], axis=2)
return mask_color
def generate_prediction(image_path, conf):

## helper function to generate predictions
image = Image.open(image_path)
transform = T.Compose([T.ToTensor()])
image = transform(image)
image = image.to(device)
predicted = final_model([image])
predicted_score = list(predicted[0][’scores’].detach().cpu().numpy())
predicted_temp = [predicted_score.index(x) for x in predicted_score if x>conf][-1]
masks = (predicted[0][’masks’]>0.5).squeeze().detach().cpu().numpy()

# print(pred[0][’labels’].numpy().max())
predicted_class_val = [CLASSES[i] for i in list(predicted[0]['labels'].cpu().numpy())]
predicted_box_val = [[(i[0], i[1]), (i[2], i[3])] for i in list(predicted[0]['boxes'].detach().cpu().numpy())]
masks = masks[:predicted_temp+1]
predicted_class_name = predicted_class_val[:predicted_temp+1]
predicted_box_score = predicted_box_val[:predicted_temp+1]
return masks, predicted_box_score, predicted_class_name
def segment_image(image_path, confidence=0.5, rect_thickness=2, text_size=2, text_thickness=2):
masks_conf, box_conf, predicted_class = generate_prediction(image_path, confidence)
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
for i in range(len(masks_conf)):
rgb_mask = get_mask_color(masks_conf[i])
image = cv2.addWeighted(image, 1, rgb_mask, 0.5, 0)
cv2.rectangle(image, box_conf[i][0], box_conf[i][1],color=(0, 255, 0), thickness=rect_thickness)
cv2.putText(image,predicted_class[i], box_conf[i][0], cv2.FONT_HERSHEY_SIMPLEX, text_size, (0,255,0),thickness=text_thickness)
plt.figure(figsize=(20,30))
plt.imshow(image)
plt.xticks([])
plt.yticks([])
plt.show()
segment_image('/content/pedestrian_img.jpg', confidence=0.7)
```

## 总结

本章讨论了图像分割的工作原理及其在市场上的不同变体。这是解决涉及图像分割问题的一个重要方面。同时，本章还通过一个示例建立了微调的概念。接下来，这里学到的概念将有助于理解计算机视觉的相关概念。

下一章将探讨如何构建管道，以解决涉及图像相似性的业务问题。

