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

