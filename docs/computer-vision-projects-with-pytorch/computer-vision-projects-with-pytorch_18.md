# 异常检测

与所有其他领域一样，视觉分析中的异常检测可分为两大类型：

- **新颖性检测：** 在训练过程中，模型会接触到由标准事件分布产生的数据。当对未知样本进行测试或预测时，算法应能发现异常数据。在此过程中，假设数据不包含任何非标准数据。这是半监督学习方法的一个例子。
- **离群点检测：** 在这种情况下，算法会同时接触到标准数据和非标准数据。由于原则上标准数据会集中分布，算法会学习它并忽略离群点。我们可以以决策树为例，其分支在分裂过程中总是会尽早尝试分离离群点。这种方法中的数据被标准数据和非标准数据污染。算法会找出哪些数据点是内点，哪些是离群点。这是一种无监督的训练方式。

我们有多种方法来检测离群点或新颖性。我们可以使用统计方法，例如总体均值和标准差来寻找离群点。然而，在这些场景中，必须了解数据的分布情况。在机器学习方法中，有一些算法可以帮助我们进行异常检测。

- **局部离群因子：** 该算法计算一个量化局部密度偏差的值。它试图定位那些与邻近样本相比密度较低的样本。
- **孤立森林：** 存在一些基于决策进行迭代分裂的方法，可用于确定样本中的离群点。如果我们能够利用基于决策树集成或随机森林的算法，就很容易得出结论：落在随机森林较短路径上的样本即为异常点。
- **单类支持向量机：** 这可以被视为支持向量机类的一个扩展，在确定阈值后，可以检查概率分布的支持度，从而在此过程中分离出离群点。

让我们看看计算机视觉领域的一些应用。

- **无监督密度估计：** 该算法试图估计特征或训练图像的概率分布。一旦模型知道了分布，对于所有未知样本，它会尝试确定该样本与分布之间的差异。
- **无监督图像重建：** 训练编码器-解码器架构的一般过程。它让网络学习向量化的潜在特征，并以一定的损失重建原始图像。与异常图像相比，正常图像的重建损失会更小。
- **单类异常检测：** 这种方法类似于前面讨论的单类支持向量机。该算法试图估计一个决策边界，以将正常类与异常类分开。

生成类算法可用于检测异常，并且已被多位研究人员证实。现在我们已经了解了一些基本概念，让我们来看一个异常检测的例子。

一些方法包括以下内容：

- 使用预训练模型，并对最后几层进行训练以进行异常分类
- 编码和解码方法
- 异常图像分类以及使用特征图定位图像中的异常

## 方法 1：使用预训练分类模型

从给定的图像数据集中找出异常图像可以被视为一个二值图像分类任务，即根据训练数据集判断图像是否为异常。这里使用一种名为 VGG-16 的成熟架构来训练最后几层。

VGG-16 架构包含 16 层，其中 13 个是卷积层，最后三个是全连接层。该网络经过训练，可以从总共 1,000 个类别中预测输入的类别。

在当前方法中，前十个卷积层使用预训练权重。最后几层用于对自定义数据集进行模型训练。输出被分类为两个类别之一。图 7-1 中突出显示的框用于训练自定义数据集。

![](img/520381_1_En_7_Fig1_HTML.png)

VGG-16 架构图像异常检测预训练分类模型 VGG-16 架构图。从左到右的水平图从输入开始，依次进入卷积层：Convi 1-1、Convi 1-2、池化层，然后按顺序模式进入 Convi 2-1、Convi 2-2。全连接层被标识为密集层。

**图 7-1** VGG-16 架构

### 步骤 1：导入所需库

```
#import torch
import torch
import torchvision
import matplotlib.pyplot as plt
#import time,os etc
import time
import os
import numpy as np
import random
from distutils.version import LooseVersion as Version
from itertools import product
```

### 步骤 2：创建种子和确定性函数

这些函数有助于为所有迭代生成相同的随机数。

```
def seed_setting(sd):
    os.environ["PL_GLOBAL_SEED"] = str(sd)
    random.seed(sd)
    np.random.seed(sd)
    torch.manual_seed(sd)
    torch.cuda.manual_seed_all(sd)

def fn_det_setting():
    #check if cuda is available
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    #check torch version
    if torch.__version__ <= Version("1.7"):
        torch.fn_det_setting(True)
    else:
        torch.use_deterministic_algorithms(True)
```

### 步骤 3：设置超参数

```
#set seed, batch size
RNDM_SEED = 245
btch_input_sz = 128
epch_nmbr = 25
DEVICE = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
seed_setting(RNDM_SEED)
#fn_det_setting() This may not work on Gpu because some algorithms are not deterministic on Gpu.
```

### 步骤 4：导入数据集

以下是训练数据：

```
tr_ds_path = "/content/drive/MyDrive/car_img/tr" #Training Images
```

以下是验证数据：

```
vd_ds_path = "/content/drive/MyDrive/car_img/vds" #Validation Images
```

以下是测试数据：

```
ts_ds_path = "/content/drive/MyDrive/car_img/ts" #Test Images
```



### 第 5 步：图像预处理阶段

图像变换包括以下步骤：

- **图像缩放**：保持训练集、测试集和验证集中图像尺寸一致
- **图像裁剪**：裁剪图像边缘
- **图像转张量**：用于 PyTorch 实现
- **图像归一化**：加速损失收敛

```python
import torch.utils.data as data
tr_data_trans = torchvision.transforms.Compose([
    torchvision.transforms.Resize((70, 70)),
    torchvision.transforms.RandomCrop((64, 64)),
    torchvision.transforms.ToTensor(), #it converts data in the range 0-255 to 0-1.
    torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
validation_data_trans = torchvision.transforms.Compose([
    torchvision.transforms.Resize((70, 70)),
    torchvision.transforms.CenterCrop((64, 64)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
tst_data_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((70, 70)),
    torchvision.transforms.CenterCrop((64, 64)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
```

`DataLoader` 函数通过并行传递数据来加速数据加载过程。

```python
train_ds_cln = torchvision.datasets.ImageFolder(root=tr_ds_path, transform=tr_data_trans)
train_loader_cln = data.DataLoader(train_ds_cln, btch_input_sz=206, shuffle=True)
test_ds_cln = torchvision.datasets.ImageFolder(root=ts_ds_path, transform=tst_data_transform)
test_loader_cln = data.DataLoader(test_ds_cln, btch_input_sz=206, shuffle=True)
valid_ds_cln = torchvision.datasets.ImageFolder(root=vd_ds_path, transform=validation_data_trans)
valid_loader_cln = data.DataLoader(valid_ds_cln, btch_input_sz=63, shuffle=True)
# Checking the dataset
for images, labels in train_loader_cln:
    print('Image batch dimensions:', images.shape)
    print('Image label dimensions:', labels.shape)
    print('Class labels of 10 examples:', labels[:10])
    break
```

输出如下：

```
Image batch dimensions: torch.Size([206, 3, 64, 64])
Image label dimensions: torch.Size([206])
Class labels of 10 examples: tensor([1, 0, 0, 1, 0, 0, 1, 1, 1, 1])
```

以下是训练数据集：

```python
for images, labels in train_loader_cln:
    print('Image batch dimensions:', images.shape)
    print('Image label dimensions:', labels.shape)
    print('Class labels of 10 examples:', labels[:10])
    break
```

输出如下：

```
Image batch dimensions: torch.Size([206, 3, 64, 64])
Image label dimensions: torch.Size([206])
Class labels of 10 examples: tensor([0, 1, 0, 1, 1, 1, 1, 1, 0, 1])
tr_ds = images
tr_ds.shape
```

输出如下：

```
torch.Size([206, 3, 64, 64])
tr_label = labels
tr_label.shape
```

输出如下：

```
torch.Size([206])
```

以下是验证数据集：

```python
for images, labels in valid_loader_cln:
    print('Image batch dimensions:', images.shape)
    print('Image label dimensions:', labels.shape)
    print('Class labels of 10 examples:', labels[:10])
    break
```

输出如下：

```
Image batch dimensions: torch.Size([63, 3, 64, 64])
Image label dimensions: torch.Size([63])
Class labels of 10 examples: tensor([1, 1, 0, 0, 1, 0, 1, 1, 1, 1])
vd_ds = images
vd_ds.shape
```

输出如下：

```
torch.Size([63, 3, 64, 64])
```

以下是测试数据集：

```python
for images, labels in test_loader_cln:
    print('Image batch dimensions:', images.shape)
    print('Image label dimensions:', labels.shape)
    print('Class labels of 10 examples:', labels[:10])
    break
```

输出如下：

```
Image batch dimensions: torch.Size([63, 3, 64, 64])
Image label dimensions: torch.Size([63])
Class labels of 10 examples: tensor([1, 1, 1, 1, 1, 0, 1, 1, 1, 0])
```

### 第 6 步：加载预训练模型

```python
model = torchvision.models.vgg16(pretrained=True)
model
```

输出如下：

```
VGG(
  (features): Sequential(
    (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): ReLU(inplace=True)
    (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (3): ReLU(inplace=True)
    (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (6): ReLU(inplace=True)
    (7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (8): ReLU(inplace=True)
    (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (10): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (11): ReLU(inplace=True)
    (12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (13): ReLU(inplace=True)
    (14): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (15): ReLU(inplace=True)
    (16): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (17): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (18): ReLU(inplace=True)
    (19): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (20): ReLU(inplace=True)
    (21): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (22): ReLU(inplace=True)
    (23): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (24): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (25): ReLU(inplace=True)
    (26): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (27): ReLU(inplace=True)
    (28): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (29): ReLU(inplace=True)
    (30): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (avgpool): AdaptiveAvgPool2d(output_size=(7, 7))
  (classifier): Sequential(
    (0): Linear(in_features=25088, out_features=4096, bias=True)
    (1): ReLU(inplace=True)
    (2): Dropout(p=0.5, inplace=False)
    (3): Linear(in_features=4096, out_features=4096, bias=True)
    (4): ReLU(inplace=True)
    (5): Dropout(p=0.5, inplace=False)
    (6): Linear(in_features=4096, out_features=1000, bias=True)
  )
)
```

### 第 7 步：冻结模型

这里，自适应平均池化层是卷积层和线性层之间的桥梁。我们只打算训练线性层。最简单的方法是先冻结整个模型。因此，我们遍历模型中的所有参数。

假设我们要微调（训练）最后三层：

```python
for param in model.parameters():
    param.requires_grad = False
```

现在，我们仍然可以运行模型的前向和反向传播，但不会更新参数。在接下来的步骤中，我们将微调最后三层。

```python
model.classifier[1].requires_grad = True
model.classifier[3].requires_grad = True
```

对于最后一层，由于类别标签数量与 ImageNet 不同，我们将输出层替换为自定义的输出层：

```python
model.classifier[6] = torch.nn.Linear(4096, 2)
```



### 第 8 步：训练模型

以下是训练过程：

```
def find_acc_metric(input_model, input_data_ldr, dvc):
    with torch.no_grad():
        correct_pred, num_examples = 0, 0
        for i, (features, targets) in enumerate(input_data_ldr):
            features = features.to(dvc)
            targets = targets.float().to(dvc)
            preds = input_model(features)
            _, predicted_labels = torch.max(preds, 1)
            num_examples += targets.size(0)
            correct_pred += (predicted_labels == targets).sum()
        return correct_pred.float()/num_examples * 100

def mdl_training(model, epch_nmbr, train_loader,
                 valid_loader, test_loader, optimizer,
                 device, logging_interval=50,
                 scheduler=None,
                 scheduler_on='valid_acc'):
    tme_strt = time.time()
    list_from_loss, accuracy_train, accuracy_validation = [], [], []
    for epoch in range(epch_nmbr):
        model.train()
        for batch_idx, (features, targets) in enumerate(train_loader):
            features = features.to(device)
            targets = targets.to(device)
            preds = model(features)
            loss = torch.nn.functional.cross_entropy(preds, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            list_from_loss.append(loss.item())
            if not batch_idx % logging_interval:
                print(f'Epoch: {epoch+1:03d}/{epch_nmbr:03d} '
                      f'| Batch {batch_idx:04d}/{len(train_loader):04d} '
                      f'| Loss: {loss:.4f}')
        model.eval()
        with torch.no_grad():
            train_acc = find_acc_metric(model, train_loader, device=device)
            valid_acc = find_acc_metric(model, valid_loader, device=device)
            print(f'Epoch: {epoch+1:03d}/{epch_nmbr:03d} '
                  f'| Train: {train_acc :.2f}% '
                  f'| Validation: {valid_acc :.2f}%')
            accuracy_train.append(train_acc.item())
            accuracy_validation.append(valid_acc.item())
        tr_time = (time.time() - tme_strt)/60
        print(f'Time tr_time: {tr_time:.2f} min')
        if scheduler is not None:
            if scheduler_on == 'valid_acc':
                scheduler.step(accuracy_validation[-1])
            elif scheduler_on == 'minibatch_loss':
                scheduler.step(list_from_loss[-1])
            else:
                raise ValueError(f'Invalid `scheduler_on` choice.')
    tr_time = (time.time() - tme_strt)/60
    print(f'Final Training Time: {tr_time:.2f} min')
    test_acc = find_acc_metric(model, test_loader, device=device)
    print(f'Test accuracy {test_acc :.2f}%')
    return list_from_loss, accuracy_train, accuracy_validation
```

### 第 9 步：评估模型

```
def Viz_acc(acc_training, val_acc, loc_res):
    epch_nmbr = len(acc_training)
    plt.plot(np.arange(1, epch_nmbr+1),
             acc_training, label='Training')
    plt.plot(np.arange(1, epch_nmbr+1),
             val_acc, label='Validation')
    plt.xlabel('# of Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    if loc_res is not None:
        image_path = os.path.join(
            loc_res, 'plot_acc_training_validation.pdf')
        plt.savefig(image_path)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(DEVICE)
optimizer = torch.optim.SGD(model.parameters(), momentum=0.9, lr=0.01)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                       factor=0.1,
                                                       mode='max',
                                                       verbose=True)
list_from_loss, accuracy_train, accuracy_validation = mdl_training(
    model=model,
    epch_nmbr=5,
    train_loader=train_loader_cln,
    valid_loader=valid_loader_cln,
    test_loader=test_loader_cln,
    optimizer=optimizer,
    device=DEVICE,
    scheduler=scheduler,
    scheduler_on='valid_acc',
    logging_interval=100)
```

以下是输出结果：

```
Epoch: 001/005 | Batch 0000/0001 | Loss: 1.4587
Epoch: 001/005 | Train: 79.13% | Validation: 76.21%
Time elapsed: 0.34 min
Epoch: 002/005 | Batch 0000/0001 | Loss: 0.8952
Epoch: 002/005 | Train: 92.72% | Validation: 90.29%
Time elapsed: 0.67 min
Epoch: 003/005 | Batch 0000/0001 | Loss: 0.3280
Epoch: 003/005 | Train: 97.57% | Validation: 96.60%
Time elapsed: 0.99 min
Epoch: 004/005 | Batch 0000/0001 | Loss: 0.1774
Epoch: 004/005 | Train: 99.03% | Validation: 96.60%
Time elapsed: 1.32 min
Epoch: 005/005 | Batch 0000/0001 | Loss: 0.0581
Epoch: 005/005 | Train: 99.51% | Validation: 98.06%
Time elapsed: 1.66 min
Total Training Time: 1.66 min
Test accuracy 100.00%
```

训练与验证的可视化：

```
Viz_acc(accuracy_train=accuracy_train,
        accuracy_validation=accuracy_validation,
        results_dir=None)
plt.ylim([60, 100])
plt.show()
```

输出结果如图 7-2 所示。

![](img/520381_1_En_7_Fig2_HTML.jpg)

输出训练与验证准确率的折线图，用于图像异常检测的预训练分类模型。Y 轴表示准确率，X 轴表示轮次。图例中蓝色代表训练准确率，橙色代表验证准确率，用于识别波长。波长从 75 和 80 开始，直至 100。

**图 7-2** 输出训练与验证准确率对比

```
def example_sample(model, data_loader, unnormalizer=None, class_dict=None):
    for batch_idx, (features, targets) in enumerate(data_loader):
        with torch.no_grad():
            features = features
            targets = targets
            preds = model(features)
            predictions = torch.argmax(preds, dim=1)
        break
    fig, axes = plt.subplots(nrows=3, ncols=5,
                             sharex=True, sharey=True)
    if unnormalizer is not None:
        for idx in range(features.shape[0]):
            features[idx] = unnormalizer(features[idx])
        nhwc_img = np.transpose(features, axes=(0, 2, 3, 1))
        if nhwc_img.shape[-1] == 1:
            nhw_img = np.squeeze(nhwc_img.numpy(), axis=3)
        for idx, ax in enumerate(axes.ravel()):
            ax.imshow(nhw_img[idx], cmap='binary')
            if class_dict is not None:
                ax.title.set_text(f'P: {class_dict[predictions[idx].item()]}'
                                  f'\nT: {class_dict[targets[idx].item()]}')
            else:
                ax.title.set_text(f'P: {predictions[idx]} | T: {targets[idx]}')
            ax.axison = False
    else:
        for idx, ax in enumerate(axes.ravel()):
            ax.imshow(nhwc_img[idx])
            if class_dict is not None:
                ax.title.set_text(f'P: {class_dict[predictions[idx].item()]}'
                                  f'\nT: {class_dict[targets[idx].item()]}')
            else:
                ax.title.set_text(f'P: {predictions[idx]} | T: {targets[idx]}')
            ax.axison = False
    plt.tight_layout()
    plt.show()

class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Parameters:
        tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
        """
```



