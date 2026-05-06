# 模型代码

我们已经讨论了生成对抗网络背后的基本概念。这引出了它的众多应用之一：超分辨率。超分辨率技术有多种应用场景，包括风格迁移、图像生成和超分辨率重建等。处理超分辨率问题的模型是`SRGAN`。它的前身之一`SRResNet`在`SSIM`和`PSNR`指标上取得了不错的结果。

让我们来看看超分辨率问题中通常使用的评估指标：

*   **结构相似性指数（SSIM）：** 该指标旨在量化因源图像到目标图像变化而导致的退化程度。它检查图像各部分之间的感知相似性，基于所选窗口的平均值和标准差。
*   **峰值信噪比：** 这是另一个重要指标，用于衡量图像或经过变换的图像与原始图像之间的重建损失。它最好通过均方误差计算来定义，可以通过以 10 为底的对数尺度来表示。
*   **平均意见得分（MOS）：** 该指标由一个序数尺度上的单一数字定义，范围从 1 到 5。1 表示最低感知质量，5 表示最高感知质量。

现在我们已经了解了用于定义和衡量差异的指标，接下来看看我们将要开发代码所基于的数据。

我们将使用`DIV2K`数据集，该数据集包含 1000 张高清图像，按 800-100-100 的比例划分为训练集、验证集和测试集。这些数据可以从 2017 年 CVPR 会议上发表的原始论文中下载，网址为[`https://data.vision.ee.ethz.ch/cvl/DIV2K/`](https://data.vision.ee.ethz.ch/cvl/DIV2K/)。

代码的搭建需要遵循应用程序的标准构建流程。通常这意味着需要有一个模型文件、一些实用脚本、一个训练文件和一个验证文件。在某些情况下，这个模型需要作为托管在服务器上的应用程序，因此还需要一个设置文件。我们一步一步来，先从模型文件开始。

## 模型开发

代码库包含一个生成模型块、一个判别模型块、一个残差块和一个内容损失计算块。

### 导入

整个代码块将使用`Torch`框架。如果在本地环境中进行开发，我们必须确保环境中已安装`Torch`及其依赖项并能正常工作。`Torch`和`TorchVision`是需要设置的两个重要包。如果配备有支持 CUDA 核心的 GPU，我们应该安装最新的 CUDA 包，以帮助`PyTorch`利用并行 GPU 核心进行计算。对于模型脚本，我们导入与`Torch`和`TorchVision`相关的函数。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch import Tensor
```

接下来，我们定义生成器类以帮助重建图像。

```python
class Generator(nn.Module):
## 定义生成器模型
## 继承类
## 初始化顺序网络 - 期望输入为 64x3
def __init__(self) -> None:
super(Generator, self).__init__()
self.convolutional_block1 = nn.Sequential(
nn.Conv2d(3, 64, (9, 9), (1, 1), (4, 4)),
nn.PReLU()
)
## 添加 16 个残差卷积块
res_trunk = []
for _ in range(16):
res_trunk.append(ResidualConvBlock(64))
self.res_trunk = nn.Sequential(*res_trunk)
self.convolutional_block2 = nn.Sequential(
nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1), bias=False),
nn.BatchNorm2d(64)
)
self.upsampling = nn.Sequential(
nn.Conv2d(64, 256, (3, 3), (1, 1), (1, 1)),
nn.PixelShuffle(2),
nn.PReLU(),
nn.Conv2d(64, 256, (3, 3), (1, 1), (1, 1)),
nn.PixelShuffle(2),
nn.PReLU()
)
self.convolutional_block3 = nn.Conv2d(64, 3, (9, 9), (1, 1), (4, 4))
self._initialize_weights()
def forward(self, x: Tensor) -> Tensor:
return self._forward_impl(x)
def _forward_impl(self, x: Tensor) -> Tensor:
## 定义前向传播 -> 3 个卷积块
out1 = self.convolutional_block1(x)
out = self.res_trunk(out1)
out2 = self.convolutional_block2(out)
output = out1 + out2
output = self.upsampling(output)
output = self.convolutional_block3(output)
return output
def _initialize_weights(self) -> None:
## 初始化权重
## 为批归一化添加支持
for m in self.modules():
if isinstance(m, nn.Conv2d):
nn.init.kaiming_normal_(m.weight)
if m.bias is not None:
nn.init.constant_(m.bias, 0)
m.weight.data *= 0.1
elif isinstance(m, nn.BatchNorm2d):
nn.init.constant_(m.weight, 1)
m.weight.data *= 0.1
```

这段代码定义了能够重建图像的卷积块类。重要的是，该代码块包含三个卷积块和一个上采样块。第一个卷积块之后是一个残差块，它充当整个生成器网络的主干。接着是第二个卷积块。上采样块由一对卷积层和随后的像素洗牌层组成。最后，添加最终的卷积块以生成输出。该块配备了批归一化层和 3x3 卷积层的组合。

前向传播在`forward`函数的实现中构建了顺序模型。还有一个用于初始化权重的函数。在介绍了基本的生成器类之后，我们将进入下一个判别器类。

判别器模块扩展了标准的`nn.module`，包含八层卷积。它们在每一层之后都使用批归一化以进行深层训练。模型结构使用带泄露的 ReLU 作为激活函数。模型以一个`torch.flatten`层结束，这有助于它进行分类。



```python
class Discriminator(nn.Module):
    ## 定义判别器
    def __init__(self) -> None:
        super(Discriminator, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, (3, 3), (1, 1), (1, 1), bias=True),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 64, (3, 3), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 128, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(128, 128, (3, 3), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(128, 256, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(256, 256, (3, 3), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(256, 512, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(512, 512, (3, 3), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True)
        )
        self.classifier = nn.Sequential(
            nn.Linear(512 * 6 * 6, 1024),
            nn.LeakyReLU(0.2, True),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )
    def forward(self, x: Tensor) -> Tensor:
        ## 定义前向传播
        output = self.features(x)
        output = torch.flatten(output, 1)
        output = self.classifier(output)
        return output
```

该模型在架构中建立了判别器类。接下来，我们来看一下`ContentLoss`类。

```python
class ContentLoss(nn.Module):
    ## 定义内容损失类
    ## 特征提取器 - 至第 36 层
    def __init__(self) -> None:
        super(ContentLoss, self).__init__()
        ## 使用预训练的 VGG 模型提取特征
        vgg19_model = models.vgg19(pretrained=True, num_classes=1000).eval()
        self.feature_extractor = nn.Sequential(*list(vgg19_model.features.children())[:36])
        for parameters in self.feature_extractor.parameters():
            parameters.requires_grad = False
        self.register_buffer("std", torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.register_buffer("mean", torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
    def forward(self, sr: Tensor, hr: Tensor) -> Tensor:
        hr = (hr - self.mean) / self.std
        sr = (sr - self.mean) / self.std
        mse_loss = F.mse_loss(self.feature_extractor(sr), self.feature_extractor(hr))
        return mse_loss
```

该类使用预训练的 VGG 网络提取特征，以计算内容损失。接下来，我们来看一个残差卷积块。

```python
class ResidualConvBlock(nn.Module):
    ## 获取残差块
    def __init__(self, channels: int) -> None:
        super(ResidualConvBlock, self).__init__()
        self.rc_block = nn.Sequential(
            nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(channels),
            nn.PReLU(),
            nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(channels)
        )
    def forward(self, x: Tensor) -> Tensor:
        identity = x
        output = self.rc_block(x)
        output = output + identity
        return output
```

至此，模型脚本部分结束。之后，我们来看几个辅助函数，首先从创建数据集开始。

```python
def main():
    r""" 训练与测试 """
    image_list = os.listdir(os.path.join("train", "input"))
    test_img_list = random.sample(image_list,
                                   int(len(image_list) / 10))
    ## 遍历测试文件
    for test_img_file in test_img_list:
        filename = os.path.join("train", "input", test_img_file)
        logger.info(f"处理: `{filename}`.")
        shutil.move(os.path.join("train", "input", test_img_file),
                    os.path.join("test", "input", test_img_file))
        shutil.move(os.path.join("train", "target", test_img_file),
                    os.path.join("test", "target", test_img_file))
```

该函数有助于定义训练集与测试集的划分，并定位文件以便训练任务执行。另一个重要的函数是裁剪函数，我们可以接下来查看。它用于返回裁剪后的图像。

```python
def crop_image(img, crop_sizes: int):
    assert img.size[0] == img.size[1]
    crop_num = img.size[0] // crop_sizes
    box_list = []
    for width_index in range(0, crop_num):
        for height_index in range(0, crop_num):
            box_info = ( (height_index + 0)*crop_sizes,(width_index + 0) * crop_sizes,
                        (height_index + 1)*crop_sizes,(width_index + 1) * crop_sizes)
            box_list.append(box_info)
    cropped_images = [img.crop(box_info) for box_info in box_list]
    return cropped_images
```

接下来需要处理的一个重要函数是数据集类。数据集类根据配置和可用性，向训练函数提供批次信息。

```python
class BaseDataset(Dataset):
    ## 基础数据集类，继承自 PyTorch 的 Dataset 类
    ## 应用随机裁剪、旋转、水平翻转和张量转换等增强技术
    ## 同时使用调整大小和中心裁剪
    ## 最终转换为张量
    def __init__(self, dataroot: str, image_size: int, upscale_factor: int, mode: str) -> None:
        super(BaseDataset, self).__init__()
        self.filenames = [os.path.join(dataroot, x) for x in os.listdir(dataroot)]
        lr_img_size = (image_size // upscale_factor, image_size // upscale_factor)
        hr_img_size = (image_size, image_size)
        if mode == "train":
            self.hr_transforms = transforms.Compose([
                transforms.RandomCrop(hr_img_size),
                transforms.RandomRotation(90),
                transforms.RandomHorizontalFlip(0.5),
                transforms.ToTensor()
            ])
        else:
            self.hr_transforms = transforms.Compose([
                transforms.CenterCrop(hr_img_size),
                transforms.ToTensor()
            ])
        self.lr_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(lr_img_size, interpolation=IMode.BICUBIC),
            transforms.ToTensor()
        ])
    def __getitem__(self, index) -> Tuple[Tensor, Tensor]:
        hr = Image.open(self.filenames[index])
        temp_lr = self.lr_transforms(hr)
        temp_hr = self.hr_transforms(hr)
```

数据集基类提供了随机裁剪、中心裁剪、随机旋转、水平翻转和调整大小等增强函数。最终，它将数据转换为 PyTorch 框架所需的张量。该类还包含`length`和`get item`函数。

在开发所需的所有重要函数之后，我们进入训练序列。训练序列用于训练生成器。代码如下：

```python
def train_generator(train_dataloader, epochs) -> None:
    ## 开始训练生成器
    ## 定义数据加载器
    ## 定义损失函数
    batch_count = len(train_dataloader)
    ## 开始训练生成器模块
    generator.train()
    for index, (lr, hr) in enumerate(train_dataloader):
        ## 将 hr 移至 cuda 或 cpu
        hr = hr.to(device)
        ## 将 lr 移至 cuda 或 cpu
        lr = lr.to(device)
        ## 将生成器梯度初始化为零，以避免梯度累积
        ## 仅在基于时间模型的情况下建议使用累积
        generator.zero_grad()
        sr = generator(lr)
        ## 定义像素损失
        pixel_losses = pixel_criterion(sr, hr)
        ## 从优化器中获取步进函数
        pixel_losses.backward()
        ## 生成器的 Adam 优化器
        p_optimizer.step()
        iteration = index + epochs * batch_count + 1
        writer.add_scalar(" 计算训练生成器损失", pixel_losses.item(), iteration)
```

类似地，对抗模块的训练如下所示。



```python
def train_adversarial(train_dataloader, epoch) -> None:
    ## 用于训练对抗网络
    batches = len(train_dataloader)
    ## 训练判别器和生成器
    discriminator.train()
    generator.train()
    for index, (lr, hr) in enumerate(train_dataloader):
        hr = hr.to(device)
        lr = lr.to(device)
        label_size = lr.size(0)
        fake_label = torch.full([label_size, 1], 0.0, dtype=lr.dtype, device=device)
        real_label = torch.full([label_size, 1], 1.0, dtype=lr.dtype, device=device)
        ## 初始化零梯度，因为我们希望避免梯度累积
        discriminator.zero_grad()
        output_dis = discriminator(hr)
        dis_loss_hr = adversarial_criterion(output_dis, real_label)
        dis_loss_hr.backward()
        dis_hr = output_dis.mean().item()
        sr = generator(lr)
        output_dis = discriminator(sr.detach())
        dis_loss_sr = adversarial_criterion(output_dis, fake_label)
        dis_loss_sr.backward()
        dis_sr1 = output_dis.mean().item()
        dis_loss = dis_loss_hr + dis_loss_sr
        d_optimizer.step()
        generator.zero_grad()
        output = discriminator(sr)
        pixel_loss = pixel_weight * pixel_criterion(sr, hr.detach())
        perceptual_loss = content_weight * content_criterion(sr, hr.detach())
        adversarial_loss = adversarial_weight * adversarial_criterion(output, real_label)
        gen_loss = pixel_loss + perceptual_loss + adversarial_loss
        gen_loss.backward()
        g_optimizer.step()
        dis_sr2 = output.mean().item()
        iteration = index + epoch * batches + 1
        writer.add_scalar("Train_Adversarial/D_Loss", dis_loss.item(), iteration)
        writer.add_scalar("Train_Adversarial/G_Loss", gen_loss.item(), iteration)
        writer.add_scalar("Train_Adversarial/D_HR", dis_hr, iteration)
        writer.add_scalar("Train_Adversarial/D_SR1", dis_sr1, iteration)
        writer.add_scalar("Train_Adversarial/D_SR2", dis_sr2, iteration)
```

最终，我们将处理一个验证模块，将生成器和对抗网络整合在一起。

以下代码将所有内容整合到主函数中，并运行整个训练序列：

```python
def main() -> None:
    ## 创建目录
    ## 构建训练和验证数据集路径
    ## 检查训练条件
    ## 检查是否可恢复训练
    if not os.path.exists(exp_dir1):
        os.makedirs(exp_dir1)
    if not os.path.exists(exp_dir2):
        os.makedirs(exp_dir2)
    train_dataset = BaseDataset(train_dir, image_size, upscale_factor, "train")
    train_dataloader = DataLoader(train_dataset, batch_size, True, pin_memory=True)
    valid_dataset = BaseDataset(valid_dir, image_size, upscale_factor, "valid")
    valid_dataloader = DataLoader(valid_dataset, batch_size, False, pin_memory=True)
    if resume:
        ## 用于恢复训练
        if resume_p_weight != "":
            generator.load_state_dict(torch.load(resume_p_weight))
        else:
            discriminator.load_state_dict(torch.load(resume_d_weight))
            generator.load_state_dict(torch.load(resume_g_weight))
    best_psnr_val = 0.0
    for epoch in range(start_p_epoch, p_epochs):
        train_generator(train_dataloader, epoch)
        psnr_val = validate(valid_dataloader, epoch, "generator")
        best_condition = psnr_val > best_psnr_val
        best_psnr_val = max(psnr_val, best_psnr_val)
        torch.save(generator.state_dict(), os.path.join(exp_dir1, f"p_epoch{epoch + 1}.pth"))
        if best_condition:
            torch.save(generator.state_dict(), os.path.join(exp_dir2, "p-best.pth"))
        ## 保存最佳模型
        torch.save(generator.state_dict(), os.path.join(exp_dir2, "p-last.pth"))
    best_psnr_val = 0.0
    generator.load_state_dict(torch.load(os.path.join(exp_dir2, "p-best.pth")))
    for epoch in range(start_epoch, epochs):
        train_adversarial(train_dataloader, epoch)
        psnr_val = validate(valid_dataloader, epoch, "adversarial")
        best_condition = psnr_val > best_psnr_val
        best_psnr_val = max(psnr_val, best_psnr_val)
        torch.save(discriminator.state_dict(), os.path.join(exp_dir1, f"d_epoch{epoch + 1}.pth"))
        torch.save(generator.state_dict(), os.path.join(exp_dir1, f"g_epoch{epoch + 1}.pth"))
        if best_condition:
            torch.save(discriminator.state_dict(), os.path.join(exp_dir2, "d-best.pth"))
            torch.save(generator.state_dict(), os.path.join(exp_dir2, "g-best.pth"))
        d_scheduler.step()
        g_scheduler.step()
    torch.save(discriminator.state_dict(), os.path.join(exp_dir2, "d-last.pth"))
    torch.save(generator.state_dict(), os.path.join(exp_dir2, "g-last.pth"))
```

至此，我们完成了代码部分，接下来可以了解如何运行它。代码块应如图 8-7 所示。完成后，我们将进入下一节，介绍如何运行应用程序。

![](img/520381_1_En_8_Fig10_HTML.jpg)

代码开发模板示意图。主要部分包括 `assets`、`data`、`results`、`scripts`、`.gitignore`、`config.py`、`dataset.py`、`image_proc_util.py`、`LICENSE`、`model_srgan.py`、`README.md`、`requirements.txt`、`setup.py`、`train.py` 和 `validate.py`。其中 `train.py` 被高亮显示。

**图 8-7** 代码开发模板

## 运行应用程序

要运行应用程序，我们首先需要将数据集下载到相应目录，或通过配置脚本将数据目录映射到训练函数。配置脚本非常重要，因为它将所有脚本和路径绑定在一起，帮助应用程序理解所需内容。

要下载数据，我们可以使用 `bash` 运行下载脚本。

```bash
! bash ./data/download_dataset.sh
```

安装完成后，我们只需运行训练脚本。

```bash
! python train.py
```

生成器训练完成后，对抗训练将自动开始。我们可以快速查看训练轮次的大致情况。



```text
Train Epoch0016/0020 Loss: 0.008974.
Train Epoch0016/0020 Loss: 0.009684.
Train Epoch0016/0020 Loss: 0.004455.
Train Epoch0016/0020 Loss: 0.008851.
Train Epoch0016/0020 Loss: 0.008883.
Valid stage: generator Epoch[0016] avg PSNR: 21.19.
Train Epoch0017/0020 Loss: 0.005397.
Train Epoch0017/0020 Loss: 0.006351.
Train Epoch0017/0020 Loss: 0.007704.
Train Epoch0017/0020 Loss: 0.007926.
Train Epoch0017/0020 Loss: 0.005559.
Valid stage: generator Epoch[0017] avg PSNR: 21.37.
Train Epoch0018/0020 Loss: 0.006054.
Train Epoch0018/0020 Loss: 0.008028.
Train Epoch0018/0020 Loss: 0.006164.
Train Epoch0018/0020 Loss: 0.006737.
Train Epoch0018/0020 Loss: 0.007716.
Valid stage: generator Epoch[0018] avg PSNR: 21.36.
Train Epoch0019/0020 Loss: 0.009527.
Train Epoch0019/0020 Loss: 0.004672.
Train Epoch0019/0020 Loss: 0.004574.
Train Epoch0019/0020 Loss: 0.005196.
Train Epoch0019/0020 Loss: 0.007712.
Valid stage: generator Epoch[0019] avg PSNR: 21.64.
Train Epoch0020/0020 Loss: 0.006843.
Train Epoch0020/0020 Loss: 0.007701.
Train Epoch0020/0020 Loss: 0.005366.
Train Epoch0020/0020 Loss: 0.004797.
Train Epoch0020/0020 Loss: 0.008607.
Valid stage: generator Epoch[0020] avg PSNR: 21.53.
Train stage: adversarial Epoch0001/0005 D Loss: 0.051520 G Loss: 0.574723 D(HR): 0.970196 D(SR1)/D(SR2): 0.019971/0.003046.
Train stage: adversarial Epoch0001/0005 D Loss: 0.001356 G Loss: 0.528222 D(HR): 0.998656 D(SR1)/D(SR2): 0.000007/0.000005.
Train stage: adversarial Epoch0001/0005 D Loss: 0.004768 G Loss: 0.574079 D(HR): 0.999959 D(SR1)/D(SR2): 0.004646/0.000619.
Train stage: adversarial Epoch0001/0005 D Loss: 0.000339 G Loss: 0.557449 D(HR): 0.999820 D(SR1)/D(SR2): 0.000159/0.000527.
Train stage: adversarial Epoch0001/0005 D Loss: 0.009615 G Loss: 0.531170 D(HR): 0.990858 D(SR1)/D(SR2): 0.000000/0.000000.
Valid stage: adversarial Epoch[0001] avg PSNR: 11.47.
Train stage: adversarial Epoch0002/0005 D Loss: 0.000002 G Loss: 0.488294 D(HR): 0.999998 D(SR1)/D(SR2): 0.000000/0.000000.
Train stage: adversarial Epoch0002/0005 D Loss: 0.114398 G Loss: 0.568630 D(HR): 0.947419 D(SR1)/D(SR2): 0.000000/0.000000.
Train stage: adversarial Epoch0002/0005 D Loss: 3.704494 G Loss: 0.580344 D(HR): 0.230086 D(SR1)/D(SR2): 0.000000/0.000000.
Train stage: adversarial Epoch0002/0005 D Loss: 0.000804 G Loss: 0.557581 D(HR): 0.999662 D(SR1)/D(SR2): 0.000464/0.000324.
Train stage: adversarial Epoch0002/0005 D Loss: 0.001132 G Loss: 0.459117 D(HR): 0.999191 D(SR1)/D(SR2): 0.000317/0.000301.
Valid stage: adversarial Epoch[0002] avg PSNR: 12.48.
Train stage: adversarial Epoch0003/0005 D Loss: 0.000187 G Loss: 0.488436 D(HR): 0.999847 D(SR1)/D(SR2): 0.000033/0.000032.
Train stage: adversarial Epoch0003/0005 D Loss: 0.001537 G Loss: 0.444651 D(HR): 0.999899 D(SR1)/D(SR2): 0.001425/0.001385.
Train stage: adversarial Epoch0003/0005 D Loss: 0.000169 G Loss: 0.493448 D(HR): 0.999877 D(SR1)/D(SR2): 0.000046/0.000041.
Train stage: adversarial Epoch0003/0005 D Loss: 0.000285 G Loss: 0.465992 D(HR): 0.999925 D(SR1)/D(SR2): 0.000210/0.000202.
Train stage: adversarial Epoch0003/0005 D Loss: 0.000720 G Loss: 0.567912 D(HR): 0.999978 D(SR1)/D(SR2): 0.000695/0.000668.
Valid stage: adversarial Epoch[0003] avg PSNR: 13.09.
Train stage: adversarial Epoch0004/0005 D Loss: 0.000293 G Loss: 0.479247 D(HR): 0.999786 D(SR1)/D(SR2): 0.000079/0.000076.
Train stage: adversarial Epoch0004/0005 D Loss: 0.000064 G Loss: 0.492225 D(HR): 0.999978 D(SR1)/D(SR2): 0.000042/0.000041.
Train stage: adversarial Epoch0004/0005 D Loss: 0.000030 G Loss: 0.444387 D(HR): 0.999984 D(SR1)/D(SR2): 0.000014/0.000014.
Train stage: adversarial Epoch0004/0005 D Loss: 0.000108 G Loss: 0.387137 D(HR): 0.999918 D(SR1)/D(SR2): 0.000025/0.000025.
Train stage: adversarial Epoch0004/0005 D Loss: 0.000224 G Loss: 0.513328 D(HR): 0.999825 D(SR1)/D(SR2): 0.000049/0.000048.
Valid stage: adversarial Epoch[0004] avg PSNR: 13.29.
```

在这个训练集上，我们使用了可配置的轮次和其他训练参数，这些都可以在配置文件中找到。一旦模型准备好下载，我们就可以用它来将图像放大四倍。我们可以在训练过程中配置放大倍数。至此，我们的训练过程就结束了。

## 总结

本章从图像放大的相关问题入手，讨论了如何进行放大。我们探讨了各种方法的优势以及当前可用的建模技术。讨论并实现了诸如 SRGAN 等最先进的算法。我们还经历了训练过程并搭建了项目。本章讨论了如何结合生成模型，使用卷积模型将图像放大一定倍数。超分辨率是一个不断发展的领域，应用广泛，例如从交通摄像头中检测车牌或增强老照片。这是计算机视觉中一个非常重要的领域，并且拥有多年的研究价值。

在下一章中，我们将从静态图像的概念转向动态图像，也就是视频。

