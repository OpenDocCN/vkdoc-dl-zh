# 3. 构建目标检测模型

目标检测是当下最受欢迎的技能之一。一张图像可能包含多个类别。此外，对物体进行分类只解决了问题的一部分，另一部分在于物体的定位。目标检测有助于通过边界框识别图像中类别的具体位置。边界框可以进一步处理以完成各种子任务。例如，想象一下交通摄像头需要检测并识别车辆。

交通摄像头需要检测车辆和车牌，然后读取车牌上的号码以识别车主。这不是一个简单的问题。我们需要带标注的注册数据。一个简单的分类卷积神经网络模型无法胜任。我们需要获取车牌的边界框，并通过一系列数据清洗、去噪和超分辨率步骤来搜索字母数字字符。

近年来，目标检测领域取得了巨大进步。在众多目标检测方法中，我们可以将其历程划分为 2012 年前（或 AlexNet 前）时代和 2012 年后时代。2012 年前时代包括多种目标检测算法，如 HOG、Haar 级联、SIFT 的某些变体、SURF 等。2012 年后时代包括 RCNN、Fast RCNN、Faster RCNN、YOLO、单次检测器（SSD）等。

我们将简要回顾 2012 年前时代的 Haar 级联，以建立基于机器学习的目标检测技术的背景。要开始使用 Haar 级联，我们需要使用图像中的特征，而不是更细粒度的像素。Haar 级联于 2001 年提出，尽管年代久远，但它仍然是目前最快的算法之一。

## 使用提升级联进行目标检测

提升级联最初是为检测人脸而构建的，但它也可用于其他目标检测任务。它包含三个部分：积分图像、用于选择特征的提升算法以及级联分类器。

首先，输入图像需要转换为所谓的积分图像。积分图像可以通过简单的计算得到。

![](img/520381_1_En_3_Fig2_HTML.jpg)

一个水平矩形的示意图。内部左上角是一个被分成 4 个部分的矩形，按顺时针方向标记为 A、B、C 和 D。线条分割的交点标记为 1、2、3 和 4。

图 3-2 中间步骤

![](img/520381_1_En_3_Fig1_HTML.jpg)

四边形状的示意图，分为标记的边缘特征、标记的线特征和标记的四矩形特征。在 a 中，有两个矩形，一个被等分为两个水平部分，另一个被等分为两个垂直部分；在 b 中，一个被等分为三个水平部分，另一个被等分为三个垂直部分；在 c 中，一个正方形被等分为四个相等的部分。

图 3-1 特征提取器示例

图 3-1 中的图像展示了三种主要类型的特征提取器。第一种是边缘提取器，其次是线和矩形特征提取器。使用这些提取器，需要选择特征，而提升算法有助于选择必要的特征。自适应提升算法提供了一组重要特征，有助于更快地进行人脸识别。

积分图像是进行特征提取的中间步骤；它通过计算像素值计算点上方和左侧所有像素的总和来实现，如图 3-2 所示。

积分图像的计算如下：

*   位置 1 = A 矩形内的像素总和（考虑左侧和上方）

*   位置 2 = 像素总和 A + B

*   位置 3 = 像素总和 A（上方）+ C（左侧）

*   位置 4 = 像素总和 (4 + 1) – 总和 (2 + 3)

提取的特征与正样本和负样本进行比对，最终选出最佳特征。用于正负图像集的训练分类器由较弱的分类器构成。对于人脸检测，从多达 16 万个特征中，一系列弱分类器的提升算法有助于识别出 6000 个有用的特征。最终，级联分类器帮助检测类别。

所谓的*注意力级联*有助于减少计算时间并提高检测器的效率。图像被分割成多个子窗口，顺序的弱分类器对这些子窗口进行处理。每个分类器使用选定的特征，尝试检查目标是否存在。如果在任何一点分类器失败，所有后续分类器都会停止，序列移动到下一个子窗口，依此类推。如果所有分类器都能对所需目标的存在进行投票并得到边界框，则检测成功。

让我们通过一系列 Python 代码来使用现有模型检测人脸和眼睛。

按如下方式导入包：

```python
import cv2
import gc
```

以下函数将从摄像头获取输入帧，并将其缩放以适应模型。由于彩色图像不会产生差异，因此考虑使用灰度图像。首先检测人脸，然后针对每张人脸，借助另一个眼睛检测器定位眼睛。

以下是处理人脸和眼睛级联的函数：

```python
def detect_face_eye(frame):

    ## 归一化并将颜色转换为灰度
    frame_to_gray = cv2.equalizeHist(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))

    ## 应用程序应能处理不同尺度的图像
    detected_faces = face_cascade.detectMultiScale(frame_to_gray)
    for (x,y,w,h) in detected_faces:
        center_face = (x + w//2, y + h//2)

        ## 绘制椭圆
        frame = cv2.ellipse(frame, center_face, (w//2, h//2), 0, 0, 360, (125, 125, 125), 6)
        face_regionofinterest = frame_to_gray[y:y+h,x:x+w]
        #检测眼睛 - 针对每张检测到的人脸

        ## 类似的多尺度操作
        detected_eyes = eyes_cascade.detectMultiScale(face_regionofinterest)
        for (x2,y2,w2,h2) in detected_eyes:
            center_eye = (x + x2 + w2//2, y + y2 + h2//2)
            radius = int(round((w2 + h2)*0.25))

            ## 绘制圆形
            frame = cv2.circle(frame, center_eye, radius, (255, 255, 255 ), 4)
    cv2.imshow('--人脸检测--', frame)
```

这些模型可以在`open-cv`维护者提供的 GitHub 仓库中找到，地址为[`https://github.com/opencv/opencv/tree/master/data/haarcascades`](https://github.com/opencv/opencv/tree/master/data/haarcascades)。这段代码使用了两个模型，一个用于检测人脸，另一个用于检测眼睛。不过，仓库中还有其他模型，你可以进行尝试。该函数还将访问连接到系统的摄像头外设，并使用它们扫描人脸。

运行该函数以启用人脸和眼睛感知过程：

```python
## 保存的 xml 路径
face_cascade_name = r' ..\chapter 3\frontal_face_alt.xml'
eyes_cascade_name = r' ..\chapter 3\eye_cascade_model.xml'

## 初始化用于检测的级联
face_cascade = cv2.CascadeClassifier()
eyes_cascade = cv2.CascadeClassifier()

## 加载级联，先加载人脸，后加载眼睛
face_cascade.load(cv2.samples.findFile(face_cascade_name))
eyes_cascade.load(cv2.samples.findFile(eyes_cascade_name))
camera_device = 0

## 启用视频处理
capture_cam_img = cv2.VideoCapture(camera_device)

## 启用分类器对人脸进行操作
if capture_cam_img.isOpened :
    while True:
        ret, frame = capture_cam_img.read()
        detect_face_eye(frame)

        ## 按下 ESC 键关闭 CV 视频感知
        if cv2.waitKey(10) == 27:
            cv2.destroyAllWindows()
            gc.collect()
            break
```

现在我们对结构化目标检测有了一些了解，可以转向另一种先进的目标检测技术，称为 R-CNN。

# R-CNN

长期以来，物体都是通过图像分割来分离的。最终，图像的层次性给开发者带来了瓶颈。试想一下，如果我们试图在车流中定位一辆汽车里的人。可以使用穷举搜索机制来扫描每辆车，以精确找到人的位置，但所需的计算量过高，难以实用。

面对这些问题，一篇题为《用于目标识别的选择性搜索》的论文试图解决生成目标位置的问题。它结合了分割和穷举搜索这两者的优点。以下步骤展示了选择性搜索机制的工作原理。

该算法的工作原理如下：

1.  该算法使用高效的基于图的分割来生成初始区域。

2.  第二阶段尝试对相似区域进行分组，以在输入图像中生成片段。对于所有已创建的区域，会计算所有相邻元素之间的相似度得分。两个最相似的区域被合并在一起，并重新计算得分。此过程重复进行，直到整个图像都被这些操作覆盖。

3.  选择过程是复杂的。它使用多种策略将相似区域聚合在一起。如果两个区域被合并，这些区域的特征可以通过层级结构进行传播。

4.  选择标准依赖于互补的颜色空间，它在多种空间中采用层次分组算法，包括寻找互补的颜色空间。总体而言，四种快速高效的策略为该算法提供了支持。

该算法的主要目标是在其策略下，找到多样且互补的特征来对区域进行分组。

目标检测领域的先驱是`HOG`（方向梯度直方图）和`SIFT`。当视觉任务的复杂性被认识到后，一种不同的方法被开发出来。

## 区域提议网络

图 3-3 描述了区域提议网络中目标检测的各个步骤。让我们来看一下重要的步骤：

1.  生成分割和多个候选区域。

2.  使用贪婪学习算法递归地将相似区域合并成更大的区域。

3.  将带有提议的图像发送到用于分类目标的卷积神经网络架构中。

4.  在标准 R-CNN 中使用的`AlexNET`的情况下，使用`227 x 227`作为图像的形状。

5.  大约 2000 个区域被发送到`AlexNET`，并传递 4096 个向量。

6.  提取的特征会与针对特定类别训练的`SVM`进行评估。

7.  在所有区域被评分后，对分类后的区域执行非极大值抑制。它会消除`IOU`低于阈值的区域，为覆盖范围更大且高于阈值的区域铺平道路。

![](img/520381_1_En_3_Fig3_HTML.jpg)

左侧是一个标记为“输入图像”的方框示意图，右侧从上到下依次是：一个包含两个方框（对象 1、对象 2）和一个未标记方框的矩形，其下方是“区域提议”，一个标记为“输入图像”的方框，一个内部包含连接方框的矩形（CNN 特征），并通过箭头连接到两个方框（对象 1 和对象 2）。

图 3-3

通过区域提议进行目标检测

图

有趣的是，当算法定位到 2000 个感兴趣区域时，它会从提议的区域生成扭曲的图像内容。由于卷积块需要固定的尺寸，信息会在空间上被扭曲并传递。

然后，这些区域中的每一个都由支持向量机进行分类。此外，该算法将执行回归，以修正或预测最初预测的边界框的任何偏移。在进入下一步之前，让我们回顾两个重要的概念，它们将在后续内容中多次使用。

-   **非极大值抑制。** 在目标检测算法中，经常会出现多个边界框重叠在单个对象周围的情况。分类器通常需要为不同大小的感兴趣区域生成概率分数。为了解决选择一个最佳边界框的问题，该算法使用分类信息和对象上的覆盖百分比。

-   **交并比（IoU）。** 用于选择与真实值最相似的边界框。在处理图像分类时，我们试图将图像映射到它们各自的类别。同样，对于目标检测，需要手动绘制边界框来定位不同的对象和类别。该公式给出了交集与并集的比率。

`IoU`的公式如下：

```
IoU = (边界框 1 ∩ 边界框 2) / (边界框 1 ∪ 边界框 2)
```

图 3-4a 显示了一个包含两个重叠边界框的图像，一个是真实框，另一个是预测框。图 3-4b 显示了由两者聚合的区域。这两个方面相互平衡，以获得真实值的最大覆盖范围。

![](img/520381_1_En_3_Fig5_HTML.jpg)

一个水平方框的示意图。内部有两个不同颜色的水平矩形。两者都被着色，并且它们的两个角连接在一起形成一个正方形。形成的正方形也被着色。

图 3-4b

边界框并集

![](img/520381_1_En_3_Fig4_HTML.jpg)

一个水平方框的示意图。内部有两个不同颜色的水平矩形。它们的两个角连接在一起形成一个正方形。形成的正方形被着色。

图 3-4a

边界框交集

总的来说，该算法能够处理许多与目标检测相关的问题，并且在发布时是最具革命性的算法之一。但它并非没有缺陷。现在让我们深入探讨它的一些明显缺陷：

-   借助复杂的图像处理技术，模型将生成 2000 个感兴趣区域。所有这些区域都需要由支持向量机进行分类。这个过程涉及巨大的计算量。

-   大多数算法在预测时，分类和处理图像需要花费大量时间。如果我们要处理实时解决方案，几乎不可能将此模型用作算法。

-   训练发生在卷积部分；分类器和回归正在修正边界框参数。

-   在算法的初始部分，使用选择性搜索机制来分割相似区域并共同生成感兴趣区域。整个过程基于复杂图像处理技术的搭配。该过程中不涉及学习，因此改进的空间很小。

尽管 R-CNN 解决了目标检测中的诸多问题，但它也留下了一系列需要解决的问题。随后出现了一种基于区域的目标检测网络的改进版本，称为快速区域卷积网络。

### 快速区域卷积神经网络

为了设定背景，假设有一张需要执行目标检测的图像，根据简单的区域卷积神经网络，它会在简单图像的基础上生成感兴趣区域。组合数量会非常庞大。但如果我们能将图像在（`x`，`y`）维度上缩小到更小的尺寸，我们仍然能获取到包含正确目标信息的图像部分。最终，这归结为我们如何将信息传递给后续层以及损失函数。快速 R-CNN 实现了更快的运算。

图 3-5 描绘了基于一个感兴趣区域的工作流程。该架构表明，对输入数据执行了卷积操作，从而减少了计算量。

![](img/520381_1_En_3_Fig6_HTML.jpg)

一张示意图，从左到右依次为：一个名为“输入图像”的方框，两个分别名为“CNN”和“ROI”的右向箭头，它们下方是“投影”，右侧是一个名为“卷积特征图”的直立长方体，一个向下的箭头，一个名为“ROI 池化层”的直立长方体，一系列名为“全连接层”的条形，以及两个名为“SoftMax”和“B BOX 回归器”的条形。

图 3-5

快速 R-CNN 架构

基于区域的快速目标检测所涉及的过程如下：

1.  通过多次卷积和池化操作创建特征图。

2.  由于全连接网络需要固定维度的向量，感兴趣区域的池化层会提取一个固定长度的向量。

3.  这些特征向量中的每一个都被输入到全连接网络中，该网络再次连接到输出层。

4.  第一个连接层包含 softmax 概率估计的计算，涵盖 *n* 个目标类别以及一个额外的背景或未知类别。

5.  第二个输出层为每个目标类别预测四个实数。每组数值定义了该类别修正后的边界框值。

我们介绍的每一种架构都使用选择性搜索算法来寻找感兴趣区域。这存在两个问题。首先，复杂的计算机视觉过程无法学习数据中的任何变化，因为它有一套关于如何识别区域的固定指令集。其次，选择性搜索是一个缓慢且耗时的过程。这些问题在算法的升级版本（称为更快的 R-CNN）中得到了解决。

### 区域提议网络的工作原理

随后出现了一个扩展思路，该思路不仅被提出，而且得到了实现。它使用神经网络来预测区域提议，而无需选择性搜索机制。区域提议网络的出现有助于识别图像中的边界框，然后将相同的块发送给卷积神经网络以生成特征图。

最终，损失函数在特征图上进行训练，并调整网络权重以适应训练。让我们逐步了解这个过程：

1.  第一步，将输入图像传递到卷积块以生成卷积特征图。

2.  区域提议网络在特征图的每个位置使用一个滑动窗口。

3.  对于每个位置，使用九个锚点框，它们具有三种不同的尺度和三种宽高比（1:1、1:2、2:1），这有助于生成区域提议。

4.  分类层输出锚点框中是否存在目标。

5.  回归层指示锚点框的坐标。

6.  锚点框被传递到快速 R-CNN 架构的感兴趣区域池化层。

我们使用神经网络来学习区域提议的位置以及如何根据数据进行调整。这也使得该过程比我们之前了解的方法快得多。图 3-6 展示了来自更快的 R-CNN 原始研究论文的架构。

![](img/520381_1_En_3_Fig7_HTML.jpg)

一张示意图，从下到上依次为：一张部分被覆盖的照片，名为“图像”，其上方是一个名为“卷积层”的长方体，一个向上的箭头，一个名为“特征图”的平行四边形，区域提议网络，一个向上的箭头，一个名为“提议”的平行四边形，向上的箭头，ROI 池化，一个名为“分类器”的平行四边形，一个向上的箭头。

图 3-6

更快的 R-CNN 总结

该网络有一个新颖的想法，即区域提议网络，它可以学习边界框并对其进行泛化。它主要有三种类型的网络：

*   **头部：** 可以是 ResNet 架构，用于生成特征图。

*   **区域提议网络：** 为分类器和回归器生成感兴趣区域。

*   **分类网络/回归网络：** 处理目标分类和目标性，或边界框坐标的正确性。

图 3-7 描绘了更快的 R-CNN 的基本层。让我们深入了解细节，这将有助于层的开发。

![](img/520381_1_En_3_Fig8_HTML.jpg)

一个流程图。流程如下：锚点生成层到区域提议层到 ROI 池化层，一个向下的箭头，以及分类层。

图 3-7

更快的 R-CNN 流程图

### 锚点生成层

该层生成一系列具有不同尺寸和宽高比的边界框，以覆盖大部分图像区域。这些边界框或锚点框将包含图像及其目标。然而，这些框将与内容无关且始终保持一致，最终区域提议网络将对它们进行处理，并识别出哪个是更好的边界框。微小的调整将带来更好的边界框。

由于预测这些坐标存在一些问题，另一种方法是将一个参考框作为边界框的标准。以一个参考框（`X[中心]`、`Y[中心]`、宽度和高度）作为基准，然后尝试预测并修正偏移值，使其更贴合。偏移值针对所有四个参数。

### 区域提议层

区域提议网络负责改变锚点框的位置、宽度和高度，以更好地贴合目标。该层可以被视为区域提议网络、提议层、锚点目标层和提议目标层的组合。

*   **区域提议网络：** 该层使用特征图并将其输入到卷积神经网络。然后将输出传递到两个 1x1 卷积层，以生成对应于边界框的回归系数、类别分数和概率。

# Mask R-CNN

在 Faster R-CNN 已有成果的基础上，Mask R-CNN 进一步扩展，能够预测检测对象上的掩码。在 ROI 池化层之后，又增加了两个卷积神经网络来生成掩码。该方法还确立了 `ROI Align`，有助于更好地将提取的特征与输入对齐，并避免 Faster R-CNN 中曾出现的变形问题。它使用双线性插值来获取输入区域的精确或近乎完美的值。

所有这些目标检测方法中产生的一个重要步骤是锚点框的使用。`YOLO` 在此基础上进行了一些额外的改进，我们将在下文探讨。

## 前提条件

![](img/520381_1_En_3_Fig9_HTML.jpg)

一张两只鸟在飞行的照片。两只鸟都被一个框标记出来。

图 3-8：带有标注的鸟类图像

- **标注。** 在分类问题中，图像需要按照其所属类别进行排序或整理。类似地，在目标检测问题中，图像需要用适当的边界框（通常称为真实标注）进行标记。边界框的作用是提供对象坐标以及所包含类别的信息。图 3-8 展示了一个已标注图像的实例，用于训练一个定位鸟类的分类器。通常，标注是手动完成的，有时会进行多次，以获得无偏的真实标注。然而，如果像现在这样只是为了练习，我们可以使用任何开源数据进行实验。

- **首选 GPU。** 在执行需要训练和运行推理的计算机视觉任务时，建议使用支持 CUDA 的 GPU 核心以加快处理速度。

- **安装支持 CUDA 的 Torch 框架。** 我们还需要在系统上安装 `PyTorch`，如前一章所述。

## YOLO

对于能够帮助实现实时推理的目标检测算法，一直存在巨大的需求。Faster R-CNN 已经非常接近这一目标，它处理了 2000 个边界框预测并超越了传统的计算机视觉方法。与之前的算法相比，它有了显著的改进，但仍有提升空间。

随后出现了革命性的算法 `YOLO`，它能够以每秒 45 帧的速度（在 TITAN X 上）检测对象。早期的模型在训练和预测的不同阶段（如锚点生成层、区域提议层、分类和边界框修正）花费了太多时间。而 `YOLO` 则试图通过一个卷积神经网络块来同时预测边界框和类别，从而减少计算时间。它采用了一种更通用的训练方式，并从整个图像中考虑信息，而不是将信息拼凑起来。最终，它超越了其他试图实现相同目标的前代算法。

图 3-9 展示了 `YOLO` 的架构，该架构受用于图像分类的 `GoogleNet` 架构启发。输入层的维度为 448x448x3。该网络包含 24 个卷积层和批量最大池化层，以及两个全连接层。

![](img/520381_1_En_3_Fig10_HTML.jpg)

一张示意图，从左到右依次为：一个内部有 7x7 立方体的层，一系列尺寸分别为 112x112、56x56、28x28、14x14、7x7 和 7x7 的 6 个层，2 个箭头，竖线，箭头，以及一个尺寸为 7x7 的立方体。方框下方是对应的卷积层维度。只有前 4 个层有最大池化层。

图 3-9：YOLO 架构

训练过程相当昂贵，因此从头开始训练一个目标检测模型需要良好的管理。该给定架构的训练通过两种方式进行。首先，模型在 `ImageNet` 数据上进行训练，使用前 20 个卷积层，并通过平均池化来匹配全连接网络的维度。这个模块训练了一周，达到了 88% 的准确率。

在这个预训练网络的基础上，添加了四个卷积层和两个全连接层，以得到最终的检测对象。输入维度也从 224x224 增加到 448x448，这有助于提升检测能力。最后一层预测分类分数和边界框坐标。边界框的宽度和高度被归一化。

图 3-10 展示了用于优化分类和回归的损失函数。对于五个锚点框中的每一个，都有一个目标性分数、四个对应于归一化边界框的坐标，以及最高的类别概率或分数。这些改进效果不错，但还需要进一步优化。让我们看看第二个版本的更新以及版本 3，它曾是最流行的模型之一。

![](img/520381_1_En_3_Fig11_HTML.jpg)

一个方程图示：lambda 下标坐标，S 上标 2 求和，B 求和，i 等于 0，j 等于 0，1 对象上标 i j，括号，x 下标 i 减去 x 下标 i，上标 2，加 y 下标 i，减去 y 下标 2，上标 2，括号，以及后续方程。

图 3-10：YOLO 损失函数

## YOLO V2/V3

YOLO 的改进非常显著，第二个版本将方法微调到了更高效的层面。以下是第二版中解决的一些关键点：

-   卷积层深度较大，因此始终存在梯度消失或梯度爆炸的风险。添加了批归一化以帮助解决学习过程中的内部协变量偏移。

-   它为每个锚框预测类别和物体置信度。

-   网络还预测五个边界框以及每个边界框的五个坐标。

-   一个重大的架构变化是移除了全连接层，并用锚框来预测边界框。

-   这些锚框是通过对真实边界框进行聚类来确定的。

即使在多次修改之后，研究人员发现还可以进行一些更改来提高准确率。他们进行了必要的修改，并将此版本命名为 YOLO V3。它可以说是最受欢迎的目标检测架构之一。YOLO 使用 `softmax` 层来获取最终的分类分数，而 YOLO V3 则采用对输入进行独立的逻辑回归或多标签分类。有趣的是，它还移除了池化层，转而使用步长为 `2` 的 `3x3` 卷积来降低维度。

该架构还对损失函数进行了修改，产生了三个主要的预测输出——边界框的坐标、物体置信度值和类别分数。YOLO V3 架构中最流行的骨干网络是 `Darnet-53`，这是一个由卷积块组成的 53 层架构，如图 3-11 所示。它使用带有 `3x3` 和 `1x1` 卷积层的残差实现来获取用于检测和分类的特征。总体而言，这些更改对架构的准确率和优化产生了巨大影响。

![](img/520381_1_En_3_Fig12_HTML.jpg)

一个表格有 4 个表头：类型、过滤器、尺寸和输出。行包含框起来的条目，并标记为 1 X、2 X、8 X、8 X 和 4 X。框起来的区域包括 2 种卷积类型的过滤器和尺寸的条目，以及一个残差输出尺寸。类型下的最后 3 个条目是平均池化、全连接和 Soft Max。

**图 3-11** Darknet 53 架构

让我们看一些使用已保存模型并针对自定义数据集进行微调的代码。为什么我们不从头开始训练呢？这些都是重量级模型，我们并不总是有足够的 GPU 能力从头开始训练。其次，使用预训练权重并相应地修改它们是一种学习体验。我们将反复提到的一个术语是*迁移学习*。

## 项目代码片段

该代码片段改编自 YOLO 的原始创建者，所有源代码归功于 Joseph Redmon 和 Ali Farhadi。尽管从头开始训练相当复杂，但我们可以尝试使用现有的开源模型对这些数据进行迁移学习。如果原始模型训练的类别与我们使用的类别非常相似，我们也可以使用现有模型对我们的数据进行推理。

文件夹设置需要遵循原始创建者的方式，因为我们将使用保存的模型来自定义训练我们的数据。如图 3-12 所示，对于任何变化，路径应根据配置文件在 `data` 目录下进行修正。

![](img/520381_1_En_3_Fig13_HTML.jpg)

代码显示为：倒 V，Yolo V r，小于 下划线 p y 缓存 下划线，小于 注释 工具，倒 V，c f g，设置图标，Yolo v3，自定义 点 c f g，设置图标，Yolo v3，s p p 点 c f g，设置图标，Yolo v3，s p p tiny 点 c f g，倒 V，data，小于 images。下方，小于 utils 被框起来。

**图 3-12** YOLO 的文件夹结构

### 步骤 1：获取标注数据

当我们想要训练自定义数据时，图像标注是目标检测算法最重要的先决条件之一。它们帮助模型处理分类和回归损失函数。它们包含了手动解析的真实值。有多个开源位置可以让我们标注图像。该工具通常有一个标记器，可以帮助在图像上绘制某种形状的边界框。该程序允许以 JSON、CSV 或 VOC/COCO 格式下载标注，具体取决于所使用的模型。训练数据和自定义数据应该保持一致。

标注正确且对标注者真实至关重要。由于这是一项手动且重复的任务，因此需要尽可能做到最好。最终，生成的文件应被下载并放置在 `data` 文件夹中。例如，每张图像可能看起来像这样：

- `0 0.41833333333333333 0.2112676056338028 0.2011111111111111 0.2007042253521127`

- `2 0.43777777777777777 0.3970070422535211 0.11555555555555555 0.15669014084507044`

- `1 0.38722222222222225 0.6813380281690141 0.47 0.4119718309859155`

一旦我们汇总了新文件，我们就可以看到如何更改数据文件。在图 3-12 中，`data` 下的文件夹主要是 `labels` 和 `images`。`images` 包含与标注图像同名的原始图像。文本文件需要包含标注信息并放置在 `labels` 中。这可以是文本文件或 JSON。

完成后，我们将检查文件的自定义数据文件，该文件需要更新信息，例如训练和测试文件信息的存储位置。我们需要在此处提供两种信息——标签和图像的路径以及实际图像。自定义数据文件将如下所示：

- `classes=4`

- `train=data/train.txt`

- `valid=data/test.txt`

- `names=data/custom.names`

这提供了关于数据及其位置的相关信息。完成后，我们需要在 `custom.names` 文件中提供类别名称。它将如下所示：

- `hardhat`

- `vest`

- `mask`

- `boots`

此文件将类别名称与前面链接的数字对应起来。如前所述，我们需要包含图像路径的 `train.txt` 和 `test.txt` 文件。这些文件应包含运行训练函数的相对路径。

还有其他文件，例如训练和测试形状（`train.shapes` 和 `test.shapes`），它们包含所有文件的形状，我们可以根据输入数据进行更改。

完成所有这些后，我们必须从来源和原始研究人员处下载保存的权重，网址为 [`https://pjreddie.com/darknet/yolo/`](https://pjreddie.com/darknet/yolo/)。根据项目工作者的 GPU 性能，有多种选项可供选择。权重和配置文件是相互关联的。因此，请务必下载与权重对应的配置文件。通过这些主要步骤，初始设置就完成了。现在我们进入下一个流程。

### 步骤 2：修复配置文件与训练

另一项重要任务是根据需求和资源修改配置文件。图 3-13 展示了训练和测试配置中的首批改动。其中提供了修改 `batch`（批次大小）、`width`（宽度）、`height`（高度）、`channels`（通道数）、`momentum`（动量）和 `decay`（衰减）等参数的设置项。

![](img/520381_1_En_3_Fig14_HTML.jpg)

表格包含两个表头：更新后的配置和原始配置。在第一列中，`batch` 和 `subdivisions` 之前的哈希键被高亮显示，`width` 和 `height` 之后的数字 `08` 也同样被高亮。在下一列中，左侧面板上的数字 `3`、`4`、`6`、`7`、`8`、`9`、`19`、`20` 和 `22` 被着色，以指示右侧的高亮数字。

**图 3-13** – 训练/测试配置文件的改动

诸如 `learning rates`（学习率）和 `burn-in`（预热）等重要参数也已提供。除了这些改动之外，还有关于 `classes`（类别数）和最终层的修改。由于我们将针对包含 80 个类别的原始训练方法进行自定义训练，因此具体情况可能会有所不同。图 3-14 展示了一些必要的改动。如果训练可以在默认的 `coco` 数据集上进行，则可以使用原始的配置文件。

![](img/520381_1_En_3_Fig15_HTML.jpg)

展示了训练流程的更新后配置和原始配置。

**图 3-14** – 训练/推理流程配置所需的改动

配置文件中所有 `classes` 和 `filters` 的实例都需要更改。我们需要将 YOLO 层之前的实例中的 `[filters=255]` 更改为 `filters=(类别数 + 5)x3`，如第 640 行所示。

完成这些更改后，我们就可以进入训练部分了。只需要运行一个任务。

```
!python train.py --data $PATH/custom.data --batch $num_batches --cache --epochs $num_epochs –nosave
```

其中：

- `$num_batches` = 批次数量
- `$num_epochs` = 训练的轮数（请记住这是迁移学习，并且我们已经在使用保存的权重）
- `$path` = 自定义数据的路径。

如果内存不足，我们可以尝试通过使用更小的已保存模型进行训练来减少模型参数，或者减小批次大小或图像分辨率。选择哪种方法，只要觉得简单且合适即可。

该项目依赖项过多，建议直接参考并使用优化后的代码版本，以节省时间。训练代码、模型代码和配置代码是相互关联的。配置文件直接影响训练过程和模型设置。让我们看看研究人员使用的源代码中用于模型定义的 Python 代码。

### 模型文件

代码中多次使用了 `torchvision` 和 `torch` 函数的标准导入。`parse` 包用于获取命令行参数。模型文件中出现的第一个函数是 `create_modules` 函数。让我们了解一些重要步骤，以防出现“从头开始训练”的情况。

# `create_modules` 函数与 `YOLOLayer` 类

该代码中的关键步骤如下：

1.  初始化一个顺序模型，该模型设置了模型块的上下文。

2.  模型从命令行接收参数，并获取与批归一化、滤波器、激活函数和卷积相关的变量。

3.  可以选择将模型保存为 ONNX 版本。

在初始模型定义之后，我们有了 `YOLOLayer` 类，它使用函数根据接收到的配置来定义模型。让我们看看源研究提供的代码。

```python
def create_modules(module_defs, img_size):

    # 根据 module_defs 中的模块配置构建层块的模块列表
    img_size = [img_size] * 2 if isinstance(img_size, int) else img_size  # 如有必要则展开
    _ = module_defs.pop(0)  # 配置训练超参数（未使用）
    output_filters = [3]  # 输入通道数
    module_list = nn.ModuleList()
    routs = []  # 路由到更深层的层列表
    yolo_index = -1
    for i, mdef in enumerate(module_defs):
        modules = nn.Sequential()
        if mdef['type'] == 'convolutional':
            bn = mdef['batch_normalize']
            filters = mdef['filters']
            k = mdef['size']  # 卷积核大小
            stride = mdef['stride'] if 'stride' in mdef else (mdef['stride_y'], mdef['stride_x'])
            if isinstance(k, int):  # 单尺寸卷积
                modules.add_module('Conv2d', nn.Conv2d(in_channels=output_filters[-1],
                                                       out_channels=filters,
                                                       kernel_size=k,
                                                       stride=stride,
                                                       padding=k // 2 if mdef['pad'] else 0,
                                                       groups=mdef['groups'] if 'groups' in mdef else 1,
                                                       bias=not bn))
            else:  # 多尺寸卷积
                modules.add_module('MixConv2d', MixConv2d(in_ch=output_filters[-1],
                                                          out_ch=filters,
                                                          k=k,
                                                          stride=stride,
                                                          bias=not bn))
            if bn:
                modules.add_module('BatchNorm2d', nn.BatchNorm2d(filters, momentum=0.03, eps=1E-4))
            else:
                routs.append(i)  # 检测输出（进入 YOLO 层）
            if mdef['activation'] == 'leaky':  # 激活函数研究 https://github.com/ultralytics/yolov3/issues/441
                modules.add_module('activation', nn.LeakyReLU(0.1, inplace=True))

            # modules.add_module('activation', nn.PReLU(num_parameters=1, init=0.10))
            elif mdef['activation'] == 'swish':
                modules.add_module('activation', Swish())
        elif mdef['type'] == 'BatchNorm2d':
            filters = output_filters[-1]
            modules = nn.BatchNorm2d(filters, momentum=0.03, eps=1E-4)
            if i == 0 and filters == 3:  # 标准化 RGB 图像

                # imagenet 均值和方差 https://pytorch.org/docs/stable/torchvision/models.html#classification
                modules.running_mean = torch.tensor([0.485, 0.456, 0.406])
                modules.running_var = torch.tensor([0.0524, 0.0502, 0.0506])
        elif mdef['type'] == 'maxpool':
            k = mdef['size']  # 卷积核大小
            stride = mdef['stride']
            maxpool = nn.MaxPool2d(kernel_size=k, stride=stride, padding=(k - 1) // 2)
            if k == 2 and stride == 1:  # yolov3-tiny
                modules.add_module('ZeroPad2d', nn.ZeroPad2d((0, 1, 0, 1)))
                modules.add_module('MaxPool2d', maxpool)
            else:
                modules = maxpool
        elif mdef['type'] == 'upsample':
            if ONNX_EXPORT:  # 明确指定大小，避免使用 scale_factor
                g = (yolo_index + 1) * 2 / 32  # 增益
                modules = nn.Upsample(size=tuple(int(x * g) for x in img_size))  # img_size = (320, 192)
            else:
                modules = nn.Upsample(scale_factor=mdef['stride'])
        elif mdef['type'] == 'route':  # 'route' 层的 nn.Sequential() 占位符
            layers = mdef['layers']
            filters = sum([output_filters[l + 1 if l > 0 else l] for l in layers])
            routs.extend([i + l if l < 0 else l for l in layers])
            modules = FeatureConcat(layers=layers)
        elif mdef['type'] == 'shortcut':  # 'shortcut' 层的 nn.Sequential() 占位符
            layers = mdef['from']
            filters = output_filters[-1]
            routs.extend([i + l if l < 0 else l for l in layers])
            modules = WeightedFeatureFusion(layers=layers, weight='weights_type' in mdef)
        elif mdef['type'] == 'reorg3d':  # yolov3-spp-pan-scale
            pass
        elif mdef['type'] == 'yolo':
            yolo_index += 1
            stride = [32, 16, 8, 4, 2][yolo_index]  # P3-P7 步长
            layers = mdef['from'] if 'from' in mdef else []
            modules = YOLOLayer(anchors=mdef['anchors'][mdef['mask']],  # 锚点列表
                                nc=mdef['classes'],  # 类别数
                                img_size=img_size,  # (416, 416)
                                yolo_index=yolo_index,  # 0, 1, 2...
                                layers=layers,  # 输出层
                                stride=stride)

            # 初始化前面的 Conv2d() 偏置 (https://arxiv.org/pdf/1708.02002.pdf 第 3.3 节)
            try:
                j = layers[yolo_index] if 'from' in mdef else -1
                bias_ = module_list[j][0].bias  # shape(255,)
                bias = bias_[:modules.no * modules.na].view(modules.na, -1)  # shape(3,85)
                bias[:, 4] += -4.5  # 目标
                bias[:, 5:] += math.log(0.6 / (modules.nc - 0.99))  # 类别 (sigmoid(p) = 1/nc)
                module_list[j][0].bias = torch.nn.Parameter(bias_, requires_grad=bias_.requires_grad)
            except:
                print('警告：智能偏置初始化失败。')
        else:
            print('警告：无法识别的层类型：' + mdef['type'])

        # 注册模块列表和输出过滤器数量
        module_list.append(modules)
        output_filters.append(filters)
    routs_binary = [False] * (i + 1)
    for i in routs:
        routs_binary[i] = True
    return module_list, routs_binary
```

这段代码定义了 YOLO 层，使用了初始化，并完美地为训练做好了所有设置。代码的重要部分如下：

1.  `YOLOLayer` 使用有用信息进行配置，例如锚框数量、类别数量、输出数量和类别数量。

2.  代码还在图像上设置了网格，这是锚框所必需的。它还设置了前向传播的参数。

```python
class YOLOLayer(nn.Module):
    def __init__(self, anchors, nc, img_size, yolo_index, layers, stride):
        super(YOLOLayer, self).__init__()
        self.anchors = torch.Tensor(anchors)
        self.index = yolo_index  # 该层在 layers 中的索引
        self.layers = layers  # 模型输出层索引
        self.stride = stride  # 层步长
        self.nl = len(layers)  # 输出层数量 (3)
        self.na = len(anchors)  # 锚框数量 (3)
        self.nc = nc  # 类别数量 (80)
        self.no = nc + 5  # 输出数量 (85)
        self.nx, self.ny, self.ng = 0, 0, 0  # 初始化 x, y 网格点数
        self.anchor_vec = self.anchors / self.stride
        self.anchor_wh = self.anchor_vec.view(1, self.na, 1, 1, 2)
        if ONNX_EXPORT:
            self.training = False
            self.create_grids((img_size[1] // stride, img_size[0] // stride))  # x, y 网格点数

    def create_grids(self, ng=(13, 13), device='cpu'):
        self.nx, self.ny = ng  # x 和 y 网格大小
        self.ng = torch.tensor(ng)
        # 构建 xy 偏移量
        if not self.training:
            yv, xv = torch.meshgrid([torch.arange(self.ny, device=device), torch.arange(self.nx, device=device)])
            self.grid = torch.stack((xv, yv), 2).view((1, 1, self.ny, self.nx, 2)).float()
        if self.anchor_vec.device != device:
            self.anchor_vec = self.anchor_vec.to(device)
            self.anchor_wh = self.anchor_wh.to(device)

    def forward(self, p, out):
        ASFF = False  # https://arxiv.org/abs/1911.09516
        if ASFF:
            i, n = self.index, self.nl  # layers 中的索引，层数
            p = out[self.layers[i]]
            bs, _, ny, nx = p.shape  # bs, 255, 13, 13
            if (self.nx, self.ny) != (nx, ny):
                self.create_grids((nx, ny), p.device)
            # 输出和权重
            # w = F.softmax(p[:, -n:], 1)  # 归一化权重
            w = torch.sigmoid(p[:, -n:]) * (2 / n)  # sigmoid 权重 (更快)
            # w = w / w.sum(1).unsqueeze(1)  # 沿层维度归一化
            # 加权 ASFF 求和
            p = out[self.layers[i]][:, :-n] * w[:, i:i + 1]
            for j in range(n):
                if j != i:
                    p += w[:, j:j + 1] * \
                         F.interpolate(out[self.layers[j]][:, :-n], size=[ny, nx], mode='bilinear', align_corners=False)
        elif ONNX_EXPORT:
            bs = 1  # 批次大小
        else:
            bs, _, ny, nx = p.shape  # bs, 255, 13, 13
            if (self.nx, self.ny) != (nx, ny):
                self.create_grids((nx, ny), p.device)
        # p.view(bs, 255, 13, 13) --> (bs, 3, 13, 13, 85)  # (bs, anchors, grid, grid, classes + xywh)
        p = p.view(bs, self.na, self.no, self.ny, self.nx).permute(0, 1, 3, 4, 2).contiguous()  # 预测
        if self.training:
            return p
        elif ONNX_EXPORT:
            # 避免 ANE 操作的广播
            m = self.na * self.nx * self.ny
            ng = 1 / self.ng.repeat((m, 1))
            grid = self.grid.repeat((1, self.na, 1, 1, 1)).view(m, 2)
            anchor_wh = self.anchor_wh.repeat((1, 1, self.nx, self.ny, 1)).view(m, 2) * ng
            p = p.view(m, self.no)
            xy = torch.sigmoid(p[:, 0:2]) + grid  # x, y
            wh = torch.exp(p[:, 2:4]) * anchor_wh  # 宽度, 高度
            p_cls = torch.sigmoid(p[:, 4:5]) if self.nc == 1 else \
                torch.sigmoid(p[:, 5:self.no]) * torch.sigmoid(p[:, 4:5])  # 置信度
            return p_cls, xy * ng, wh
        else:  # 推理
            io = p.clone()  # 推理输出
            io[..., :2] = torch.sigmoid(io[..., :2]) + self.grid  # xy
            io[..., 2:4] = torch.exp(io[..., 2:4]) * self.anchor_wh  # wh yolo 方法
            io[..., :4] *= self.stride
            torch.sigmoid_(io[..., 4:])
            return io.view(bs, -1, self.no), p  # 将 [1, 3, 13, 13, 85] 视为 [1, 507, 85]
```

3.  它还提供了一个设置 ONNX 模型的条款。

最后，放置了检测模型代码，它使用 `Darknet` 框架为对象检测创建了一个高度优化的工作流程。

```python
class Darknet(nn.Module):
    # YOLOv3 目标检测模型
    def __init__(self, cfg, img_size=(416, 416), verbose=False):
        super(Darknet, self).__init__()
        self.module_defs = parse_model_cfg(cfg)
        self.module_list, self.routs = create_modules(self.module_defs, img_size)
        self.yolo_layers = get_yolo_layers(self)
        # torch_utils.initialize_weights(self)
        # Darknet 头部信息 https://github.com/AlexeyAB/darknet/issues/2914#issuecomment-496675346
        self.version = np.array([0, 2, 5], dtype=np.int32)  # (int32) 版本信息：主版本号、次版本号、修订号
        self.seen = np.array([0], dtype=np.int64)  # (int64) 训练期间已处理的图像数量
        self.info(verbose) if not ONNX_EXPORT else None  # 打印模型描述

def forward(self, x, augment=False, verbose=False):
        if not augment:
            return self.forward_once(x)
        else:  # 图像增强（仅用于推理和测试）https://github.com/ultralytics/yolov3/issues/931
            img_size = x.shape[-2:]  # 高度，宽度
            s = [0.83, 0.67]  # 缩放比例
            y = []
            for i, xi in enumerate((x,
                                    torch_utils.scale_img(x.flip(3), s[0], same_shape=False),  # 水平翻转并缩放
                                    torch_utils.scale_img(x, s[1], same_shape=False),  # 缩放
                                    )):
                # cv2.imwrite('img%g.jpg' % i, 255 * xi[0].numpy().transpose((1, 2, 0))[:, :, ::-1])
                y.append(self.forward_once(xi)[0])

y[1][..., :4] /= s[0]  # 缩放
            y[1][..., 0] = img_size[1] - y[1][..., 0]  # 水平翻转
            y[2][..., :4] /= s[1]  # 缩放
            # for i, yi in enumerate(y):  # coco small, medium, large =  32\. ** 2).float()
            #     y[i] = yi
            y = torch.cat(y, 1)
            return y, None

def forward_once(self, x, augment=False, verbose=False):
        img_size = x.shape[-2:]  # 高度，宽度
        yolo_out, out = [], []
        if verbose:
            print('0', x.shape)
        str = ''
        # 图像增强（仅用于推理和测试）
        if augment:  # https://github.com/ultralytics/yolov3/issues/931
            nb = x.shape[0]  # 批次大小
            s = [0.83, 0.67]  # 缩放比例
            x = torch.cat((x,
                           torch_utils.scale_img(x.flip(3), s[0]),  # 水平翻转并缩放
                           torch_utils.scale_img(x, s[1]),  # 缩放
                           ), 0)

for i, module in enumerate(self.module_list):
            name = module.__class__.__name__
            if name in ['WeightedFeatureFusion', 'FeatureConcat']:  # 求和，拼接
                if verbose:
                    l = [i - 1] + module.layers  # 层
                    sh = [list(x.shape)] + [list(out[i].shape) for i in module.layers]  # 形状
                    str = ' >> ' + ' + '.join(['layer %g %s' % x for x in zip(l, sh)])
                x = module(x, out)  # WeightedFeatureFusion(), FeatureConcat()
            elif name == 'YOLOLayer':
                yolo_out.append(module(x, out))
            else:  # 直接运行模块，例如 mtype = 'convolutional', 'upsample', 'maxpool', 'batchnorm2d' 等
                x = module(x)
                out.append(x if self.routs[i] else [])
            if verbose:
                print('%g/%g %s -' % (i, len(self.module_list), name), list(x.shape), str)
                str = ''

if self.training:  # 训练
            return yolo_out
        elif ONNX_EXPORT:  # 导出
            x = [torch.cat(x, 0) for x in zip(*yolo_out)]
            return x[0], torch.cat(x[1:3], 1)  # 分数，边界框：3780x80, 3780x4
        else:  # 推理或测试
            x, p = zip(*yolo_out)  # 推理输出，训练输出
            x = torch.cat(x, 1)  # 拼接 yolo 输出
            if augment:  # 反增强结果
                x = torch.split(x, nb, dim=0)
                x[1][..., :4] /= s[0]  # 缩放
                x[1][..., 0] = img_size[1] - x[1][..., 0]  # 水平翻转
                x[2][..., :4] /= s[1]  # 缩放
                x = torch.cat(x, 1)
            return x, p

def fuse(self):
        # 融合整个模型中的 Conv2d + BatchNorm2d 层
        print('正在融合层...')
        fused_list = nn.ModuleList()
        for a in list(self.children())[0]:
            if isinstance(a, nn.Sequential):
                for i, b in enumerate(a):
                    if isinstance(b, nn.modules.batchnorm.BatchNorm2d):
                        # 将此 bn 层与之前的 conv2d 层融合
                        conv = a[i - 1]
                        fused = torch_utils.fuse_conv_and_bn(conv, b)
                        a = nn.Sequential(fused, *list(a.children())[i + 1:])
                        break
            fused_list.append(a)
        self.module_list = fused_list
        self.info() if not ONNX_EXPORT else None  # yolov3-spp 从 225 层减少到 152 层

def info(self, verbose=False):
        torch_utils.model_info(self, verbose)
```

这些步骤使用了 `Darknet` 框架，该框架可在 [`https://pjreddie.com/darknet/`](https://pjreddie.com/darknet/) 获取。它速度快，并且针对计算机视觉问题进行了高度优化。除此之外，模型文件包含配置细节，用于寻找现有权重及其他信息。训练文件包含大部分可配置的细节，包括处理数据路径、配置文件路径以及其他架构细节的设置。它还设置并冻结了已经完成训练的权重，并且只训练那些需要训练和更新的层。至此，我们完成了 YOLO 的训练过程。

## 总结

目标检测是一个困难的过程，需要同时解决多个任务。它需要针对实时使用进行优化。在本章中，我们探讨了允许模型学习目标分类和定位的机制。

这一切都归结为一个事实：如果允许，机器可以发挥强大的能力来学习约束条件。目标检测算法可以用于日常工作中，包括自动驾驶汽车、交通摄像头、安全无人机以及更多应用场景。

在下一章中，我们将探讨图像分割，这与我们之前讨论的过程类似。图像分割和目标检测经常在相似的场景中使用。