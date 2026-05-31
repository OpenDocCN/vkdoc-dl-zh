# 1. 深度学习简介

我们介绍了深度学习的基本概念。我们使用 TensorFlow 2.x、谷歌云服务和谷歌驱动交互，通过 Python 编码示例使这些概念生动起来。

章节的笔记本位于以下 URL：[`https://github.com/paperd/tensorflow`](https://github.com/paperd/tensorflow)。

那么，深度学习是什么？**深度学习**是一种机器学习技术，通过自动学习算法从数据中提供见解，目的是为决策提供信息。深度学习算法使用连续层从原始输入中逐步提取更高层次的特征。哇，这听起来很复杂。让我们稍微分解一下。深度学习强调从数据中学习越来越有意义的表示的连续层。深度学习模型的每一层都从数据中学习。因此，每一层将其学到的知识传递给下一层。在图像处理中，较低层可能识别边缘，而较高层可能识别与人类相关的概念，例如数字、字母或面孔。不用担心这会让人困惑，因为我们还没有定义基础知识，我们马上就要做了。

## 神经网络

连续层几乎总是由称为神经网络的模型学习。**神经网络**是一组算法，这些算法松散地模仿了人脑，旨在识别模式。这些网络通过基于标记或聚类的原始输入进行的一种机器感知来解释感官数据。

层是深度学习中的核心构建块。**层**是一个包含人工神经元的容器，通常接收加权输入，通过一组主要是非线性函数对其进行转换，然后将这些值作为输出传递给下一层。**人工神经元**是神经网络中的基本单元，它接收一个或多个输入并将它们相加以产生一个输出。神经网络中的每一层都由人工神经元组成。

层可以被视为一个数据处理模块，它充当数据过滤器。数据进入一个层，并以更有用的形式输出。也就是说，层从它们接收到的数据中提取表示。当然，我们希望这些表示对我们解决问题有意义。需要大量的**实践**和**实验**才能获得有意义的收益。因此，我们展示了多个代码示例并对其进行解释，以帮助您获得见解。但我们建议多次练习这些示例，因为深度学习是一个非常复杂和精细的主题。

一个非常常见的深度学习问题就是识别数字 0 到 9。我们可以通过创建一个由连续层组成的神经网络来解决此问题，以帮助我们**自动**从其图像数据中预测一个数字。

例如，假设在我们的数据集中有一个数字 8 的图像。如果我们的神经网络是健壮的，它应该能够从图像数据中正确预测该数字是 8，而不需要人为干预！也就是说，网络模型能够以高精度*预测*数字的图像。当然，人类可以轻易地区分 0 到 9 之间的数字，但计算机模型能够做到这一点是惊人的，这也是深度学习的核心所在。

神经网络是由*神经元*组成的集合，它们通过*突触*连接。它被组织成三个主要部分：

+   输入层

+   隐藏层

+   输出层

当训练神经网络时，数据最初被传递到输入层。因此，**输入层**将初始数据带入系统，以便后续的人工神经元层进一步处理。然后它通过激活函数传递数据，再将其传递到第一个隐藏层。**激活函数**是一个算法，它定义了给定输入或一组输入的神经元输出。

**隐藏层**位于输入层和输出层之间，其中人工神经元接收一组加权输入并通过激活函数产生输出。一个网络可以有多个隐藏层。**输出层**为给定的输入产生结果。它是所有计算完成的地方。神经元通常非常简单，只有浮点值、输入和输出。这个浮点值就是我们所说的神经元的*权重*。

所以神经元从它们的前一层接收输入，通过激活函数将它们转换以保持值在可管理的范围内，并将转换后的输入及其权重发送到下一层的神经元。由于输入层的值通常以零为中心并且已经适当地缩放，因此它们不需要通过激活函数进行转换。

## 从数据中学习表示

机器学习算法发现执行数据处理任务的规则。因此，为了进行机器学习预测，我们需要三样东西：

1.  输入数据点

1.  预期输出的例子

1.  一种衡量算法性能的方法

*输入数据点*是某种类型的数据。输入数据的例子可以是图片。深度学习中的图像识别需要图片。当然，深度学习模型需要数值数据。那么模型如何解释图像呢？图片必须以某种方式转换！不用担心这是如何完成的，因为我们在下一章会介绍。

注意

所有数据，而不仅仅是图像数据，在可以被深度学习模型消费之前，都必须适合数值表示。

要从深度学习中的数据中进行预测，我们需要 *预期的输出示例*。因此，数据必须包含每个数据示例的表示以及每个数据示例所代表的含义。我们可以用一个例子来更好地理解这一点。在预测数字时，数据必须包含每个数字的表示以及该数字所代表的含义。如果数据中的一个示例是数字 9，我们必须有数字 9 的表示以及目标值 9。我们将在下一章中介绍如何表示数字及其目标值。

最后，我们需要 *评估算法性能*。我们通过确定 *算法当前输出与预期输出之间的距离* 来做到这一点。这个距离通常被称为损失或误差。*损失* 被用作反馈来调整算法的工作方式。这种调整被称为 *学习*。例如，如果我们的神经网络模型预测一个数字是 3 但实际上它是 8，那么我们的模型至少有一些损失。也就是说，模型预测的结果与预期输出（实际目标值）之间存在一些距离。

## TensorFlow 2.x

**TensorFlow** 是一个用于数值计算的 *Python* 开源库，旨在促进机器学习和深度学习问题的解决。TensorFlow 将机器学习模块、深度学习模块和相关算法捆绑到一个共同的编程环境中。TensorFlow 2.x 是该软件的最新版本。我们使用 2.x 标识是因为该软件变化非常快。

## Google Colab

**Google Colab**（简称 Google Colaboratory）是一种云服务，为 *Python* 提供了一个与 *Jupyter Notebook* 非常相似的数据科学工作空间。实际上，任何 Jupyter Notebook 都可以直接加载到 Colab 云服务中。

Colab 笔记本存储在 Google Drive 中，可以像 Google Docs 或 Sheets 一样共享。只需点击任何 Colaboratory 笔记本右上角的共享按钮，或遵循 Google Drive 的共享说明。

浏览 [`colab.research.google.com/notebooks/welcome.ipynb`](https://colab.research.google.com/notebooks/welcome.ipynb) 以开始使用。该网站提供了一个很好的教程，但您也可以浏览 YouTube 视频或其他教程来加深您的 Colab 技能。当然，我们也会向您介绍基础知识。

## Google Drive

**Google Drive** 是一个基于云的文件存储和同步服务，您可能已经非常熟悉，所以我们不会在这方面花费太多时间。但我们需要展示如何将 Google Colab 与 Google Drive 连接起来。这只需要几个简单的步骤：

1.  使用您的 Google 邮箱账户登录。

1.  打开一个新的浏览器标签页并浏览到 *Google Colab*。

1.  点击 *Google Colab* 链接。

1.  从弹出窗口的顶部菜单中点击 *Google Drive*。

你 Google Drive 账户上的所有笔记本都会显示在窗口中。除非你之前与 Colab 一起工作过，否则你应该看不到任何笔记本。笔记本保存在 Google Drive 的 *我的驱动器* 中的 *Colab 笔记本* 目录下。当 Colab 连接到 Google Drive 时，此目录会自动创建。

如果你想要创建一个新的笔记本，请点击 *新建笔记本*。或者点击 *取消*，这将带你回到 *欢迎使用 Colaboratory* 屏幕。此屏幕提供了 Google Colab 的主菜单，以及帮助你开始的目录。

注意

我们刚刚建立的 Google Colab 和 Google Drive 之间的连接是 *持久的*。也就是说，除非清除浏览器历史记录，否则我们只需要建立这个连接 *一次*。

## 创建一个新的笔记本

在 Colab 环境中，创建一个新的笔记本非常容易。在浏览器中打开 Google Colab（如果尚未打开）。从弹出窗口中，点击 *新建笔记本*。如果在 Colab 环境中，请点击顶部左侧菜单下的 *欢迎使用 Colaboratory*。从下拉菜单中点击 *新建笔记本*。现在有一个代码单元格可以执行 Python 代码！通过点击 *+ 代码* 或 *+ 文本* 按钮添加代码或文本单元格。要获取更多选项，请从菜单中点击 *插入*。

要创建你的第一段代码，请在代码单元格中添加以下内容：

```py
string = 'Peter picked a pail of pickled peppers'
string
```

要执行代码，请点击左边的 *小箭头*。代码单元格的输出显示了 *字符串* 变量的内容。

小贴士

我们建议从网站上复制并粘贴代码。

## GPU 硬件加速器

为了极大地加快处理速度，我们使用 Google Colab 云服务提供的 GPU。Colab 提供了一个大约 12 GB 的免费 Tesla K80 GPU。在 Colab 笔记本中启用 GPU 非常简单：

1.  点击顶部左侧菜单中的 *运行*。

1.  从下拉菜单中选择 *更改运行时类型*。

1.  从 *硬件加速器* 下拉菜单中选择 *GPU*。

1.  点击 *保存*。

注意

每个笔记本都必须启用 GPU。但只需启用一次。

测试 GPU 是否激活：

```py
import tensorflow as tf
# display tf version and test if GPU is active
tf.__version__, tf.test.gpu_device_name()
```

导入 *tensorflow* 库并显示 TensorFlow 的版本以及 GPU 的状态。如果显示 ‘/device:GPU:0’，则 GPU 已激活。如果显示 ‘..’，则常规 CPU 已激活。

## 从 URL 下载文件

让我们开始工作！我们可以使用 *tf.keras.utils.get_file* 工具直接从 URL 下载文件。我们需要 tensorflow 库，但我们已经导入了它。我们建议通过点击 *+ 代码* 创建一个新的代码单元格。

以下代码单元格从 URL 下载一个 CSV 文件：

```py
# import keras module
from tensorflow import keras
ds = 'auto-mpg.data'
url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
dataset_path = tf.keras.utils.get_file(ds, url)
dataset_path
```

从 tensorflow 库中导入 keras 模块。使用 *tf.keras.utils.get_file* 从 *UCI 机器学习仓库* 下载数据集。这个仓库是用于机器学习算法经验分析的数据库、领域理论和数据生成器的集合。

小贴士

我们强烈建议在各自的代码单元格中测试小块代码，以减少调试时间和努力。

## 准备数据集

如此，数据集需要一些预处理。例如，它没有特征标题。我们建议为每个示例创建一个新的代码单元。现在通过点击**+ 代码**来完成此操作。

以下代码单元通过访问之前代码单元中创建的路径中的 CSV 文件创建一个 pandas 数据框：

```py
# import the pandas library
import pandas as pd
cols = ['MPG','Cylinders','Displacement','Horsepower','Weight',
'Acceleration', 'Model Year', 'Origin']
raw_dataset = pd.read_csv(dataset_path, names=cols,
na_values = "?", comment="\t",
sep=" ", skipinitialspace=True)
```

导入 pandas 并创建一个列表来保存特征名称。使用**read_csv()**函数将 CSV 数据放入 pandas 数据框中。**pandas 数据框**是一个二维（或 2D）数据结构，数据以表格行和列的形式排列。让我们看看最后五条记录：

```py
raw_dataset.tail()
```

Pandas 的**tail**方法返回最后 n 行。默认情况下，它返回最后五行。它对于快速验证数据很有用。

我们可以通过创建一个副本来保存原始数据：

```py
# create a copy
data = raw_dataset.copy()
# verify contents by displaying some data
data.tail()
```

## Colab 崩溃

AbEnd（也称为异常结束或崩溃）是软件或程序的异常终止。当我们长时间（数小时）运行 Colab，将大量数据集读入内存，处理这些数据或创建一个非常大的笔记本时，它可能会崩溃。当这种情况发生时，我们有两个选择：

1.  重新启动运行时。

1.  关闭笔记本并从头开始重新启动。

要重新启动运行时，请点击顶部菜单中的**运行时**，从下拉菜单中选择**重新启动运行时**，并在提示时点击**是**。然后从笔记本的开始处重新运行您的笔记本。Colab 推荐此选项。要从头开始重新启动，请清除浏览器历史记录并打开 Colab。

## Colab 奇怪的结果

有时在处理 Colab 时会出现意外的错误或其他奇怪的结果。如果发生这种情况，请重新启动运行时或按照“Colab 崩溃”部分中描述的方法从头开始打开 Colab。

## 张量

**张量**是**数值**数据的容器。张量可以包含任意数量的维度。维度通常被称为**轴**。

在深度学习中，张量被认为是通过**n 维**数组表示的矩阵的推广。张量的维度通常由其轴的数量来描述。因此，张量是通过它们总共拥有多少轴来定义的。**秩**是张量表示的轴的数量。

通过例子来理解张量是最好的方式。所以让我们从最简单的类型开始。

## 标量（0D 张量）

**标量**是只有一个数字的张量。因此，标量被认为是零维（或 0D）张量。例如，包括 numpy float32 或 float64 数字。

让我们看看一个例子：

```py
import numpy as np
# create numpy scalar 9
scalar = np.array(9)
scalar
```

导入 numpy 模块。将包含 9 的 numpy 数组赋值给一个变量并显示结果。

如此表示标量张量的秩：

```py
# signal its rank
print (str(scalar.ndim) + 'D')
```

**ndim**属性表示 numpy 张量的轴的数量（或维度）。由于我们的变量是一个标量，所以我们看到显示 0D。

## 向量（1D 张量）

**向量**是一组数字的数组。因此，向量是一个一维（或 1D）张量。一维张量恰好有一个轴。

创建一个包含六个元素的 numpy 向量：

```py
# create numpy vector [0, 1, 0, 0, 0, 0]
vector = np.array([0, 1, 0, 0, 0, 0])
vector
```

我们看到显示的是[0, 1, 0, 0, 0, 0]。

信号向量的秩：

```py
# signal its rank
print (str(vector.ndim) + 'D')
```

我们看到显示 1D。

## 矩阵（2D 张量）

**矩阵** 是向量的数组。最简单的矩阵是二维（或 2D）张量。2D 张量有两个轴。其轴通常被称为 *行* 和 *列*。

创建一个 numpy 矩阵：

```py
# create a numpy matrix
matrix = np.array([[0, 1, 0, 0, 0, 0],
[0, 0, 1, 0, 0, 0],
[0, 0, 0, 1, 0, 0],
[0, 0, 0, 0, 1, 0],
[0, 0, 0, 0, 0, 1]])
matrix
```

行是第一轴的条目，列是第二轴的条目。因此，我们的例子中的第一行是 [0, 1, 0, 0, 0, 0]，第一列是 [0, 0, 0, 0, 0, 0]。由于矩阵有五行六列，它被认为是一个 5 `×` 6 的矩阵。

通知矩阵的秩：

```py
# signal its rank
print (str(matrix.ndim) + 'D')
```

我们看到显示的是 2D。

## 3D 矩阵（3D 张量）

我们可以通过将 2D 张量打包到一个新数组中来创建 3D 张量。3D 张量可以直观地解释为一个数字的立方体。

让我们看看一个例子：

```py
# create a 3D tensor
D3 = np.array([[[0, 1, 2]],
[[3, 4, 5]],
[[6, 7, 8]]])
# signal its rank
print (str(D3.ndim) + 'D')
```

通过将 3D 张量打包到一个数组中，我们可以创建 4D 张量，依此类推。在深度学习中，我们通常操作 0D–4D 的张量。在视频处理中，我们可以达到 5D。

## 张量的关键属性

1.  秩

1.  形状

1.  数据类型

如前所述，**秩** 是轴的数量，它通过 *ndim* 属性显示。0D 张量没有轴，1D 张量有一个轴，2D 张量有两个轴，3D 张量有三个轴。

**形状** 是一个整数元组，描述了每个轴上的维度数量。因此，我们的 3D 矩阵的形状是 (3, 1, 3)，2D 矩阵的形状是 (5, 6)，向量的形状是 (6,)，而标量的形状是空的 ()。我们可以使用 *shape* 属性来显示形状。

让我们看看我们 3D 矩阵的形状：

```py
# 3 instances of 1 x 3 matrices
D3.shape
```

显示我们 2D 矩阵的形状：

```py
# 5 rows and 6 columns (or 5 x 6 matrix)
matrix.shape
```

显示我们向量的形状：

```py
# 6 element vector
vector.shape
```

显示我们标量的形状：

```py
# just a scalar number
scalar.shape
```

**数据类型** 是张量中包含的数据的描述。使用 *dtype* 属性来显示数据类型。

显示我们张量的数据类型：

```py
# dtype of tensors
print (scalar.dtype)
print (vector.dtype)
print (matrix.dtype)
print (D3.dtype)
```

我们看到我们所有的张量都包含 int64 类型的值。

## 输入管道

**输入管道** 是一系列数据处理组件，它们操纵并应用数据转换。管道在机器学习和深度学习系统中非常常见，因为这些系统是 *数据密集型* 的。也就是说，它们需要大量数据才能表现良好。输入管道是转换大型数据集的最佳方式，因为它们将数据分解成可管理的组件。

输入管道的每个组件都会拉取大量数据，以某种方式处理它，并输出结果。下一个组件会拉取结果数据，以另一种方式处理它，并输出自己的输出。管道会继续进行，直到所有组件都完成了它们的工作。

## tf.data API

TensorFlow 的应用围绕着 tf.data API 中封装的数据集概念。*tf.data API* 允许您从简单的、可重用的组件中构建快速、灵活且易于使用的输入管道。它是 TensorFlow 中构建输入管道的推荐 API。

让我们创建一个简单的 TensorFlow 张量：

```py
import tensorflow as tf
X = tf.range(5)
X
```

如有必要，导入 tensorflow 库。创建一个形状为 (5,) 且数据类型为 int32 的 TensorFlow 张量。该向量包含值 [0, 1, 2, 3, 4]，对应于形状 (5,)。

小贴士

将你的输出与网站代码输出进行比较，以验证结果。

使用 *numpy()* 属性显示 TensorFlow 张量中的值：

```py
X.numpy()
```

我们也可以从张量中访问单个元素：

```py
# first element from tensor
X[0].numpy()
```

或者访问多个元素：

```py
# 2nd, 3rd, and 4th elements from tensor
X[1:4].numpy()
```

代码从张量中切取第二、第三和第四个元素。

## 函数 from_tensor_slices

函数 *from_tensor_slices* 从张量的所有切片创建一个 tf.data.Dataset。在我们的例子中，它创建了一个数据集，其元素是 *X* 的所有切片（沿第一个维度）。**tf.data.Dataset** 是标准的 TensorFlow API，用于构建输入管道。它表示一个可能很大的元素集。

让我们用一个例子来演示：

```py
dataset = tf.data.Dataset.from_tensor_slices(X)
dataset
```

我们刚刚创建了一个从 X 中切分的 TensorSliceDataset 对象，形状为 ()，数据类型为 tf.int32。

## 遍历 tf.data.Dataset

使用简单的循环遍历一个 *tf.data.Dataset*：

```py
for item in dataset:
print (item)
```

我们看到每个值及其形状和数据类型。

## 张量和 numpy

TensorFlow 张量与 numpy 交互良好。我们可以从 numpy 数组创建 TensorFlow 张量，反之亦然。我们甚至可以将 TensorFlow 操作应用于 numpy 数组，以及将 numpy 操作应用于 TensorFlow 张量。

从我们刚刚创建的 TensorFlow 数据集中创建一个 numpy 数组，如列表 1-1 所示。

```py
# create a variable to hold a line break
br = '\n' # this is just a convenient way to include a line break
# import numpy
import numpy as np
# technique 1
ls = []
for item in dataset:
e = item.numpy()
ls.append(e)
np_arr = np.asarray(ls, dtype=np.float32)
print (type(np_arr))
print (np_arr, br)
# technique 2
ls = [item.numpy() for item in dataset]
np_arr = np.asarray(ls, dtype=np.float32)
print (type(np_arr))
print (np_arr)
Listing 1-1
Create a numpy array from a TensorFlow dataset
```

我们展示了两种从 TensorFlow 数据集创建 numpy 数组的技术。技术 1 初始化一个列表，遍历张量，将每个值转换为 numpy，并将其追加到列表中。然后，将列表转换为 numpy 数组。技术 2 执行相同的逻辑，但使用列表推导式以更紧凑的代码。

我们可以使用 *constant* 方法将 numpy 数组转换回 TensorFlow 数据集：

```py
tf_arr = tf.constant(np_arr)
tf_arr
```

然而，常量是不可变的。也就是说，它们的值不能被修改。因此，如果我们需要修改值，我们可以使用 *variable* 方法。

使用变量方法：

```py
tf_arr = tf.Variable(np_arr)
tf_arr
```

## 链式转换

我们可以通过调用其转换方法来对 tf.data.Dataset 应用转换。每个方法返回一个新的数据集，这允许我们链式转换。让我们从一个转换开始。

创建一个 tf.data.Dataset 并显示其值：

```py
dataset = tf.data.Dataset.range(5)
for item in dataset:
print (item)
```

数据集包含值 [0, 1, 2, 3, 4]。

使用 *repeat()* 转换方法重复张量中的元素：

```py
data_rep = dataset.repeat(3)
for item in data_rep:
print (item)
```

新数据集包含原始数据的三倍。我们重复数据以扩大数据集，从而提高模型性能，而不需要获取新数据。

现在，让我们链式转换：

```py
data_batch = dataset.repeat(3).batch(7)
for item in data_batch:
print (item)
```

发生了什么？第一个转换 *repeat(3)* 创建了原始数据集的三份副本。我们将第一个转换链入第二个转换中，使用 *batch(7)*，这会创建包含七个元素的批次。

因此，新数据集由三个张量组成。第一个张量包含 [0, 1, 2, 3, 4, 0, 1]，第二个张量包含 [2, 3, 4, 0, 1, 2, 3]，第三个张量包含 [4]。当我们到达第三个批次时，数据就耗尽了。

我们可以丢弃最后一个批次：

```py
data_drop = dataset.repeat(3).batch(7, drop_remainder=True)
for item in data_drop:
print (item)
```

新数据集由两个张量组成。第一个张量包含 [0, 1, 2, 3, 4, 0, 1]，第二个张量包含 [2, 3, 4, 0, 1, 2, 3]。转换方法不会修改数据集，它们创建新的数据集，因此我们可以通过不同的名称来跟踪每个数据集。

创建相等批次：

```py
data_equal = dataset.repeat(3).batch(5)
for item in data_equal:
print (item)
```

新数据集由三个张量组成。每个张量包含 [0, 1, 2, 3, 4]。

## 映射张量

使用 *map* 方法转换张量中的元素。**map** 函数在将给定的函数应用于给定张量的每个元素后返回一个结果映射对象。返回的对象是一个迭代器。**迭代器**是一个能够一次返回其成员的 Python 对象。列表、元组和字符串是 Python 中常见的迭代器。

哇！让我们看看列表 1-2 中的示例，以帮助我们理解 map 函数的工作方式。

```py
# create a dataset
dataset = tf.data.Dataset.range(7)
# repeat and batch it
data_batch = dataset.repeat(3).batch(7)
# display the batched dataset
for row in data_batch:
print (row)
# map() a function on it
data_map = data_batch.map(lambda x: x ** 2)
# display the first batch
print ()
for item in data_map.take(1):
print (item)
Listing 1-2
A simple map function example
```

我们创建了一个包含值 [0, 1, 2, 3, 4, 5, 6] 的新数据集。我们将 repeat 转换与 batch 转换链式连接，以创建一个包含三个张量的新数据集。每个张量包含 [0, 1, 2, 3, 4, 5, 6]。我们通过映射 lambda 函数对每个元素进行平方。**lambda 函数**是一个没有名称的单行函数，可以接受任何数量的参数，但只能有一个表达式。我们不需要迭代整个数据集，可以使用 *take* 方法取一个或多个样本。在我们的例子中，我们只从数据集中取出第一个样本（或第一个张量）。

因此，映射后的数据集包含三个张量。每个张量包含 [0, 1, 4, 9, 16, 25, 36]，因为 lambda 函数对值进行平方，map 函数将 lambda 函数表达式映射到张量中的每个元素。

## 过滤 tf.data.Dataset

如果我们想过滤数据集怎么办？**filter** 方法通过一个函数过滤给定的序列，该函数测试序列中的每个元素是否为真。

让我们看看列表 1-3 中的示例。

```py
# create a dataset
dataset = tf.data.Dataset.range(7)
# display the dataset
for row in dataset:
print (row)
# apply a filter
data_filter = dataset.filter(lambda x: x  3)
print ()
for item in data_filter:
print (item)
Listing 1-3
A simple filter() function example
```

我们创建了一个包含值 [0, 1, 2, 3, 4, 5, 6] 的新数据集。我们过滤数据集以提取介于 3 和 6 之间的值。由于我们在 lambda 函数中使用小于和大于，因此我们不包含 3 或 6 的值。所以过滤后的数据集包含 [4, 5]，因为 lambda 函数提取了介于 3 和 6 之间的非包含值。

## 打乱数据集

深度学习算法在训练集中的实例独立且同分布时效果最佳。确保这一点的一个简单方法是使用 *shuffle* 方法打乱实例。

创建用于打乱的 tf.data.Dataset：

```py
# create a dataset
dataset = tf.data.Dataset.range(10).repeat(3)
print ('dataset has', len(list(dataset)), 'elements')
```

打乱数据集：

```py
# shuffle data into batches of 7
ds = dataset.shuffle(buffer_size=5).batch(7)
for item in ds:
print (item)
```

我们得到七个元素的张量。注意，最后一个张量只有两个元素。我们有四个大小为七的张量，总共 28 个。由于数据集有 30 个元素，我们剩下两个元素。

我们将缓冲区大小设置为 5。因此，TensorFlow 保持下一个五个样本（或张量）的缓冲区，并随机选择这五个样本中的一个。然后，它将下一个元素添加到缓冲区中。每个样本包含一个数据批次。因此，我们例子中的每个样本包含七个元素，因为我们设置了批大小为 7。通过实验不同的批量和缓冲区大小可以提高性能，但正确设置需要时间和精力。

一旦数据集被洗牌，每次数据集迭代都会创建一个新的洗牌：

```py
# rerun to get a different shuffle
for item in ds:
print (item)
```

## TensorFlow 数学

TensorFlow 提供了多个操作，用于 *tf.math* 模块中的数学计算。查阅 [www.tensorflow.org/api_docs/python/tf/math](http://www.tensorflow.org/api_docs/python/tf/math) 了解所有可能的数学运算。要执行数学运算，张量必须具有相同的形状。

### 向量张量

让我们创建一些数据：

```py
# create data
v1 = np.array([0, 1, 4, 8, 16])
v2 = np.array([0, 3, 9, 27, 81])
```

将 numpy 数组转换为张量常量并相加：

```py
conv1 = tf.constant(v1)
conv2 = tf.constant(v2)
result = tf.add(conv1, conv2)
result
```

结果是 [0, 4, 13, 35, 97]。

将 numpy 数组转换为张量变量并相加：

```py
varv1 = tf.Variable(v1)
varv2 = tf.Variable(v2)
result = tf.add(varv1, varv2)
result
```

我们得到相同的结果。常量和变量之间的唯一区别是常量值不能更改。也就是说，它们是不可变的。

减去张量变量：

```py
result = tf.subtract(varv2, varv1)
result
```

从 var2 中减去 var1 的结果是 [0, 2, 5, 19, 65]。

混合常量和变量：

```py
result = tf.add(conv1, varv2)
result
```

结果是 [0, 4, 13, 35, 97]。

测试等价性：

```py
result = tf.equal(varv1, varv2)
result
```

结果是 [True, False, False, False, False]。

乘以张量常量：

```py
result = tf.multiply(conv1, conv2)
result
```

结果是 [0, 3, 36, 216, 1296]。

将张量除以一个值：

```py
result = tf.divide(conv2, 3)
result
```

结果是 [0, 1, 3, 9, 27]。

### 矩阵张量

数学运算适用于 n 维张量。因此，让我们在矩阵张量上执行数学运算。

创建一些数据：

```py
# create data
m1 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
m2 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
m1.shape, m2.shape
```

我们刚刚创建了一对 3 `×` 3 的矩阵。也就是说，它们都有三行三列。

将 numpy 矩阵转换为张量并相加：

```py
conm1 = tf.constant(m1)
conm2 = tf.constant(m2)
result = tf.add(conm1, conm2)
result
```

结果是 [[2, 0, 0], [0, 2, 0], [0, 0, 2]]。

测试等价性：

```py
result = tf.equal(conm1, conm2)
result
```

结果是 [[True, True, True], [True, True, True], [True, True, True]]，因为张量是等价的。

### tf.data.Dataset 张量

我们可以在 tf.data.Dataset 张量上执行数学运算。

创建一个数据集：

```py
# create a dataset
m = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
m
```

我们刚刚创建了一个 3 `×` 4 的矩阵。因此，矩阵由三行四列组成。

将数据集转换为 tf.data.Dataset：

```py
dataset = tf.data.Dataset.from_tensor_slices(m)
dataset
```

tf.data.Dataset 是 *TensorSliceDataset*，因为我们使用 *from_tensor_slices* 方法转换了 numpy 数据集。数据集包含三个张量。每个张量的形状是 (4,), 这意味着每个张量都有四个元素。元素的数据类型是 int64。

显示张量：

```py
for t in dataset:
print (t)
```

tf.data.Dataset 包含三个张量，其值分别为 [1, 2, 3, 4]，[5, 6, 7, 8]，和 [9, 10, 11, 12]。

转换张量值：

```py
squared_data = dataset.map(lambda x: x ** 2)
# display tensors
for item in squared_data:
print (item)
```

我们将 lambda 函数映射到张量元素。因此，每个元素都被平方。

## 保存笔记本

虽然在 Google Colab 中实现了 *自动保存*，但执行单元格和保存发生之间的延迟是存在的。因此，我们建议定期保存。

手动保存笔记本只需两步：

1.  在左上角菜单中点击 *文件*。

1.  从下拉菜单中选择 *保存*。

笔记本已保存在 Google Drive 的 *My Drive* 目录下的 *Colab Notebooks* 文件夹中。

## 将笔记本下载到本地驱动器

Google Drive 是存储 Colab 笔记本的绝佳地点。但我们也更喜欢将笔记本保存在本地驱动器上。

将笔记本下载到本地驱动器：

1.  确保保存笔记本。

1.  在左上角的菜单中点击 *File*。

1.  从下拉菜单中选择 *Download .ipynb*。

笔记本现在位于 *Downloads* 文件夹中。

## 从本地驱动器加载笔记本

我们使用 Google Drive 作为备份，因为它只提供 15 GB 的免费空间。如果您在公司工作，他们可能提供额外的存储空间。在这种情况下，您可能希望将 Google Drive 用作主要存储。

要从本地驱动器加载笔记本

1.  打开 *Google Colab*。

1.  在弹出菜单中，点击 *Upload*。

1.  点击 *Choose File*。

1.  在您的本地驱动器上找到笔记本并打开它。
