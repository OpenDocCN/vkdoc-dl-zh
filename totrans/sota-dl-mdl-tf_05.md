# 5. 张量处理单元简介

我们通过代码示例向您介绍张量处理单元。**张量处理单元（TPU）**是一种专为加速机器学习工作负载而设计的专用集成电路（ASIC）。TensorFlow 中可用的 TPU 是由 Google Brain 团队从头开始定制开发的，基于其在机器学习社区中的丰富经验和领导地位。“Google Brain”是谷歌的一个深度学习人工智能（AI）研究团队，他们研究使机器智能以改善人们生活的途径。

各章节的 Notebooks 位于以下 URL：

[`github.com/paperd/deep-learning-models`](https://github.com/paperd/tensorflow)

## TPU 硬件加速器

我们展示了使用来自 Google Colab 云服务的 TPU 硬件加速器的学习实验。我们的示例并不展示 TPU 的全部功效，但可以为您提供一个起点。我们使用高级 TensorFlow API 在 Cloud TPU 硬件上运行模型。

我们在 Colab 中使用的硬件加速器通常被称为 Cloud TPU。**Cloud TPU**旨在在 Google Cloud 上运行具有 AI 服务的尖端机器学习模型。它允许使用 TensorFlow 在 Google 的 TPU 加速器硬件上处理机器学习工作负载。其定制的快速网络在一个 Pod 中提供超过 100 petaflops 的性能，这足以将企业转变为 AI 就绪或创造下一个研究突破。Cloud TPU 旨在提供最大性能和灵活性，以帮助研究人员、开发人员和企业在可以利用 CPU、GPU 和 TPU 的 TensorFlow 计算集群中构建。

对于部署 Cloud TPU 的全面指南，请参阅

[`cloud.google.com/tpu/docs/tpus`](https://cloud.google.com/tpu/docs/tpus)

对于本写作时所有可用的 Cloud TPU 教程，请参阅

[`cloud.google.com/tpu/docs/tutorials`](https://cloud.google.com/tpu/docs/tutorials)

### Cloud TPU 的优势

Cloud TPU 资源加速了线性代数计算的性能，这在机器学习应用中得到了广泛的应用。Cloud TPU 最小化了我们在训练大型、复杂的神经网络模型时的准确度时间。在其他硬件平台上原本需要数周时间训练的模型，在 Cloud TPU 上可以数小时内收敛。

### 当使用 Cloud TPU 时

Cloud TPU 针对特定工作负载进行了优化。在某些情况下，我们使用 GPU 或 CPU 来运行机器学习工作负载。通常，根据以下指南决定使用哪种硬件。

#### CPU

+   * 需要最大灵活性的快速原型设计

+   * 训练时间不长的简单模型

+   * 带有小型有效批次的简单模型

+   * 以 C++编写的自定义 TensorFlow 操作为主的模型

+   * 受限于可用 I/O 或主机系统网络带宽的模型

#### GPU

+   * 源代码不存在或修改过于繁琐的模型

+   * 包含大量必须至少部分在 CPU 上运行的定制 TensorFlow 操作的模型

+   * 包含在 Cloud TPU 上不可用的 TensorFlow 操作的模型

+   * 中等至大型模型，具有更大的有效批量大小

#### TPUs

+   * 以矩阵计算为主的模型

+   * 主训练循环中没有定制 TensorFlow 操作的模型

+   * 训练数周或数月的模型

+   * 非常大的模型，具有非常大的有效批量大小

Cloud TPU 不适合需要频繁分支或主要由元素级代数主导的线性代数程序。Cloud TPU 优化以快速执行大量矩阵乘法。因此，非矩阵乘法主导的工作负载在与其他平台相比不太可能在 Cloud TPU 上表现良好。需要高精度算术的工作负载不适合 Cloud TPU（例如，双精度算术）。以稀疏方式访问内存的工作负载可能在 Cloud TPU 上不可用。

关于 Cloud TPU 的丰富资源，请参阅

[`cloud.google.com/tpu/docs/tpus`](https://cloud.google.com/tpu/docs/tpus)

## 导入 TensorFlow 库

导入库并将其别名为 **tf**：

```py
import tensorflow as tf
```

## 启用 TPU 运行时

在 Colab 笔记本中启用 TPU 非常简单：

1.  在右上角菜单中点击 *运行时*。

1.  从下拉菜单中选择 *更改运行时类型*。

1.  从 *硬件加速器* 下拉菜单中选择 *TPU*。

1.  点击 *保存*。

注意

TPU 必须在每个笔记本中启用。但只需启用一次。

## TPU 检测

设置 TPU 解析器并验证其是否正在运行：

```py
tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
print('Running on TPU ', tpu.cluster_spec().as_dict()['worker'])
```

消息看起来可能像这样：

在 TPU [‘10.112.96.162:8470’] 上运行

小贴士

如果出现错误 **NAME ‘TF’ IS NOT DEFINED**，请重新执行代码以导入 TensorFlow 库！由于某些原因，我们有时不得不在 Colab 中重新执行 TensorFlow 库的导入。我们不知道这是为什么。

*tf.distribute.cluster_resolver.TPUClusterResolver()* API 是 Google Cloud TPU 的集群解析器。*集群解析器* 为 TensorFlow 提供了一种与各种集群管理系统（例如，GCE、AWS 等）通信的方式，并访问设置分布式训练所需的信息。通过让 TensorFlow 与这些系统通信，它能够自动发现并解析各种 TensorFlow 工作器的 IP 地址。因此，它能够最终从底层机器故障中自动恢复，并调整 TensorFlow 工作器集群的系统管道的上下文。

## 为此笔记本配置 TPU

使集群上的设备可用于使用：

```py
tf.config.experimental_connect_to_cluster(tpu)
```

初始化 TPU 设备：

```py
tf.tpu.experimental.initialize_tpu_system(tpu)
```

注意

这两种配置都是实验性的，这意味着它们将来可能会更改。

## 创建分发策略

**分布策略**是一种用于在多个 CPU、GPU 或 TPU 之间分配训练的抽象。要更改模型在给定设备上的运行方式，只需更换分布策略。*tf.distribute.Strategy*是 TensorFlow API，用于将分布策略应用于模型。应用此 API 允许我们在最小代码更改的情况下将现有模型的训练分配到多个设备。

该 API 的设计考虑了三个关键目标：

+   *易于使用并支持多个用户群体（例如，数据科学家）

+   * 提供了开箱即用的良好性能

+   *易于在策略之间切换

为此笔记本创建一个 TPU 策略：

```py
tpu_strategy = tf.distribute.TPUStrategy(tpu)
```

显示可用的 TPU 和其他设备列表。

## 手动设备放置

在 TPU 系统初始化后，我们可以使用手动设备放置来指导计算在单个 TPU 设备上执行，如列表 5-1 所示。

```py
a = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
b = [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]
with tf.device('/TPU:7'):
c = tf.matmul(a, b)
I = [[1.0, 0.0], [0.0, 1.0]]
with tf.device('/TPU:6'):
d = tf.matmul(c, I)
print('c device:', c.device)
print(c)
print('d device:', d.device)
print(d)
Listing 5-1
Run Computations on TPU Devices
```

在 TPU 设备 7 的范围内将矩阵 a 乘以矩阵 b，并将结果放置在矩阵 c 中。接下来，将矩阵 c 乘以单位矩阵，并将结果放置在 TPU 设备 6 的范围内矩阵 d 中。

## 在所有 TPU 核心上运行计算

为了使计算可以在所有 TPU 核心上运行，将其传递给*strategy.run* API：

```py
@tf.function
def matmul_fn(x, y):
z = tf.matmul(x, y)
return z
z = tpu_strategy.run(matmul_fn, args=(a, b))
print(z)
```

创建一个乘以两个矩阵的函数。如果启用贪婪行为，则使该函数成为*tf.function*或在 tf.function 内部调用*strategy.run*。在 Colab 中，贪婪行为自动启用！

使用我们创建的*strategy.run* API 调用函数，这确保所有 TPU 核心获得相同的输入（a，b），并且在每个核心上独立应用矩阵乘法。输出是所有核心副本的值。

## 贪婪执行

Cloud TPU 的操作必须启用贪婪执行。*贪婪执行*（或贪婪行为）是一种命令式编程环境，因为它立即评估操作而不构建图。因此，TensorFlow 操作返回具体值而不是构建用于稍后运行的计算图。它还提供了一个直观的接口，允许您自然地组织代码并使用 Python 数据结构。其他好处包括对小型模型和小型数据的快速迭代以及更易于调试，因为您可以直接调用 ops 来检查正在运行的模式并测试更改。

注意

TensorFlow 2.x 默认启用贪婪执行。

## 实验

我们包括四个 Cloud TPU 实验。实验从简单的数据集开始，逐渐过渡到更复杂的数据集。与多个数据集一起工作的意义在于获得经验和信心。每个数据集都不同，因此实验略有不同。学习过程通常是相同的，但每个数据集都有不同的特征。我们展示多个经验的主要原因是为了给您提供更多的练习机会。通过与研讨会参与者一起工作，我们发现他们通过使用多个数据集学习得更快，他们也更加享受学习过程！

## 数字实验

*digits*数据集嵌入在*sklearn.datasets*包中。它由 1797 个 0 到 9 的手写数字的 8 × 8 图像组成。*Scikit-learn*（也称为 sklearn）是 Python 中的一个机器学习库。它包含一些小的标准数据集（例如 digits），这些数据集不需要从外部网站下载任何文件。

导入必需的库：

```py
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
```

加载数据集：

```py
digits = load_digits()
```

获取键：

```py
digits.keys()
```

键包括“data”、“target”、“target_names”、“images”和“DESCR”。键*data*包含形状为(1797, 64)的图像作为展平的数据矩阵。键*target*包含形状为(1797,)的标签向量。键*target_names*是目标类别的名称列表。键*images*包含形状为(1797, 8, 8)的图像矩阵。键*DESCR*包含数据集的完整描述。因此 digits 包含 1797 个 8 × 8 像素的图像和 1797 个相应的标量标签。

这个数据集有键的原因是因为它是一个 scikit-learn (sklearn)数据集。它包含各种分类、回归和聚类算法，包括支持向量机、随机森林、梯度提升、k-means 和 DBSCAN。它被设计为与 Python 数值和科学库 NumPy 和 SciPy 互操作。它还包含像 digits 这样的实践数据集。

显示图像：

```py
images = digits.images
image = images[0]
fig = plt.imshow(image, cmap='binary')
fig = plt.axis('off')
```

### 预处理数据

由于 digits 是一个 scikit-learn 实践集，我们可以非常容易地预处理它。*train_test_split*方法自动将数据分割成训练集和测试集。你可以通过*test_size*参数调整分割比例。

创建一个函数来预处理数据，如列表 5-2 所示。

```py
def load_data(digits, splits, random, scale):
X = digits.images
y = digits.target
x_train, x_test, y_train, y_test = train_test_split(
X, y, test_size=splits, random_state=random)
x_train, x_test = x_train / scale, x_test / scale
return (x_train, y_train), (x_test, y_test)
Listing 5-2
Function to Preprocess Data
```

预处理数据：

```py
splits, seed, scale = 0.33, 0, 255.0
(x_train, y_train), (x_test, y_test) = load_data(
digits, splits, seed, scale)
```

数据分割为 67%的训练和 33%的测试。当然，你可以通过将 0.33 更改为另一个值来调整分割比例。

获取目标名称和类别数量：

```py
target_names = digits.target_names
num_classes = len(target_names)
num_classes, target_names
```

我们利用内置的 sklearn 方法*target_names*来获取目标（类别标签）名称。

### 构建输入管道

准备数据以供 TensorFlow 使用：

```py
train_dataset = tf.data.Dataset.from_tensor_slices(
(x_train, y_train))
test_dataset = tf.data.Dataset.from_tensor_slices(
(x_test, y_test))
```

检查训练张量：

```py
for img, lbl in train_dataset.take(1):
print (img.shape, lbl)
```

设置参数并构建管道：

```py
BATCH_SIZE = 64
SHUFFLE_BUFFER_SIZE = 100
train_ds = train_dataset\
.shuffle(SHUFFLE_BUFFER_SIZE)\
.batch(BATCH_SIZE)
test_ds = test_dataset.batch(BATCH_SIZE)
```

### 模型数据在 TPU 范围内

保留模型的输入形状：

```py
for item in train_ds.take(1):
s = item[0].shape
in_shape = s[1:]
in_shape
```

我们保存输入形状以供模型后续使用。

导入库：

```py
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
```

创建一个函数来构建模型，如列表 5-3 所示。

```py
def get_model():
return tf.keras.Sequential([
Flatten(input_shape=in_shape),
Dense(256, input_shape=in_shape, activation='relu'),
Dense(num_classes, activation='softmax')])
Listing 5-3
Function to Create the Model
```

在 TPU 分布策略范围内创建和编译模型，如列表 5-4 所示。

```py
with tpu_strategy.scope():
model = get_model()
model.compile(
optimizer='adam',
loss='sparse_categorical_crossentropy',
metrics=['accuracy'])
Listing 5-4
Create and Compile the Model Within TPU Scope
```

要在 Cloud TPU 上训练一个模型，我们必须在章节中之前创建的 TPU 策略范围内创建和编译它。

检查模型：

```py
model.summary()
```

训练模型：

```py
epochs = 60
history = model.fit(train_ds, epochs=epochs,
validation_data=(test_ds))
```

我们可以用我们想要的任何多或少个 epoch 来训练一个模型。然而，大数据集比小数据集消耗内存要快得多。由于 digits 非常小且图像简单，我们可以放心地使用更多 epoch 进行训练，而不必担心内存消耗。记住，我们通过大量的实验达到了 60 个 epoch。这就是我们鼓励你多练习的原因！

## MNIST 实验

虽然 MNIST 正在被 Fashion-MNIST 取代用于深度学习实践，但它仍然比数字数据集大得多。

将数据集作为 TFDS 对象加载：

```py
import tensorflow_datasets as tfds
train, info = tfds.load(name='mnist', split='train',
as_supervised=True, try_gcs=True,
with_info=True, shuffle_files=True)
test = tfds.load(name='mnist', split='test',
as_supervised=True, try_gcs=True)
```

我们将其留给你去探索元数据，因为我们已经在早期章节中详细描述了此数据集对象。

### 构建输入管道

创建一个函数来缩放数据：

```py
def scale(image, label):
image = tf.cast(image, tf.float32) / 255.0
return image, label
```

构建管道：

```py
BATCH_SIZE = 200
SHUFFLE_SIZE = 10000
train_dataset = train.map(scale)\
.shuffle(SHUFFLE_SIZE).repeat()\
.batch(BATCH_SIZE).prefetch(1)
test_dataset = test.map(scale)\
.batch(BATCH_SIZE).prefetch(1)
```

仅对训练数据集进行洗牌和重复。*repeat()* 方法将数据集计数重复 *n* 次。如果没有包含数字，则数据集计数是无限的。无限数据集用于训练的优势是避免每个 epoch 中潜在的最后一个部分批次，这样用户就不需要根据实际的批次大小调整梯度缩放。由于模型只从训练集中学习，所以我们不对测试集进行洗牌和重复。

### TPU 范围内的模型数据

创建变量来保存图像数量、步长和验证步骤：

```py
num_train_img = info.splits['train'].num_examples
num_test_img = info.splits['test'].num_examples
steps_per_epoch = num_train_img // BATCH_SIZE
validation_steps = num_test_img // BATCH_SIZE
```

当重复数据时，我们必须包括每轮步骤和验证步骤参数的值。

保留模型的输入形状：

```py
for item in train_dataset.take(1):
s = item[0].shape
mnist_shape = s[1:]
mnist_shape
```

创建一个函数来构建模型，如列表 5-5 所示。

```py
def create_model():
return Sequential([
Flatten(input_shape=mnist_shape),
Dense(512, activation='relu'),
Dense(mnist_classes, activation='softmax')
])
Listing 5-5
Function to Build the Model
```

清除模型并生成一个种子：

```py
import numpy as np
tf.keras.backend.clear_session()
np.random.seed(0)
tf.random.set_seed(0)
```

导入损失函数的库：

```py
from tensorflow.keras.losses import SparseCategoricalCrossentropy
```

我们导入损失函数以减少模型中包含的字符数。如果我们不这样做，我们就必须告诉模型损失函数的位置。因此，我们必须包括整个库的名称。我们将此代码作为之前实验中做法的替代方案。

获取模型的类别数量：

```py
mnist_classes = info.features['label'].num_classes
mnist_classes
```

在 TPU 策略范围内，创建并编译，如列表 5-6 所示。

```py
with tpu_strategy.scope():
model = create_model()
model.compile(
optimizer='adam',
steps_per_execution = 50,
loss=SparseCategoricalCrossentropy(from_logits=True),
metrics=['sparse_categorical_accuracy'])
Listing 5-6
Create and Compile the Model Within TPU Scope
```

在 TPUStrategy 范围内创建模型意味着我们在 TPU 系统上训练模型。这个想法是利用 TPU 来加速学习。在工业界，我们必须与系统工程师合作，以利用多个 TPU 和其他内存源之间的并行处理。尝试 *steps_per_execution*。介于 2 和 *steps_per_epoch* 之间的任何值都可能提高性能。

训练模型：

```py
history = model.fit(
train_dataset, epochs=5,
steps_per_epoch=steps_per_epoch,
validation_data=test_dataset,
validation_steps=validation_steps)
```

## Fashion-MNIST 实验

由于 Fashion-MNIST 是当前 MNIST 的直接替换，我们用它来演示 TPU 学习。

*Fashion-MNIST* 是由 Zalando 的文章图像组成的数据库，包括 60,000 个示例的训练集和 10,000 个示例的测试集。该数据集旨在作为基准机器学习算法的直接替换 MNIST 数据集。

将 Fashion-MNIST 作为 tf.keras 数据集加载：

```py
fashion_train, fashion_test = tf.keras.datasets\
.fashion_mnist.load_data()
```

### 将数据集转换为图像和标签集

创建图像和标签集：

```py
train_img, train_lbl = fashion_train
test_img, test_lbl = fashion_test
```

将灰度颜色维度添加到图像张量中：

```py
fashion_train_img = np.expand_dims(train_img, -1)
fashion_test_img = np.expand_dims(test_img, -1)
```

检查图像张量：

```py
fashion_train_img.shape, fashion_test_img.shape
```

创建一个将张量转换为浮点数的函数：

```py
def float_it(x):
return x.astype(np.float32)
```

通过将张量转换为 NumPy 张量，训练性能应该会提高。

调用函数：

```py
fash_train_img, fash_train_lbl = float_it(
fashion_train_img), float_it(train_lbl)
fash_test_img, fash_test_lbl = float_it(
fashion_test_img), float_it(test_lbl)
```

### TPU 范围内的模型数据

导入库：

```py
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization,\
Conv2D, MaxPooling2D, Dropout
```

创建一个函数来构建模型，如列表 5-7 所示。

```py
def create_model():
return Sequential([
BatchNormalization(input_shape=(28,28,1)),
Conv2D(64, (5, 5), padding='same', activation='elu'),
MaxPooling2D(pool_size=(2, 2), strides=(2,2)),
Dropout(0.25),
BatchNormalization(),
Conv2D(128, (5, 5), padding='same', activation='elu'),
MaxPooling2D(pool_size=(2, 2)),
Dropout(0.25),
BatchNormalization(),
Conv2D(256, (5, 5), padding='same', activation='elu'),
MaxPooling2D(pool_size=(2, 2), strides=(2,2)),
Dropout(0.25),
Flatten(),
Dense(256, activation='elu'),
Dropout(0.5),
Dense(10, activation='softmax')
])
Listing 5-7
Function to Build the Model
```

**批量归一化**是一种用于训练非常深的神经网络的技巧，它为每个小批量标准化层的输入。该技术稳定了学习过程，显著减少了训练深度网络所需的训练周期数。

清除和设置随机种子：

```py
tf.keras.backend.clear_session()
np.random.seed(0)
tf.random.set_seed(0)
```

在 TPU 策略范围内，创建并编译模型，如列表 5-8 所示。

```py
with tpu_strategy.scope():
model = create_model()
model.compile(
optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3, ),
loss='sparse_categorical_crossentropy',
metrics=['accuracy'])
Listing 5-8
Create and Compile the Model Within TPU Scope
```

训练模型：

```py
history = model.fit(fash_train_img, fash_train_lbl,
epochs=17, steps_per_epoch=60,
validation_data=(fash_test_img, fash_test_lbl),
validation_freq=17)
```

我们添加了*validation_freq*参数作为另一个选项。它仅在提供验证数据时相关。它指定在执行新的验证运行之前要运行多少个训练周期。因此，我们每 17 个周期进行一次验证。我们包括这个参数只是为了在实验的上下文中展示另一个参数。

评估模型：

```py
loss, acc = model.evaluate(fash_test_img, fash_test_lbl)
print ('loss:', loss)
print ('accuracy:', acc)
```

### 保存训练好的模型

我们可以保留训练模型的权重：

```py
model.save_weights('./fashion_mnist.h5', overwrite=True)
```

### 进行推理

我们对这个数据集进行推理（预测），因为它目前是深度学习竞赛的首选数据集。这样的竞赛对于学生和/或从业者来说非常有帮助，可以积累经验并与志同道合的人建立联系。前述数据集非常适合练习，但并不具有挑战性。也就是说，获得高精度（或低损失）太容易了。

现在我们已经完成了训练，让我们看看模型在预测时尚类别方面的表现如何！

获取标签名称：

```py
class_labels = ['t_shirt', 'trouser', 'pullover', 'dress',
'coat', 'sandal', 'shirt', 'sneaker',
'bag', 'ankle_boots']
```

从训练好的模型中创建一个新的模型：

```py
new_model = create_model()
new_model.load_weights('./fashion_mnist.h5')
```

从测试集中获取 40 个预测数组进行绘图：

```py
preds = new_model.predict(fash_test_img)[:40]
```

将预测数组转换为标量预测值：

```py
pred_40 = [tf.argmax(i).numpy() for i in preds]
```

由于预测数组不能识别实际的预测，我们使用 tf.argmax。

获取图像和标签以进行显示，如列表 5-9 所示。

```py
images, labels = [], []
for i in range(40):
img = tf.squeeze(fash_test_img[i])
images.append(img)
labels.append(int(fash_test_lbl[i]))
Listing 5-9
Images and Labels for Display
```

创建一个函数来显示预测结果，如列表 5-10 所示。

```py
def display_test(feature, target, num_images,
n_rows, n_cols, cl, p):
for i in range(num_images):
plt.subplot(n_rows, 2*n_cols, 2*i+1)
plt.imshow(feature[i], cmap='nipy_spectral')
pred = cl[p[i]]
actual = cl[int(target[i])]
title_obj = plt.title(actual + ' (' +\
pred + ') ')
if pred == actual:
title_obj
else:
plt.getp(title_obj, 'text')
plt.setp(title_obj, color='r')
plt.tight_layout()
plt.axis('off')
Listing 5-10
Function to Display Prediction Performance
```

调用函数：

```py
num_rows, num_cols = 10, 4
num_images = num_rows*num_cols
plt.figure(figsize=(20, 20))
display_test(images, labels, num_images, num_rows,
num_cols, class_labels, pred_40)
```

红色图像表示预测错误。黑色图像表示正确预测。

## 花卉实验

花卉数据集并不大，但图像比 Fashion-MNIST 中的图像复杂得多。花卉图像包含比本章中其他数据集更多的像素。我们认为，与花卉这样的复杂数据集一起工作是非常好的实践。

TPUs 非常快。因此，流入模型的数据流必须与模型的训练速度保持一致，以充分利用 TPUs 的强大功能。TPU 使用的首选方法是存储数据到基于 protobuf 的 TFRecord 格式。*TFRecord 格式*将数据存储为一系列二进制字符串作为 TFRecords。二进制字符串在存储和数据传输方面效率高。

花卉数据集存储在 Google Cloud Storage（GCS）上作为 TFRecords。为了充分利用 TPU 提供的并行性并避免数据传输瓶颈，我们以每文件约 230 张图像的方式读取数据。230 这个数字用于将花卉数据均匀分配到几个桶中。我们通过实验发现了这个数字。我们使用 tf.data.experimental.AUTOTUNE 来优化输入加载的不同部分。我们在这个实验中使用 AUTOTUNE 是因为数据集包含复杂的图像。这种优化技术对早期的实验不是必需的。

对于 Colab 中的 TPU 教程，请参阅

[Colab 中的 TPU 教程](https://colab.research.google.com/notebooks/tpu.ipynb)

对于 TPU 数据管道处理的良好介绍，请参阅

[Keras 花卉数据代码实验室](https://codelabs.developers.google.com/codelabs/keras-flowers-data)

### 以 TFRecord 文件读取花卉数据

建立 GCS 文件名模式：

```py
piece1 = 'gs://flowers-public/'
piece2 = 'tfrecords-jpeg-192x192-2/*.tfrec'
TFR_GCS_PATTERN = piece1 + piece2
tfr_filenames = tf.io.gfile.glob(TFR_GCS_PATTERN)
```

获取桶的数量：

```py
num_images = len(tfr_filenames)
print ('Pattern matches {} image buckets.'.format(num_images))
```

图像包含在 16 个桶（或 TFRecord 文件）中。我们知道从早期的花卉实验中，花卉数据集中有 3670 张图像。前 15 个 TFRecord 文件每个包含 230 张图像，最后一个 TFRecord 文件包含 220 张图像。

显示所有 TFRecord 文件（或桶）：

```py
filenames_tfrds = tf.data.Dataset.list_files(TFR_GCS_PATTERN)
for filename in filenames_tfrds.take(16):
print (filename.numpy())
```

数据项（花卉图像）的数量写在 TFRecord 文件的名字中。例如，TFRecord 文件 b’gs://flowers-public/tfrecords-jpeg-192x192-2/flowers02-*230*.tfrec’ 有 230 个数据项。

### 设置训练参数

设置图像调整大小、管道化和 epoch 的参数：

```py
IMAGE_SIZE = [192, 192]
AUTO = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 64
SHUFFLE_SIZE = 100
EPOCHS = 9
```

设置数据分割和标签的参数：

```py
VALIDATION_SPLIT = 0.19
CLASSES = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']
```

我们为这个实验使用了不同的分割方式，因为我们发现这种方式表现最好。

创建数据分割、验证步骤和每个 epoch 的步骤，如列表 5-11 所示。

```py
split = int(len(tfr_filenames) * VALIDATION_SPLIT)
training_filenames = tfr_filenames[split:]
validation_filenames = tfr_filenames[:split]
print ('Splitting dataset into {} training files and {} '\
'validation files'\
.format(len(tfr_filenames), len(training_filenames),
len(validation_filenames)), end = ' ')
print ('with a batch size of {}.'.format(BATCH_SIZE))
validation_steps = int(3670 // len(tfr_filenames) *\
len(validation_filenames)) // BATCH_SIZE
steps_per_epoch = int(3670 // len(tfr_filenames) *\
len(training_filenames)) // BATCH_SIZE
print ('There are {} batches per training epoch and {} '\
'batches per validation run.'\
.format(BATCH_SIZE, steps_per_epoch, validation_steps))
Listing 5-11
Create Data Splits, Validation Steps, and Steps per Epoch
```

我们首先将数据分割为训练和测试分割。然后显示分割的结果。我们为模型创建验证步骤和每个 epoch 的步骤。最后，我们显示这些信息。

### 创建加载和处理 TFRecord 文件的函数

创建一个函数来解析 TFRecord 文件，如列表 5-12 所示。

```py
def read_tfrecord(example):
features = {
'image': tf.io.FixedLenFeature([], tf.string),
'class': tf.io.FixedLenFeature([], tf.int64)
}
example = tf.io.parse_single_example(example, features)
image = tf.image.decode_jpeg(example['image'], channels=3)
image = tf.cast(image, tf.float32) / 255.0
image = tf.reshape(image, [*IMAGE_SIZE, 3])
class_label = example['class']
return image, class_label
Listing 5-12
Function to Parse a TFRecord File
```

该函数接受来自 TFRecord 文件的示例。一个字典包含 TFRecords 中常见的数据类型。tf.string API 将图像转换为字节字符串（字节列表）。tf.int64 API 将类别标签转换为 64 位整数标量值。示例被解析为（图像，标签）元组。图像元素（一个 JPEG 编码的图像）被解码为一个 uint8 图像张量。图像张量被缩放到 [0, 1] 范围以提高训练速度。然后将其重塑为模型消费的标准大小。类别标签元素被转换为标量。

创建一个函数来将 TFRecord 文件作为 tf.data.Dataset 加载，如列表 5-13 所示。

```py
def load_dataset(filenames):
option_no_order = tf.data.Options()
option_no_order.experimental_deterministic = False
dataset = tf.data.TFRecordDataset(
filenames, num_parallel_reads=AUTO)
dataset = dataset.with_options(option_no_order)
dataset = dataset.map(read_tfrecord, num_parallel_calls=AUTO)
return dataset
Listing 5-13
Function to Load TFRecord Files as a tf.data.Dataset
```

函数接受 TFRecord 文件。为了获得最佳性能，代码中包含了同时读取多个 TFRecord 文件的实现。选项设置允许改变顺序的优化。因此，*n* 个文件并行读取，并为了读取速度而忽略数据顺序。

创建一个函数来增强训练数据：

```py
def data_augment(image, label):
modified = tf.image.random_flip_left_right(image)
modified = tf.image.random_saturation(modified, 0, 2)
return modified, label
```

创建一个函数来从 TFRecord 文件构建输入管道，如列表 5-14 所示。

```py
def get_batched_dataset(filenames, train=False):
dataset = load_dataset(filenames)
dataset = dataset.cache()
if train:
dataset = dataset.map(data_augment, num_parallel_calls=AUTO)
dataset = dataset.repeat()
dataset = dataset.shuffle(SHUFFLE_SIZE)
dataset = dataset.batch(BATCH_SIZE)
dataset = dataset.prefetch(AUTO)
return dataset
Listing 5-14
Build an Input Pipeline from TFRecord Files
```

函数接受 TFRecord 文件并调用 *load_dataset* 函数。函数接着通过缓存、重复、打乱、分批和预取数据集来构建输入管道。重复和打乱仅映射到训练数据。我们遵循最佳实践，只对训练数据进行重复和打乱。

### 创建训练和测试集：

实例化数据集：

```py
training_dataset = get_batched_dataset(
training_filenames, train=True)
validation_dataset = get_batched_dataset(
validation_filenames, train=False)
training_dataset, validation_dataset
```

我们为训练和测试集构建输入管道。

显示图像：

```py
for img, lbl in training_dataset.take(1):
plt.axis('off')
plt.title(CLASSES[lbl[0].numpy()])
fig = plt.imshow(img[0])
tfr_flower_shape = img.shape[1:]
```

我们显示一个图像并保留图像形状以供模型使用。

### 模型数据

清除和设置种子：

```py
tf.keras.backend.clear_session()
np.random.seed(0)
tf.random.set_seed(0)
```

创建一个函数来构建模型，如列表 5-15 所示。

```py
def create_model():
return Sequential([
Conv2D(32, (3, 3), activation = 'relu'),
MaxPooling2D(2, 2),
Conv2D(64, (3, 3), activation='relu'),
MaxPooling2D(2, 2),
Conv2D(128, (3, 3), activation='relu'),
MaxPooling2D(2),
Conv2D(128, (3, 3), activation='relu'),
MaxPooling2D(2, 2),
Flatten(),
Dense(512, activation='relu'),
Dense(num_classes, activation='sigmoid')
])
Listing 5-15
Function to Create the Model
```

在 TPU 策略范围内，创建并编译模型，如列表 5-16 所示。

```py
with tpu_strategy.scope():
flower_model = create_model()
flower_model.compile(
optimizer='adam',
loss=tf.losses.SparseCategoricalCrossentropy(),
metrics=['accuracy'])
Listing 5-16
Create and Compile the Model Within TPU Scope
```

训练模型：

```py
history = flower_model.fit(training_dataset, epochs=EPOCHS,
verbose=1, steps_per_epoch=steps_per_epoch,
validation_steps=validation_steps,
validation_data=validation_dataset)
```

### 进行推理

让我们看看模型预测花朵类别的效果如何！

从验证（测试）集中抓取 40 个预测结果：

```py
preds = flower_model.predict(validation_dataset)[:40]
```

将预测数组转换为标量预测值：

```py
pred_40 = [tf.argmax(i).numpy() for i in preds]
```

创建用于显示的图像和标签：

```py
images, labels = [], []
for img, lbl in validation_dataset.take(1):
for i in range(40):
actual_img = tf.squeeze(img[i])
images.append(actual_img)
labels.append(lbl[i].numpy())
```

创建一个函数来显示预测结果，如列表 5-17 所示。

```py
def display_test(feature, target, num_images,
n_rows, n_cols, cl, p):
for i in range(num_images):
plt.subplot(n_rows, 2*n_cols, 2*i+1)
plt.imshow(feature[i])
pred = cl[p[i]]
actual = cl[int(target[i])]
title_obj = plt.title(actual + ' (' +\
pred + ') ')
if pred == actual:
title_obj
else:
plt.getp(title_obj, 'text')
plt.setp(title_obj, color='r')
plt.tight_layout()
plt.axis('off')
Listing 5-17
Function to Display Predictions
```

调用该函数：

```py
num_rows, num_cols = 10, 4
num_images = num_rows*num_cols
plt.figure(figsize=(20, 20))
display_test(images, labels, num_images, num_rows,
num_cols, CLASSES, pred_40)
```
