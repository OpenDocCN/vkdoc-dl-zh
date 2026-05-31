# 7. 高级转移学习

我们通过基于几个转移学习架构的代码示例介绍了高级转移学习。这些代码示例使用这些架构训练学习模型。

章节的笔记本位于以下 URL：

[`github.com/paperd/deep-learning-models`](https://github.com/paperd/deep-learning-models)

## 转移学习

转移学习基于这样一个理念：网络为某个问题学习到的特征可以用于各种其他任务。在机器学习和模式识别中，**特征**是指被观察现象的个别可测量的属性或特征。因此，有效的学习算法能够从数据中提取出信息丰富、具有区分性和独立性的特征。

转移学习模型非常神奇，因为它们可以在新数据上重用它们所学习的特征！利用转移学习可以节省创建新模型、测试它以及调整它直到它提供所需结果所需的时间。此外，转移学习模型是由经验丰富的数据科学家创建的，他们花费了多年时间调整和优化可用的转移学习模型。

与从头开始训练的模型相比，转移学习模型可以在不花费太多时间收敛的情况下提高准确性。但这并不意味着预训练模型明显优于从头创建的模型。当机器学习模型在训练过程中达到一个状态，损失在最终值周围的误差范围内稳定时，它就达到了收敛。因此，当额外的训练不再提高模型时，模型就收敛了。

在与一位实践中的数据科学家最近的一次交谈中，他告诉我们，他从头开始创建了一些自己的模型。在当前的项目中，他正在使用一个现有的（预训练的）模型。对于其他项目，他则构建自己的模型。这完全取决于任务。他还让我们知道，他更关心收敛性而不是速度。请记住，这个人拥有数据科学博士学位，并在业界拥有多年的经验。

如果一个预训练的神经网络有效，它所学习的特征可以用于其他任务。当人类学习如何执行一项新任务时，我们很少从头开始。我们将我们一生中所学的一切都转移到快速学习新事物上。我们通常可以从单个训练示例中学习。但有时这实际上阻碍了我们的发展。当然，婴儿不会以这种方式学习，因为他们没有相同水平的前置知识。

在前一章中，我们展示了转移学习可以用于它从未见过的数据集。在这一章中，我们介绍了一个新的预训练模型，以展示它是如何与不同的数据集一起工作的。我们还向您展示了如何直接实现转移学习，而不使用 TensorFlow Hub 库。直接实现转移学习提供了更多的灵活性，正如我们在实验中所展示的那样。

我们展示了四个迁移学习实验。我们首先进行豆子实验。然后继续进行斯坦福狗、花朵和剪刀石头布实验。我们坚信熟能生巧。每个实验的代码通常相似，但每个数据集的处理方式都不同。

在我们所有的实验中，我们搭建了 Colab 生态系统。因此，首先导入主要的 TensorFlow 库并实例化 GPU。

## 导入 TensorFlow 库

导入库并将其别名设置为 **tf**：

```py
import tensorflow as tf
```

将 TensorFlow 库别名设置为 tf 是常见的做法。

## GPU 硬件加速器

为了方便，我们提供了在 Colab 笔记本中启用 GPU 的步骤：

1.  点击左上角的菜单中的 *运行*。

1.  从下拉菜单中点击 *更改运行时类型*。

1.  从“硬件加速器”下拉菜单中选择 *GPU*。

1.  点击 *保存*。

验证 GPU 是否已激活：

```py
tf.__version__, tf.test.gpu_device_name()
```

如果显示 “/device:GPU:0”，则 GPU 已激活。如果显示 “..”，则常规 CPU 已激活。

备注

如果你得到错误 **NAME** **‘****TF****’** **IS NOT DEFINED**，重新执行导入 TensorFlow 库的代码！

## 豆子实验

*豆子* 是使用智能手机相机在田间拍摄的豆子植物图像的 TensorFlow 数据集 (TFDS)。它由三个类别组成（bean_rust、angular_leaf_spot、healthy）。其中两个类别是 *angular leaf spot* 和 *bean rust*，这两种都是可能影响豆子植物的疾病。第三个类别是 *healthy*。因此，在这个数据集中，豆子植物要么是健康的，要么患有这两种疾病中的一种。数据由乌干达国家作物资源研究学院 (NaCRRI) 的专家标注，并由 Makerere AI 研究实验室收集。

我们使用两个预训练模型在豆子数据上训练。其中一个模型对你来说是新的，另一个是之前章节中演示的 Inception 模型。我们展示两者以供对比。即使其中一个模型在这个数据集上比另一个表现更好，也不能保证它在另一个数据集上也会有同样的表现！但一旦你习惯了与预训练模型一起工作，使用它们就相当直接了。所以尝试不同的模型可能是一个不错的策略。

### 加载豆子

从 Google Cloud 服务 (GCS) 加载豆子作为 TFDS：

```py
import tensorflow_datasets as tfds
beans, beans_info = tfds.load(
'beans', with_info=True, as_supervised=True,
try_gcs=True)
```

虽然不是必需的，但我们建议从 GCS 加载 TFDS。

### 探索数据

显示元数据：

```py
beans_info
```

为了简单起见，将分割分配给变量：

```py
train = beans['train']
valid = beans['validation']
test = beans['test']
```

从元数据中，我们知道训练数据有 1,034 个示例，验证数据有 133 个示例，测试数据有 128 个示例。

获取标签和类别数量：

```py
class_labels = beans_info.features['label'].names
num_classes = beans_info.features['label'].num_classes
class_labels, num_classes
```

检查图像大小：

```py
for img, lbl in train.take(10):
print (img.shape)
```

从样本中可以看出，所有图像都是 500 × 500 × 3。

### 可视化

使用 *show_examples* 进行可视化：

```py
fig = tfds.show_examples(train, beans_info)
```

### 重新格式化图像

调整并处理图像以适应 Xception 模型：

```py
def preprocess(image, label):
resized_image = tf.image.resize(image, [224, 224])
final_image = tf.keras.applications.xception.\
preprocess_input(resized_image)
return final_image, label
```

我们将图像调整大小为 224 × 224，这是 Xception 图像的预期大小。我们使用 Xception 预处理 API 预处理调整大小的图像，并返回最终图像和标签。该函数将在下一节中使用。

### 构建输入管道

将训练、验证和测试数据转换为 TensorFlow 可消费对象：

```py
BATCH_SIZE = 32
shuffle = 250
train_ds = train.shuffle(shuffle).\
map(preprocess).batch(BATCH_SIZE).prefetch(1)
valid_ds = valid.map(preprocess).batch(BATCH_SIZE).prefetch(1)
test_ds = test.map(preprocess).batch(BATCH_SIZE).prefetch(1)
```

如列表 7-1 所示进行可视化。

```py
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 12))
for img, lbl in train_ds.take(1):
for index in range(9):
plt.subplot(3, 3, index + 1)
plt.imshow(img[index] / 2 + 0.5)
plt.title(class_labels[lbl[index]])
plt.axis('off')
Listing 7-1
Visualize Training Examples
```

我们为当前任务提供不同的可视化代码。您可以自由创建自己的可视化代码，以获得 Python 数据科学任务的实践经验。

### 使用 Xception 模型

我们引入 Xception 预训练模型以拓宽您的经验。此外，Xception 是最近公开的最新模型。

Xception 是六个在 ImageNet 数据集上预训练的最先进图像分类器之一。其他五个是 MobileNet、VGG16、VGG19、ResNet50 和 Inception-v3。Xception 在 ImageNet 数据集上略优于 Inception-v3，在包含 17,000 个类别（或更多）的更大图像分类数据集上则远远优于它。最重要的是，它与 Inception 具有相同数量的模型参数，这意味着更高的计算效率。VGG16 和 VGG19 具有较小的卷积层，但训练时间比 Xception 长得多。ResNet 模型大小相当，但 Xception 训练更快，并产生更好的训练结果。

Xception 模型由 Francois Chollet 于 2017 年提出。*Xception*是 Inception 架构的扩展，它用深度可分离卷积替换了标准的 Inception 模块。Xception 在 ImageNet 上进行了预训练。Xception 通常优于 VGGNet、ResNet 和 Inception-v3 模型。作为旁注，Chollet 也是 Keras 的作者。

#### 创建模型

清除并设置随机种子：

```py
import numpy as np
tf.keras.backend.clear_session()
np.random.seed(0)
tf.random.set_seed(0)
```

创建一个带有 Xception 的基础模型：

```py
Xception = tf.keras.applications.xception.Xception
xception_model = Xception(
weights='imagenet', include_top=False)
```

通过设置 *include_top=False* 排除网络的顶层。这意味着全局平均池化层和 Dense 输出层将不被包含在 Xception 中，因此我们必须在分类头中包含这两个层。

我们排除了顶层以加快训练速度。然而，使用所有层进行训练可能会增加学习。我们将在本章后面向您展示如何做到这一点。

探索基础模型层：

```py
tf.keras.utils.plot_model(
xception_model,
show_shapes=True,
show_layer_names=True)
```

tf.keras.utils.plot_model API 提供了 Xception 模型的详细图形描述。

获取层数：

```py
len(xception_model.layers)
```

Xception 有 132 层。

导入库：

```py
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout,\
GlobalAveragePooling2D
```

构建最终模型（分类头）：

```py
x_model = tf.keras.Sequential([
xception_model,
GlobalAveragePooling2D(),
Dropout(0.5),
Dense(num_classes, activation='softmax')])
```

我们排除了预训练网络中的顶层，该层包含全局平均池化层和 Dense 输出层。因此，我们必须添加我们自己的全局平均池化层和一个具有三个类别和 softmax 激活的 Dense 输出层。

获取最终模型的布局：

```py
tf.keras.utils.plot_model(
x_model,
show_shapes=True,
show_layer_names=True)
```

注意我们最终模型的简洁性！

#### 模型化数据

冻结预训练层的权重：

```py
for layer in xception_model.layers:
layer.trainable = False
```

我们添加前面的步骤来通知编译器我们的意图是冻结顶层。

编译：

```py
optimizer = tf.keras.optimizers.SGD(
lr=0.2, momentum=0.9, decay=0.01)
x_model.compile(
loss='sparse_categorical_crossentropy',
optimizer=optimizer,
metrics=['accuracy'])
```

训练：

```py
history = x_model.fit(
train_ds, validation_data=valid_ds, epochs=10)
```

注意训练时间是最小的。

创建一个函数来可视化训练性能，如列表 7-2 所示。

```py
def visualize(span):
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = span
plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')
plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
Listing 7-2
Visualization Function
```

调用：

```py
visualize(range(10))
```

注意

确保向可视化函数提供与模型训练的 epoch 数量相等的范围。

我们设置了一个非常激进的学习率，但仍然获得了可尊敬的性能。尝试不同的学习率以查看是否可以提高性能。

备注

将学习率设置得较高通常可以使模型更快地学习。然而，代价可能是一个次优的最终权重集。将学习率设置得较低可能允许模型学习到一个更优甚至全局最优的权重集，但可能需要更长的时间来训练。

#### 使用未冻结层的训练数据模型

验证准确率相当不错，但没有进一步提高。所以顶层训练得相当好。也就是说，准确率达到了平台期。

现在我们准备解冻所有层并继续训练，如列表 7-3 所示。

```py
for layer in xception_model.layers:
layer.trainable = True
optimizer = tf.keras.optimizers.SGD(
learning_rate=0.01, momentum=0.9,
nesterov=True, decay=0.001)
x_model.compile(
loss='sparse_categorical_crossentropy',
optimizer=optimizer, metrics=['accuracy'])
history = x_model.fit(
train_ds, validation_data=valid_ds, epochs=10)
Listing 7-3
Continue Training with All Layers Unfrozen
```

初始时，一个好的策略是冻结顶层以加快训练速度。设置高学习率可以进一步提高训练速度。一旦验证准确率停滞不前，我们就知道顶层已经训练好了。然后我们可以通过解冻顶层并设置较低的学习率来继续训练。我们使用*较低的学习率*以避免损坏预训练的权重。

我们仍然在同一个模型上进行训练，直到我们清除模型会话。这就是为什么我们在训练新模型之前清除模型会话的原因。

可视化：

```py
visualize(range(10))
```

### 使用 Inception 对模型进行 Bean 化

让我们看看 Inception 与 Xception 的比较。在本节中，我们展示了如何轻松地交换预训练模型。

*Inception-v3*是一个预训练的卷积神经网络模型，深度为 48 层。它是已经在 ImageNet 数据库中超过一百万张图像上训练过的网络的版本。预训练的网络可以将图像分类到 1000 个对象类别中，如键盘、鼠标、铅笔和许多动物。

#### 创建模型

清除并设置随机种子：

```py
tf.keras.backend.clear_session()
np.random.seed(0)
tf.random.set_seed(0)
```

创建基础模型：

```py
inception_v3 = tf.keras.applications.InceptionV3
inception_model = inception_v3(
include_top=False, weights='imagenet',
input_shape=(224, 224, 3))
```

探索基础模型层：

```py
tf.keras.utils.plot_model(
inception_model,
show_shapes=True,
show_layer_names=True)
```

获取层数数量：

```py
len(inception_model.layers)
```

呼！Inception 非常大（311 层）且复杂！

创建最终模型：

```py
i_model = tf.keras.Sequential([
inception_model,
GlobalAveragePooling2D(),
Dropout(0.5),
Dense(num_classes, activation='softmax')])
```

我们可以在形状为(224, 224, 3)的情况下进行训练，因为 include_top 是 False。否则，输入形状必须是(299, 299, 3)以适应 Inception。

#### 模型数据

冻结预训练层的权重：

```py
for layer in inception_model.layers:
layer.trainable = False
```

编译：

```py
optimizer = tf.keras.optimizers.RMSprop(lr=0.1)
i_model.compile(
loss='sparse_categorical_crossentropy',
optimizer=optimizer, metrics=['accuracy'])
```

训练：

```py
history = i_model.fit(
train_ds, validation_data=valid_ds, epochs=10)
```

可视化：

```py
visualize(range(10))
```

#### 使用未冻结层的训练数据模型

解冻所有层并继续训练，如列表 7-4 所示。

```py
for layer in inception_model.layers:
layer.trainable = True
optimizer = tf.keras.optimizers.RMSprop(lr=0.0001)
i_model.compile(
loss='sparse_categorical_crossentropy',
optimizer=optimizer, metrics=['accuracy'])
history = i_model.fit(
train_ds, validation_data=valid_ds, epochs=10)
Listing 7-4
Continue Training with All Layers Unfrozen
```

备注

自适应梯度下降算法包括 Adagrad、Adadelta、RMSprop 和 Adam，为经典的 SGD 算法提供了一种替代方案。自适应算法提供了一种启发式方法，可以自动调整超参数，从而避免了手动调整学习率调度超参数时的昂贵工作。

可视化：

```py
visualize(range(10))
```

#### 在未见数据上泛化

在 Xception 的未见测试数据集上进行泛化：

```py
x_model.evaluate(test_ds)
```

在 Inception 的未见测试数据集上进行泛化：

```py
i_model.evaluate(test_ds)
```

## 斯坦福狗实验

我们已经展示了大数据集实验，但其中没有一个包含大量的类别标签。*斯坦福狗* 数据集包含来自世界各地的 120 种狗的图像。该数据集使用来自 ImageNet 的图像和注释构建，用于细粒度图像分类任务。斯坦福狗包含 20,580 张图像，分为 12,000 张训练图像和 8,580 张测试图像。为所有 12,000 张图像提供了类别标签和边界框注释。

### 使用 MobileNet 对斯坦福狗进行建模

*MobileNets* 是小型、低延迟、低功耗的模型，参数化以满足各种用例的资源限制。预训练模型被认为是低功耗模型，因为它们消耗很少的计算机资源。原因是它们已经过训练。初始训练确实消耗了大量的计算机资源。但一旦训练完成，它们消耗的计算机资源非常少。它们可以用于分类、检测、嵌入和分割，类似于其他流行的规模化模型的操作，如 Inception。

*MobileNet-v2* 模型是在谷歌开发的。它在 ImageNet 数据集上预训练，这是一个包含 140 万张图像和 1,000 个类别的庞大数据集。*ImageNet* 是一个具有广泛类别（如芒果和注射器）的研究训练数据集。其基础知识帮助我们对我们特定的数据集中的狗进行分类。

### 加载数据

从斯坦福狗训练数据（训练分割）中加载训练集：

```py
train_pups, dogs_info = tfds.load(
'stanford_dogs', with_info=True,
as_supervised=True, try_gcs=True,
split='train')
```

从斯坦福狗测试数据（测试分割）中加载验证集和测试集，每个分割 50%：

```py
(validation_pups, test_pups) = tfds.load(
'stanford_dogs',
split=['test[:50%]', 'test[50%:]'],
as_supervised=True, try_gcs=True)
```

### 元数据

显示元数据：

```py
dogs_info
```

### 可视化示例

创建一个函数来从整数标签获取 *命名* 标签：

```py
get_name = dogs_info.features['label'].int2str
```

通过试错，我们得到了所有的整数标签并将它们转换为命名标签：

```py
lbls = []
for image, label in train_pups.take(464):
lbls.append(get_name(label))
set_lbl = set(lbls)
len(set_lbl)
```

变量 *set_lbl* 存储数据集的类别标签。

从元数据中我们知道斯坦福狗有 120 个类别。因此，我们通过试错调整了示例的数量，直到我们得到了 120 个独特的标签！

捕获一些图像和标签进行可视化：

```py
img, lbl = [], []
for image, label in train_pups.take(9):
img.append(image)
lbl.append(get_name(label)[10:])
```

显示第一个：

```py
lbl[0]
```

按照列表 7-5 所示显示一些示例。

```py
plt.figure(figsize=(12, 12))
for index in range(9):
plt.subplot(3, 3, index + 1)
plt.imshow(img[index])
plt.title(lbl[index])
plt.axis('off')
Listing 7-5
Display Examples
```

以简单的方式显示示例：

```py
fig = tfds.show_examples(train_pups, dogs_info)
```

### 检查图像形状

取一些示例并显示图像形状：

```py
for img, lbl in train_pups.take(10):
print (img.shape)
```

由于图像大小不一，我们必须调整它们的大小：

### 构建输入管道

获取类别数量：

```py
num_classes = dogs_info.features['label'].num_classes
num_classes
```

我们已经知道类别的数量，但我们将向您展示如何用一行代码获取这个数字。

创建管道变量：

```py
IMG_LEN = 224
IMG_SHAPE = (IMG_LEN,IMG_LEN,3)
N_BREEDS = num_classes
```

按照列表 7-6 创建预处理函数。

```py
def preprocess(img, lbl):
resized_image = tf.image.resize(img, [IMG_LEN, IMG_LEN])
final_image = tf.keras.applications.mobilenet.preprocess_input(
resized_image)
label = tf.one_hot(lbl, N_BREEDS)
return final_image, label
Listing 7-6
Preprocessing Function
```

创建一个函数来构建输入管道，如列表 7-7 所示。

```py
def prepare(dataset, batch_size=None, shuffle_size=None):
ds = dataset.map(preprocess, num_parallel_calls=4)
ds = ds.shuffle(buffer_size=1000)
if batch_size:
ds = ds.batch(batch_size)
if shuffle_size:
ds = ds.shuffle(shuffle_size)
ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
return ds
Listing 7-7
Function to Build the Input Pipeline
```

构建管道：

```py
BATCH_SIZE = 32
SHUFFLE_SIZE = 1000
train_dogs = prepare(train_pups, batch_size=BATCH_SIZE,
shuffle_size=SHUFFLE_SIZE)
validation_dogs = prepare(validation_pups, batch_size=32)
test_dogs = prepare(test_pups, batch_size=32)
```

### 创建模型

创建基础模型：

```py
mobile_v2 = tf.keras.applications.MobileNetV2
mobile_model = mobile_v2(
input_shape=IMG_SHAPE, include_top=False,
weights='imagenet')
```

探索基础模型层：

```py
tf.keras.utils.plot_model(
mobile_model,
show_shapes=True,
show_layer_names=True)
```

获取层数：

```py
len(mobile_model.layers)
```

清除和设置随机种子：

```py
tf.keras.backend.clear_session()
np.random.seed(0)
tf.random.set_seed(0)
```

创建模型（冻结顶层）：

```py
mobile_model.trainable = False
sd_model = tf.keras.Sequential([
mobile_model,
GlobalAveragePooling2D(),
Dropout(0.5),
Dense(num_classes, activation='softmax')
])
```

### 编译和训练

按照列表 7-8 所示编译和训练模型。

```py
EPOCHS = 5
sd_model.compile(
optimizer=tf.keras.optimizers.Adamax(learning_rate=0.005),
loss='categorical_crossentropy',
metrics=['accuracy', 'top_k_categorical_accuracy'])
history = sd_model.fit(
train_dogs, epochs=EPOCHS, validation_data=validation_dogs)
Listing 7-8
Compile and Train
```

由于我们训练了更多的图像和更多的类别，因此训练时间要长得多。所以请耐心等待。如果你的电脑崩溃，请不要担心，这不是错误。需要更多的 RAM。我们使用 Colab Pro，并且似乎没有问题。

小贴士

如果你想要对现实世界数据进行建模，你可能愿意每月花费几美元迁移到 Colab Pro。

可视化：

```py
visualize(range(EPOCHS))
```

可视化前五个预测结果，如列表 7-9 所示。

```py
acc = history.history['top_k_categorical_accuracy']
val_acc = history.history['val_top_k_categorical_accuracy']
epochs_range = range(EPOCHS)
plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Top 5 Training and Validation Accuracy')
plt.grid(b=None)
Listing 7-9
Top Five Predictions
```

还不错！我们在品种检测中获得了超过 80% 的准确率。如果我们查看前五个预测，猜测正确品种的机会跃升至超过 97%！由于数据集庞大且复杂，我们展示了使用预训练模型可以具有实际应用场景。而且令人惊讶的是，我们只需几行代码就能构建如此强大的模型！

### 使用未冻结的层训练模型

解冻所有层并继续训练，如列表 7-10 所示。

```py
mobile_model.trainable = True
sd_model.compile(
optimizer=tf.keras.optimizers.Adamax(0.00001),
loss='categorical_crossentropy',
metrics=['accuracy', 'top_k_categorical_accuracy'])
history = sd_model.fit(
train_dogs, epochs=3,
validation_data=validation_dogs)
Listing 7-10
Unfreeze All Layers and Continue Training
```

我们使用一个*较低的学习率*来避免损坏预训练的权重。我们只运行三个周期，因为训练时间非常长。

注意

对于像斯坦福狗那样的现实世界用例，设置非常低的学习率。

可视化：

```py
visualize(range(3))
```

### 归纳

从未见过的数据中归纳：

```py
sd_model.evaluate(test_dogs)
```

## 花朵实验

让我们处理一个存储为 TFRecords 的数据集。我们坚信，使用各种数据集进行实践有助于学习。

由于我们在前面的章节中处理过花朵，我们不会在本节中描述数据集。我们以 TFRecords 格式加载花朵，并使用预训练模型进行学习。

### 读取作为 TFRecords 格式的花朵

从 GCS 读取 TFRecord 文件：

```py
piece1 = 'gs://flowers-public/'
piece2 = 'tfrecords-jpeg-192x192-2/*.tfrec'
TFR_GCS_PATTERN = piece1 + piece2
tfr_filenames = tf.io.gfile.glob(TFR_GCS_PATTERN)
```

### 创建数据拆分

设置管道和训练参数：

```py
IMAGE_SIZE = [192, 192]
AUTO = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 64
SHUFFLE_SIZE = 100
EPOCHS = 5
VALIDATION_SPLIT = 0.19
CLASSES = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']
```

创建如列表 7-11 所示的拆分。

```py
split = int(len(tfr_filenames) * VALIDATION_SPLIT)
training_filenames = tfr_filenames[split:]
validation_filenames = tfr_filenames[:split]
print ('Splitting dataset into {} training files and {}'
'validation files'.\
format(
len(tfr_filenames), len(training_filenames),
len(validation_filenames)), end = ' ')
print ('with a batch size of {}.'.format(BATCH_SIZE))
validation_steps = int(3670 // len(tfr_filenames) *\
len(validation_filenames)) // BATCH_SIZE
steps_per_epoch = int(3670 // len(tfr_filenames) *\
len(training_filenames)) // BATCH_SIZE
print ('There are {} batches per training epoch and {} '\
'batches per validation run.'\
.format(BATCH_SIZE, steps_per_epoch, validation_steps))
Listing 7-11
Create Splits
```

### 创建加载和处理 TFRecord 文件的函数

如列表 7-12 所示演示 one-hot 编码。

```py
named_lbl = 'sunflowers'
indx = CLASSES.index(named_lbl)
encode = tf.one_hot([indx], 5)
one_hot = encode[0].numpy()
print ('encoded label:', one_hot)
pos = tf.math.argmax(one_hot).numpy()
print ('integer label:', pos)
Listing 7-12
Demonstrate One-Hot Encoding
```

使用 tf.one_hot() API 对命名标签进行编码。使用 tf.math.argmax() API 将 one-hot 编码的标签转换为整数标签。

创建一个函数，解析 TFRecord 文件，如列表 7-13 所示。

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
one_hot = tf.one_hot(class_label, 5)
return image, one_hot
Listing 7-13
Function to Parse a TFRecord File
```

创建一个函数，将 TFRecord 文件加载为 tf.data.Dataset，如列表 7-14 所示。

```py
def load_dataset(filenames):
option_no_order = tf.data.Options()
option_no_order.experimental_deterministic = False
dataset = tf.data.TFRecordDataset(
filenames, num_parallel_reads=AUTO)
dataset = dataset.with_options(option_no_order)
dataset = dataset.map(read_tfrecord, num_parallel_calls=AUTO)
return dataset
Listing 7-14
Function to Load TFRecords as a tf.data.Dataset
```

创建一个函数，从 TFRecord 文件构建输入管道，如列表 7-15 所示。

```py
def get_batched_dataset(filenames, train=False):
dataset = load_dataset(filenames)
dataset = dataset.cache()
if train:
dataset = dataset.repeat()
dataset = dataset.shuffle(SHUFFLE_SIZE)
dataset = dataset.batch(BATCH_SIZE)
dataset = dataset.prefetch(AUTO)
return dataset
Listing 7-15
Function to Build an Input Pipeline from TFRecord Files
```

### 创建训练和测试集

实例化数据集：

```py
training_dataset = get_batched_dataset(
training_filenames, train=True)
validation_dataset = get_batched_dataset(
validation_filenames, train=False)
training_dataset, validation_dataset
```

显示图像，如列表 7-16 所示。

```py
for img, lbl in training_dataset.take(1):
plt.axis('off')
label = tf.math.argmax(lbl[0]).numpy()
plt.title(CLASSES[label])
fig = plt.imshow(img[0])
tfr_flower_shape = img.shape[1:]
Listing 7-16
Display an Image
```

### 创建模型

创建预训练模型列表：

```py
ptm =\
[tf.keras.applications.MobileNetV2,
tf.keras.applications.VGG16,
tf.keras.applications.MobileNet,
tf.keras.applications.xception.Xception,
tf.keras.applications.InceptionV3,
tf.keras.applications.ResNet50]
```

通过索引选择任何预训练模型。我们使用 Xception 创建基础模型：

```py
pre_trained_model = ptm3
```

清除和设置随机种子：

```py
tf.keras.backend.clear_session()
np.random.seed(0)
tf.random.set_seed(0)
```

创建最终模型：

```py
pre_trained_model.trainable = True
flower_model = tf.keras.Sequential([
pre_trained_model,
GlobalAveragePooling2D(),
Dense(5, activation='softmax')])
```

我们使用 Xception 预训练模型。我们移除了 ImageNet 特定的顶层，设置 *include_top=false*，并添加一个最大池化和一个 softmax 层来预测五种花朵类别。请注意，我们还解冻了所有顶层！不要一开始就尝试使用更大的数据集解冻所有层！

### 编译和训练

编译：

```py
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
flower_model.compile(
optimizer=optimizer,
loss = 'categorical_crossentropy',
metrics=['accuracy'])
```

初始学习率通常是单个最重要的超参数。如果只能调整一个超参数，那么学习率就是值得调整的。方便的是，我们可以使用 *Adam* 优化器来自动调整学习率！但我们仍然必须设置初始学习率。

我们设置了一个非常低的学习率，以便模型（希望）学习到一个更优甚至全局最优的权重集。但训练时间会显著增加。我们将学习率设置得很低，以确保梯度下降不会增加训练错误。最初，我们希望允许神经网络随机调整其权重。较低的学习率增加了随机性。较高的学习率减少了随机性。

训练：

```py
history = flower_model.fit(
training_dataset, epochs=EPOCHS,
verbose=1, steps_per_epoch=steps_per_epoch,
validation_steps=validation_steps,
validation_data=validation_dataset)
```

### 可视化

可视化训练性能：

```py
visualize(range(EPOCHS))
```

### 泛化

我们在验证集上泛化，因为我们没有划分测试集：

```py
flower_model.evaluate(validation_dataset)
```

## 石头-剪刀-布实验

最后的实验是使用 *rock_paper_scissors* 数据集。该数据集是从石头、剪刀、布游戏中改编的。数据包含玩石头、剪刀、布游戏的手部图像。

掷骰子、剪刀、布是一种通常由两个人玩的手游戏，每个玩家同时用伸展的手形成三种形状之一。可能的形状是石头、布和剪刀。规则很简单。石头胜剪刀。布胜石头。剪刀胜布。比喻来说，石头砸剪刀，布包石头，剪刀剪布。

### 加载数据

加载训练集：

```py
train_digits, rps_info = tfds.load(
'rock_paper_scissors', with_info=True,
split='train', as_supervised=True,
try_gcs=True)
```

加载测试集：

```py
test_digits = tfds.load(
'rock_paper_scissors',  try_gcs=True,
as_supervised=True, split='test')
```

检查张量：

```py
for image, label in train_digits.take(5):
print (image.shape, label.numpy())
```

显示元数据：

```py
rps_info
```

### 可视化

显示一些示例：

```py
fig = tfds.show_examples(train_digits, rps_info)
```

### 构建输入管道

创建一个函数来处理图像和标签，如列表 7-17 所示。

```py
def process_digits(image, label):
resized_image = tf.image.resize(image, [224, 224])
final_image = tf.keras.applications.xception.\
preprocess_input(resized_image)
one_hot = tf.one_hot(label, 3)
return final_image, one_hot
Listing 7-17
Function to Process Data
```

预期标签作为 one-hot 编码的对象。

构建管道：

```py
BATCH_SIZE = 64
shuffle = 250
train_fingers = train_digits.shuffle(shuffle).\
map(process_digits).batch(BATCH_SIZE).prefetch(1)
test_fingers = test_digits.map(process_digits).\
batch(BATCH_SIZE).prefetch(1)
```

### 创建模型

创建基础模型：

```py
Xception = tf.keras.applications.xception.Xception
xception_model = Xception(
weights='imagenet', include_top=False)
```

清除和设置种子：

```py
tf.keras.backend.clear_session()
np.random.seed(0)
tf.random.set_seed(0)
```

创建最终模型：

```py
pre_trained_model.trainable = True
fingers_model = tf.keras.Sequential([
xception_model,
GlobalAveragePooling2D(),
Dense(3, activation='softmax')])
```

解冻所有层！

### 编译和训练

编译：

```py
optimizer = tf.keras.optimizers.Adam(learning_rate=0.00001)
fingers_model.compile(
optimizer=optimizer,
loss = 'categorical_crossentropy',
metrics=['accuracy'])
```

由于我们解冻了所有层，因此设置一个非常低的学习率，以便在网络在早期训练时期随机平衡神经元权重。

训练：

```py
history = fingers_model.fit(
train_fingers, epochs=10,
validation_data=test_fingers)
```

还不错！

### 可视化

可视化训练性能：

```py
visualize(range(10))
```

### 泛化

在测试数据上泛化：

```py
fingers_model.evaluate(test_fingers)
```

## 技巧和概念

要获取更多调整迁移学习模型的技巧，请参阅

[使用 ResNet50 进行 Keras 迁移学习的指南](https://medium.com/@kenneth.ca95/a-guide-to-transfer-learning-with-keras-using-resnet50-a81a4a28084b)

要全面（但易于阅读）地了解该主题，请参阅

[使用深度学习中的实际应用全面动手指南进行迁移学习](https://towardsdatascience.com/a-comprehensive-hands-on-guide-to-transfer-learning-with-real-world-applications-in-deep-learning-212bf3b2f27a)
