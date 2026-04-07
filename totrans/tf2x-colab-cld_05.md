# 5. 分类

**分类** 是一种监督学习方法，用于预测给定输入数据的类别标签。尽管我们通过 MNIST 介绍了分类，但我们通过著名的 Fashion-MNIST 数据集深入研究这一主题。

*Fashion-MNIST* 旨在作为 MNIST 的直接替代品，以更好地基准测试机器学习算法。它具有相同的图像大小和训练/测试分割的结构，但是一个更具挑战性的分类问题。

MNIST 基准测试有几个相关的问题。对于标准机器学习算法来说，达到超过 97% 的准确率太容易了。对于深度学习模型来说，达到超过 99% 的准确率则更加容易。数据集被过度使用。最后，MNIST 无法代表现代计算机视觉任务。

笔记本位于以下 URL：[`https://github.com/paperd/tensorflow`](https://github.com/paperd/tensorflow)。

启用 GPU（如果尚未启用）：

1.  在右上角菜单中点击 *运行时*。

1.  从下拉菜单中选择 *更改运行时类型*。

1.  从 *硬件加速器* 下拉菜单中选择 *GPU*。

1.  点击 *保存*。

测试 GPU 是否激活：

```py
import tensorflow as tf
# display tf version and test if GPU is active
tf.__version__, tf.test.gpu_device_name()
```

导入 *tensorflow* 库。如果显示 ‘/device:GPU:0’，则表示 GPU 正在运行。如果显示 ‘..’，则表示常规 CPU 正在运行。

## Fashion-MNIST 数据集

**Fashion-MNIST** 是由 Zalando Research 创建的服装图像数据集，包含 60,000 个示例的训练集和 10,000 个示例的测试集。每个示例是一个与十个类别之一的标签关联的 28 × 28 灰度图像。

Zalando Research 是一个利用敏捷设计流程的组织，该流程结合了宝贵的人类经验和机器学习的力量。Zalando 致力于探索在时尚设计中使用生成模型的新方法，以实现快速可视化和原型设计。

## 将 Fashion-MNIST 加载为 TFDS

由于 Fashion-MNIST 是一个 tfds.data.Dataset，我们可以很容易地使用 tfds.load 加载它。要获取所有 TFDS 的列表，只需运行第三章中演示的 tfds.list_builders 方法即可。

将训练和测试示例作为 tf.data.Dataset 加载：

```py
import tensorflow_datasets as tfds
train, info = tfds.load('fashion_mnist', split="train",
with_info=True, shuffle_files=True)
test = tfds.load('fashion_mnist', split="test")
```

由于我们已经有训练数据的 *info*，因此不需要再次加载测试数据。

验证训练和测试数据：

```py
train.element_spec, test.element_spec
```

每个图像由 28 `×` 28 像素组成。*1* 维度表示图像是灰度的。每个标签都是一个标量。

### 探索数据集

显示数据集信息：

```py
info
```

我们可以看到名称、描述和主页。我们还可以看到特征图像和标签的形状和数据类型。我们还可以看到有 70,000 个示例，其中训练和测试分割分别为 60,000 和 10,000。还包括许多其他信息。

提取类别数量和类别标签：

```py
br = '\n'
num_classes = info.features['label'].num_classes
class_labels = info.features['label'].names
print ('number of classes:', num_classes, br)
print ('class labels:', class_labels)
```

我们有十个类别，代表十种服装。

让我们看看一些训练示例：

```py
fig = tfds.show_examples(train, info)
```

show_examples 方法显示样本图像和标签。每个图像下方显示标签名称及其关联的类号。例如，*Pullover* 的类号是 *2*，因为它是在类标签列表中的第三个标签。

列表 5-1 是一个自定义函数，用于显示训练数据集的样本。

```py
import matplotlib.pyplot as plt, numpy as np
def display_samples(data, num, cmap):
for example in data.take(num):
image, label = example['image'], example['label']
print ('Label:', class_labels[label.numpy()], end=', ')
print ('Index:', label.numpy())
plt.imshow(image.numpy()[:, :, 0].astype(np.float32),
cmap=plt.get_cmap(cmap))
plt.show()
Listing 5-1
Custom function for displaying sample data
```

导入几个库。该函数接受数据集、要显示的样本数量和颜色图。**颜色图**是一组颜色，用于将像素数据映射到实际的颜色值。matplotlib 库提供了多种内置颜色图。

我们将一个示例图像及其标签分配给变量。然后显示标签名称及其关联的类号。最后，使用 *imshow()* 函数显示图像。我们使用 *[:, :, 0]* 来获取每个图像的所有像素。

调用该函数以显示几幅训练图像：

```py
# choose colormap by changing 'indx'
indx = 5
cmap = ['coolwarm', 'viridis', 'plasma',
'seismic', 'copper', 'twilight']
samples = 2
display_samples(train, samples, cmap[indx])
```

现在，让我们构建一个自定义函数来显示样本网格。

在创建函数之前，从训练集中取出 30 个样本，如下所示：5-2。

```py
num = 30
images, labels = [], []
for example in train.take(num):
image, label = example['image'], example['label']
images.append(tf.squeeze(image.numpy()))
labels.append(tf.squeeze(label.numpy()))
Listing 5-2
Take samples from the train set
```

按照如下所示创建函数：5-3。

```py
def display_grid(feature, target, n_rows, n_cols, cl):
plt.figure(figsize=(n_cols * 1.5, n_rows * 1.5))
for row in range(n_rows):
for col in range(n_cols):
index = n_cols * row + col
plt.subplot(n_rows, n_cols, index + 1)
plt.imshow(feature[index], cmap="binary",
interpolation='nearest')
plt.axis('off')
plt.title(cl[target[index]], fontsize=12)
plt.subplots_adjust(wspace=0.2, hspace=0.5)
Listing 5-3
Function that displays a grid of examples
```

调用该函数：

```py
rows = 5
cols = 6
display_grid(images, labels, rows, cols, class_labels)
```

我们还可以使用 DatasetInfo 精确定位元数据：

```py
print ('Number of training examples:', end=' ')
print (info.splits['train'].num_examples)
print ('Number of test examples:', end=' ')
print (info.splits['test'].num_examples)
```

### 构建输入管道

按照如下所示构建训练和测试数据的输入管道：5-4。

```py
BATCH_SIZE = 128
SHUFFLE_SIZE = 5000
train_f1 = train.shuffle(SHUFFLE_SIZE).batch(BATCH_SIZE)
train_f2 = train_f1.map(lambda items: (
tf.cast(items['image'], tf.float32) / 255., items['label']))
train_fs = train_f2.cache().prefetch(1)
test_f1 = test.batch(BATCH_SIZE)
test_f2 = test_f1.map(lambda items: (
tf.cast(items['image'], tf.float32) / 255., items['label']))
test_fs = test_f2.cache().prefetch(1)
Listing 5-4
Build the input pipeline
```

洗牌并批处理训练数据。缩放训练图像。缓存并预取训练图像。批处理测试数据。缩放测试图像。缓存并预取测试图像。在此实验中使用 128 个批大小和 5,000 个洗牌缓冲区大小。

注意

不要对测试数据进行洗牌，因为它被认为对神经网络模型来说是新的。

缓存 TFDS 可以显著提高性能。tf.data.Dataset 的 *cache* 方法可以将数据集缓存到内存或本地存储中，从而避免在每个 epoch 期间执行文件打开和数据读取等操作。

添加预取是一个好主意，因为它可以提高批处理过程的效率。当我们的训练算法正在处理一个批次时，TensorFlow 正在并行处理数据集以准备下一个批次。因此，预取可以显著提高训练性能。

关于 TFDS 性能提升的更多信息，请参阅

+   [`www.tensorflow.org/datasets/performances`](http://www.tensorflow.org/datasets/performances)

+   [`www.tensorflow.org/guide/data_performance`](http://www.tensorflow.org/guide/data_performance)

验证训练和测试张量：

```py
train_fs.element_spec, test_fs.element_spec
```

训练和测试图像都是 28 `×` 28 `×` 1。1 的值表示图像是灰度的。也就是说，图像是黑白两色的。

### 构建模型

让我们构建一个简单的前馈神经网络，如下所示：5-5。

```py
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
# clear previous model and generate a seed
tf.keras.backend.clear_session()
np.random.seed(0)
tf.random.set_seed(0)
model = Sequential([
Flatten(input_shape=[28, 28, 1]),
Dense(512, activation="relu"),
Dropout(0.4),
Dense(10, activation="softmax")
])
Listing 5-5
Simple feedforward neural network
```

导入所需的库。清除任何先前的模型会话并生成一个种子以促进结果的重复性。第一层将图像展平。第二层使用 *relu* 激活对 512 个神经元进行数据处理。第三层使用 dropout 来减少过拟合。第四层使用 softmax 激活对十个神经元进行分类标签的处理。

### 模型摘要

显示模型的摘要：

```py
model.summary()
```

第一层的输出形状是 (None, 784)。我们通过将 28 乘以 28 得到 784。在这一层我们没有参数，因为它仅用于将数据带入模型。

第二层的输出形状是 (None, 512)，因为我们在这层有 512 个神经元。通过将这层的 512 个神经元乘以前一层的 784 个神经元，并加上这层的 512 个神经元，我们得到 401,920 个参数。

第三层的输出形状是 (None, 512)，参数为 0，因为 dropout 不影响神经元或参数。第四层的输出形状是 (None, 10)，因为我们在这层有十个神经元来处理十个输出类别。通过将这层的 10 个神经元乘以前一层的 512，并加上这层的 10 个神经元，我们得到 5,130 个参数。

注意

使用 *None* 是因为 TensorFlow 模型可以接受任何批大小。

### 编译模型

定义一个梯度下降优化器，调整模型参数以最小化损失函数：

```py
model.compile(optimizer='adam',
loss='sparse_categorical_crossentropy',
metrics=['accuracy'])
```

### 训练模型

使用十个周期训练模型：

```py
epochs = 10
history = model.fit(train_fs, epochs=epochs,
verbose=1, validation_data=test_fs)
```

我们得到了相当不错的准确率，并且过拟合不多。

### 在测试数据上泛化

虽然训练提供了准确性和损失指标，但始终明确地在测试数据上评估模型以查看模型在新数据上的泛化能力是一个好主意：

```py
print('Test accuracy:', end=' ')
test_loss, test_acc = model.evaluate(test_fs, verbose=2)
```

### 可视化性能

*fit* 方法自动将训练过程的历史记录作为一个字典。因此，我们可以将训练信息分配给一个变量。在这种情况下，我们将其分配给 history。变量的 history 属性包含字典信息。

显示字典键以告知我们如何绘制结果：

```py
hist_dict = history.history
print (hist_dict, '\n')
print (hist_dict.keys())
```

字典 history.history 包含损失、准确率、val_loss 和 val_accuracy 指标，这些指标是模型在每个周期结束时在训练集和验证（或测试）集上测量的。

*params* 变量提供了所有与训练相关的参数：

```py
history.params
```

如列表 5-6 所示，绘制训练性能图。

```py
plt.plot(history.history['accuracy'], label="accuracy")
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
plt.show()
plt.plot(history.history['loss'], label="loss")
plt.plot(history.history['val_loss'], label = 'val_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.ylim([0.05, .7])
plt.legend(loc='lower right')
plt.show()
Listing 5-6
Plot training performance
```

我们没有太多过拟合，并且对于这样一个简单的神经网络，我们的模型准确率相当不错。

我们也可以使用 pandas 来绘制训练性能：

```py
import pandas as pd
pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1)
```

### 预测测试图像的标签

现在我们已经训练了一个模型，我们可以根据测试图像进行预测。我们使用测试图像进行预测，因为模型将这些图像视为 *新* 数据。

首先预测测试集中的每个图像的标签：

```py
tf.random.set_seed(0)
predictions = model.predict(test_fs)
```

我们在处理过的测试集 *test_fs* 上使用 *predict* 方法。

由于 Fashion-MNIST 有十个类别标签，每个预测由一个包含十个数字的数组组成，这些数字代表模型对图像与十个不同服装类别对应程度的**置信度**。

让我们看看 *第一个* 预测：

```py
predictions[0]
```

第一预测位于索引 0，因为 Python 的索引范围从 0 到 9999，对于测试集大小为 10,000。通过查看浮点数值数组，很难判断哪个值具有最高的数值。

将预测数组中的数字四舍五入，以便更容易看到具有最高置信度的数组位置：

```py
np.round(predictions[0], 2)
```

现在，我们可以清楚地看到数值最高的值。最高置信度的位置对应于类别标签数组中的位置。

我们可以使用以下代码直接推导出预测的置信度：

```py
c = 100*np.max(predictions[0])
c
```

显示置信度：

```py
'{:.2%}'.format(np.max(predictions[0]))
```

显示测试集中*第一个*图像的预测结果：

```py
np.argmax(predictions[0])
```

预测值必须在 0 到 9 之间，因为我们的目标是介于 0 到 9 之间的。

使用我们之前创建的*class_labels*数组来查看实际的时尚衣物：

```py
class_labels[np.argmax(predictions[0])]
```

因此，我们得到了测试集中第一张图像的预测衣物。

显示*第一个*实际测试图像：

```py
# take the first batch of images
for image, label in test_fs.take(1):
label
class_labels[label[0].numpy()]
```

我们从 128 张图像的一批中取出第一张图像，因为我们设置了批大小为 128。如果预测与测试图像匹配，则是一个正确的预测。

### 构建预测图

现在我们已经有一个训练好的模型，我们可以构建一个预测图。

由于我们想看看模型在新数据上的表现如何，所以我们从测试集中取出 30 个样本，如列表 5-7 所示。

```py
num = 30
images, labels = [], []
for example in test.take(num):
image, label = example['image'], example['label']
images.append(tf.squeeze(image.numpy()))
labels.append(tf.squeeze(label.numpy()))
Listing 5-7
Take samples from the test set
```

为了绘图目的，将每个图像压缩以去除 1 维。

如列表 5-8 所示构建一个绘图函数。

```py
def display_test(feature, target, num_images,
n_rows, n_cols, cl, p):
for i in range(num_images):
plt.subplot(n_rows, 2*n_cols, 2*i+1)
if cl[target[i]] != cl[np.argmax(p[i])]:
plt.imshow(feature[i], cmap="Reds")
else:
plt.imshow(feature[i], cmap="Blues")
val = 100*np.max(p[i])
rounded = str(np.round(val, 2)) + '%'
plt.title(cl[target[i]] + ' (' +\
cl[np.argmax(p[i])] + ') ' +\
rounded )
plt.tight_layout()
plt.show()
Listing 5-8
Function to build a prediction plot
```

如列表 5-9 所示调用函数。

```py
num_rows, num_cols = 6, 5
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
display_test(images, labels, num_images, num_rows,
num_cols, class_labels, predictions)
Listing 5-9
Invoke the prediction plot function
```

所有标有*红色*的衣物意味着预测是错误的。每件衣物的上方是实际标签，括号内是预测结果，以及预测的置信度。

## 将 Fashion-MNIST 作为 Keras 数据集加载

虽然 TFDS 推荐用于 TensorFlow 2.x 应用程序，但因为我们展示了如何与 Keras 数据集一起工作，所以这得益于其在行业中的普及。TensorFlow 2.x 的新颖性意味着它还没有像 Keras 那样在行业中渗透。

加载 Keras 数据集：

```py
train, test = tf.keras.datasets.fashion_mnist.load_data()
```

### 探索数据

让我们看看数据形状：

```py
print ('train data:', br)
print (train[0].shape)
print (train[1].shape, br)
print ('test data:', br)
print (test[0].shape)
print (test[1].shape)
```

Keras 数据集包含与 Fashion-MNIST TFDS 相同的数据。训练数据包括 60,000 个 28 `×` 28 的特征图像和 60,000 个标签。测试数据包括 10,000 个 28 `×` 28 的特征图像和 10,000 个标签。训练图像包含在 train 元组中。因此，train[0]代表图像，train[1]代表标签。

让我们看看第一张图像代表什么：

```py
class_labels[train[1][0]]
```

由于我们使用 train[1]访问训练标签，所以我们通过 train[1][0]获取第一个标签。

### 可视化第一张图像

使用 Matplotlib 的 imshow 函数绘制第一张图像：

```py
plt.imshow(train[0][0], cmap="binary")
plt.axis('off')
```

将图像和标签分配给变量以方便使用：

```py
train_images = train[0]
train_labels = train[1]
```

根据 train_images 变量绘制第一张图像：

```py
plt.imshow(train_images[0], cmap="binary")
plt.axis('off')
```

访问第一张图像的标签名称：

```py
class_labels[train_labels[0]]
```

标签是范围从 0 到 9 的类别 ID：

```py
train_labels
```

### 可视化样本图像

由于我们不能使用 TFDS 的 show_examples 方法，所以我们创建如列表 5-10 所示的代码。

```py
n_rows = 5
n_cols = 6
plt.figure(figsize=(n_cols * 1.5, n_rows * 1.5))
for row in range(n_rows):
for col in range(n_cols):
index = n_cols * row + col
plt.subplot(n_rows, n_cols, index + 1)
plt.imshow(train_images[index], cmap="binary",
interpolation='nearest')
plt.axis('off')
plt.title(class_labels[train_labels[index]], fontsize=12)
plt.subplots_adjust(wspace=0.2, hspace=0.5)
Listing 5-10
Code to visualize examples
```

### 准备训练数据

为了准备 TensorFlow 消耗的训练数据，我们需要从训练和测试数据中获取图像和标签。我们缩放图像以确保每个输入参数（在我们的例子中是一个像素）具有相似的数据分布。此类数据的分布类似于以零为中心的高斯曲线。缩放数据使网络训练更快收敛。

如我们所知，像素数据由 0 到 255 的范围表示。因此，我们将每个特征图像除以 255 以对其进行缩放。一旦图像被缩放，我们就使用 from_tensor_slices 将其转换为 tf.Tensor 对象，如列表 5-11 所示。

```py
# add test images and labels to the mix
test_images, test_labels = test
train_pictures = train_images / 255\.  # divide by 255 to scale
train_targets = train_labels.astype(np.int32)
test_pictures = test_images / 255\.  # divide by 255 to scale
test_targets = test_labels.astype(np.int32)
print ('train images:', len(train_pictures))
print ('train labels:', len(train_targets), br)
print ('test images', len(test_pictures))
print ('test labels', len(test_targets))
train_ds = tf.data.Dataset.from_tensor_slices(
(train_pictures, train_targets))
test_ds = tf.data.Dataset.from_tensor_slices(
(test_pictures, test_targets))
Listing 5-11
Prepare the input pipeline
```

显示张量：

```py
train_ds.element_spec, test_ds.element_spec
```

通过洗牌、分批和预取训练数据以及分批和预取测试数据来完成输入管道：

```py
BATCH_SIZE = 128
SHUFFLE_BUFFER_SIZE = 5000
train_ks = train_ds.shuffle(
SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE).prefetch(1)
test_ks = test_ds.batch(BATCH_SIZE).prefetch(1)
```

将 BATCH_SIZE 设置为 128，这样模型运行速度会比使用更小的批量大。尝试这个数字并看看会发生什么。我们还设置了 SHUFFLE_BUFFER_SIZE 为 5000，以便 shuffle 方法能够很好地工作。再次尝试这个数字并看看会发生什么。

显示最终化的输入管道张量：

```py
train_ks.element_spec, test_ks.element_spec
```

### 构建模型

创建与 TFDS 相同的模型，如列表 5-12 所示。

```py
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
# clear previous model and generate a seed
tf.keras.backend.clear_session()
np.random.seed(0)
tf.random.set_seed(0)
model = Sequential([
Flatten(input_shape=[28, 28]),
Dense(512, activation="relu"),
Dropout(0.4),
Dense(10, activation="softmax")
])
Listing 5-12
Keras model
```

如我们所知，清除任何先前的建模会话总是一个好主意。记住，我们已经在本章的早期创建了一个模型并对其进行了训练。此外，生成一个种子确保结果的一致性。在机器学习中，这种一致性称为*可重复性*。也就是说，种子为随机生成器提供了一个起点，使我们能够以一致的方式重现结果。您可以使用任何整数作为随机种子数字。我们使用*0*。

第一层，Flatten()，将 28 `×` 28 的二维图像矩阵展平为 784 像素的一维数组。第二层是第一个真正的层。它接受 128 个神经元的输入图像并执行*relu*激活。最后一层是输出层。它接受输入层的输出，包含十个神经元，代表十种服装类别，并执行 softmax 激活。

### 模型摘要

检查模型：

```py
model.summary()
```

### 编译模型

编译：

```py
model.compile(optimizer='adam',
loss='sparse_categorical_crossentropy',
metrics=['accuracy'])
```

### 训练模型

调整模型。传递一个包含(特征，标签)对的数据集就足够了，用于 Model.fit 和 Model.evaluate：

```py
history = model.fit(train_ks, epochs=10, validation_data=test_ks)
```

### 在测试数据上推广

虽然模型拟合信息在训练期间提供了验证损失和准确度，但始终明确地在测试数据上评估模型是一个好主意，因为准确度和损失值可能不同：

```py
print('Test accuracy:', end=' ')
test_loss, test_acc = model.evaluate(test_ks, verbose=2)
```

### 可视化训练

fit 方法自动将训练过程的历史记录记录为字典。因此，我们可以将训练历史记录分配给一个变量。在这种情况下，我们将其分配给*history*。变量的 history 属性包含字典信息。

显示字典键以告知我们如何绘制结果：

```py
hist_dict = history.history
print (hist_dict, '\n')
print (hist_dict.keys())
```

字典*history.history*包含模型在每个训练集和验证集的每个 epoch 结束时测量的损失和其他指标。

*params* 变量提供了与模型相关的所有参数：

```py
history.params
```

列表 5-13 绘制了训练历史。

```py
plt.plot(history.history['accuracy'], label="accuracy")
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
plt.show()
plt.plot(history.history['loss'], label="loss")
plt.plot(history.history['val_loss'], label = 'val_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.ylim([0.05, .7])
plt.legend(loc='lower right')
plt.show()
Listing 5-13
Training history plots
```

过拟合最小，因为训练准确率与测试准确率紧密一致。我们在模型中使用了 dropout 来减少过拟合。**Dropout** 是一种正则化方法，它通过并行训练具有不同架构的大量神经网络来近似训练。这个定义对于任何刚开始接触深度学习的人来说都不容易理解。所以让我们解释一下它是如何工作的。

在训练过程中，一些层输出被随机忽略或 *丢弃*。通过随机丢弃层输出，实际上层看起来就像并且被当作具有不同节点数量和与前一层连接性的层来处理。因此，在训练过程中对层的每次更新都是使用配置层的不同 *视图* 来执行的。Dropout 是一种简单但非常有效的技术，可以减少过拟合。

将 dropout 设置为 *0.4* 意味着我们将随机丢弃 40%的层输出。你可以通过改变这个值轻松地尝试 dropout。然而，你应该将 dropout 率保持在或低于 *0.5*，否则你将删除太多数据！

小贴士

尝试不同的 dropout 水平，但保持它不超过 0.5，以避免删除太多数据。

如果模型过拟合（训练准确率大于测试准确率）且 dropout 值设置不当，可以增加它。如果模型欠拟合（训练准确率小于测试准确率），可以减少它。

使用 pandas 进行绘图：

```py
import pandas as pd
pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1)
```

### 预测测试图像的标签

由于我们有一个训练好的模型，我们可以对测试图像进行预测。

从测试集 *test_ks* 对每张图像进行标签预测：

```py
predictions = model.predict(test_ks)
```

与 Fashion-MNIST TFDS 一样，预测是一个包含十个数字的数组，代表模型对图像对应十个服装类别的置信度。

#### 预测第一张图像

让我们看看 *第一个* 预测：

```py
predictions[0]
```

为了方便起见，对数组值进行四舍五入：

```py
np.round(predictions[0], 2)
```

直接访问预测的置信度：

```py
100*np.max(predictions[0])
```

将值转换为百分比：

```py
str(np.round(100*np.max(predictions[0]), 2)) + '%'
```

显示第一张图像的预测结果：

```py
np.argmax(predictions[0])
```

预测值只能在 0 到 9 之间，因为我们的目标是介于 0 到 9 之间。

使用之前创建的 class_labels 数组来查看实际的时尚文章：

```py
class_labels[np.argmax(predictions[0])]
```

显示用于比较的第一个实际测试标签：

```py
class_labels[test_labels[0]]
```

如果预测和实际图像匹配，则预测正确。

#### 预测四张图像

从测试集中进行四次预测：

```py
pred_4 = predictions[:4]
```

显示预测结果：

```py
ls = [np.argmax(row) for row in pred_4]
ls
```

获取类别标签：

```py
np.array(class_labels)[ls]
```

将预测与测试数据集进行比较：

```py
actual_4 = test_labels[:4]
actual_4
```

以类别标签显示实际值：

```py
np.array(class_labels)[actual_4]
```

如列表 5-14 所示进行可视化。

```py
# slice off the first four images from the test data
img_4 = test_images[:4]
# plot images
plt.figure(figsize=(7.2, 2.4))
for index, image in enumerate(img_4):
plt.subplot(1, 4, index + 1)
plt.imshow(image, cmap="twilight", interpolation="nearest")
plt.axis('off')
plt.title(class_labels[actual_4[index]], fontsize=12)
plt.subplots_adjust(wspace=0.2, hspace=0.5)
Listing 5-14
Visualize the first four actual test images
```

### 探索错误分类

让我们探索预测和实际测试图像标签以找到一些错误分类。

第一步是识别预测标签：

```py
rng = (len(predictions))
y_pred = [np.argmax(row) for row in predictions]
```

*y_pred* 变量包含预测标签的列表。

接下来，获取一些预测和实际目标：

```py
# find first n predictions and actual targets
n = 20
y_n = [y_pred[i] for i, row in enumerate(range(n))]
y_actual = [test_labels[i] for i, row in enumerate(range(n))]
y_n, y_actual
```

寻找错误分类：

```py
# compare predictions against actual targets
miss_indx_list = [index for index, (x, y) in
enumerate(zip(y_n, y_actual))
if x != y]
miss_indx_list
```

如果没有错误分类，增加 *n* 的大小并重新运行最后两个代码片段。

我们现在有一个按索引排列的错误分类数组。

列表 5-15 显示了错误分类的置信度。

```py
# display confidence for each misclassification:
for row in miss_indx_list:
val = 100*np.max(predictions[row])
rounded = str(round(val, 2)) + '%'
print ('index:', row, 'confidence:', rounded,
'pred:', class_labels[np.argmax(predictions[row])],
'actual:', class_labels[test_labels[row]])
Listing 5-15
Confidence in misclassifications
```

### 可视化错误分类

创建一个函数，如列表 5-16 所示，来可视化错误分类。

```py
def see_misses(indx):
plt.imshow(test_images[indx], cmap="nipy_spectral")
plt.show()
print ('actual:', class_labels[test_labels[indx]])
print ('predicted:',
class_labels[np.argmax(predictions[indx])])
print ('confidence', rounded)
Listing 5-16
Function to visualize misclassifications
```

调用函数：

```py
for row in miss_indx_list:
val = 100*np.max(predictions[row])
rounded = str(round(val, 2)) + '%'
see_misses(row)
```

即使预测不正确，置信度可能仍然相当高。尽管神经网络在预测方面通常表现良好，但它们通过 softmax 层的输出估计的预测概率往往过高。也就是说，它们对自己的预测过于自信。记住，预测数组中值最高的概率是预测的类别。这个概率就是置信度。然而，除非模型有 100% 的准确率，否则预测不一定正确。

如列表 5-17 所示，创建一个更复杂的可视化。

```py
# Plot the first X test images, their true labels,
# their predicted labels, and prediction confidence.
num_rows = 8
num_cols = 8
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
plt.subplot(num_rows, 2*num_cols, 2*i+1)
if class_labels[test_labels[i]] != class_labels[y_pred[i]]:
plt.imshow(test_images[i], cmap="Reds")
else:
plt.imshow(test_images[i], cmap="Blues")
val = 100*np.max(predictions[i])
rounded = str(np.round(val, 2)) + '%'
plt.title(class_labels[test_labels[i]] + ' (' +\
class_labels[y_pred[i]] + ') ' + rounded )
plt.tight_layout()
plt.show()
Listing 5-17
Sophisticated visualization of misclassifications
```

如果有的话，错误分类将以红色显示。每件衣服上方是实际标签，括号内是预测，以及预测的置信度。

### 从单张图片进行预测

我们可以对单张图片进行预测。我们选择一个介于 0 和 9,999 之间的数字，因为我们想要测试集中的图片。或者，我们也可以生成一个介于 0 和 9,999 之间的随机数。

从测试集中随机选择一张图片：

```py
beg, end = 0, len(test_images) - 1
rng = np.random.default_rng()
indx = int(rng.uniform(beg, end, size=1))
indx
```

显示的是随机选择的图片的索引。

从测试集中获取图片：

```py
# Grab the image from the test dataset
img = test_images[indx]
label = class_labels[test_labels[indx]]
img.shape, label
```

TensorFlow 模型被优化为一次对一批或一系列示例进行预测。所以将单张图片添加到一个批次中，使其成为唯一的成员。

创建一个包含单个图片的批次：

```py
img_batch = (np.expand_dims(img, 0))
img_batch.shape
```

*expand_dims* 方法在扩展数组的形状中插入一个新的轴，该轴位于扩展数组的轴位置。因此，新的形状是 (1, 28, 28)。

现在，我们可以进行预测：

```py
pred_single = model.predict(img_batch)
```

显示预测：

```py
np.argmax(pred_single)
```

预测数组中的索引位置被显示出来。

为了简化，通过标签名称显示预测：

```py
class_labels[np.argmax(pred_single)]
```

显示实际标签：

```py
class_labels[test_labels[indx]]
```

### 可视化单张图片预测

如列表 5-18 所示，可视化预测结果。

```py
pred = class_labels[np.argmax(pred_single)]
actual = class_labels[test_labels[indx]]
# get confidence from the predictions object
val = 100*np.max(predictions[indx])
rounded = str(np.round(val, 2)) + '%'
# display actual image
plt.imshow(test_images[indx], cmap=plt.cm.binary)
plt.show()
print ('actual:', actual)
print ('predicted:', pred)
print ('confidence', rounded)
Listing 5-18
Visualization of single image prediction
```

通过标签名称和实际标签名称获取图片的预测。从我们之前创建的预测对象中获取预测置信度。该预测对象包含基于测试集的所有预测。*indx* 值提供了置信值在预测对象中的位置。

### 混淆矩阵

创建一个混淆矩阵，以直观地展示模型对 Fashion-MNIST 衣服类别的分类效果：

```py
tf.math.confusion_matrix(test_labels, y_pred)
```

要理解 TensorFlow 混淆矩阵，想象矩阵的第一行上方是 0-9 类别，从左到右排列。这些代表预测。想象矩阵的第一列左侧是 0-9 类别，从上到下排列。这些代表实际标签。正确的分类位于对角线上。错误分类不在对角线上。

第一组正确分类位于第一行第一列，代表类别 0。第二组位于第二行第二列，代表类别 1。以此类推…

一个例子中的错误分类集位于第一行第二列。这代表了对类别 1 的预测错误地被分类为类别 0 的数量。关于 TensorFlow 混淆矩阵的技术解释，请参阅以下网址：

[理解 TensorFlow 二分类混淆矩阵](https://stackoverflow.com/questions/46616837/understanding-a-tensorflow-confusion-matrix-for-binary-classification)

关于混淆矩阵的一般解释，请参阅以下网址：

[混淆矩阵术语简单指南](http://www.dataschool.io/simple-guide-to-confusion-matrix-terminology/)

## 隐藏层数量

对于许多问题，我们可以从一个隐藏层开始，并像本章中我们的 Fashion-MNIST 模型那样获得合理的结果。对于更复杂的问题，我们应该添加层，直到我们开始过拟合训练集。

## 隐藏层中的神经元数量

输入层和输出层中的神经元数量取决于你的任务所需的输入和输出类型。例如，Fashion-MNIST 任务需要 28 `×` 28 = 784 个输入神经元和十个输出神经元。对于隐藏层，这非常难以确定。你可以尝试逐渐增加神经元数量，直到网络开始过拟合。在实践中，我们通常选择比所需更多层和神经元的模型，然后使用提前停止和其他正则化技术来减少过拟合。由于这是一个介绍，我们不会深入探讨网络调优。

通常，通过增加层数而不是每层的神经元数量，我们可以得到更好的模型。当然，我们包含的层数和神经元数量受限于我们可用的计算资源。
