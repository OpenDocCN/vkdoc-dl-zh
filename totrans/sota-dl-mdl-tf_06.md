# 6. 使用 TensorFlow Hub 的简单迁移学习

**迁移学习**是通过微调先前训练好的神经网络来创建新的学习模型的过程。我们不是从头开始训练网络，而是下载一个预训练的开源学习模型，并针对我们的目的对其进行微调。一个**预训练模型**是由其他人创建来解决类似问题的。我们可以使用这些模型之一，而不是构建自己的模型。一个很大的优势是，预训练模型是由专家精心制作的，因此我们可以有信心它在大多数情况下表现得很出色。另一个优势是，我们不需要大量数据就可以使用预训练模型。

在迁移学习中，机器利用从先前任务中获得的知识来提高对另一个任务的泛化能力。例如，在训练一个分类器来预测图像是否包含食物时，我们可以使用它在训练期间获得的知识来识别饮料。因此，迁移学习可以节省时间，在大多数情况下提供更好的神经网络性能，并且不需要大量数据。

*TensorFlow Hub*是一个存储库，其中包含易于集成到深度学习实验中的训练好的机器学习模型。我们通过 TensorFlow Hub 的代码示例演示简单的迁移学习。

图像分类模型可能有数百万个参数。从头开始训练它们需要大量的标记训练数据和大量的计算能力。使用迁移学习，我们不必从头开始训练！我们可以取一个已经在相关任务上训练好的模型的一部分，并在新的模型中重用它。由于我们使用了预训练模型的权重，训练时间大大减少！因此，我们不必训练一个巨大的神经网络，因为它已经被训练过了！而且，由于我们在数据集上使用了预训练模型，训练时间也减少了。

章节的笔记本位于以下 URL：

[`github.com/paperd/deep-learning-models`](https://github.com/paperd/deep-learning-models)

## 迁移学习的预训练模型

如果我们没有足够的训练数据，通常重用预训练模型的底层是一个好主意。重用预训练模型的底层通常被称为迁移学习。我们只训练底层，而将其他层冻结。**底层**指的是模型的一般（或问题无关）特征。高层指的是模型的具体（或问题相关）特征。预训练模型的最后几层也被称为最终层。

简单来说，就是通过微调先前训练的模型来创建新的深度学习模型，以适应我们的目的。简单地导入预训练模型的训练权重，并更改模型的最后层（或几层）。然后，我们在自己的数据集上重新训练这些层。我们节省了训练时间，并获得了可接受的性能。

我们通过示例演示如何使用预训练模型。我们从使用 MobileNet-v2 的实验开始，以使用 Inception-v3 的实验结束。每个预训练模型都在章节的相应部分中进行了解释。

## 导入 TensorFlow 库

导入库并将其别名为**tf**：

```py
import tensorflow as tf
```

## GPU 硬件加速器

为了方便起见，我们提供了在 Colab 笔记本中启用 GPU 的步骤：

1.  点击左上角的菜单中的*运行时*。

1.  从下拉菜单中选择*更改运行时类型*。

1.  从*硬件加速器*下拉菜单中选择*GPU*。

1.  点击*保存*。

验证 GPU 是否正在使用：

```py
tf.__version__, tf.test.gpu_device_name()
```

如果显示“/device:GPU:0”，则表示 GPU 正在使用中。如果显示“..”，则表示常规 CPU 正在使用中。

注意

如果出现错误**NAME** **‘****TF****’** **IS NOT DEFINED**，请重新执行代码以导入 TensorFlow 库！

## TensorFlow Hub

我们使用来自 TensorFlow Hub 的预训练 TensorFlow SavedModels 对花朵数据进行建模，以进行图像特征提取。*SavedModel*是一个包含序列化签名和运行它们所需的状态的目录，包括变量值和词汇表。*TensorFlow Hub*是一个预训练的 TensorFlow 模型的存储库。这些预训练模型是在非常大的通用数据集上训练的。

关于 TensorFlow Hub 的更多信息，请参阅

[`https://tfhub.dev/`](https://tfhub.dev/)

关于 TensorFlow Hub 的教程，请参阅

[`www.tensorflow.org/hub/tutorials`](http://www.tensorflow.org/hub/tutorials)

我们使用两个预训练的 TensorFlow Hub 模型进行迁移学习。我们首先使用 MobileNet-v2 预训练模型。然后使用 Inception-v3 预训练模型，并比较两者的结果。

## MobileNet-v2

*MobileNet-v2*是一个 53 层的卷积神经网络。网络的预训练版本在 ImageNet 数据库中的 140 万张图像和 1000 个网络图像类别上进行了训练。该模型在 224 × 224 像素的图像上运行。默认的训练批量大小为 1024，这意味着每次迭代都会处理 1024 张这样的图像。

*ImageNet 项目*是一个大型视觉数据库，旨在用于视觉对象识别软件研究。项目手动标注了 1400 多万张图像，以指示图片中的对象。

## 花朵 MobileNet-v2 实验

第一个实验使用 MobileNet-v2 预训练模型对花朵进行分类。

### 将 Flowers 作为 TFDS 对象加载

加载 TFDS 对象，将训练集分为 75%，验证集分为 15%，测试集分为 10%：

```py
import tensorflow_datasets as tfds
(test, valid, train), info = tfds.load(
'tf_flowers', as_supervised=True,
split = ['train[:10%]', 'train[10%:25%]', 'train[25%:]'],
with_info=True, try_gcs=True)
```

### 探索元数据

使用 info 对象显示一般信息：

```py
info
```

显示数据分割中的示例数量：

```py
num_train_img = info.splits['train[25%:]'].num_examples
num_valid_img = info.splits['train[10%:25%]'].num_examples
num_test_img = info.splits['train[:10%]'].num_examples
print ('train images:', num_train_img)
print ('valid images:', num_valid_img)
print ('test images:', num_test_img)
```

我们喜欢在每个实验中显示这些信息，只是为了确保我们按预期分割数据。

手动计算数据分割中的示例数量，如列表 6-1 所示。

```py
num_train_examples = 0
num_valid_examples = 0
num_test_examples = 0
for example in train:
num_train_examples += 1
for example in valid:
num_valid_examples += 1
for example in test:
num_test_examples += 1
print('Total Number of Training Images: {}'\
.format(num_train_examples))
print('Total Number of Validation Images: {}'\
.format(num_valid_examples))
print('Total Number of Testing Images: {}'\
.format(num_test_examples))
Listing 6-1
Manually Verify Examples in Each Split
```

列表 6-1 中的代码是可选的。我们只是想展示如何手动检查学习集的大小。

获取标签和类别数量：

```py
class_labels = info.features['label'].names
num_classes = info.features['label'].num_classes
class_labels, num_classes
```

### 显示图像和形状

显示数据集的一些图像：

```py
fig = tfds.show_examples(train, info)
```

显示图像形状：

```py
for i, example in enumerate(train.take(5)):
print('Image {} shape: {} label: {}'\
.format(i+1, example[0].shape,
example[1]))
```

花卉数据集中的图像大小并不相同。因此，我们必须将图像调整到标准大小，以便它们可以被 TensorFlow 模型消费。

### 构建输入管道

创建一个重新格式化图像的函数：

```py
def format_image(image, label):
image = tf.image.resize(image, (224, 224)) /255.0
return image, label
```

函数调整图像大小并缩放。MobileNet-v2 期望的分辨率是（224，224）。该函数接受“图像”和“标签”作为参数，并返回所需形式的新的“图像”和相应的“标签”。

将函数映射到训练、验证和测试示例，并应用其他转换：

```py
BATCH_SIZE = 367
train_batches = train.shuffle(num_train_img//4).\
map(format_image).batch(BATCH_SIZE).prefetch(1)
validation_batches = valid.map(format_image).\
batch(BATCH_SIZE).prefetch(1)
test_batches = test.map(format_image).\
batch(BATCH_SIZE).prefetch(1)
```

意外的是，我们发现 367 的批量大小效果非常好！我们尝试了默认的 1024 批量大小，并基于这个批量大小显示标签。令人惊讶的是，我们得到了一个错误信息，类似于*我们的索引超出了 367 的大小范围*。因此，我们将批量大小更改为 367，并注意到与 1024 的批量大小相比，模型在验证准确率提高方面表现更好。此外，训练时间也有所减少。

### 创建特征向量

**特征**是观察到的现象的个别可测量属性或特征。**特征向量**是一个包含关于对象多个元素的向量。特征由特征向量表示。将特征向量组合在一起创建特征空间。**特征空间**是与特征向量关联的向量空间。一个特征可能只代表一个像素或整个图像。在我们的情况下，特征由像素的特征向量表示的整个图像。

**特征提取**是从数据集中提取初始特征子集的过程。提取的特征应包含来自输入数据的相关信息。特征提取的目的是通过使用减少的表示（或子集）而不是完整的数据来执行所需的任务。选择信息丰富、有区分性和独立的特征是模式识别、分类和回归中有效算法的关键步骤。

使用预训练模型创建特征提取器：

```py
import tensorflow_hub as hub
piece1 = 'https://tfhub.dev/google/tf2-preview/'
piece2 = 'mobilenet_v2/feature_vector/4'
URL = piece1 + piece2
feature_extractor_mn = hub.KerasLayer(
URL, input_shape=(224, 224, 3))
```

使用 MobileNet-v2 特征向量模型创建特征提取器。特征提取器是来自 TensorFlow Hub 的部分模型（没有最终的分类层）。

### 冻结预训练模型

在特征提取器层中冻结变量，以便训练只修改最终的分类层：

```py
feature_extractor_mn.trainable = False
```

如果不冻结预训练层，那么在未来的训练轮次中会破坏它们所包含的信息。

### 创建分类头

创建一个分类头，利用预训练模型对数据集进行分类。分类头由一个简单的顺序模型组成，包括预训练模型和新的分类层。

导入库：

```py
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
```

清除和设置随机种子：

```py
import numpy as np
tf.keras.backend.clear_session()
np.random.seed(0)
tf.random.set_seed(0)
```

创建分类头：

```py
mobile_model = tf.keras.Sequential([
feature_extractor_mn,
Dropout(0.5),
Dense(num_classes)])
```

*分类头*只是一个容器，它包含预训练模型以及我们可能希望添加的任何内容。请注意，我们刚刚创建的模型的第一层是特征提取器。也请注意，由于我们利用了 MobileNet-v2 的预训练权重，所以模型**极其**简单！

注意

对于我们的实验，分类头是我们创建的新模型，它包含预训练模型（在这种情况下是特征提取器）、一个 Dropout 层（用于减少过拟合）和一个 Dense 层（用于实现分类）。要实现预训练模型，需要一个机制，这被称为分类头。

### 编译和训练模型

编译：

```py
from tensorflow.keras.losses import SparseCategoricalCrossentropy
mobile_model.compile(
optimizer='adam',
loss=SparseCategoricalCrossentropy(from_logits=True),
metrics=['accuracy'])
```

训练：

```py
EPOCHS = 6
history = mobile_model.fit(
train_batches, epochs=EPOCHS,
validation_data=validation_batches)
```

尽管我们的模型非常简单，没有调整，但我们仅用六个 epoch 就获得了相当好的准确率，因为 MobileNet-v2 是由专家经过长时间精心设计的，并在庞大的 ImageNet 数据集上进行了训练。此外，训练时间非常合理。

注意

使用像 MobileNet-v2 这样的大型模型，如果我们不使用其预训练的权重，训练时间将会非常显著。

### 可视化性能

如列表 6-2 所示绘制验证准确率和损失：

```py
import matplotlib.pyplot as plt
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(EPOCHS)
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
Listing 6-2
Visual Accuracy and Loss Performance
```

我们发现，从它没有训练过的数据集上从预训练模型中获得可尊敬的结果非常令人惊讶。

### 从测试数据中做出预测

我们在测试数据上做出预测，因为模型**从未**见过它！

在测试数据上做出预测：

```py
predictions = mobile_model.predict(test_batches)
```

显示类标签：

```py
class_labels
```

#### 检查第一个预测

我们检查数据集中的一个预测（或推理）以向您展示*predict*方法返回的原始值。由于推理是分类学习的主要目标，我们认为展示底层发生的事情（换句话说）非常有价值。

获取第一个预测数组：

```py
predictions[0]
```

返回的数组是原始预测。

使用*np.argmax()*函数获取预测，这是预测数组中第一个图像的最高概率值：

```py
predicted_id = np.argmax(predictions[0])
predicted_id
```

将标签转换为其类名：

```py
class_labels[predicted_id]
```

从第一个批次获取标签：

```py
for img, lbl in test_batches.take(1):
print (lbl)
```

标签的数量与批大小匹配。

获取第一个标签：

```py
class_labels[lbl[0].numpy()]
```

如果第一个标签与第一个图像的预测相匹配，则预测是正确的！

#### 检查预测的第一个批次

或者，我们可以将*test_batches*转换为迭代器：

```py
image_batch, label_batch = next(iter(test_batches))
images = image_batch.numpy()
labels = label_batch.numpy()
class_labels[labels[0]]
```

从迭代器中获取第一个批次，将图像和标签转换为 NumPy，并显示第一个标签。

显示第一个批次的标签：

```py
labels
```

将标签批次转换为命名标签：

```py
named_labels = [class_labels[labels[i]]
for i, lbl in enumerate(range(BATCH_SIZE))]
named_labels
```

从第一个批次获取预测：

```py
predicted_batch = [np.argmax(predictions[i])
for i, _ in enumerate(range(BATCH_SIZE))]
predicted_batch
```

将预测转换为命名预测：

```py
named_pred = [class_labels[predicted_batch[i]]
for i, lbl in enumerate(range(BATCH_SIZE))]
named_pred
```

### 绘制预测

可视化显示了第一个测试批次中的实际图像。如果预测正确，标题是蓝色。如果不正确，标题是红色。如果预测不正确，预测将显示在括号内的实际标签旁边。

如列表 6-3 所示显示预测：

```py
plt.figure(figsize=(20,20))
for n in range(30):
plt.subplot(6,5,n+1)
plt.subplots_adjust(hspace = 0.3)
plt.imshow(images[n])
color = 'blue' if labels[n] == predicted_batch[n] else 'red'
if labels[n] != predicted_batch[n]:
t = named_pred[n].title() +\
' (' +named_labels[n].title() + ')'
else:
t = named_pred[n].title()
plt.title(t, color=color)
plt.axis('off')
st = 'Model predictions (blue: correct, red: incorrect)'
_ = plt.suptitle(st)
Listing 6-3
Prediction Plot
```

## 花卉 Inception-v3 实验

对于第二个实验，我们使用 Inception-v3 预训练模型来分类花朵。*Inception-v3* 是一个用于图像识别的 48 层深度卷积神经网络，已经在 ImageNet 数据集上显示出超过 78.1% 的准确率。预训练模型在 299 × 299 的图像上运行。默认训练批次大小为 1024，这意味着每次迭代都会处理 1024 张这样的图像。

有关 Cloud TPU 上 Inception-v3 的高级指南，请参阅

[`cloud.google.com/tpu/docs/inception-v3-advanced`](https://cloud.google.com/tpu/docs/inception-v3-advanced)

### 构建 Input Pipeline

重新创建函数以重新格式化图像：

```py
def format_image(image, label):
image = tf.image.resize(image, (299, 299)) / 255.0
return image, label
```

该函数调整图像大小并缩放。Inception-v3 预期分辨率为 (299, 299)。该函数接受“图像”和“标签”作为参数，并返回所需形式的新的“图像”和相应的“标签”。

构建 Inception-v3 的输入管道：

```py
BATCH_SIZE = 367
train_im = train.shuffle(num_train_img//4).\
map(format_image).batch(BATCH_SIZE).prefetch(1)
validation_im = valid.map(format_image).\
batch(BATCH_SIZE).prefetch(1)
test_im = test.map(format_image).\
batch(BATCH_SIZE).prefetch(1)
```

将函数映射到训练、验证和测试示例，并应用其他转换。

创建特征提取器：

```py
piece1 = 'https://tfhub.dev/google/tf2-preview/'
piece2 = 'inception_v3/feature_vector/4'
URL = piece1 + piece2
feature_extractor_im = hub.KerasLayer(URL,
input_shape=(299, 299, 3),
trainable=False)
```

冻结预训练模型：

```py
feature_extractor_im.trainable = False
```

### 构建模型

清除并设置随机种子：

```py
tf.keras.backend.clear_session()
np.random.seed(0)
tf.random.set_seed(0)
```

创建模型：

```py
inception_model = tf.keras.Sequential([
feature_extractor_im,
Dropout(0.5),
Dense(num_classes)])
```

我们只需替换 Inception-v3 特征提取器以获取其预训练权重！

### 编译和训练

编译：

```py
inception_model.compile(
optimizer='adam',
loss=SparseCategoricalCrossentropy(from_logits=True),
metrics=['accuracy'])
```

训练：

```py
EPOCHS = 6
history = inception_model.fit(
train_im, epochs=EPOCHS,
validation_data=validation_im)
```

### 可视化性能

如列表 6-4 所示，可视化训练损失和准确率。

```py
import matplotlib.pyplot as plt
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(EPOCHS)
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
Listing 6-4
Training Performance
```

注意

我们并不真正关心比较 Inception 和 MobileNet 之间的模型性能，因为结果总是与你要建模的数据相关。也就是说，一个模型可能在某个数据集上表现更好，但在另一个数据集上则不那么好。这就是为什么我们展示了如何实现这两个模型。

### 预测

进行预测：

```py
im_predictions = inception_model.predict(test_im)
```

获取一批预测并将它们转换为命名预测：

```py
im_pred_batch = [np.argmax(im_predictions[i])
for i, _ in enumerate(range(BATCH_SIZE))]
im_named_pred = [class_labels[im_pred_batch[i]]
for i, lbl in enumerate(range(BATCH_SIZE))]
```

从测试集中获取第一批图像和标签：

```py
im_image_batch, im_label_batch = next(iter(test_im))
im_images = im_image_batch.numpy()
im_labels = im_label_batch.numpy()
```

将标签转换为命名标签：

```py
im_named_labels = [class_labels[im_labels[i]]
for i, lbl in enumerate(range(BATCH_SIZE))]
```

### 绘制模型预测

创建一个如列表 6-5 所示的绘图函数。

```py
def plot_pred(images, labels, named_labels, named_pred):
plt.figure(figsize=(20,20))
for n in range(30):
plt.subplot(6,5,n+1)
plt.subplots_adjust(hspace = 0.3)
plt.imshow(images[n])
color = 'blue' if named_labels[n] == named_pred[n] else 'red'
if named_labels[n] != named_pred[n]:
t = named_pred[n].title() +\
' (' +named_labels[n].title() + ')'
else:
t = named_pred[n].title()
plt.title(t, color=color)
plt.axis('off')
st = 'Model predictions (blue: correct, red: incorrect)'
_ = plt.suptitle(st)
Listing 6-5
Plot Function
```

将绘图逻辑封装在函数中。

调用函数：

```py
plot_pred(im_images, im_labels, im_named_labels, im_named_pred)
```

## 摘要

MobileNet-v2 和 Inception-v3 在花朵数据集上的性能相当相似，两者都没有出现过拟合。两个模型的准确率都超过了 80%。当然，我们只训练了六个周期。但我们无法预测预训练模型在给定数据集上的表现如何。预训练模型的想法是节省你的时间和精力。这也是为什么有这么多可用的预训练模型。我们推测在不久的将来将会创建许多新的模型。
