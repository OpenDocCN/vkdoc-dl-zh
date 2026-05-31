# 8. 堆叠自动编码器

前七章专注于监督学习算法。**监督学习**是机器学习的一个子类别，它使用*标记*数据集来训练算法以准确地对数据进行分类和预测结果。剩余的章节专注于无监督学习算法。**无监督学习**使用机器学习算法来分析和聚类*未标记*数据集。这些算法在不需要人为干预的情况下发现隐藏的模式或数据分组。

**自动编码器**是人工神经网络，它们在没有任何监督的情况下学习输入数据的密集表示。学习到的密集表示通常被称为潜在表示（或编码）。编码用于重建原始输出。

编码通常比输入数据具有更低的维度，这使得自动编码器对于降维很有用。自动编码器还可以用于特征提取、深度神经网络的未监督预训练和生成模型。作为生成模型，它们可以随机生成看起来非常类似于训练数据的新数据。

如果自动编码器仅仅从输入数据产生相同的输出，那么训练自动编码器可能看起来有些反直觉。但我们展示了如何通过训练的自动编码器从*新*数据中重建图像！我们还展示了它们作为有效的降维机制的有效性。

自动编码器由编码器组件、代码组件和解码器组件组成。编码器压缩输入并生成代码。然后解码器从代码中重建原始输入。

章节的笔记本位于以下 URL：

[`https://github.com/paperd/deep-learning-models`](https://github.com/paperd/deep-learning-models)

我们展示了四个堆叠自动编码器实验。我们从一个基本的堆叠编码器实验开始。我们继续进行绑定权重实验。我们以降噪和简单的调整实验结束。

我们从一个简单的堆叠编码器开始，因为它是最基本的自动编码器类型。每个后续实验都增加了复杂性，例如通过绑定权重、降噪和调整自动编码器来展示。

通过导入主 TensorFlow 库并实例化 GPU 来开始设置 Colab 生态系统。

## 导入 TensorFlow 库

导入库并将其别名为 **tf**：

```py
import tensorflow as tf
```

将 TensorFlow 库别名为 tf 是一种常见的做法。

## GPU 硬件加速器

为了方便起见，我们包括在 Colab 笔记本中启用 GPU 的步骤：

1.  在右上角菜单中点击*运行*。

1.  从下拉菜单中选择*更改运行时类型*。

1.  从*硬件加速器*下拉菜单中选择*GPU*。

1.  点击*保存*。

验证 GPU 是否处于活动状态：

```py
tf.__version__, tf.test.gpu_device_name()
```

如果显示“/device:GPU:0”，则表示 GPU 正在运行。如果显示“..”，则表示常规 CPU 正在运行。

备注

如果出现错误**名称****‘****TF****’****IS NOT DEFINED**，请重新执行代码以导入 TensorFlow 库！

## 基本堆叠编码器实验

*堆叠编码器*具有多个隐藏层。该架构在中心隐藏层方面通常是对称的，该层被称为 *编码层*。在这个实验中，我们将向您展示如何创建和训练堆叠编码器。

### 加载数据

将 Fashion-MNIST 加载为 NumPy 数组：

```py
import tensorflow_datasets as tfds
(x_train_img, _), (x_test_img, _) = tfds.as_numpy(
tfds.load('fashion_mnist', split=['train','test'],
batch_size=-1, as_supervised=True,
try_gcs=True))
```

注意，我们没有从 Fashion-MNIST 加载类标签，因为自动编码器是无监督模型！也就是说，它们 *不需要标记数据来学习*！

### 缩放数据

通过将数据集除以表示图像的像素数进行缩放：

```py
import numpy as np
x_train, x_test = x_train_img.astype(np.float32) / 255.,\
x_test_img.astype(np.float32) / 255.
```

在深度学习实验中推荐对数据进行缩放，因为这通常可以平滑训练过程中的优化。

### 构建堆叠编码器模型

清除和设置随机种子：

```py
tf.keras.backend.clear_session()
np.random.seed(0)
tf.random.set_seed(0)
```

获取输入形状：

```py
in_shape = x_train.shape[1:]
in_shape
```

导入库：

```py
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten,\
Reshape
```

自动编码器分为编码器和解码器。编码器将示例编码为更小、更密集的表示。解码器接受这些密集表示，并尝试将输出重建回原始输入。

创建编码器：

```py
stacked_encoder = Sequential([
Flatten(input_shape=in_shape),
Dense(128, activation='relu'),
Dense(64, activation='relu'),
Dense(32, activation='relu')
])
```

编码器接受 28 × 28 像素的灰度图像，将它们展平，使每个图像都表示为一个大小为 784 的向量，并通过三个大小递减的 Dense 层（128 个单位到 64 个单位到 32 个单位）进行处理。32 个单位的层是编码层（中心隐藏层）。对于每个输入图像，编码器输出一个大小为 32 的向量。

创建解码器：

```py
stacked_decoder = Sequential([
Dense(64, activation='relu'),
Dense(128, activation='relu'),
Dense(28 * 28, activation='sigmoid'),
Reshape(in_shape)
])
```

解码器接受来自编码器的大小为 32 的编码，并通过三个大小递增的 Dense 层（64 个单位到 128 个单位到 784 个单位）进行处理。然后它将最终的向量重塑为 28 × 28 数组，这样解码器的输出形状与编码器的输入相同。

创建堆叠自动编码器：

```py
stacked_ae = Sequential([stacked_encoder, stacked_decoder])
```

### 编译和训练

创建一个用于准确度指标的函数：

```py
def rounded_accuracy(y_true, y_pred):
return tf.keras.metrics.binary_accuracy(tf.round(y_true),
tf.round(y_pred))
```

重建是一个二进制问题。输出要么与输入匹配，要么不匹配。重建损失会惩罚网络创建与输入不同的输出。因此，二进制准确度对于这个实验是理想的。

编译：

```py
opt = tf.keras.optimizers.SGD(lr=1.5)
stacked_ae.compile(
loss='binary_crossentropy',
optimizer=opt, metrics=[rounded_accuracy])
```

训练：

```py
sae_history = stacked_ae.fit(
x_train, x_train, epochs=10,
validation_data=(x_test, x_test))
```

### 可视化性能

导入绘图库：

```py
import matplotlib.pyplot as plt
```

创建一个如图 8-1 所示的绘图函数。

```py
def viz_history(training_history):
loss = training_history.history['loss']
val_loss = training_history.history['val_loss']
accuracy = training_history.history['rounded_accuracy']
val_accuracy = training_history.history['val_rounded_accuracy']
plt.figure(figsize=(14, 4))
plt.subplot(1, 2, 1)
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.plot(loss, label='Training set')
plt.plot(val_loss, label='Test set', linestyle='--')
plt.legend()
plt.grid(linestyle='--', linewidth=1, alpha=0.5)
plt.subplot(1, 2, 2)
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.plot(accuracy, label='Training set')
plt.plot(val_accuracy, label='Test set', linestyle='--')
plt.legend()
plt.grid(linestyle='--', linewidth=1, alpha=0.5)
plt.show()
Listing 8-1
Training Performance Visualization Function
```

可视化：

```py
viz_history(sae_history)
```

### 可视化重建结果

创建一个用于绘制灰度 28 × 28 图像的函数：

```py
def plot_image(image):
plt.imshow(image, cmap='binary')
plt.axis('off')
```

创建一个函数来可视化原始图像和重建结果，如图 8-2 所示。

```py
def show_reconstructions(model, images, n_images):
reconstructions = model.predict(images[:n_images])
reconstructions = tf.squeeze(reconstructions)
fig = plt.figure(figsize=(n_images * 1.5, 3))
for image_index in range(n_images):
plt.subplot(2, n_images, 1 + image_index)
plot_image(images[image_index])
plt.subplot(2, n_images, 1 + n_images + image_index)
plot_image(reconstructions[image_index])
Listing 8-2
Visualization Function for Reconstructions
```

该函数接受训练好的模型、一组 *测试* 图像和一个表示图像批次的数字。该函数计算图像批次的 *测试* 图像的预测。然后它从预测中挤压出 *1* 维度以进行可视化。请注意，我们的自动编码器模型是在训练数据 (*x_train*) 上训练的，重建是基于 Fashion-MNIST 数据集的测试数据 (*x_test*)。还请注意，重建是在 *没有任何标记数据* 的情况下创建的。

检查测试数据的维度：

```py
x_test.shape
```

形状为 (10000, 28, 28, 1)。因此测试数据集中有 10,000 个 28 × 28 × 1 的图像。

为了使用 imshow()函数进行可视化，从张量中移除尺寸为“1”的维度：

```py
x_test_imgs = tf.squeeze(x_test)
x_test_imgs.shape
```

现在的形状是 TensorShape([10000, 28, 28])，这是 imshow()函数期望的图像形状。

可视化：

```py
show_reconstructions(stacked_ae, x_test_imgs, 6)
```

重建图像是根据训练模型的预测从测试图像生成的。自动编码器能够学习如何将图像分解成小块数据。这些小块数据提供了图像的表示。自动编码器学习如何从这些表示中重建原始图像。当然，重建的图像并不完全相同于原始图像，因为我们使用了一个简单的堆叠自动编码器。

### 分解

让我们分解 8-2 中的函数，看看它是如何工作的。

从测试集中获取第一个图像作为单个批次的批次：

```py
img = x_test[:1]
```

由于预测方法计算是在批量中完成的，因此获取第一个图像作为单个批次的批次。

基于第一个图像批次进行预测：

```py
reconstruction = stacked_ae.predict(img)
```

删除“1”维度：

```py
reconstruction = tf.squeeze(reconstruction)
```

绘制重建图：

```py
plot_image(reconstruction)
```

绘制实际图像：

```py
plot_image(tf.squeeze(x_test[0]))
```

从图像中挤压出“1”维度以进行绘图。

### 降维

自动编码器在降维方面很有用。**降维**是通过获得一组主变量来减少深度学习实验中考虑的随机变量数量（或输入特征）的过程。主变量是数据集中的最重要的特征。降维在深度学习中常用，因为大量的输入特征可能导致机器学习算法性能不佳。

为了进行降维，*我们需要标签*。因此，从测试数据集中加载标签：

```py
test = tfds.as_numpy(
tfds.load('fashion_mnist', split=['test'],
batch_size=-1, as_supervised=True,
try_gcs=True))
```

从测试数据集中切片测试标签：

```py
y_test = test[0][1]
```

使用编码器按 8-3 所示降低维度。

```py
from sklearn.manifold import TSNE
np.random.seed(0)
x_test_compressed = stacked_encoder.predict(x_test_imgs)
tsne = TSNE()
x_test_2D = tsne.fit_transform(x_test_compressed)
x_test_2D = (x_test_2D - x_test_2D.min()) /\
(x_test_2D.max() - x_test_2D.min())
Listing 8-3
Dimensionality Reduction Algorithm
```

使用 scikit-learn 实现的 t-SNE（t-分布随机邻域嵌入）算法将维度降低到 2D 以进行可视化。该算法将维度降低到 32。*t-SNE*算法是一种统计方法，通过在二维或三维地图中为每个数据点指定一个位置来可视化高维数据。

可视化：

```py
plt.scatter(x_test_2D[:, 0], x_test_2D[:, 1],
c=y_test, s=10, cmap='tab10')
plt.axis('off')
plt.show()
```

每个类别都由不同的颜色表示。

创建一个更美观的如图 8-4 所示的可视化。

```py
import matplotlib as mpl
plt.figure(figsize=(10, 8))
cmap = plt.cm.tab10
plt.scatter(x_test_2D[:, 0], x_test_2D[:, 1],
c=y_test, s=10, cmap=cmap)
image_positions = np.array([[1., 1.]])
for index, position in enumerate(x_test_2D):
dist = np.sum((position - image_positions) ** 2, axis=1)
if np.min(dist) > 0.02: # if far enough from other images
image_positions = np.r_[image_positions, [position]]
imagebox = mpl.offsetbox.AnnotationBbox(
mpl.offsetbox.OffsetImage(x_test_imgs[index],
cmap='binary'),
position, bboxprops={
'edgecolor': cmap(y_test[index]), 'lw': 2})
plt.gca().add_artist(imagebox)
plt.axis('off')
plt.show()
Listing 8-4
Pretty Dimensionality Reduction Visualization
```

通过这种可视化，我们可以借助降维看到提取的类别标签。

## 权重绑定实验

当自动编码器结构对称时，我们可以将解码器层的权重与编码器层的权重绑定。*结构对称*意味着编码器和解码器具有对称的层结构。因此，我们模型中的权重数量减半，这加快了训练速度并减少了过拟合。

权重绑定的自动编码器的解码器权重是编码器权重的转置，这是一种参数共享的形式。我们通过参数共享减少了参数数量。

### 定义一个自定义层

要绑定编码器和解码器的权重，使用编码器权重的转置作为解码器权重，如列表 8-5 所示。

```py
class DenseTranspose(tf.keras.layers.Layer):
def __init__(self, dense, activation=None, **kwargs):
self.dense = dense
self.activation = tf.keras.activations.get(activation)
super().__init__(**kwargs)
def build(self, batch_input_shape):
self.biases = self.add_weight(
name='bias', shape=[self.dense.input_shape[-1]],
initializer='zeros')
super().build(batch_input_shape)
def call(self, inputs):
z = tf.matmul(
inputs, self.dense.weights[0], transpose_b=True)
return self.activation(z + self.biases)
Listing 8-5
Custom Layer Class
```

该类接受来自模型的层和一个激活函数（如果包含在层中）并转置数据。很多时候，我们必须预处理输入到机器学习算法中的数据。原因是数据可能以行存储，但机器学习算法期望输入为列或反之亦然。因此，转置在机器学习活动中是一个非常有用的操作。

### 构建绑定权重的模型：

清除和设置随机种子：

```py
tf.keras.backend.clear_session()
np.random.seed(0)
tf.random.set_seed(0)
```

为了方便起见，为模型创建三个 Dense 层：

```py
dense_1 = Dense(128, activation='relu')
dense_2 = Dense(64, activation='relu')
dense_3 = Dense(32, activation='relu')
```

我们可以创建这些变量以在编码器中使用。当然，如果您愿意，可以直接在模型中包含层语法。

构建编码器：

```py
tied_encoder = Sequential([
Flatten(input_shape=in_shape),
dense_1,
dense_2,
dense_3
])
```

构建解码器：

```py
tied_decoder = Sequential([
DenseTranspose(dense_3, activation='relu'),
DenseTranspose(dense_2, activation='relu'),
DenseTranspose(dense_1, activation='sigmoid'),
Reshape([28, 28])
])
```

由于自动编码器是对称的，我们可以将 DenseTranspose 类映射到编码器的层。DenseTranspose 层类似于 Dense 层，但输入矩阵以列主序存储而不是行主序，也就是说，它是普通 Dense 矩阵的转置。

构建绑定权重的自动编码器：

```py
tied_ae = Sequential([tied_encoder, tied_decoder])
```

### 编译和训练：

编译：

```py
tied_ae.compile(loss='binary_crossentropy',
optimizer=opt, metrics=[rounded_accuracy])
```

训练：

```py
tied_history = tied_ae.fit(
x_train, x_train, epochs=10,
validation_data=(x_test, x_test))
```

### 可视化训练性能：

可视化：

```py
viz_history(tied_history)
```

### 可视化重建结果：

根据训练模型的预测显示测试图像重建：

```py
show_reconstructions(tied_ae, x_test_imgs, 6)
plt.show()
```

## 去噪实验：

自动编码器也可以训练以从图像中去除噪声。想法是在输入中添加*随机噪声*并训练以恢复原始的无噪声输入。这听起来有些反直觉，但它确实有效！

图像中固有的噪声可能会造成麻烦，因为算法可能会认为噪声是应该学习的模式。但是添加随机噪声可以防止网络记住训练样本，因为它们总是在变化。

随机噪声使我们能够在已知噪声量的情况下测试算法的鲁棒性和性能。也就是说，我们知道我们添加了什么噪声。通过向输入数据添加噪声，我们得到了更多数据，我们的深度神经网络可以对其进行训练。但在噪声数据上训练意味着模型在噪声数据上泛化。

### 构建去噪模型：

清除和设置随机种子：

```py
tf.keras.backend.clear_session()
np.random.seed(0)
tf.random.set_seed(0)
```

导入高斯噪声库：

```py
from tensorflow.keras.layers import GaussianNoise
```

该库使我们能够应用加性零均值高斯噪声。**高斯噪声**（以卡尔·弗里德里希·高斯命名）是一种具有与正态分布（也称为高斯分布）相同的概率密度函数的统计噪声。因此，添加的噪声值是高斯分布的（或正态分布的）。

我们添加高斯噪声，因为它是在信息理论中用于模拟自然界中自然发生的许多随机过程的基本噪声模型。高斯噪声是正态分布的噪声，因此它适合许多自然现象。

直接将高斯噪声添加到编码器中：

```py
gaussian_encoder = Sequential([
Flatten(input_shape=in_shape),
GaussianNoise(0.2),
dense_1,
dense_2,
dense_3
])
```

构建解码器：

```py
gaussian_decoder = Sequential([
DenseTranspose(dense_3, activation='relu'),
DenseTranspose(dense_2, activation='relu'),
DenseTranspose(dense_1, activation='sigmoid'),
Reshape([28, 28])
])
```

为了获得更好的性能，将解码器层的权重绑定到编码器层的权重。

构建去噪自动编码器：

```py
gaussian_ae = Sequential([gaussian_encoder, gaussian_decoder])
```

### 编译和训练：

编译：

```py
gaussian_ae.compile(
loss='binary_crossentropy',
optimizer=opt, metrics=[rounded_accuracy])
```

训练：

```py
gae_history = gaussian_ae.fit(
x_train, x_train, epochs=10,
validation_data=(x_test, x_test))
```

### 可视化训练性能

可视化：

```py
viz_history(gae_history)
```

### 可视化重建

向测试图像添加相同数量的高斯噪声：

```py
noise = GaussianNoise(0.2)
show_reconstructions(gaussian_ae, noise(x_test_imgs), 6)
plt.show()
```

## Dropout 实验

**Dropout**是一种正则化技术，有助于防止过拟合。在训练过程中，Dropout 随机*关闭*一些神经元，将神经网络转换为一个神经网络集合。这种想法是并行训练大量具有不同架构的神经网络。

Dropout 的效果使得层看起来像，并且被当作具有不同节点数量和与前一层连接性的层来处理。因此，在训练过程中对层的每次更新都是使用配置层的一个*不同视角*来执行的。

然而，任何数量的 Dropout 都代表了一种信息损失！通过将层设置为 0.5 的 Dropout 概率，我们在每个 epoch（或训练迭代）中失去了该层一半的信息！在大多数情况下推荐使用 Dropout，但不是在每个层，并且它不应设置在 0.5 以上。

Dropout 使训练过程变得嘈杂，这迫使层内的节点以概率承担更多或更少的输入责任。通过丢弃神经元，它及其所有传入和传出连接暂时从网络中移除。

Dropout 在神经网络中的每一层都实现。它可以与大多数类型的层一起使用，例如全连接层、卷积层和循环层。

Dropout 可以应用于任何或所有隐藏层。一个超参数指定了层输出被丢弃的概率。对于隐藏层，一个常见的值是 0.5 的概率，这似乎对于广泛的网络和任务来说接近最优。对于输入层，最优值必须更接近 0 而不是 0.5。

该实验展示了构建和训练 Dropout 自动编码器是多么容易。唯一的变化是在编码器中添加一个 Dropout 值。

### 构建 Dropout 模型

清除并设置随机种子：

```py
tf.keras.backend.clear_session()
np.random.seed(0)
tf.random.set_seed(0)
```

导入 Dropout 库：

```py
from tensorflow.keras.layers import Dropout
```

使用 0.5 的 Dropout 构建编码器：

```py
dropout_encoder = Sequential([
Flatten(input_shape=in_shape),
Dropout(0.5),
dense_1,
dense_2,
dense_3
])
```

使用绑定权重构建解码器以获得更好的性能：

```py
dropout_decoder = Sequential([
DenseTranspose(dense_3, activation='relu'),
DenseTranspose(dense_2, activation='relu'),
DenseTranspose(dense_1, activation='sigmoid'),
Reshape([28, 28])
])
```

构建 Dropout 自动编码器：

```py
dropout_ae = Sequential([dropout_encoder, dropout_decoder])
```

### 编译和训练

编译：

```py
dropout_ae.compile(
loss='binary_crossentropy',
optimizer=opt, metrics=[rounded_accuracy])
```

训练：

```py
drop_history = dropout_ae.fit(
x_train, x_train, epochs=10,
validation_data=(x_test, x_test))
```

### 可视化训练性能

可视化：

```py
viz_history(drop_history)
```

### 可视化重建

向测试图像添加相同数量的 Dropout 噪声：

```py
dropout = Dropout(0.5)
show_reconstructions(dropout_ae, dropout(x_test_imgs), 6)
plt.show()
```

## 摘要

在本章中，我们展示了带有一些调整的基本堆叠自动编码器。在下一章中，我们将介绍更强大的自动编码器。
