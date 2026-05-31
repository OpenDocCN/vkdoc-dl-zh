# 9. 卷积和变分自编码器

自编码器通常不适用于图像，除非它们非常小。但卷积和变分自编码器与大型彩色图像相比，效果要好得多。

**变分自编码器（VAE）**是一种深度学习技术，用于学习潜在表示。它们将学习的潜在空间表示应用于绘制图像和句子之间的插值。VAE 通过将输入压缩到潜在空间表示，然后从该表示重建输出而工作。**潜在空间**是压缩数据的数学表示。

VAE 通过使编码器返回潜在空间上的高斯分布而不是单个点，并在返回的分布上添加损失函数的正则化来解决潜在空间不规则性的问题。在深度学习中，数据可以压缩以更好地管理计算资源。但潜在空间必须分布在高斯分布上以进行归一化，以便模型使用。

VAE 的一个主要优点是它以高斯概率方式描述（在潜在空间中的）观察结果，以更好地模拟自然现象。因此，而不是构建一个输出单个值来描述每个潜在状态属性的编码器，VAE 构建一个来描述每个潜在属性的概率分布。结果是生成的高质量图像。

各章的笔记本位于以下 URL：

[deep-learning-models 仓库](https://github.com/paperd/deep-learning-models)

我们通过代码示例演示卷积和变分自编码器。我们从一个卷积自编码器实验开始。我们继续进行 VAE 实验。我们以使用 Tensorflow Probability (TFP)层进行的 VAE 实验结束。这三个实验通过产生原始图像的更清晰的渲染来展示它们相对于基本堆叠自编码器的优越性。

通过导入主 TensorFlow 库和实例化 GPU 来开始设置 Colab 生态系统。

## 导入 TensorFlow 库

导入库并将其别名为**tf**：

```py
import tensorflow as tf
```

将 TensorFlow 库别名为 tf 是常见做法。

## GPU 硬件加速器

为了方便起见，我们包括在 Colab 笔记本中启用 GPU 的步骤：

1.  在左上角菜单中点击*运行时*。

1.  从下拉菜单中选择*更改运行时类型*。

1.  从*硬件加速器*下拉菜单中选择*GPU*。

1.  点击*保存*。

验证 GPU 是否处于活动状态：

```py
tf.__version__, tf.test.gpu_device_name()
```

如果显示“/device:GPU:0”，则 GPU 处于活动状态。如果显示“..”，则常规 CPU 处于活动状态。

注意

如果出现错误**“NAME”** **“****TF****”** **“IS NOT DEFINED”**，请重新执行代码以导入 TensorFlow 库！

## 卷积编码器实验

*卷积自编码器*是用于无监督学习卷积滤波器的卷积神经网络的一个变体。卷积自编码器通过在图像重建过程中学习最优滤波器来最小化重建误差。在实验中，卷积自编码器学习如何生成新的图像，这些图像是数据集中输入特征的真实副本。

### 加载数据

我们使用*horses_or_humans*数据集进行此实验。horses_or_humans 数据集包含 1,283 张马和人类图像。

将数据集作为 TFDS 对象加载以进行检查：

```py
import tensorflow_datasets as tfds
data, hh_info = tfds.load(
'horses_or_humans', with_info=True,
split='train', try_gcs=True)
```

### 检查数据

获取元数据：

```py
hh_info
```

获取标签和类别数量：

```py
class_labels = hh_info.features['label'].names
num_classes = hh_info.features['label'].num_classes
class_labels, num_classes
```

### 显示示例

使用 tfds API 显示一些示例：

```py
fig = tfds.show_examples(data, hh_info)
```

以 pandas 数据框的形式显示示例：

```py
tfds.as_dataframe(data.take(4), hh_info)
```

手动显示图像：

```py
import matplotlib.pyplot as plt
for element in data.take(1):
plt.imshow(element['image'])
plt.axis('off')
```

手动显示如清单 9-1 所示的示例网格。

```py
img, lbl = [], []
for element in data.take(9):
img.append(element['image'])
lbl.append(element['label'].numpy())
fig=plt.figure(figsize=(8, 8))
columns = 3
rows = 3
for i in range(1, columns*rows+1):
fig.add_subplot(rows, columns, i)
plt.imshow(img[i-1])
plt.title(class_labels[lbl[i-1]])
plt.axis('off')
plt.show()
Listing 9-1
Grid of Examples
```

创建一组图像和标签。以网格形式绘制图像和标签。

### 获取训练数据

从元数据中，我们知道数据拆分：

```py
(x_train_img, _), (x_test_img, _) = tfds.as_numpy(
tfds.load(
'horses_or_humans', split=['train','test'],
batch_size=-1, as_supervised=True,
try_gcs=True))
```

由于自编码器是无监督模型，我们不需要标签。

获取训练和测试元素的数量：

```py
len(x_train_img), len(x_test_img)
```

### 检查形状

检查训练和测试形状：

```py
x_train_img.shape, x_test_img.shape
```

检查图像大小：

```py
for element in range(10):
print (x_train_img.shape)
```

由于图像形状相同，我们不需要为训练调整大小。

### 预处理图像数据

缩放图像：

```py
import numpy as np
x_train, x_test = x_train_img.astype(np.float32) / 255,\
x_test_img.astype(np.float32) / 255
```

在缩放前后检查训练图像中的向量：

```py
x_train_img[0][0][0], x_train[0][0][0]
```

### 创建一个卷积自编码器

清除和设置随机种子：

```py
tf.keras.backend.clear_session()
np.random.seed(0)
tf.random.set_seed(0)
```

获取输入形状：

```py
hh_shape = hh_info.features['image'].shape
hh_shape
```

导入库：

```py
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D,\
Dense, Flatten, Input, Conv2DTranspose, Reshape
from tensorflow.keras.models import Model
```

创建一个如清单 9-2 所示的编码器。

```py
conv_encoder = Sequential([
Input(shape=hh_shape),
Conv2D(16, kernel_size=3, padding='SAME', activation='selu'),
MaxPool2D(pool_size=2),
Conv2D(32, kernel_size=3, padding='SAME', activation='selu'),
MaxPool2D(pool_size=2),
Conv2D(64, kernel_size=3, padding='SAME', activation='selu'),
MaxPool2D(pool_size=2)
])
Listing 9-2
Encoder for a Convolutional Autoencoder
```

编码器由卷积层和池化层组成。编码器在增加深度（特征图数量）的同时，减少了输入的空间维度（高度和宽度）。

创建如清单 9-3 所示的解码器。

```py
conv_decoder = Sequential([
Conv2DTranspose(32, kernel_size=3, strides=2, padding='VALID',
activation='selu'),
Conv2DTranspose(16, kernel_size=3, strides=2, padding='SAME',
activation='selu'),
Conv2DTranspose(3, kernel_size=3, strides=2, padding='SAME',
activation='sigmoid')
])
Listing 9-3
Decoder for a Convolutional Autoencoder
```

解码器通过上采样图像并减少其深度回到原始尺寸来执行编码器的逆操作。用于此目的的是 Conv2DTranspose 层。

创建卷积自编码器：

```py
conv_ae = Sequential([conv_encoder, conv_decoder])
```

### 编译和训练

创建一个用于准确度指标的函数：

```py
def rounded_accuracy(y_true, y_pred):
return tf.keras.metrics.binary_accuracy(
tf.round(y_true), tf.round(y_pred))
```

编译：

```py
conv_ae.compile(
loss='binary_crossentropy',
optimizer=tf.keras.optimizers.SGD(lr=1.0),
metrics=[rounded_accuracy])
```

训练：

```py
cae_history = conv_ae.fit(
x_train, x_train, epochs=5,
validation_data=(x_test, x_test))
```

### 可视化训练性能

创建一个如清单 9-4 所示的可视化函数。

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
Listing 9-4
Performance Visualization Function
```

调用可视化函数：

```py
viz_history(cae_history)
```

### 可视化重建结果

创建一个如清单 9-5 所示的重建可视化函数。

```py
def show_reconstructions(model, images, n_images, reshape=False):
reconstructions = model.predict(images[:n_images])
if reshape:
reconstructions = tf.squeeze(reconstructions)
fig = plt.figure(figsize=(n_images * 1.5, 3))
for image_index in range(n_images):
plt.subplot(2, n_images, 1 + image_index)
plot_image(images[image_index])
plt.subplot(2, n_images, 1 + n_images + image_index)
plot_image(reconstructions[image_index])
Listing 9-5
Reconstruction Visualization Function
```

该函数从图像预测中生成重建。

创建一个显示图像的函数：

```py
def plot_image(image):
plt.imshow(image, cmap='binary')
plt.axis('off')
```

显示原始和重建的图像：

```py
show_reconstructions(conv_ae, x_test, 5)
```

## 变分自编码器实验

**变分自编码器**（VAE）是一种无监督机器学习技术，它以概率学习高效的数据编码。VAE 的训练被正则化以减轻过拟合并确保潜在空间在与其他训练输入相比时有效地生成相似的输出。VAE 与其他自编码器非常不同，因为它将输入编码为潜在空间上的分布，而不是一个单独的点。

初始时，输入被编码为潜在空间上的一个分布。然后从这个分布中采样一个潜在空间中的点。采样点被解码，并计算重建误差。然后重建误差通过网络反向传播。

在实践中，编码分布是**正态**的，因此编码器可以被训练以返回描述高斯分布的均值和协方差矩阵。输入被编码为分布而不是单个点的原因是因为潜在空间正则化可以非常自然地表示为高斯。编码器返回的分布被 VAE 模型强制接近标准正态分布。

对于 VAE 模型讨论的精彩讨论，请参阅

[《理解变分自编码器 VAE》](https://towardsdatascience.com/understanding-variational-autoencoders-vaes-f70510919f73)

### 加载数据

将 Fashion-MNIST 训练和测试图像作为单个批次的 NumPy 数组加载：

```py
(x_train_fm, _), (x_test_fm, _) = tfds.as_numpy(
tfds.load('fashion_mnist', split=['train','test'],
batch_size=-1, as_supervised=True,
try_gcs=True))
```

### 检查数据

获取训练和测试张量的形状：

```py
x_train_fm.shape, x_test_fm.shape
```

获取编码器的输入形状：

```py
fmnist_shape = x_train_fm.shape[1:]
fmnist_shape
```

### 缩放

通过将每个图像中的像素数除以特征图像的规模：

```py
x_train_fds, x_test_fds = x_train_fm.astype(np.float32) / 255,\
x_test_fm.astype(np.float32) / 255
```

### 创建一个用于采样编码的自定义层

编码器的采样层接受两个输入，即均值 μ 和对数方差 γ。它使用 tf.random.normal API 从均值为零（μ=0）和标准差为 1（σ=1）的正态分布中采样与 γ 形状相同的随机向量，将随机向量乘以 exp(γ/2)，加上 μ，并返回结果。

创建如列表 9-6 所示的采样类。

```py
class Sampling(tf.keras.layers.Layer):
def call(self, inputs):
mean, log_var = inputs
return tf.random.normal(tf.shape(log_var)) *\
tf.math.exp(log_var / 2) + mean
Listing 9-6
Sampling Class for the Encoder
```

### 创建 VAE 模型

清除和设置随机种子：

```py
tf.keras.backend.clear_session()
np.random.seed(0)
tf.random.set_seed(0)
```

创建如列表 9-7 所示的编码器。

```py
codings_size = 10
inputs = Input(shape=fmnist_shape)
z = Flatten()(inputs)
z = Dense(128, activation='relu')(z)
z = Dense(64, activation='relu')(z)
z = Dense(32, activation='relu')(z)
codings_mean = Dense(codings_size)(z)
codings_log_var = Dense(codings_size)(z)
codings = Sampling()([codings_mean, codings_log_var])
variational_encoder = Model(
inputs=[inputs],
outputs=[codings_mean, codings_log_var, codings])
Listing 9-7
VAE Encoder
```

我们使用功能 API，因为模型并不完全是序列的。密集层输出编码 _mean μ 和编码 _log_var γ，它们都有相同的输入（即第二个密集层的输出）。编码 _mean 和编码 _log_var 然后传递到采样层。variational_encoder 有三个输出，即编码 _mean、编码 _log_var 和编码。但我们只使用编码输出。

创建如列表 9-8 所示的解码器。

```py
decoder_inputs = Input(shape=[codings_size])
x = Dense(32, activation='relu')(decoder_inputs)
x = Dense(64, activation='relu')(x)
x = Dense(128, activation='relu')(x)
x = Dense(28 * 28, activation='sigmoid')(x)
outputs = Reshape(fmnist_shape)(x)
variational_decoder = Model(
inputs=[decoder_inputs], outputs=[outputs])
Listing 9-8
VAE Decoder
```

我们本可以使用序列 API 而不是功能 API，因为实际上它只是一个简单的层堆叠。但我们以这种方式创建 VAE 解码器是为了与 VAE 编码器的结构相匹配。

构建 VAE 模型：

```py
_, _, codings = variational_encoder(inputs)
reconstructions = variational_decoder(codings)
variational_ae = Model(
inputs=[inputs], outputs=[reconstructions])
```

忽略前两个输出，因为我们只需要编码。

### 编译和训练

#### 将潜在损失和重建损失添加到模型

计算潜在损失为 1 加上编码 _log_var 减去编码 _log_var 的指数减去编码 _mean 的平方（1 + codings_log_var − e^(codings_log_var) − codings_mean²）。将此结果乘以 –0.5。计算重建损失为批中所有实例的平均损失除以 784，以确保适当的规模。我们之所以除以 784，是因为图像是 28 × 28 像素。

作为复习，自动编码器是一种神经网络，它接收一个图像并通过学习图像的输入特征来重建它。编码器从图像中创建一个潜在向量。*潜在向量*是图像的压缩表示。潜在向量被送入解码器，解码器从潜在向量中重建输入图像的复制品。

VAE 还可以从图像的学习输入特征中重建一个新的图像。但编码器基于高斯分布创建潜在向量的样本。解码器从编码器创建的分布中随机抽取潜在向量以创建新的复制品图像。新图像质量更好，因为它们基于从高斯分布中抽取的随机潜在表示。

我们对 VAE 使用潜在损失进行限制。*潜在损失*衡量潜在向量与编码器创建的高斯分布之间的损失。因此，潜在损失根据潜在向量与单位高斯分布的接近程度或距离来惩罚网络。实际图像损失也进行衡量，以查看复制品图像与输入图像的匹配程度。当结合这两个损失指标时，网络必须在低潜在损失（潜在向量的单位高斯分布）和低图像损失（输入与复制品输出图像之间的高相似度）之间找到最佳权衡。只有当均值是 0 且标准差是 1 时，潜在损失才评估为零（完美），这对应于单位高斯分布。

创建如列表 9-9 所示的限制 VAE。

```py
latent_loss = -0.5 * tf.math.reduce_sum(
1 + codings_log_var - tf.math.exp(codings_log_var) -\
tf.math.square(codings_mean), axis=-1)
variational_ae.add_loss(
tf.math.reduce_mean(latent_loss) / 784.)
Listing 9-9
Restricted VAE
```

编译：

```py
variational_ae.compile(
loss='binary_crossentropy', optimizer='rmsprop',
metrics=[rounded_accuracy])
```

训练：

```py
vae_history = variational_ae.fit(
x_train_fds, x_train_fds, epochs=10,
batch_size=128,
validation_data=(x_test_fds, x_test_fds))
```

可视化训练性能：

```py
viz_history(vae_history)
```

### 可视化重建结果

检查测试集的形状：

```py
x_test_fds.shape
```

为了绘图目的，移除 *1* 维度：

```py
x_test_fds_imgs = tf.squeeze(x_test_fds)
x_test_fds_imgs.shape
```

可视化：

```py
show_reconstructions(
variational_ae, x_test_fds_imgs, 5, reshape=True)
```

### 生成新图像

创建一个如列表 9-10 所示的绘图函数。

```py
def plot_multiple_images(images, n_cols=None):
n_cols = n_cols or len(images)
n_rows = (len(images) - 1) // n_cols + 1
if images.shape[-1] == 1:
images = np.squeeze(images, axis=-1)
plt.figure(figsize=(n_cols, n_rows))
for index, image in enumerate(images):
plt.subplot(n_rows, n_cols, index + 1)
plt.imshow(image, cmap='binary')
plt.axis('off')
Listing 9-10
Plotting Function for New Images
```

我们不是从图像预测中生成重建，而是创建一个随机的高斯编码集，并直接从解码器中重建图像。

生成一些随机编码，解码它们，并绘制结果图像：

```py
tf.random.set_seed(0)
codings = tf.random.normal(shape=[12, codings_size])
images = variational_decoder(codings).numpy()
plot_multiple_images(images, 4)
```

创建 12 个随机编码，用解码器解码它们，并用绘图函数绘制它们。

如列表 9-11 所示，在图像之间执行语义插值。

```py
tf.random.set_seed(0)
np.random.seed(0)
codings_grid = tf.reshape(codings, [1, 3, 4, codings_size])
larger_grid = tf.image.resize(codings_grid, size=[5, 7])
interpolated_codings = tf.reshape(
larger_grid, [-1, codings_size])
images = variational_decoder(interpolated_codings).numpy()
images.shape
Listing 9-11
Semantic Interpolation Between the Codings
```

使用隐式模型如 VAE，我们在潜在空间中的采样点之间进行插值。我们这样做是为了使代码向量的分布假设与插值路径的几何形状相匹配。否则，关于编码点之间质量和语义的典型假设可能是不合理的。

为了绘图目的，移除 *1* 维度：

```py
images = tf.squeeze(images)
```

如列表 9-12 所示进行可视化。

```py
plt.figure(figsize=(7, 5))
for index, image in enumerate(images):
plt.subplot(5, 7, index + 1)
if index%7%2==0 and index//7%2==0:
plt.gca().get_xaxis().set_visible(False)
plt.gca().get_yaxis().set_visible(False)
else:
plt.axis('off')
plt.imshow(image, cmap='binary')
Listing 9-12
Visualize Interpolated Codings
```

## TFP 实验

在这个实验中，我们使用 TensorFlow Probability Layers 拟合一个 VAE。*TensorFlow Probability (TFP)* 是 TensorFlow 中进行概率推理和统计分析的库。作为 TensorFlow 生态系统的一部分，TFP 提供了概率方法与深度网络的集成、基于梯度的推理使用自动微分，以及通过硬件加速（例如 GPU）和分布式计算扩展到大型数据集和模型的可伸缩性。

TFP 使得在现代硬件（例如 TPU、GPU）上结合概率模型和深度学习变得容易。TFP 是为数据科学家、统计学家、机器学习研究人员和希望将领域知识编码以理解数据和做出预测的从业者设计的。

TFP 包括广泛的概率分布和双射。*Bijector API*提供了构建广泛类概率分布的模块化构建块。双射封装了概率密度的变量变化。因此，它们可以用来将张量分布转换为另一种类型的分布。简单来说，它们将一个随机结果转换为来自不同分布的另一个随机结果。

由于 TFP 继承了 TensorFlow 的优点，我们可以在模型探索和生产整个生命周期中使用单一语言来构建、拟合和部署模型。TFP 是开源的，可在 GitHub 上获取。

加载 Fashion-MNIST 数据：

```py
fmnist, fmnist_info = tfds.load(
name='fashion_mnist', try_gcs=True,
with_info=True, as_supervised=False)
```

创建一个预处理数据的函数：

```py
def _preprocess(sample):
image = tf.cast(sample['image'], tf.float32) / 255.
image = image < tf.random.uniform(tf.shape(image))
return image, image
```

将图像转换为浮点数并缩放。接着随机将图像像素转换为二进制形式，作为*True*或*False*值。tf.random.uniform API 输出均匀分布的随机值。注意，该函数返回*image, image*而不是仅仅 image，因为 Keras 旨在与具有*(example, label)*输入格式的判别性模型一起使用。由于 VAE 的目标是从 x 本身恢复输入 x，因此数据对是*(example, example)*。

建立输入管道的参数：

```py
auto = tf.data.experimental.AUTOTUNE
BATCH_SIZE, SHUFFLE_SIZE = 256, int(10e3)
```

创建训练集：

```py
train_tpl = (fmnist['train']
.map(_preprocess)
.batch(BATCH_SIZE)
.prefetch(auto)
.shuffle(SHUFFLE_SIZE))
```

创建测试集：

```py
test_tpl = (fmnist['test']
.map(_preprocess)
.batch(BATCH_SIZE)
.prefetch(auto))
```

检查来自批次的图像切片：

```py
for example in train_tpl.take(1):
print (example[0][0][0])
print (example[0].shape)
```

注意，像素现在要么是布尔值 True，要么是 False！

检查训练批次的形状：

```py
for row in train_tpl.take(1):
print (row[0].shape)
```

### 创建 TFP VAE 模型

创建一个没有学习参数的 TFP 独立高斯分布。潜在变量 z（encoded_size）分配了 16 个维度。将分布分配到先验，如列表 9-13 所示。

```py
import tensorflow_probability as tfp
tfd = tfp.distributions
encoded_size = 16
prior = tfd.Independent(
tfd.Normal(
loc=tf.zeros(encoded_size), scale=1),
reinterpreted_batch_ndims=1)
Listing 9-13
TFP Independent Gaussian Distribution
```

注意，先验是独立归一化的。**先验**是我们对世界的潜在假设。一个常见的先验是我们假设一枚硬币是公平的（50%正面和 50%反面）。常识是先验不一定总是正确的，但大多数情况下它是正确的。

获取输入形状并分配深度：

```py
input_shape = fmnist_info.features['image'].shape
base_depth = 32
input_shape
```

基础深度是神经元的基本数量。

为方便起见，分配别名：

```py
tfpl = tfp.layers
tfd = tfp.distributions
leaky = tf.nn.leaky_relu
```

#### TFP VAE 编码器

创建一个具有全协方差高斯分布的编码器，其均值和协方差矩阵由神经网络的输出参数化。TFP 层允许以简单格式构建这个复杂的编码器。

创建如图 9-14 所示的编码器。

```py
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from tensorflow.keras.layers import InputLayer, Lambda
encoder = Sequential([
InputLayer(input_shape=input_shape),
Lambda(lambda x: tf.cast(x, tf.float32) - 0.5),
Conv2D(base_depth, 5, strides=1, padding='same',
activation=leaky),
Conv2D(base_depth, 5, strides=2, padding='same',
activation=leaky),
Conv2D(base_depth * 2, 5, strides=1,
padding='same', activation=leaky),
Conv2D(base_depth * 2, 5, strides=2, padding='same',
activation=leaky),
Conv2D(4 * encoded_size, 7, strides=1, padding='valid',
activation=leaky),
Flatten(),
Dense(tfpl.MultivariateNormalTriL.params_size(encoded_size),
activation=None),
tfpl.MultivariateNormalTriL(
encoded_size,
activity_regularizer=tfpl.KLDivergenceRegularizer(
prior, weight=1.0))
])
Listing 9-14
TFP VAE Encoder
```

编码器是一个普通的 Keras 顺序模型，包含卷积和密集层。但输出被传递到一个 TFP 层（MultivariateNormalTril()），它透明地将最终密集层的激活分解为指定均值和下三角协方差矩阵所需的各个部分。我们使用 tfpl 辅助函数 MultivariateNormalTriL.params_size(encoded_size)来确保密集层输出正确的激活数量。我们还向最终损失中添加了一个正则化项以减轻过拟合。具体来说，我们在编码器和先验之间添加了 Kullback-Leibler (KL)散度正则化到损失中。

#### TFP VAE 解码器

创建解码器，如图 9-15 所示，作为一个像素独立的伯努利分布。

```py
decoder = Sequential([
InputLayer(input_shape=[encoded_size]),
Reshape([1, 1, encoded_size]),
Conv2DTranspose(2 * base_depth, 7, strides=1,
padding='valid', activation=leaky),
Conv2DTranspose(2 * base_depth, 5, strides=1,
padding='same', activation=leaky),
Conv2DTranspose(2 * base_depth, 5, strides=2,
padding='same', activation=leaky),
Conv2DTranspose(base_depth, 5, strides=1,
padding='same', activation=leaky),
Conv2DTranspose(base_depth, 5, strides=2,
padding='same', activation=leaky),
Conv2DTranspose(base_depth, 5, strides=1,
padding='same', activation=leaky),
Conv2D(filters=1, kernel_size=5, strides=1,
padding='same', activation=None),
Flatten(),
tfpl.IndependentBernoulli(
input_shape, tfd.Bernoulli.logits)
])
Listing 9-15
TFP VAE Decoder
```

此处的形式基本上与编码器相同，但我们使用转置卷积将我们的 16 维向量的潜在表示转换回 28 × 28 × 1 张量。最后一层参数化了像素独立的伯努利分布。

#### 构建 TFP VAE 模型

构建模型：

```py
from tensorflow.keras import Model
tpl_vae = Model(inputs=encoder.inputs,
outputs=decoder(encoder.outputs[0]))
```

检查输入是否符合预期：

```py
encoder.inputs
```

### 编译和训练

设置学习率：

```py
lr = 1e-3
lr
```

我们根据试错实验设置这个学习率。请随意尝试不同的学习率。

编译：

```py
negloglik = lambda x, rv_x: -rv_x.log_prob(x)
tpl_vae.compile(
optimizer=tf.optimizers.Adam(learning_rate=lr),
loss=negloglik)
```

我们的模型只是一个 Keras 模型，其中输出被定义为编码器和解码器的组合。由于编码器已经将 Kullback-Leibler (KL)散度项添加到损失中，我们只需要指定重建损失（ELBO 的第一个项）。KL 散度是衡量一个概率分布与第二个分布差异的度量。它也被称为相对熵。

ELBO（证据下界）是 log p(x)（或观察数据点的对数概率）的下界。ELBO 方程中的第一个积分是重建项。它询问我们从一个图像 x 开始，将其编码为 z，解码，并返回原始 x 的可能性有多大。第二项是 KL 散度项。它衡量我们的编码器与分配给先验变量的值有多接近。我们可以将这个项视为只是试图让我们的编码器诚实。如果我们的编码器生成的 z 样本在先验值下太不可能，那么目标函数比它生成更符合先验值的 z 样本时的目标函数更差。因此，编码器应该只在这样做所带来的成本超过重建项带来的好处时才与先验值不同。

训练：

```py
_ = tpl_vae.fit(train_tpl, epochs=15,
validation_data=test_tpl)
```

### 效率测试

检查十个随机数字：

```py
x = next(iter(test_tpl))[0][:10]
xhat = tpl_vae(x)
assert isinstance(xhat, tfd.Distribution)
```

创建一个函数以显示如图 9-16 所示的随机图像。

```py
def display_imgs(x, y=None):
if not isinstance(x, (np.ndarray, np.generic)):
x = np.array(x)
plt.ioff()
n = x.shape[0]
fig, axs = plt.subplots(1, n, figsize=(n, 1))
if y is not None:
fig.suptitle(np.argmax(y, axis=1))
for i in range(n):
axs.flat[i].imshow(x[i].squeeze(),
interpolation='none',
cmap='gray')
axs.flat[i].axis('off')
plt.show()
plt.close()
Listing 9-16
Plotting Function for Random Images
```

可视化：

```py
print('Originals:')
display_imgs(x)
print('Decoded Random Samples:')
display_imgs(xhat.sample())
print('Decoded Modes:')
display_imgs(xhat.mode())
print('Decoded Means:')
display_imgs(xhat.mean())
```

Python 的 sample() 方法返回一个从序列中选择的特定长度的项目列表。它用于随机抽样而不进行替换。Python 的 mode() 方法适用于名义（非数值）数据。记住，我们将像素图像数据转换为布尔值 True 或 False。它用于定位数值或名义数据的中心趋势。Python 的 mean() 方法计算给定数字列表的算术平均值。为了获得更清晰的重建图像，运行模型更多轮次（以减少损失）。

## 摘要

在本章中，我们展示了三个最先进的自编码器实验。这些自编码器中的每一个都比基本的堆叠自编码器产生了更逼真的输入图像渲染。
