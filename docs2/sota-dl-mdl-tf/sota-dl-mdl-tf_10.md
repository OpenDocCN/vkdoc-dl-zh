# 10. 生成对抗网络

生成建模是一种无监督学习技术，它涉及自动发现和学习输入数据中的规律性（或模式），以便训练后的模型可以生成新的示例，这些示例可能来自原始数据集。一种流行的生成模型是生成对抗网络。**生成对抗网络**（GANs）是生成模型，可以创建与训练数据相似的新数据实例。

GANs 将问题框架化为一个带有两个子模型的监督学习问题，即生成器模型和判别器模型。*生成器*模型被训练以生成新的示例。*判别器*模型试图将示例分类为真实（来自域）或伪造（生成）。域表示来自原始训练集的图像。这两个模型在零和对抗游戏中一起训练，直到判别器模型大约一半的时间被欺骗。训练的结果是生成可信的示例。

GANs 在图像到图像的翻译任务中表现出色，例如将夏季照片转换为冬季或白天转换为夜晚。它们还成功地生成了逼真的物体、场景和人物照片，即使人类也无法分辨出这些照片是伪造的。GANs 可以创建看起来像人类面部照片的图像，即使这些面部不属于任何真实人物。

我们通过代码示例演示了一个 GAN。我们还通过代码示例演示了深度卷积生成对抗网络（DCGANs）。GAN 使用前馈网络来学习，而深度卷积 GAN 使用卷积网络来学习。

各章节的 Notebooks 位于以下 URL：

[`github.com/paperd/deep-learning-models`](https://github.com/paperd/deep-learning-models)

我们从一个 GAN 实验开始。我们继续进行两个深度卷积生成对抗网络（DCGAN）实验。GAN 从输入图像生成类似图像。DCGAN 做同样的事情，但生成的类似图像更加逼真。

通过导入主 TensorFlow 库和实例化 GPU 来开始设置 Colab 生态系统。

## 导入 TensorFlow 库

导入库并将其别名为**tf**:

```py
import tensorflow as tf
```

## GPU 硬件加速器

为了方便，我们包括在 Colab 笔记本中启用 GPU 的步骤：

1.  在右上角菜单中点击*运行时*。

1.  从下拉菜单中选择*更改运行时类型*。

1.  从*硬件加速器*下拉菜单中选择*GPU*。

1.  点击*保存*。

验证 GPU 是否处于活动状态：

```py
tf.__version__, tf.test.gpu_device_name()
```

如果显示“/device:GPU:0”，则 GPU 处于活动状态。如果显示“..”，则常规 CPU 处于活动状态。

注意

如果出现错误**NAME** **‘****TF****’** **IS NOT DEFINED**，请重新执行代码以导入 TensorFlow 库！

## GAN 实验

通过将一个学习生成目标输出的生成器与一个学习区分真实数据与生成器输出的判别器配对，GAN 可以在其创建的新数据实例中生成高水平的现实感。生成器试图欺骗判别器，而判别器则试图不被欺骗。

在训练过程中，生成器和判别器有相反的目标。判别器试图区分伪造图像和真实图像。生成器试图生成足够真实的图像来欺骗判别器。

我们从一个简单的前馈 GAN 开始，向您展示如何在简单数据集上对其进行训练。

### 加载数据

将 Fashion-MNIST 加载为 NumPy 数组：

```py
import tensorflow_datasets as tfds
x_train_img, _ = tfds.as_numpy(
tfds.load('fashion_mnist', split='train',
batch_size=-1, as_supervised=True,
try_gcs=True, shuffle_files=True))
```

由于我们的实验是*无监督的*，我们只需要训练图像。

获取示例数量：

```py
len(x_train_img)
```

### 规模

缩放示例：

```py
import numpy as np
images = x_train_img.astype(np.float32) / 255
images.shape
```

验证缩放：

```py
x_train_img[0][0], images[0][0]
```

缩放效果如预期。

### 构建生成对抗网络（GAN）

获取输入形状：

```py
in_shape = images.shape[1:]
in_shape
```

清除并设置随机种子：

```py
tf.keras.backend.clear_session()
np.random.seed(0)
tf.random.set_seed(0)
```

导入库：

```py
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten,\
Reshape
```

按照列表 10-1 构建生成器。

```py
codings_size = 30
generator = Sequential([
Dense(32, activation='selu', input_shape=[codings_size]),
Dense(64, activation='selu'),
Dense(128, activation='selu'),
Dense(28 * 28, activation='sigmoid'),
Reshape(in_shape)
])
Listing 10-1
The Generator
```

如我们所知，*生成器*用于从问题域生成新的合理示例。它接受一个随机分布作为输入（通常是高斯分布）并输出一些数据（通常是图像）。随机输入可以被认为是待生成图像的潜在表示（或编码）。因此，生成器具有与变分自编码器中的解码器相同的功能，并且可以以相同的方式用于生成新图像。然而，它们的训练方式非常不同。

生成器通过结合判别器的反馈来学习创建伪造数据。它学习使判别器将其输出分类为真实。

生成器训练需要比判别器训练更紧密的生成器和判别器之间的集成。训练生成器的 GAN 部分包括随机输入、生成器网络、判别器输出和生成器损失。生成器网络将随机输入转换为数据实例。判别器网络对生成的数据进行分类。生成器损失对生成器未能欺骗判别器进行惩罚。

现在，按照列表 10-2 构建判别器。

```py
discriminator = Sequential([
Flatten(input_shape=in_shape),
Dense(128, activation='selu'),
Dense(64, activation='selu'),
Dense(32, activation='selu'),
Dense(1, activation='sigmoid')
])
Listing 10-2
The Discriminator
```

如我们所知，*判别器*用于将示例分类为真实（来自域）或伪造（由生成器生成）。它接受来自生成器的伪造图像或来自训练集的真实图像作为输入，并猜测输入图像是伪造的还是真实的。判别器是一个常规的二分类器（真实或伪造图像）。它接受一个图像作为输入，并以包含单个单元的 Dense 层结束。

创建模型：

```py
gan = Sequential([generator, discriminator])
```

### 编译判别器模型

由于判别器本质上是一个二元分类器（伪造或真实图像），我们自然使用二元交叉熵损失。由于生成器仅通过 GAN 模型进行训练，我们**不需要**编译它。GAN 模型也是一个二元分类器，因此它可以使用相同的损失函数。重要的是，在第二阶段训练判别器之前，不应对其进行训练。因此，我们在编译 GAN 模型之前将其设置为不可训练。

```py
discriminator.compile(
loss='binary_crossentropy', optimizer='rmsprop')
discriminator.trainable = False
gan.compile(loss='binary_crossentropy', optimizer='rmsprop')
```

输入到判别器的训练数据来自两个来源——真实数据和伪造数据。在我们的实验中，真实数据实例是 Fashion-MNIST 图像。判别器将这些实例用作训练中的正例。伪造数据实例由生成器创建。判别器将这些实例用作训练中的负例。

在判别器训练期间，生成器不进行训练。其权重保持不变，同时它为判别器生成训练示例。

判别器连接到两个损失函数——生成器和判别器。在判别器训练期间，判别器忽略生成器损失，仅使用判别器损失。判别器对来自生成器的真实数据和伪造数据进行分类。判别器损失对判别器因错误地将真实实例分类为伪造实例或伪造实例分类为真实实例进行惩罚。判别器通过从判别器损失通过判别器网络进行反向传播来更新其权重。

### 构建输入管道

使用 32 个批次的管道构建：

```py
batch_size = 32
dataset = tf.data.Dataset.from_tensor_slices(
images).shuffle(1000)
dataset = dataset.batch(
batch_size, drop_remainder=True).prefetch(1)
```

### 创建自定义训练循环

由于训练循环不寻常，我们无法使用常规的 fit 方法。相反，我们创建一个自定义循环，该循环接受一个数据集以迭代图像。训练不寻常，因为它不是顺序的。在第一阶段，判别器在生成器提供的真实和伪造图像上进行训练。在第二阶段，生成器在判别器从其学习内容中产生的结果上进行训练。因此，在训练过程中存在反馈循环。

创建一个如列表 10-3 所示的训练循环函数。

```py
def train_gan(gan, dataset, batch_size,
codings_size, n_epochs=50):
generator, discriminator = gan.layers
for epoch in range(n_epochs):
print('Epoch {}/{}'.format(epoch + 1, n_epochs))
for X_batch in dataset:
# phase 1 - training the discriminator
noise = tf.random.normal(
shape=[batch_size, codings_size])
generated_images = generator(noise)
X_fake_and_real = tf.concat(
[generated_images, X_batch], axis=0)
y1 = tf.constant([[0.]] * batch_size + [[1.]] * batch_size)
discriminator.trainable = True
discriminator.train_on_batch(X_fake_and_real, y1)
# phase 2 - training the generator
noise = tf.random.normal(
shape=[batch_size, codings_size])
y2 = tf.constant([[1.]] * batch_size)
discriminator.trainable = False
gan.train_on_batch(noise, y2)
plot_multiple_images(generated_images, 8)
plt.show()
Listing 10-3
Custom Training Loop Function
```

每个训练迭代分为两个阶段。第一阶段训练判别器。从训练集中采样一批真实图像，生成器生成相同数量的假图像。标签设置为 0 表示假图像，1 表示真实图像。判别器使用二元交叉熵损失在一个步骤上对标记的批次进行训练。重要的是，在这个阶段，反向传播只优化判别器的权重。第二阶段训练生成器。生成器生成另一批假图像，判别器区分假图像和真实图像。批中没有添加真实图像，所有标签都设置为 1 以表示它们是真实的（尽管它们不是真实的）。目的是让生成器生成判别器错误地认为是真实的图像！关键的是，在这个步骤中冻结判别器的权重，所以反向传播只影响生成器的权重。

从技术角度讲，第一阶段向生成器输入高斯噪声以生成假图像。目标*y1*设置为 0 表示假图像，1 表示真实图像。然后，在批次上训练判别器。第二阶段向 GAN 输入一些高斯噪声。生成器首先生成假图像。然后判别器尝试猜测图像是真是假。目的是让判别器相信假图像是真的，因此目标*y2*设置为 1。

人类可以轻易忽略噪声，但机器学习算法却很困难。微小的、人类难以察觉的像素变化可以显著改变神经网络做出准确预测的能力。研究表明，*噪声*和*高斯模糊*在测试模型上显示出几乎立即的平滑效果。**高斯模糊**（也称为高斯平滑）是通过高斯函数模糊图像的结果。它在机器学习中广泛用于减少图像噪声和减少细节。

关于添加噪声的绝佳资源，请参阅

[为什么要在机器学习中向图像添加噪声](https://blog.roboflow.com/why-to-add-noise-to-images-for-machine-learning/)

创建一个如列表 10-4 所示的绘图函数。

```py
import matplotlib.pyplot as plt
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
Listing 10-4
Plotting Function for Generated Images
```

使用未训练的生成器生成图像：

```py
tf.random.set_seed(0)
np.random.seed(0)
noise = tf.random.normal(shape=[batch_size, codings_size])
generated_images = generator(noise)
plot_multiple_images(generated_images, 8)
```

如预期的那样，生成的图像并不太令人印象深刻。当然，我们还没有训练好模型！

### 训练 GAN

训练 GAN 几个时期：

```py
n = 5
train_gan(gan, dataset, batch_size, codings_size, n_epochs=n)
```

生成的图像比完全没有训练要好，但仍然不太令人印象深刻！我们训练了 50 个时期，但几乎没有改进。也就是说，图像质量不再提高。我们在这里只训练了几个时期，因为需要相当多的时间。你可以自由地训练更多的时期，但可能需要吃个午餐，因为训练会花费很多时间。

注意

如果图像看起来像团块，请重新启动 Colab 笔记本运行时并重新运行实验。

从训练好的 GAN 生成图像：

```py
np.random.seed(0)
tf.random.set_seed(0)
noise = tf.random.normal(shape=[batch_size, codings_size])
generated_images = generator(noise)
plot_multiple_images(generated_images, 8)
```

添加一些高斯噪声，并使用生成器模型生成一些图像。由于批量大小为 32，因此生成了 32 张图像。我们添加高斯噪声以补偿使用高斯噪声训练 GAN。

## 在小图像上实验 DCGAN

使用 CNN 架构与 GAN 结合可以产生比简单的前馈网络与 GAN 结合更好的结果。**深度卷积生成对抗网络**（DCGAN）是一种 GAN，其生成器和分辨器使用卷积神经网络。我们不是使用前馈网络作为生成器和分辨器，而是用 CNN 代替。DCGAN 架构在许多情况下在图像处理任务中实现了优越的性能。

### 创建生成器

导入库：

```py
from tensorflow.keras.layers import BatchNormalization,\
Conv2D, Conv2DTranspose, LeakyReLU, Dropout
```

创建与列表 10-5 中所示的生成器。

```py
codings_size = 100
dc_generator = Sequential([
Dense(7 * 7 * 128, input_shape=[codings_size]),
Reshape([7, 7, 128]),
BatchNormalization(),
Conv2DTranspose(
64, kernel_size=5, strides=2, padding='SAME',
activation='selu'),
BatchNormalization(),
Conv2DTranspose(
1, kernel_size=5, strides=2, padding='SAME',
activation='tanh'),
])
Listing 10-5
DCGAN Generator
```

首先向生成器提供小张量（7 × 7 像素），并将其投影到 128 个维度，形成一个 6,272 维的空间。将 7 × 7 × 128 相乘以获得维空间。目标是增加张量图像的大小以匹配 28 × 28 的 Fashion-MNIST 图像，并将深度减少到 1 以匹配通道大小。我们通过试错实验创建了此网络。尝试不同的初始张量大小，但请注意，这样做会改变生成器和分辨器的神经元计算。

具体来说，生成器接受大小为 100 的编码，并将它们投影到 6,272 个维度（7 × 7 × 128）。然后生成器将投影重塑为一个 7 × 7 × 128 的张量，该张量经过批量归一化，并输入到一个步长为 2 的转置卷积层。步长将张量上采样到 14 × 14，因为它将 7 × 7 维度翻倍。该层还将张量的深度从 128 减少到 64。结果是再次进行批量归一化，并输入到另一个步长为 2 的转置卷积层。步长将其上采样到 28 × 28，因为它将 14 × 14 维度翻倍。该层还将张量的深度从 64 减少到 1。输出张量的形状为(28, 28, 1)，这是目标，因为 Fashion-MNIST 图像的形状为 28 × 28 × 1。由于最终层使用*tanh*激活，输出在-1 和 1 之间重塑。

双曲正切激活函数也被称为 tanh 函数。它与 sigmoid 激活函数非常相似，甚至具有相同的 S 形状。但该函数接受任何实数值作为输入，并输出范围在-1 到 1 之间的值。我们之所以使用 tanh 而不是 sigmoid 作为输出层，是因为在这种情况下它的表现更好。

使用未训练的生成器生成图像：

```py
tf.random.set_seed(0)
np.random.seed(0)
noise = tf.random.normal(shape=[batch_size, codings_size])
generated_images = dc_generator(noise)
plot_multiple_images(generated_images, 8)
```

### 创建分辨器

创建与列表 10-6 中所示的分辨器。

```py
dc_discriminator = Sequential([
Conv2D(64, kernel_size=5, strides=2, padding='SAME',
activation=LeakyReLU(0.2),
input_shape=[28, 28, 1]),
Dropout(0.4),
Conv2D(128, kernel_size=5, strides=2, padding='SAME',
activation=LeakyReLU(0.2)),
Dropout(0.4),
Flatten(),
Dense(1, activation='sigmoid')
])
Listing 10-6
DCGAN Discriminator
```

分辨器看起来像用于二分类的常规 CNN（最终密集层为 1），但它使用步长进行下采样而不是最大池化层。

### 创建 DCGAN

从 DCGAN 生成器和分辨器创建 DCGAN：

```py
dcgan = Sequential([dc_generator, dc_discriminator])
```

### 编译分辨器模型

由于判别器自然是一个二元分类器（伪造或真实图像），我们自然使用二元交叉熵损失。由于生成器仅通过 DCGAN 模型进行训练，所以我们*不需要*编译它。DCGAN 模型也是一个二元分类器，因此它可以使用相同的损失函数。重要的是，判别器在第二阶段不应该进行训练。因此，我们在编译 DCGAN 模型之前将其设置为不可训练。

编译 DCGAN：

```py
dc_discriminator.compile(
loss='binary_crossentropy', optimizer='rmsprop')
dc_discriminator.trainable = False
dcgan.compile(loss='binary_crossentropy', optimizer='rmsprop')
```

### 重新塑形

由于生成器最终层的 tanh 激活导致输出范围从-1 到 1，因此将训练集重新缩放到相同的范围：

```py
images_dcgan = tf.reshape(
images, [-1, 28, 28, 1]) * 2\. - 1.
```

### 构建输入管道

使用 32 个批次的批量大小构建管道：

```py
batch_size = 32
dataset = tf.data.Dataset.from_tensor_slices(images_dcgan)
dataset = dataset.shuffle(1000)
dataset = dataset.batch(
batch_size, drop_remainder=True).prefetch(1)
```

### 训练

清除并设置随机种子：

```py
tf.keras.backend.clear_session()
np.random.seed(0)
tf.random.set_seed(0)
```

训练几个周期：

```py
n = 5
train_gan(
dcgan, dataset, batch_size, codings_size, n_epochs=n)
```

哇！好多了！

### 使用训练好的生成器生成图像

使用训练好的 DC 生成器生成图像：

```py
tf.random.set_seed(0)
np.random.seed(0)
noise = tf.random.normal(shape=[batch_size, codings_size])
generated_images = dc_generator(noise)
plot_multiple_images(generated_images, 8)
```

## 大图像深度卷积生成对抗网络实验

一个非常简单的深度卷积生成对抗网络在 Fashion-MNIST 上表现良好。但这个数据集中的图像很小，是灰度的。让我们看看深度卷积生成对抗网络在包含玩剪刀石头布游戏的手的大彩色图像的*rock_paper_scissors*数据集上的表现如何。

### 检查元数据

加载数据集以检查其元数据：

```py
rps, info = tfds.load('rock_paper_scissors', with_info=True,
split='train', try_gcs=True)
```

检查元数据对象：

```py
info
```

获取类别标签和类别数量：

```py
num_classes = info.features['label'].num_classes
classes = info.features['label'].names
classes, num_classes
```

可视化一些示例：

```py
fig = tfds.show_examples(rps, info)
```

### 加载训练数据

将训练和测试图像作为 NumPy 数组加载：

```py
(x_train_img, _), (x_test_img, _) = tfds.as_numpy(
tfds.load('rock_paper_scissors', split=['train','test'],
batch_size=-1, as_supervised=True,
try_gcs=True))
```

检查形状：

```py
for element in range(10):
print (x_train_img.shape)
```

该数据集包含 2,520 个 300 × 300 × 3 的图像。由于图像大小相同，我们不需要调整大小。然而，我们调整大小是为了在章节后面解释的另一个原因。

### 处理数据

将数据集缩小到 2 的幂以保证批次相等：

```py
x_train_rps = x_train_img[:2048]
len(x_train_rps)
```

当我们尝试使用原始数据集大小进行训练时，我们遇到了错误。因此，我们创建了相等的批次以消除错误。

创建一个重新格式化图像的函数：

```py
IMAGE_RES = 256
def format_image(image):
image = tf.image.resize(image, (IMAGE_RES, IMAGE_RES))/255.0
return image
```

虽然我们不需要调整图像大小，但我们这样做是为了更容易创建具有适当神经元数量的网络层的模型。我们在本章前面提到，所有图像都是 300 × 300 像素大小。但我们将其调整到 2 的幂以方便。

生成器开始时，将相对较小的图像投影到具有大深度的多个维度上，并逐渐增加图像大小并减少深度大小，直到输出我们想要的图像大小。因此，使用能被 2 整除和乘以 2 的值更容易工作。深度学习模型在模型中神经元的数量基于 2 的幂时也表现更好。

### 构建输入管道

从特征图像创建张量：

```py
train_slice = tf.data.Dataset.from_tensor_slices(x_train_rps)
```

转换图像以获得最佳性能：

```py
BATCH_SIZE = 32
SHUFFLE_SIZE = 500
train_rps = (train_slice.
shuffle(SHUFFLE_SIZE).
map(format_image).
batch(BATCH_SIZE).
cache().
prefetch(1))
train_rps
```

如列表 10-7 所示，可视化一批示例。

```py
plt.figure(figsize=(10, 10))
for images in train_rps.take(1):
for i in range(9):
ax = plt.subplot(3, 3, i + 1)
plt.imshow(images[i])
plt.axis('off')
Listing 10-7
Visualize Examples from a Batch
```

我们没有显示标签，因为我们不需要它们进行我们的无监督实验。

### 构建模型

清除并设置随机种子：

```py
tf.keras.backend.clear_session()
np.random.seed(0)
tf.random.set_seed(0)
```

如列表 10-8 所示创建生成器。

```py
codings_size = 100
gencolor = Sequential([
Dense(32 * 32 * 256, input_shape=[codings_size]),
Reshape([32, 32, 256]),
BatchNormalization(),
Conv2DTranspose(128, kernel_size=5, strides=2, padding='SAME',
activation='selu'),
BatchNormalization(),
Conv2DTranspose(64, kernel_size=5, strides=2, padding='SAME',
activation='selu'),
BatchNormalization(),
Conv2DTranspose(3, kernel_size=5, strides=2, padding='SAME',
activation='tanh'),
])
Listing 10-8
The Generator
```

神经元的数量基于 2 的幂，只有一个例外。最终层包含三个神经元来表示类别数量。

由于我们想要生成大型的彩色图像，首先将一个更大的（32 × 32）张量投影到 256 维，然后输入给生成器。因此，结果维度空间是 262,144。将 32 乘以 32 乘以 256 得到这个值。

生成器将投影重塑为一个 32 × 32 × 256 的张量，该张量经过批量归一化，然后输入到一个步长为 2 的转置卷积层。步长将张量上采样到 64 × 64，因为它将 32 × 32 维度加倍。该层还将张量的深度从 256 减少到 128。结果是再次批量归一化，然后输入到另一个步长为 2 的转置卷积层。步长将其上采样到 128 × 128，因为它将 64 × 64 维度加倍。该层还将张量的深度从 128 减少到 64。结果是再次批量归一化，然后输入到另一个步长为 2 的转置卷积层。步长将其上采样到 256 × 256，因为它将 128 × 128 维度加倍。该层还将张量的深度从 64 减少到 3。

输出张量的形状为（256，256，3），这是我们想要的目标，因为我们希望它们是原始的 256 × 256 × 3 形状。由于最终层使用 tanh 激活，输出在-1 和 1 之间重塑。

注意

希望你能看到我们为什么使用 2 的幂次来分层模型。我们以一个大的维度空间开始我们的生成器，然后一层层地减少张量维度，回到我们数据集图像的原始形状。通过 2 上采样和减少张量是非常容易的数学运算。

检查：

```py
gencolor.summary()
```

创建如图 10-9 所示的判别器。

```py
discolor = Sequential([
Conv2D(64, kernel_size=5, strides=2, padding='SAME',
activation=LeakyReLU(0.2),
input_shape=[256, 256, 3]),
Dropout(0.4),
Conv2D(128, kernel_size=5, strides=2, padding='SAME',
activation=LeakyReLU(0.2)),
Dropout(0.4),
Flatten(),
Dense(1, activation='sigmoid')
])
Listing 10-9
The Discriminator
```

判别器接受 256 × 256 × 3 图像，将张量下采样到 128 × 128，并将深度从 32 增加到 64。张量被输入到一个转置卷积层，该层将张量下采样到 64 × 64，并将深度增加到 128。结果是展平到一个 524,288 维的空间（64 × 64 × 128）。由于我们正在进行二元分类，最终的密集层只包含**一个**神经元。

生成器上采样张量以激活更高维度空间中的神经元。判别器将张量下采样回原始图像大小，以对数据进行有效特征分类。

检查：

```py
discolor.summary()
```

创建 DCGAN：

```py
dcgan_color = Sequential([gencolor, discolor])
```

### 编译判别器模型

由于判别器天然是一个二元分类器（伪造或真实图像），我们自然使用二元交叉熵损失。由于生成器仅通过 DCGAN 模型进行训练，所以我们**不需要**编译它。DCGAN 模型也是一个二元分类器，因此它可以使用相同的损失函数。重要的是，在第二阶段不应该训练判别器。因此，在编译 DCGAN 模型之前，我们将其设置为不可训练：

```py
discolor.compile(
loss='binary_crossentropy', optimizer='rmsprop')
discolor.trainable = False
dcgan_color.compile(
loss='binary_crossentropy', optimizer='rmsprop')
```

### 重新缩放

创建一个重新缩放的函数：

```py
def rescale(image):
image = tf.math.multiply(image, image * 2\. -1)
return image
```

由于生成器最终层的 tanh 激活导致输出范围从-1 到 1，因此将训练集重新缩放到相同的范围。

重新缩放图像：

```py
train_color = train_rps.map(rescale)
```

### 训练模型并生成图像

创建一个函数来绘制如图 10-10 所示的图像。

```py
def plot_color(images, n_cols=None):
n_cols = n_cols or len(images)
n_rows = (len(images) - 1) // n_cols + 1
images = np.clip(images, 0, 1)
if images.shape[-1] == 1:
images = np.squeeze(images, axis=-1)
plt.figure(figsize=(n_cols, n_rows))
for index, image in enumerate(images):
plt.subplot(n_rows, n_cols, index + 1)
plt.imshow(image, cmap='binary')
plt.axis('off')
Listing 10-10
Plotting Function for Generated Images
```

创建一个训练循环函数来训练 DCGAN，如图 10-11 所示。

```py
def train_gan(
gan, dataset, batch_size, codings_size, n_epochs=50):
generator, discriminator = gan.layers
for epoch in range(n_epochs):
print('Epoch {}/{}'.format(epoch + 1, n_epochs))
for X_batch in dataset:
# phase 1 - training the discriminator
noise = tf.random.normal(
shape=[batch_size, codings_size])
generated_images = generator(noise)
X_fake_and_real = tf.concat(
[generated_images, X_batch], axis=0)
y1 = tf.constant(
[[0.]] * batch_size + [[1.]] * batch_size)
discriminator.trainable = True
discriminator.train_on_batch(X_fake_and_real, y1)
# phase 2 - training the generator
noise = tf.random.normal(
shape=[batch_size, codings_size])
y2 = tf.constant([[1.]] * batch_size)
discriminator.trainable = False
gan.train_on_batch(noise, y2)
plot_color(generated_images, 8)
plt.show()
Listing 10-11
Training Loop Function
```

训练 DCGAN：

```py
train_gan(
dcgan_color, train_color, BATCH_SIZE, codings_size, 10)
```

只需经过几个 epoch，我们就可以看到生成的图像开始代表石头、剪刀、布的手势。但我们所构建的生成器和判别器模型非常简单。为了生成逼真的图像，我们需要创建一个更加健壮的模型。

## 摘要

在本章中，我们构建了一个具有正向网络的 GAN，其生成的复制品质量一般。对于 Fashion-MNIST，使用 DCGAN 生成的复制品更加逼真。对于石头剪刀布，复制品并不出色。我们可以尝试训练模型更多的 epoch，但这需要花费大量时间，并且可能需要比你的电脑能处理的更多内存。
