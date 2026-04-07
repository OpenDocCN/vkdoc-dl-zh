# 11. 逐步增长生成对抗网络

GANs 在生成清晰的合成图像方面非常有效，但大小有限，约为 64 × 64 像素。**逐步增长 GAN**是 GAN 的一种扩展，它使生成器模型能够生成高达约 1024 × 1024 像素（截至本文撰写时）的大尺寸高质量图像。这种方法已被证明在生成高度逼真的合成人脸方面非常有效。

逐步增长 GANs 的关键创新是生成器输出图像大小的增量增加。通过在训练开始时生成小图像，并逐渐向生成器和判别器模型添加卷积层，生成越来越大的图像，分辨率越来越细。

该技术从低分辨率向量作为输入开始。它通过逐步增加生成器和判别器的新层来增长生成器和判别器，从而使模型在训练过程中能够越来越多地学习细节。该技术还通过生成前所未有的高质量图像来加速和稳定训练。

## 潜在空间学习

**潜在空间**是压缩数据的一种表示，其中相似的数据点在空间中更靠近。潜在空间对于学习数据特征和找到数据更简单的表示以进行分析很有用。创建潜在空间背后的想法是压缩现实，以便向量数学可以工作。潜在空间也可以称为*潜在向量空间*。

当我们观察我们的世界时，我们看到的是一个由像素（或*观察像素空间*）组成的广阔景观。但我们如何从这样巨大的数据画布中学习呢？一种解决方案是创建一个潜在空间，将观察像素空间压缩成可管理的像素图像。例如，为了教会模型学习人脸，我们首先获取（或使用现有的）他们的照片。然后我们将照片转换成一系列像素图像。因此，每个面孔都由一组像素表示。现在我们有了潜在表示，我们可以在图像像素上应用微积分和向量代数，以教学习模型人类面孔的本质（至少从像素图像的角度来看）。实质上，我们实验的潜在空间是从实际人类面孔的观察像素空间中提取的压缩像素空间表示。

在观察像素空间中，任何两个图像之间可能没有立即的相似性。但是将像素空间映射到潜在空间可以将图像压缩得更加接近，这样我们就可以更容易地了解这类图像。

GAN 架构中的生成模型学习将潜在空间中的点映射到图像生成。它从潜在空间中取一个点作为输入，并应用向量运算来生成一个新的图像。在潜在空间中两点之间的线性路径上也可以创建一系列点，以生成多个生成的图像。在实践中，生成模型有效地使用其潜在空间表示来在其潜在空间中的点之间进行插值，目的是从其生成的图像中得出有意义和针对性的效果。但是，潜在空间只有在应用于正在训练的生成模型时才有意义。也就是说，每个学习实验都有自己的潜在空间。

各章节的笔记本位于以下 URL：

[`github.com/paperd/deep-learning-models`](https://github.com/paperd/deep-learning-models)

我们展示了两个渐进式增长 GAN 实验。第一个实验从预训练模型生成图像。第二个实验创建一个自定义训练循环，从初始生成的图像中学习目标图像。通过导入主 TensorFlow 库和实例化 GPU 来开始设置 Colab 生态系统。

## 导入 TensorFlow 库

导入库并将其别名为**tf**：

```py
import tensorflow as tf
```

将 TensorFlow 库别名为 tf 是常见做法。

## GPU 硬件加速器

作为便利，我们包括在 Colab 笔记本中启用 GPU 的步骤：

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

如果出现错误**名称** **‘****TF****’** **未定义**，重新执行代码以导入 TensorFlow 库！

## 创建实验环境

两个实验都期望包含图像和动画显示的包、库和函数。

### 安装用于创建动画的包

安装 imageio、scikit-image 和 TensorFlow 文档包以启用实验的动画创建：

```py
!pip -q install imageio
!pip -q install scikit-image
!pip install -q git+https://github.com/tensorflow/docs
```

### 安装库

在标准日志模块之上安装日志模块：

```py
from absl import logging
```

日志模块来自 Abseil Python 包，该包为构建 Python 应用程序提供各种库。该模块定义了函数和类，允许设计人员为应用程序和库构建灵活的事件日志系统。日志 API 的关键优势是所有 Python 模块都可以参与日志记录。因此，我们的应用程序日志可以包括其自己的消息，这些消息与第三方模块的消息集成。

安装图像处理库：

```py
import imageio
import PIL.Image
import matplotlib.pyplot as plt
from IPython import display
from skimage import transform
```

*imageio* 库提供了一个易于使用的接口，用于读取和写入各种图像数据，包括动画图像、体积数据和科学格式。*PIL.Image* 模块用于表示 Python Imaging Library (PIL) 图像。该库增加了对打开、操作和保存许多不同图像文件格式的支持。*plt* 库用于显示图像。*display* 模块是 IPython 中显示工具的公共 API。*IPython* 是一个用 Python 构建的交互式外壳。*transform* 模块用于图像处理。

导入其他必需的库：

```py
import numpy as np
import tensorflow_hub as hub
from tensorflow_docs.vis import embed
import time
```

*NumPy* 是一个用于处理标量、数组和矩阵的 Python 库。*hub* 模块允许访问 TensorFlow Hub，这是一个训练好的机器学习模型的存储库。*embed* 模块用于在笔记本中嵌入动画。*time* 库提供了许多在代码中表示时间的方法，如对象、数字和字符串。

## 创建用于图像显示的函数

创建一个用于显示 PIL 图像的函数：

```py
def display_image(image):
image = tf.constant(image)
image = tf.image.convert_image_dtype(image, tf.uint8)
return PIL.Image.fromarray(image.numpy())
```

创建一个用于显示图像的函数：

```py
def show_image(image):
plt.imshow(image)
plt.axis('off')
plt.show()
```

使用 *imshow()* 函数来显示数据作为图像。

创建一个用于显示如图 11-1 所示动画的函数。

```py
def animate(images):
images = np.array(images)
converted_images = np.clip(
images * 255, 0, 255).astype(np.uint8)
imageio.mimsave('./animation.gif', converted_images)
return embed.embed_file('./animation.gif')
Listing 11-1
Function to Display Animation
```

## 创建潜在空间维度

潜在空间对于学习数据特征和找到数据分析的简化表示非常有用。人类对广泛的主题及其相关事件有理解。潜在空间旨在通过定量的空间表示为计算机模型提供类似的理解。因此，潜在空间实际上只是特定训练实验中压缩现实的可测量空间表示。

将数据集压缩到潜在空间有助于模型更好地理解观察到的数据，因为模型处理的变化比整个数据集要小得多。因此，模型从比实际观察到的像素空间更小的空间中学习。

高维和低维这些术语有助于定义我们希望我们的潜在空间学习并表示的特征的具体性或普遍性。*高维* 潜在空间对输入数据的更具体特征敏感，但如果没有足够的训练数据，有时会导致过拟合。*低维* 潜在空间旨在捕捉学习并表示输入数据所需的最重要特征（或方面）。

设置高维潜在空间为 512，因为我们用于此实验的预训练模型是在这个潜在空间上训练的：

```py
latent_dim = 512
```

预训练模型将 512 维的潜在空间映射到图像。如果我们事先不知道使用的模块，我们可以从 *module.structured_input_signature* 方法中检索潜在维度。我们将在本章后面展示如何做到这一点。

## 设置错误日志的详细程度

要查看日志错误：

```py
logging.set_verbosity(logging.ERROR)
```

## 图像生成实验

我们使用 progan-128 预训练模型生成逼真的人脸图像。*progan-128* 模型是在 CelebA（CelebFaces Attributes Dataset）上训练的 Progressive GAN，用于 128 × 128 像素图像。它从 512 维潜在空间映射到图像。在训练过程中，潜在空间向量是从正态分布中采样的。

该模块接受一个表示一批潜在向量的张量（Tensor(tf.float32, shape=[?, 512]）作为输入，并输出一个表示一批 RGB 图像的张量（Tensor(tf.float32, shape=[?, 128, 128, 3]））。原始模型在 GPU 上训练了 636,801 步，批大小为 16。

*CelebA* 是一个包含超过 200,000 张名人图像的大规模人脸属性数据集。每个图像有 40 个属性注释。该数据集中的图像覆盖了大量的姿态变化和背景杂乱。CelebA 还具有多样性大、数量多和丰富的注释，包括

+   * 10,177 张独特名人图像

+   * 202,599 张人脸图像

+   * 5 个地标位置

+   * 每个图像 40 个二进制属性注释

CelebA 可以用作计算机视觉任务（如人脸属性识别、人脸检测、地标（或面部部分）定位、人脸编辑和合成）的训练和测试集。

### 创建一个用于插值超球面的函数

创建一个函数来插值潜在空间中向量之间的空间，如列表 11-2 所示。

```py
def interpolate_hypersphere(v1, v2, num_steps):
v1_norm = tf.norm(v1)
v2_norm = tf.norm(v2)
v2_normalized = v2 * (v1_norm / v2_norm)
vectors = []
for step in range(num_steps):
interpolated =\
v1 + (v2_normalized - v1) *\
step / (num_steps - 1)
interpolated_norm = tf.norm(interpolated)
interpolated_normalized =\
interpolated * (v1_norm / interpolated_norm)
vectors.append(interpolated_normalized)
return tf.stack(vectors)
Listing 11-2
Function to Interpolate the Hypersphere
```

该函数从两个随机初始化的向量开始，在潜在空间中进行潜在空间插值。它插值非零向量，并且这两个向量不都位于通过原点的直线上，然后将归一化的插值向量返回到调用环境。**图像** **插值**是从一系列图像生成中间图像的过程。

函数首先创建两个欧几里得范数向量 v1 和 v2。然后它将 v2 归一化，使其具有与 v1 相同的范数。然后它在超球面（或潜在空间）上在这两个向量之间进行插值，以产生基于插值步数的向量集。

**超球面**是球面的四维类似物。虽然球面存在于三维空间中，但其表面是二维的。同样，超球面有一个三维的表面，它弯曲到四维空间中。我们的宇宙可能是超球面的超曲面。

### 加载预训练模型

设置全局随机种子以保持可重复性：

```py
tf.random.set_seed(7)
```

设置随机种子值。可以使用任何你想要的数字。但为了可重复性，所有实验中都要使用相同的数字。

加载预训练的 Progressive GAN (progan-128)：

```py
hub_model = hub.load(
'https://tfhub.dev/google/progan-128/1')\
.signatures['default']
```

获取输出形状：

```py
hub_model.output_shapes
```

Progressive GAN (progan-128) 在 CelebA 上使用 128 × 128 × 3 像素大小的图像进行训练。

获取潜在空间的维度：

```py
hub_model.structured_input_signature
```

Progan-128 模型将 512 维的潜在空间映射到图像。在训练过程中，潜在空间向量是从正态分布中采样的。

### 生成并显示图像

可以从潜在空间中的随机点生成一个新图像。首先在潜在空间中创建一个随机正态向量。然后继续将向量输入到 progan-128 中。预训练模型通过最小化真实和生成分布之间的总距离来算法性地识别最近的潜在向量，并从最近的向量生成一个图像。

创建一个函数来找到潜在空间中最接近的向量：

```py
def get_module_space_image():
vector = tf.random.normal([1, latent_dim])
image = hub_model(vector)['default'][0]
return image
```

函数在 1 到 512（我们潜在空间的大小）之间创建一个随机正态向量。然后使用 progan-128 从创建的随机潜在向量最近的潜在向量生成一个图像。

显示一个生成的图像：

```py
generated_image = get_module_space_image()
display_image(generated_image)
```

还不错！预训练模型从潜在空间中抽取的向量生成了一个相对逼真的图像。

### 创建一个生成多个图像的函数

函数创建两个随机向量，在潜在空间中插值它们之间的空间，并使用 progan-128 生成如列表 11-3 中所示的图像。

```py
def interpolate_between_vectors(steps):
v1 = tf.random.normal([latent_dim])
v2 = tf.random.normal([latent_dim])
vectors = interpolate_hypersphere(v1, v2, steps)
interpolated_images = hub_model(vectors)['default']
return interpolated_images
Listing 11-3
Function to Generate Interpolated Images
```

函数从 512 维潜在空间中创建两个随机正态变量。然后创建一个张量，包含从 v1 到 v2 之间的*n*个插值步。最后，使用 progan-128 根据张量生成*n*个图像。由于张量由 n 个插值步组成，生成的图像位于由 v1 和 v2 之间潜在空间中最近的向量所界定的插值线上。

### 创建一个动画

基于 n 个生成的图像创建一个动画：

```py
interpolation_steps = 100
interpolated_images = interpolate_between_vectors(
interpolation_steps)
animate(interpolated_images)
```

非常惊人！借助 progan-128，我们根据潜在空间中两个向量之间的插值创建了一个动画。步骤的数量在动画过程中有很大影响！步骤数越多，在潜在空间中的向量之间生成的插值图像就越多。所以尝试不同的步骤数，看看会发生什么。但不要使用太大的 n 值，因为您可能会耗尽内存！

小贴士

保持插值步数相对较低以避免内存崩溃！尽管 Colab 是一个云服务，但免费版本在为笔记本分配的 RAM 数量上相当有限。

### 显示插值图像向量

显示插值图像向量以分解图像生成过程。

获取由随机向量 v1 和 v2 界定的潜在空间中插值图像的数量：

```py
num_imgs = len(interpolated_images)
num_imgs
```

我们将插值步数设置为 100，因此我们有 100 个插值图像。

显示初始插值图像：

```py
show_image(interpolated_images[0])
```

显示最终的插值图像：

```py
show_image(interpolated_images[num_imgs - 1])
```

因此，动画从初始图像开始，逐渐变为最终图像，因为插值向量位于由随机向量 v1 和 v2 界定的潜在空间线上的一个点。

创建一个函数来显示从潜在空间生成的图像向量，如列表 11-4 所示。

```py
def generated_images(images, cols, rows):
columns, rows = cols, rows
ax = []
fig = plt.figure(figsize=(20, 20))
for i in range(columns*rows):
img = images[i].numpy()
ax.append(fig.add_subplot(rows, columns, i+1))
plt.imshow(img)
plt.axis('off')
Listing 11-4
Function to Display Generated Image
```

显示插值图像：

```py
generated_images(interpolated_images, 10, 10)
```

看到模型如何根据潜在空间中的插值向量从初始图像到最终图像创建一系列图像，这是一个非常迷人的过程！因此，学习模型可以在潜在空间中两点之间创建一系列线性路径上的向量，正如我们通过我们的动画所展示的那样。

### 从单个向量显示多张图像

创建一个函数以返回一个潜在向量和图像：

```py
def get_vector():
vector = tf.random.normal([1, latent_dim])
image = hub_model(vector)['default'][0]
return vector, image
```

函数返回一个随机正态潜在向量和由 progan-128 从随机潜在向量生成的图像。每次函数执行时，图像都不同，因为每个潜在向量都是从随机向量生成的，输入到 progan-128 中，然后生成的。

显示：

```py
for _ in range(2):
latent_vector, image = get_vector()
print (latent_vector[0][0:3])
show_image(image)
```

对于每个循环周期，都会显示每个潜在向量及其对应的图像的一个切片。这两个图像彼此不同且没有关系，因为我们为每个循环周期创建了一个接近*随机向量*的潜在向量。但是，向量之间没有路径，因为我们没有在两个随机向量定义的线性路径上创建一系列潜在向量！我们只为每个循环周期从潜在空间中插值了一个单个向量。

显示如列表 11-5 中所示的多张图像。

```py
rows, cols = 2, 2
plt.figure(figsize=(10, 10))
for i in range(rows*cols):
plt.subplot(rows, cols, i + 1)
plt.imshow(get_module_space_image())
plt.axis('off')
Listing 11-5
Display Several Images Based on a Single Random Vector
```

再次强调，由于每次调用函数时都会创建一个接近随机潜在向量的潜在向量，因此图像彼此不同且没有关系！

### 从上传的图像创建一个目标潜在向量

如我们之前所展示的，我们可以创建一个随机正态潜在向量并从中生成一个新的图像。或者，我们可以从一个上传的图像向量创建一个向量并从中生成一个新的图像。

导入所需的库：

```py
from google.colab import files
```

创建一个函数以从您的本地驱动器获取上传的图像：

```py
def upload_image():
uploaded = files.upload()
image = imageio.imread(uploaded[list(uploaded.keys())[0]])
return transform.resize(image, [128, 128])
```

函数使用*files.upload()*从*google.colab*库启用文件上传到笔记本。它使用*imread()*从*imageio*库读取本地驱动器上的图像。该函数使用*skimage*库中的*transform.resize()*返回一个符合 progan-128 期望的调整大小后的图像。

注意

请确保您的本地驱动器上有一个图像可以上传。为了方便，我们建议使用本章网站上包含的图像之一。只需将图像复制到您的本地驱动器上。

从您的本地驱动器获取一个图像：

```py
local_image = upload_image()
```

点击*选择文件*以从您的本地驱动器选择要加载的图像。

显示图像：

```py
display_image(local_image)
```

基于本地图像向量创建一个生成的图像：

```py
vector = tf.dtypes.cast(local_image, tf.float32)
generated_image = hub_model(vector)['default'][0]
display_image(generated_image)
```

而不是创建一个随机正态潜在向量来生成图像，从上传的图像创建一个向量来生成图像。我们确实需要将本地图像张量转换为浮点张量。接着，在 progan-128 的帮助下从浮点张量生成图像。最后，显示图像。progan-128 模型根据从 CelebA 学习到的内容生成新的图像。所以生成的图像类似于人脸，无论上传什么图像，因为 progan-128 是在人脸上学习的！尽管深度学习很神奇，但它仍然受限于其训练的内容（至少目前是这样）。

## 自定义循环学习实验

监督学习实验的目的是预测新呈现的输入数据的正确标签。模型以非常精确的方式学习如何将输入特征与预测结果相连接。尽管 GANs 是无监督实验，但我们可以通过在两个随机生成的向量之间定义的线性路径上创建一系列潜在向量来创建一个**伪监督**实验。我们识别其中一个生成的向量为特征向量，另一个为预测（或目标）图像。**潜在向量**是从潜在空间中抽取的，在训练期间无法访问或操作。这个想法与前馈神经网络无法操作隐藏层输出的值类似。

我们通过在特征向量和目标图像之间定义损失函数，并在特征向量上使用梯度下降来找到最小化损失的可变值来训练模型。训练开始于在特征图像和目标图像之间定义损失函数。一个自定义的训练循环使梯度下降算法和损失函数能够找到最小化特征向量和特征向量之间损失的可变值。

### 创建特征向量

生成一个种子以实现可重复性并创建特征向量：

```py
seed_value = 777
tf.random.set_seed(seed_value)
feature_vector = tf.random.normal([1, latent_dim])
```

更改种子以生成不同的图像。请注意，使用相同的种子值以保持实验之间的可重复性。

注意

我们为这个实验设置了一个不同的随机种子。我们本可以保留在第一个实验中设置的全球随机种子。但我们决定将两个实验的种子分开。

特征向量可以被视为初始向量，因为模型通过梯度步骤学习如何从初始潜在向量过渡到最终的目标图像。

验证特征向量是否是从潜在空间中抽取的：

```py
feature_vector.shape
```

特征向量是从形状为 1 × 512 的 512 维潜在空间中抽取的。因此，特征向量是一个包含 512 个元素的单一维向量。

### 从特征向量显示图像

使用 progan-128 从特征向量显示图像：

```py
display_image(hub_model(feature_vector)['default'][0])
```

### 创建目标图像

从潜在空间创建目标图像：

```py
target_image = get_module_space_image()
```

验证其形状：

```py
target_image.shape
```

目标张量的形状是 128 × 128 × 3，这是 progan-128 期望的形状。

显示目标图像：

```py
display_image(target_image)
```

### 创建查找最近似潜在向量的函数

函数在特征向量和目标图像之间定义了一个损失函数。它使用梯度下降来找到最小化损失变量的值，如列表 11-6 所示。

```py
def find_closest_latent_vector(
initial_vector, target_image,
num_optimization_steps,
steps_per_image, loss_alg):
images = []
losses = []
vector = tf.Variable(initial_vector)
optimizer = tf.optimizers.Adam(learning_rate=0.01)
loss_fn = loss_alg
for step in range(num_optimization_steps):
if (step % 100)==0:
print()
print('.', end='')
with tf.GradientTape() as tape:
image = hub_model(vector.read_value())['default'][0]
if (step % steps_per_image) == 0:
images.append(image.numpy())
target_image_difference = loss_fn(
image, target_image[:,:,:3])
regularizer = tf.abs(tf.norm(vector) - np.sqrt(latent_dim))
loss = target_image_difference + regularizer
losses.append(loss.numpy())
grads = tape.gradient(loss, [vector])
optimizer.apply_gradients(zip(grads, [vector]))
return images, losses
Listing 11-6
Custom Training Loop Function
```

函数接受特征向量、步骤数、每图像步骤数和损失算法。然后初始化包括优化器和损失函数在内的变量。它还将特征向量转换为 tf.Variable。*tf.Variable* 是一个张量，其值可以在训练期间更改。

函数继续使用自定义训练循环。对于自定义循环学习，TensorFlow 提供了 *tf.GradientTape* API 用于自动微分。TensorFlow 将在 tf.GradientTape 上下文中执行的相关操作记录到一个虚拟带上。然后，TensorFlow 使用虚拟带通过反向模式微分来计算记录计算的梯度。

**自动微分** (AD) 是一套用于数值评估由计算机程序指定的函数导数的技巧。我们使用 AD 来计算某个输入（例如，tf.Variables）相对于计算梯度的值。AD 实现了两种不同的模式。*前向模式微分* 从内部向外遍历链式法则。*反向模式微分* 从外部向内部遍历链式法则。自动微分对于实现机器学习算法，如用于训练神经网络的反向传播非常有用。

关于自动微分的优秀资源，请参阅

[《https://rufflewind.com/2016-12-30/reverse-mode-automatic-differentiation》](https://rufflewind.com/2016-12-30/reverse-mode-automatic-differentiation)

为了自动微分，TensorFlow 需要记住在正向传递过程中发生的操作顺序。然后，在反向传递过程中，TensorFlow 以 *反向顺序* 遍历这个操作列表来计算梯度。

梯度带使用预训练模型生成图像，找到特征图像和目标图像之间的空间，使用正则化得到更逼真的图像，计算损失，应用梯度下降算法，并优化梯度。一旦训练完成，该函数返回一个生成的图像数组以及训练期间计算的损失数组。

由于潜在向量是从正态分布中采样的，如果我们把潜在向量的长度正则化为所有向量的平均长度，我们可以得到更逼真的图像。我们在函数的梯度带部分使用 *regularizer* 变量来实现正则化。

### 创建损失函数

创建损失函数算法：

```py
reduction = tf.keras.losses.Reduction.SUM
mae_loss_algorithm = tf.losses.MeanAbsoluteError(reduction)
```

使用平均绝对误差 (MAE) 减少损失函数算法。**MAE** 衡量两个连续变量之间的差异。

通常，*tf.losses.MeanAbsoluteError* API 计算标签和预测之间的绝对差异的平均值。在我们的实验中，它计算潜在向量和实际目标之间的平均绝对差异。

### 训练模型

清除先前的模型会话：

```py
tf.keras.backend.clear_session()
```

小贴士

当在笔记本中运行多个训练会话时，在用 tf.keras.backend.clear_session API 训练模型之前清除先前的模型会话是一个好主意。

调用训练函数：

```py
num_optimization_steps = 200
steps_per_image = 5
mae_images, mae_loss = find_closest_latent_vector(
feature_vector, target_image, num_optimization_steps,
steps_per_image, mae_loss_algorithm)
```

调整优化步骤和每张图片的步骤以查看对可视化的影响。

### 训练损失

绘制损失图：

```py
plt.plot(mae_loss)
fig = plt.ylim([0, max(plt.ylim())])
```

使用 MAE 减少计算最终损失：

```py
MAE_loss = mae_loss[num_optimization_steps - 1]
MAE_loss
```

### 动画

从训练生成的图像创建动画：

```py
animate(np.stack(mae_images))
```

### 将学习图像与目标进行比较

获取生成的图像数量：

```py
len(mae_images)
```

获取图像类型：

```py
type(mae_images[0])
```

创建一个函数来显示如图 11-7 所示的学习图像：

```py
def closest_latent_images(faces, rows, cols):
fig = plt.figure(1, (20., 40.))
for i in range(40):
plt.subplot(10, 4, i+1)
plt.imshow(faces[i])
plt.axis('off')
Listing 11-7
Function to Display Generated Images
```

显示特征和目标之间的最接近的潜在图像：

```py
closest_latent_images(mae_images, 10, 4)
```

在训练循环的每一步中，该函数通过利用 progan-128 的预训练权重生成一个新的图像。然后，它计算新生成的图像与目标之间的损失。通过梯度下降和损失最小化技术，图像逐渐变得更加接近目标。这是魔法！

将第一个生成的图像与目标进行对比：

```py
display_image(np.concatenate(
[mae_images[0], target_image], axis=1))
```

展示模型的表现如何：

```py
display_image(np.concatenate(
[mae_images[-1], target_image], axis=1))
```

获取‘mae_images[-1]’以显示最终生成的图像。

为了获取更大的图像，使用其他显示函数：

```py
show_image(np.concatenate(
[mae_images[-1], target_image], axis=1))
```

完全不错！

### 尝试不同的损失函数算法

使用 MSE 而不是 MAE：

```py
reduction = tf.keras.losses.Reduction.SUM
mse_loss_algorithm = tf.losses.MeanSquaredError(reduction)
```

**均方误差** (MSE)衡量的是估计值与目标之间的误差平方的平均值（平均平方差异）。*tf.losses.MeanSquaredError* API 计算标签和预测之间误差平方的平均值。

清除先前的模型会话：

```py
tf.keras.backend.clear_session()
```

使用 MSE 减少训练模型：

```py
num_optimization_steps = 200
steps_per_image = 5
mse_images, mse_loss = find_closest_latent_vector(
feature_vector, target_image, num_optimization_steps,
steps_per_image, mse_loss_algorithm)
```

绘制损失图：

```py
plt.plot(mse_loss)
fig = plt.ylim([0, max(plt.ylim())])
```

使用 MSE 减少计算最终损失：

```py
MSE_loss = mse_loss[num_optimization_steps - 1]
MSE_loss
```

MSE 损失要低得多。

显示动画：

```py
animate(np.stack(mse_images))
```

将最终生成的图像与目标图像进行比较：

```py
display_image(np.concatenate(
[mse_images[-1], target_image], axis=1))
```

与 MAE 减少进行比较：

```py
display_image(np.concatenate(
[mae_images[-1], target_image], axis=1))
```

### 从上传的图像创建一个目标：

而不是从潜在向量使用 progan-128 创建目标图像，而是从上传的图像向量使用 progan-128 创建目标图像。

生成一个种子并从潜在空间创建一个初始特征向量：

```py
seed_value = 0
tf.random.set_seed(seed_value)
feature_vector = tf.random.normal([1, latent_dim])
```

注意，我们使用了不同的种子值。当然，您可以使用任何您想要的种子值。但始终在实验中使用相同的种子值以确保可重复性。

从本地驱动器获取一张图像并显示：

```py
uploaded_image = upload_image()
display_image(uploaded_image)
```

将上传的图像转换为 float32 以供 TensorFlow 使用：

```py
uploaded_vector = tf.dtypes.cast(uploaded_image, tf.float32)
display_image(uploaded_vector)
```

注意

上传的张量**不是**目标图像，因为它并非来自潜在空间。目标图像是根据上传的向量基于 progan-128 创建的！为了方便起见，我们建议使用本章节网站上包含的其中一张图片。只需将图片复制到您的本地驱动器即可。

使用 progan-128 从我们刚刚从上传的图像创建的矢量生成目标图像：

```py
uploaded_target = hub_model(uploaded_vector)['default'][0]
display_image(uploaded_target)
```

因此，上传的图像不是目标图像。它只是一个矢量（一旦我们将图像转换为浮点张量），progan-128 使用它来创建目标。

创建一个具有 MSE 减少的损失算法：

```py
reduction = tf.keras.losses.Reduction.SUM
loss_algorithm = tf.losses.MeanSquaredError(reduction)
```

我们使用 MSE 减少法，但如果你愿意，也可以替换为 MAE 减少法。

清除先前的模型会话：

```py
tf.keras.backend.clear_session()
```

训练：

```py
num_optimization_steps = 300
steps_per_image = 5
mse_images, mse_loss = find_closest_latent_vector(
feature_vector, uploaded_target, num_optimization_steps,
steps_per_image, loss_algorithm)
```

我们通过更多的优化步骤训练模型以生成目标图像的更好复制品。

计算最终损失：

```py
MSE_loss = mse_loss[num_optimization_steps - 1]
MSE_loss
```

动画：

```py
animate(np.stack(mse_images))
```

将最终生成的图像与目标图像进行比较：

```py
display_image(np.concatenate(
[mse_images[-1], uploaded_target], axis=1))
```

还不错。我们可以增加优化步骤的数量以生成更逼真的图像。但要注意，设置步骤数量过高可能会影响可用的 RAM。

### 从 Google Drive 图像创建目标：

而不是从本地驱动器中获取图像，从 Google Drive 中获取。我们创建了一个新的特征矢量，但如果你愿意，可以使用我们为上传图像练习创建的那个。

从潜在空间生成一个种子并创建一个初始特征矢量：

```py
seed_value = 0
tf.random.set_seed(seed_value)
feature_vector = tf.random.normal([1, latent_dim])
```

挂载 Google Drive：

```py
from google.colab import drive
drive.mount('/content/gdrive')
```

点击 URL，选择一个 Gmail 账户，复制授权代码，将其粘贴到文本框中，然后点击键盘上的 *Enter* 按钮。

获取并显示图像：

```py
from PIL import Image
p1 = 'gdrive/My Drive/Colab Notebooks/'
p2 = 'images/honest_abe.jpeg'
path = p1 + p2
img_path = path
gdrive_image = Image.open(img_path)
plt.axis('off')
_ = plt.imshow(gdrive_image)
```

创建 Google Drive 中图像的路径。使用路径打开图像并显示。确保图像位于 *Colab Notebooks* 目录中！

注意

你希望加载到 Colab 笔记本中的任何图像（或文件）都必须位于你的 Google Drive 上的“Colab 笔记本”目录（或“Colab 笔记本”目录内的子目录）中。第一次保存 Colab 笔记本时，会自动创建“Colab 笔记本”目录。所有 Colab 笔记本都保存在“Colab 笔记本”目录中。为了方便，我们建议使用该章节网站上包含的图像之一。只需将图像复制到你的 Google Drive 中的“Colabs Notebook”目录即可。

创建一个函数将 Google Drive 中的图像转换为 TensorFlow 可消费的张量：

```py
def reformat(img, size):
img = tf.keras.preprocessing.image.img_to_array(img) / 255.
img = tf.image.resize(img, size)
return img
```

Keras 工具将 PIL 图像实例转换为 NumPy 数组，并将其调整大小为 progan-128 预期的尺寸，即 128 × 128 × 3。

调用函数：

```py
img_size = (128, 128)
gdrive_vector = reformat(gdrive_image, img_size)
gdrive_vector.shape
```

显示图像：

```py
display_image(gdrive_vector)
```

注意

上传的张量**不是**目标图像，因为它不是从潜在空间中生成的。目标图像是从基于上传张量的 progan-128 生成的！

使用 progan-128 从我们刚刚从 Google Drive 图像创建的矢量生成目标图像：

```py
gdrive_target = hub_model(gdrive_vector)['default'][0]
display_image(gdrive_target)
```

再次，从 Google Drive 加载的图像不是目标图像。它只是一个矢量（一旦我们将图像转换为 NumPy 数组），基于 Google Drive 图像，progan-128 使用它来创建目标图像。

创建一个具有 MSE 减少的损失算法：

```py
reduction = tf.keras.losses.Reduction.SUM
loss_algorithm = tf.losses.MeanSquaredError(reduction)
```

我们使用 MSE 减少法，但 MAE 减少法也应该有效。

清除先前的模型会话：

```py
tf.keras.backend.clear_session()
```

训练：

```py
num_optimization_steps = 300
steps_per_image = 5
mse_images, mse_loss = find_closest_latent_vector(
feature_vector, gdrive_target, num_optimization_steps,
steps_per_image, loss_algorithm)
```

我们通过更多的优化步骤训练模型以生成目标图像的更好复制品。

计算最终损失：

```py
MSE_loss = mse_loss[num_optimization_steps - 1]
MSE_loss
```

动画：

```py
animate(np.stack(mse_images))
```

将最终生成的图像与目标进行比较：

```py
display_image(np.concatenate(
[mse_images[-1], gdrive_target], axis=1))
```

### 从维基共享资源图像创建目标：

*维基共享资源* 是一个免费使用的图像、声音、其他媒体和 JSON 文件的在线存储库。它是维基媒体基金会的项目。维基共享资源仅接受免费内容，如图像和其他不受版权限制的媒体文件，这些文件在任何时间、任何目的下都可以被任何人使用。

生成种子并创建特征向量：

```py
seed_value = 0
tf.random.set_seed(seed_value)
feature_vector = tf.random.normal([1, latent_dim])
```

捕获一个图像：

```py
p1 = 'http://upload.wikimedia.org/wikipedia/commons/'
p2 = 'd/de/Wikipedia_Logo_1.0.png'
URL = p1 + p2
im = imageio.imread(URL)
im.shape
```

导入将图像张量转换为 NumPy 数组的必需库：

```py
from keras.preprocessing.image import img_to_array
```

将图像转换为 NumPy 数组并显示：

```py
img_array = img_to_array(im)
print(img_array.dtype)
print(img_array.shape)
plt.imshow(tf.squeeze(img_array))
fig = plt.axis('off')
```

Keras 工具将 PIL 图像实例转换为 NumPy 数组，以便我们可以显示图像。

调整图像大小以供 progan-128 使用：

```py
wiki_vector = tf.image.resize(img_array, (128, 128))
plt.imshow(tf.squeeze(wiki_vector))
fig = plt.axis('off')
```

将图像调整到 progan-128 预期的尺寸并显示：

创建目标：

```py
wiki_target = hub_model(wiki_vector)['default'][0]
display_image(wiki_target)
```

创建一个具有 MSE 减少的损失算法：

```py
reduction = tf.keras.losses.Reduction.SUM
loss_algorithm = tf.losses.MeanSquaredError(reduction)
```

清除之前的模型会话：

```py
tf.keras.backend.clear_session()
```

训练：

```py
num_optimization_steps = 300
steps_per_image = 5
mse_images, mse_loss = find_closest_latent_vector(
feature_vector, wiki_target, num_optimization_steps,
steps_per_image, loss_algorithm)
```

计算最终损失：

```py
MSE_loss = mse_loss[num_optimization_steps - 1]
MSE_loss
```

动画：

```py
animate(np.stack(mse_images))
```

比较：

```py
display_image(np.concatenate(
[mse_images[-1], wiki_target], axis=1))
```

还不错。

虽然我们生成了相当逼真的图像和目标图像的复制品，但仍需要更稳健的模型和训练算法来持续生成与实际图像难以区分的图像。当然，还需要更多的计算资源！

## 潜在向量和图像数组

progan-128 模块可以从大小为 (1, 512) 的潜在向量或大小为 128 × 128 × 3 的 float 向量生成新的图像。progan-128 接受的潜在向量是一个大小为 512 的一维向量。progan-128 接受的 float 向量是一个 128 × 128 × 3 像素的向量。

progan-128 模块是基于 TensorFlow 重实现的渐进式 GANs 的图像生成器。它将 512 维的潜在空间映射到图像。在训练过程中，潜在空间向量是从正态分布中采样的。

根据文档，progan-128 接收一个数据类型为 float32、形状为 (?, 512) 的输入张量。progan-128 的输入张量代表一批潜在向量。progan-128 的输出是一个形状为 (?, 128, 128, 3) 的 float 张量，它代表一批 RGB 图像。我们还可以从一个图像数组生成一个新的图像，但这在文档中并未提及。

要查看 progan-128 文档，请查阅

[`https://tfhub.dev/google/progan-128/1`](https://tfhub.dev/google/progan-128/1)

### 从潜在向量生成新的图像：

从潜在空间创建一个随机正态向量：

```py
random_normal_latent_vector = tf.random.normal([1, latent_dim])
random_normal_latent_vector.shape
```

tf.random.normal API 从正态分布输出随机值。因此，新向量由从我们的潜在空间中随机抽取的 512 个正态分布值组成。

将张量转换为 NumPy 以便检查：

```py
rnlv = random_normal_latent_vector.numpy()
len(rnlv[0])
```

检查 NumPy 数组的一些元素：

```py
for i, element in enumerate(rnlv[0]):
if i < 5:
print (element)
else: break
```

新向量中的每个元素代表一个潜在维度（或潜在变量），这些维度无法直接观察到，但可以假设其存在。由于潜在维度存在，它们可以用来解释观察变量中的变化模式。在我们的实验中，观察变量代表 CelebA 图片。因此，我们可以将新向量输入到 progan-128 中以生成新的图片。

使用 progan-128 从潜在空间创建一个浮点输出张量：

```py
float_output_tensor = hub_model(
random_normal_latent_vector)['default'][0]
float_output_tensor.shape
```

将浮点输出张量显示为图片：

```py
display_image(generated_image)
```

因此，progan-128 从一个潜在向量生成了一个 128 × 128 × 3 的图片。

### 从图像向量生成新的图片

从 Google Drive 获取图片：

```py
p1 = 'gdrive/My Drive/Colab Notebooks/'
p2 = 'images/honest_abe.jpeg'
path = p1 + p2
img_path = path
abe_image = Image.open(img_path)
plt.axis('off')
_ = plt.imshow(abe_image)
```

将 JPEG 图片转换为适当数据类型和大小的图像向量：

```py
img_size = (128, 128)
abe_vector = reformat(abe_image, img_size)
abe_vector.shape
```

显示向量的一个切片：

```py
abe_vector[0][0].numpy()
```

这个向量肯定不是来自潜在空间！

从我们刚刚创建的向量生成新的图片：

```py
image_from_abe_vector = hub_model(abe_vector)['default'][0]
display_image(image_from_abe_vector)
```

因此，progan-128 从一个未从潜在空间抽取的特征图像向量生成了一个 128 × 128 × 3 的图片！

## 摘要

我们以详细和逐步的方式演示了两个渐进式增长生成对抗网络（Progressive Growing GAN）实验。我们还验证了 progan-128 也可以从未从潜在空间抽取的特征向量生成逼真的图片。
