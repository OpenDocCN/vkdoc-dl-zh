# 11. 机器学习的对比解释

对比学习是一种较新的方法，用于在机器学习流程中寻找相似和相异的候选对象。对比解释旨在寻找两个特征之间的相似性，以帮助对某个类别进行预测。典型的黑盒模型，基于不同类型的超参数以及在不同轮次和学习率下优化的大量参数进行训练，非常难以解释，更难以推理模型为何预测 A 类而非 B 类。为了生成更多解释以便业务用户理解预测结果的需求，吸引了许多开发者创建能够产生价值的创新框架。对比解释侧重于解释模型为何预测 A 类而非 B 类。寻找原因有助于业务用户理解模型的行为。在本章中，你将使用基于 TensorFlow 框架的 Alibi 库来处理图像分类任务。

## 什么是机器学习中的对比解释？

为了理解机器学习中的对比解释，让我们以银行的贷款审批流程或信用风险评估流程为例。没有银行会向高风险客户提供贷款。同样，没有银行会拒绝向非风险客户提供贷款。当贷款申请或信用申请被拒绝时，这并非银行某人的随意决定。相反，这是一个 AI 模型做出的决定，该模型考虑了关于个人的众多特征，包括财务历史和其他因素。贷款被拒绝的个人可能会思考：哪些方面决定了他的贷款资格？在贷款资格方面，是什么将一个人与另一个人区分开来？他的个人资料中缺少哪些要素？等等。同样，在从多个可用图像中识别感兴趣的对象时，是什么使感兴趣的对象与其他图像不同？对比解释就是将一个类别与其他类别区分开来的特征差异。

对比解释就像人类的对话，有助于人类进行更深入的交流。对比解释涉及两个概念：

*   相关正特征
*   相关负特征

相关正特征解释是指，找到那些对于机器学习模型识别出与预测类别相同的类别所必需的特征的存在。例如，一个人的收入和年龄决定了个人的净资产。你希望你的机器学习模型能够根据高收入和高年龄段的存在来识别高净值人群类别。这在多个方面与机器学习模型的锚点解释相似。相关负特征是相关正特征的反面；它解释了在保持原始输出类别不变的情况下，应该从记录中缺失的特征。一些研究人员也将其称为*反事实解释*。



### 使用 Alibi 实现对比解释方法（CEM）

模型的对比解释有助于向最终用户阐明，为何某个事件或预测结果会与另一种情况不同。为了解释模型对比解释方法（CEM）的概念以及如何通过基于 Python 的库来实现它，我们以 Alibi 为例。首先，你需要使用 TensorFlow 作为后端，开发一个基于 Keras 的深度学习模型。以下脚本包含了开发深度学习模型时可用的各种模块和方法的导入语句。CEM 可在 Alibi 库的 `explainer` 模块中找到。

```
import tensorflow as tf
tf.get_logger().setLevel(40) # suppress deprecation messages
tf.compat.v1.disable_v2_behavior() # disable TF2 behaviour as alibi code still relies on TF1 constructs
import tensorflow.keras as keras
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D, Input, UpSampling2D
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.utils import to_categorical
import matplotlib
%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import os
from alibi.explainers import CEM
print('TF version: ', tf.__version__)
print('Eager execution enabled: ', tf.executing_eagerly()) # False
```

在解释“必要存在特征”（PP）和“必要缺失特征”（PN）时，识别并组织相关特征，以及从非重要特征中分类出重要特征至关重要。如果在模型训练步骤中有更多非重要特征，那么它们属于 PP 还是 PN 实际上并不重要。模型中的非重要特征完全不相关。之所以展示 MNIST 手写数字分类数据集用于 CEM 解释，是因为许多开发者和机器学习工程师都熟悉 MNIST 数据集，因此他们能很好地理解 CEM 的概念。以下脚本将数据集分为训练集和测试集，图 11-1 中还显示了一张数字 4 的示例图像：

![../images/506619_1_En_11_Chapter/506619_1_En_11_Fig1_HTML.jpg](img/506619_1_En_11_Fig1_HTML.jpg)

图 11-1

数字 4 的示例图像

```
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
print('x_train shape:', x_train.shape, 'y_train shape:', y_train.shape)
plt.gray()
plt.imshow(x_test[4]);
```

接下来，你需要开发一个分类模型。作为下一步，你需要对特征进行归一化，以加快深度学习模型的训练过程。因此，将像素值除以最高像素值（255）。

```
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
x_train = np.reshape(x_train, x_train.shape + (1,))
x_test = np.reshape(x_test, x_test.shape + (1,))
print('x_train shape:', x_train.shape, 'x_test shape:', x_test.shape)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print('y_train shape:', y_train.shape, 'y_test shape:', y_test.shape)
xmin, xmax = -.5, .5
x_train = ((x_train - x_train.min()) / (x_train.max() - x_train.min())) * (xmax - xmin) + xmin
x_test = ((x_test - x_test.min()) / (x_test.max() - x_test.min())) * (xmax - xmin) + xmin
```

以下脚本展示了创建卷积神经网络模型的步骤：

*   输入数据集形状为 28x28 像素。
*   卷积 2D 层应用了 64 个滤波器，内核大小为 2，填充方式为“相同”，激活函数为修正线性单元。
*   应用卷积层后，需要应用最大池化，以生成可供后续层使用的抽象特征。
*   通常应用 Dropout 来限制模型过拟合。
*   共有三层卷积 2D 滤波器，随后依次应用最大池化层和 Dropout 层，以降低数据维度。
*   应用卷积和最大池化的目标是得到一个神经元数量较少的层，用于训练全连接神经网络模型。
*   如果需要展平数据以改变形状，则应用全连接神经网络模型——密集层。
*   最后，应用 Softmax 层以生成相对于每个数字类别的类别概率。
*   在编译步骤中，你需要提供分类交叉熵作为损失函数，`adam` 作为优化器，以及准确率作为评估指标。

```
def cnn_model():
x_in = Input(shape=(28, 28, 1)) #input layer
x = Conv2D(filters=64, kernel_size=2, padding='same', activation='relu')(x_in) #conv layer
x = MaxPooling2D(pool_size=2)(x) #max pooling layer
x = Dropout(0.3)(x) #drop out to avoid overfitting
x = Conv2D(filters=32, kernel_size=2, padding='same', activation='relu')(x) # second conv layer
x = MaxPooling2D(pool_size=2)(x) #max pooling layer
x = Dropout(0.3)(x) #drop out to avoid overfitting
x = Conv2D(filters=32, kernel_size=2, padding='same', activation='relu')(x) # third conv layer
x = MaxPooling2D(pool_size=2)(x) #max pooling layer
x = Dropout(0.3)(x) # drop out to avoid overfitting
x = Flatten()(x) # flatten for reshaping the matrix
x = Dense(256, activation='relu')(x) #this is for Fully Connected Neural Network Layer training
x = Dropout(0.5)(x) # drop out again to avoid overfitting
x_out = Dense(10, activation='softmax')(x) #final output layer
cnn = Model(inputs=x_in, outputs=x_out)
cnn.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
return cnn
```

卷积神经网络模型摘要如下表所示。在模型训练步骤中，你需要提供批次大小和训练轮数。然后，你可以将训练好的模型对象保存为 `h5` 格式。

```
cnn = cnn_model()
cnn.summary()
cnn.fit(x_train, y_train, batch_size=64, epochs=5, verbose=1)
cnn.save('mnist_cnn.h5', save_format='h5')
```

为了生成有意义的 CEM 解释，你必须确保模型具有很高的准确率，否则 CEM 解释将缺乏一致性。因此，建议先训练、微调或搜索能够产生至少 85% 准确率的最佳模型。在当前示例中，模型在测试数据集上的准确率为 98.73%，因此你可以期待有意义的 CEM 解释。

```
# Evaluate the model on test set
cnn = load_model('mnist_cnn.h5')
score = cnn.evaluate(x_test, y_test, verbose=0)
print('Test accuracy: ', score[1])
```

为了生成 CEM 解释，你需要一个能将输入数据分类到特定类别的模型对象。CEM 尝试生成两种可能的解释：

*   尝试找出输入数据中必须存在的最少信息量，这些信息足以产生相同的类别分类。这被称为 PP。
*   尝试找出输入数据中缺失的最少信息量，这些信息足以阻止类别预测发生变化。这被称为 PN。



