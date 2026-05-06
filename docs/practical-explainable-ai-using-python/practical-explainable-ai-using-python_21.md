# 自编码器模型训练与分析

寻找可能改变预测结果或有助于维持预测的最小信息，通常是输入数据集中最重要的特征值。将输入数据与从数据中生成的抽象层进行匹配，用于识别最小信息的存在与否。这个抽象层被称为自编码器。对于图像分类问题，它可以是卷积自编码器。自编码器通过在输入层和输出层同时使用输入数据进行训练。模型被训练为精确预测与输入相同的输出。一旦输入与输出匹配，神经网络模型中最内层隐藏层的权重就可以推导为自编码器值。这些自编码器值有助于识别任何输入数据集中的最小信息可用性。以下脚本展示了如何训练自编码器模型。它是一个神经网络模型，接收 28x28 像素的输入，并生成 28x28 像素的输出。

```python
# 定义并训练一个自编码器，其工作原理类似于主成分分析模型
def ae_model():
    x_in = Input(shape=(28, 28, 1))
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x_in)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    encoded = Conv2D(1, (3, 3), activation=None, padding='same')(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    decoded = Conv2D(1, (3, 3), activation=None, padding='same')(x)
    autoencoder = Model(x_in, decoded)
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder

ae = ae_model()
ae.summary()
ae.fit(x_train, x_train, batch_size=128, epochs=4, validation_data=(x_test, x_test), verbose=0)
ae.save('mnist_ae.h5', save_format='h5')
```

模型的摘要显示了架构和可训练参数的总数。可训练参数在模型的每次迭代中都会更新权重。没有不可训练的参数。一旦自编码器模型准备就绪，就可以将模型对象加载到会话中。使用测试集上的 `predict` 函数，可以生成解码后的 MNIST 图像。以下脚本的输出显示，测试图像与使用自编码器函数生成的图像完全匹配，这意味着您的自编码器模型在生成精确输出方面相当稳健。自编码器模型包含两部分：编码器和解码器。编码器的作用是将任何输入转换为抽象层，解码器的作用是从抽象层重建相同的输入。

### 原始图像与自编码器生成图像的对比

您可以将训练集中的原始图像与基于自编码器生成的模型图像集（如图 11-2 所示）进行对比。从图像中可以看出，自编码器生成的模型与原始图像完全匹配。由于这是一个示例数据集，匹配度非常高；然而，对于其他示例，需要大量的训练才能生成如此接近的图像。一个训练有素的自编码器模型对于生成对比解释将非常有帮助。

![../images/506619_1_En_11_Chapter/506619_1_En_11_Fig2_HTML.jpg](img/506619_1_En_11_Fig2_HTML.jpg)

**图 11-2** — 自编码器模型生成的图像与原始图像的对比

```python
ae = load_model('mnist_ae.h5')
decoded_imgs = ae.predict(x_test)
n = 5
plt.figure(figsize=(20, 4))
for i in range(1, n+1):
    # 显示原始图像
    ax = plt.subplot(2, n, i)
    plt.imshow(x_test[i].reshape(28, 28))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    # 显示重建图像
    ax = plt.subplot(2, n, i + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
```

第一行显示测试数据集第一条记录的实际图像，第二行显示自编码器生成的预测图像。

```python
ae = load_model('mnist_ae.h5')
decoded_imgs = ae.predict(x_test)
n = 5
plt.figure(figsize=(20, 4))
for i in range(1, n+1):
    # 显示原始图像
    ax = plt.subplot(2, n, i)
    plt.imshow(x_test[i].reshape(28, 28))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    # 显示重建图像
    ax = plt.subplot(2, n, i + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
```

![../images/506619_1_En_11_Chapter/506619_1_En_11_Fig3_HTML.jpg](img/506619_1_En_11_Fig3_HTML.jpg)

**图 11-3** — 相关负样本解释

上述脚本展示了同一图像的显示和重建（图 11-3），该图像在下方显示为数字 5（图 11-4）。

![../images/506619_1_En_11_Chapter/506619_1_En_11_Fig4_HTML.jpg](img/506619_1_En_11_Fig4_HTML.jpg)

**图 11-4** — 数字 5 的原始图像显示

```python
idx = 15
X = x_test[idx].reshape((1,) + x_test[idx].shape)
plt.imshow(X.reshape(28, 28));
```

```python
# 模型预测
cnn.predict(X).argmax(), cnn.predict(X).max()
```

CNN 模型以 99.95% 的概率将输入预测为数字 5（表 11-1）。

**表 11-1** — CEM 参数说明

| 参数 | 说明 |
| --- | --- |
| `Mode` | PN 或 PP |
| `Shape` | 输入实例的形状 |
| `Kappa` | 扰动实例在预测类别（与原始实例相同）上的预测概率与在其他类别上的最大概率之间所需的最小差值，用于最小化第一个损失项 |
| `Beta` | L1 损失项的权重 |
| `Gamma` | 自编码器损失项的权重 |
| `C_steps` | 更新次数 |
| `Max iterations` | 迭代次数 |
| `Feature_range` | 扰动实例的特征范围 |
| `Lr` | 初始学习率 |

以下脚本展示了从样本实例 `X` 生成的解释对象中的相关负样本预测。相关负样本分析表明，数字 5 缺少一些重要特征；否则，它会被分类为数字 8。这些缺失的信息正是使数字 5 类别预测与数字 8 类别预测保持不同的最小信息。图 11-5 将数字 8 叠加在数字 5 上。

![../images/506619_1_En_11_Chapter/506619_1_En_11_Fig5_HTML.jpg](img/506619_1_En_11_Fig5_HTML.jpg)

**图 11-5**



