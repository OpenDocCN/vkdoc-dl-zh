# 相关否定与相关肯定解释

相关否定预测结果为数字 8，而原始图像为数字 5

```
mode = 'PN'  # 'PN'（相关否定）或 'PP'（相关肯定）
shape = (1,) + x_train.shape[1:]  # 实例形状
kappa = 0.  # 扰动实例在原始实例预测类别上的预测概率，与其他类别最大概率之间所需的最小差值
# 用于最小化第一个损失项
beta = .1  # L1 损失项的权重
gamma = 100  # 可选自编码器损失项的权重
c_init = 1.  # 损失项的初始权重 c，该损失项鼓励扰动实例预测与待解释原始实例不同的类别（PN）或
# 相同的类别（PP）
c_steps = 10  # c 的更新次数
max_iterations = 1000  # 每个 c 值对应的迭代次数
feature_range = (x_train.min(),x_train.max())  # 扰动实例的特征范围
clip = (-1000.,1000.)  # 梯度裁剪
lr = 1e-2  # 初始学习率
no_info_val = -1. # 一个值（浮点数或按特征指定），可视为不包含任何用于预测的信息
# 向该值方向扰动意味着移除特征，远离该值则意味着添加特征
# 对于我们的 MNIST 图像，背景（-0.5）信息量最小，
# 因此正向/负向扰动分别意味着添加/移除特征
# 初始化 CEM 解释器并解释实例
cem = CEM(cnn, mode, shape, kappa=kappa, beta=beta, feature_range=feature_range, gamma=gamma, ae_model=ae, max_iterations=max_iterations,
c_init=c_init, c_steps=c_steps, learning_rate_init=lr, clip=clip, no_info_val=no_info_val)
explanation = cem.explain(X)
print(f'相关否定预测: {explanation.PN_pred}')
plt.imshow(explanation.PN.reshape(28, 28));
```

对于同一个数字 5，也可以生成相关肯定解释，这意味着为了将数字分类为 5，你在图像中绝对要寻找哪些特征。这被称为相关肯定解释。

```
# 现在生成相关肯定
mode = 'PP'
# 初始化 CEM 解释器并解释实例
cem = CEM(cnn, mode, shape, kappa=kappa, beta=beta, feature_range=feature_range, gamma=gamma, ae_model=ae, max_iterations=max_iterations,
c_init=c_init, c_steps=c_steps, learning_rate_init=lr, clip=clip, no_info_val=no_info_val)
explanation = cem.explain(X)
print(f'相关肯定预测: {explanation.PP_pred}')
plt.imshow(explanation.PP.reshape(28, 28));
```

![../images/506619_1_En_11_Chapter/506619_1_En_11_Fig6_HTML.jpg](img/506619_1_En_11_Fig6_HTML.jpg)

**图 11-6** – 数字 5 的相关肯定解释

在上述脚本中，你生成了相关肯定解释。该解释表明，图 11-6 中显示的像素值是将图像预测为数字 5 所需的最小必要条件。

### 表格数据的 CEM 解释

对于任何表格数据（也称为结构化数据），行代表样本，列代表特征。你可以使用与上述卷积神经网络模型相同的过程，以熟悉的 IRIS 数据集为例，处理一个简单的多类分类问题。

```
# 结构化数据集的 CEM
import tensorflow as tf
tf.get_logger().setLevel(40) # 抑制弃用消息
tf.compat.v1.disable_v2_behavior() # 禁用 TF2 行为，因为 alibi 代码仍依赖 TF1 结构
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.utils import to_categorical
import matplotlib
%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_iris
from alibi.explainers import CEM
print('TF 版本: ', tf.__version__)
print('启用了即时执行: ', tf.executing_eagerly()) # False
```

上述脚本展示了 Alibi 模型生成 PP 和 PN 解释所需的导入语句。

```
dataset = load_iris()
feature_names = dataset.feature_names
class_names = list(dataset.target_names)
# 缩放数据
dataset.data = (dataset.data - dataset.data.mean(axis=0)) / dataset.data.std(axis=0)
idx = 145
x_train,y_train = dataset.data[:idx,:], dataset.target[:idx]
x_test, y_test = dataset.data[idx+1:,:], dataset.target[idx+1:]
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
```

IRIS 模型有四个特征，目标列中的三个类别分别是`setosa`、`versicolor`和`virginica`。前 145 条记录是训练数据集，5 条记录留作测试模型。由于目标列包含字符串值，因此需要对目标列（即`y_train`和`y_test`数据集）进行分类编码。

以下神经网络模型函数将四个特征作为输入，并使用 Keras 的密集函数通过全连接网络训练模型。你使用分类交叉熵作为损失函数，随机梯度下降作为优化器。训练模型的批量大小为 16，迭代 500 轮。需要训练的参数数量非常少。

```
def lr_model():
x_in = Input(shape=(4,))
x_out = Dense(3, activation='softmax')(x_in)
lr = Model(inputs=x_in, outputs=x_out)
lr.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
return lr
lr = lr_model()
lr.summary()
lr.fit(x_train, y_train, batch_size=16, epochs=500, verbose=0)
lr.save('iris_lr.h5', save_format='h5')
```

训练完成后，模型被保存为`iris_lr.h5`。在以下脚本中，你加载训练好的模型对象，并使用表 11-1 中先前解释的所有参数初始化 CEM 函数。


```python
idx = 0
X = x_test[idx].reshape((1,) + x_test[idx].shape)
print('对要解释的实例的预测结果: {}'.format(class_names[np.argmax(lr.predict(X))]))
print('该实例上每个类别的预测概率: {}'.format(lr.predict(X)))
mode = 'PN'  # 'PN' (相关负例) 或 'PP' (相关正例)
shape = (1,) + x_train.shape[1:]  # 实例形状
kappa = .2  # 扰动实例在原始实例预测类别上的预测概率与其他类别最大概率之间所需的最小差值
# 用于最小化第一个损失项
beta = .1  # L1 损失项的权重
c_init = 10\.  # 损失项的初始权重 c，鼓励扰动实例预测与原始实例不同的类别(PN)或
# 相同的类别(PP)
c_steps = 10  # c 的更新次数
max_iterations = 1000  # 每个 c 值的迭代次数
feature_range = (x_train.min(axis=0).reshape(shape)-.1,  # 扰动实例的特征范围
x_train.max(axis=0).reshape(shape)+.1)  # 可以是浮点数或形状为(1x 特征数)的数组
clip = (-1000.,1000.)  # 梯度裁剪
lr_init = 1e-2  # 初始学习率
# 定义模型
lr = load_model('iris_lr.h5')
# 初始化 CEM 解释器并解释实例
cem = CEM(lr, mode, shape, kappa=kappa, beta=beta, feature_range=feature_range, max_iterations=max_iterations, c_init=c_init, c_steps=c_steps, learning_rate_init=lr_init, clip=clip)
cem.fit(x_train, no_info_type='median')  # 我们需要定义哪些特征值包含最少
# 关于预测的信息
# 这里我们将天真地假设特征中位数
# 不包含任何信息；领域知识会有所帮助！
explanation = cem.explain(X, verbose=False)
```

在上述脚本中，原始实例是`virginica`，但相关负例的解释将其预测为`versicolor`。只有第三个特征的差异导致了预测的不同。你也可以用同样的例子来预测相关正例的类别。

```python
print(f'原始实例: {explanation.X}')
print('预测类别: {}'.format(class_names[explanation.X_pred]))
print(f'相关负例: {explanation.PN}')
print('预测类别: {}'.format(class_names[explanation.PN_pred]))
expl = {}
expl['PN'] = explanation.PN
expl['PN_pred'] = explanation.PN_pred
mode = 'PP'
# 定义模型
lr = load_model('iris_lr.h5')
# 初始化 CEM 解释器并解释实例
cem = CEM(lr, mode, shape, kappa=kappa, beta=beta, feature_range=feature_range, max_iterations=max_iterations, c_init=c_init, c_steps=c_steps, learning_rate_init=lr_init, clip=clip)
cem.fit(x_train, no_info_type='median')
explanation = cem.explain(X, verbose=False)
print(f'相关正例: {explanation.PP}')
print('预测类别: {}'.format(class_names[explanation.PP_pred]))
```

在上述脚本中，你解释了相关正例的预测类别。它是`virginica`类别，实际类别也是`virginica`类别。你正在创建一种可视化方式来展示 PP 和 PN，使用 CEM 模型的特征和结果。你创建了一个`expl`对象，并创建了名为`PN`、`PN_pred`、`PP`和`PP_pred`的列。你创建了一个包含原始数据和特征名称（包括目标类别）的数据框。这是数据可视化所必需的。Seaborn Python 库用于以图形方式展示 PN 和 PP，如图 11-7 所示。

![../images/506619_1_En_11_Chapter/506619_1_En_11_Fig7_HTML.png](img/506619_1_En_11_Fig7_HTML.png)

**图 11-7** 相关正例和相关负例可视化

```python
expl['PP'] = explanation.PP
expl['PP_pred'] = explanation.PP_pred
df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
df['species'] = np.array([dataset.target_names[i] for i in dataset.target])
pn = pd.DataFrame(expl['PN'], columns=dataset.feature_names)
pn['species'] = 'PN_' + class_names[expl['PN_pred']]
pp = pd.DataFrame(expl['PP'], columns=dataset.feature_names)
pp['species'] = 'PP_' + class_names[expl['PP_pred']]
orig_inst = pd.DataFrame(explanation.X, columns=dataset.feature_names)
orig_inst['species'] = 'orig_' + class_names[explanation.X_pred]
df = df.append([pn, pp, orig_inst], ignore_index=True)
fig = sns.pairplot(df, hue='species', diag_kind='hist');
```

对比解释通常通过将特征投影到潜在空间作为抽象特征，然后仅考虑特征空间中对模型区分目标类别有用的特征来生成。

## 结论

在本章中，你探索了能够为图像分类问题（使用 MNIST 数据集进行手写识别）和结构化数据分类问题（使用简单的 IRIS 数据集）建立对比解释的方法和库。相关正例和相关负例均从 Alibi 库的对比解释模块中捕获。这种 CEM 方法为类别预测提供了更清晰的解释，并总结了为什么预测某个特定类别，而不是为什么没有预测到该类别。

