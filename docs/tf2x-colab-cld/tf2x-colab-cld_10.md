# 10. 使用 RNN 进行时间序列预测

我们已经利用 RNN 进行了 NLP。在本章中，我们创建实验以使用时间序列数据进行预测。我们使用著名的 *Weather* 数据集来演示单变量和多变量示例。

RNN 非常适合进行时间序列预测，因为它能记住过去，并且它的决策会受到从过去学到的内容的影响。因此，当数据变化时，它能够做出良好的决策。**时间序列预测**是指基于先前观察到的值部署模型来预测未来的值。

时间序列数据与我们之前处理的数据不同，因为它是一系列按时间顺序连续观察到的数据。时间序列数据包括一个时间维度，这是观察之间的显式顺序依赖性。

## 天气预报

预测天气是一项困难和复杂的任务。但像 Google、IBM、Monsanto 和 Facebook 这样的领先公司正在利用人工智能技术来实现准确和及时的天气预报。鉴于我们课程的入门性质，我们无法展示如此复杂的 AI 实验。但我们向您展示如何使用天气数据构建简单的时间序列预测模型。

章节的笔记本位于以下网址：[`https://github.com/paperd/tensorflow`](https://github.com/paperd/tensorflow).

## 天气数据集

我们介绍了使用 RNN 进行单变量问题的时序预测。然后我们预测多变量时间序列。我们使用天气时间序列数据来训练我们的模型。我们使用的数据是由马克斯·普朗克生物地球化学研究所记录的。

通过以下网址了解更多关于该研究所的信息：

[马克斯·普朗克生物地球化学研究所主页](http://www.bgc-jena.mpg.de/index.php/Main/HomePage)

启用 GPU（如果尚未启用）：

1.  在右上角菜单中点击 *Runtime*。

1.  从下拉菜单中选择 *Change runtime type*。

1.  从 *Hardware accelerator* 下拉菜单中选择 *GPU*。

1.  点击 *SAVE*。

测试 GPU 是否激活：

```py
import tensorflow as tf
# display tf version and test if GPU is active
tf.__version__, tf.test.gpu_device_name()
```

导入 *tensorflow* 库。如果显示‘/device:GPU:0’，则 GPU 正在运行。如果显示‘ .. ’，则常规 CPU 正在运行。

获取如列表 10-1 所示的数据集。

```py
import os
p1 = 'https://storage.googleapis.com/tensorflow/'
p2 = 'tf-keras-datasets/jena_climate_2009_2016.csv.zip'
url = p1 + p2
zip_path = tf.keras.utils.get_file(
origin = url,
fname='jena_climate_2009_2016.csv.zip',
extract=True)
csv_path, _ = os.path.splitext(zip_path)
Listing 10-1
Get weather data
```

使用 *splitext* 方法从 URL 中提取 CSV 文件。为方便加载到 pandas 中创建适当的路径。

将 CSV 文件加载到 pandas 中：

```py
import pandas as pd
df = pd.read_csv(csv_path)
```

## 探索数据

显示数据框的特征：

```py
list(df)
```

显示前几条记录：

```py
df.head(3)
```

显示最后几条记录：

```py
df.tail(3)
```

我们可以看到数据收集始于 2009 年 1 月 1 日，结束于 2016 年 12 月 31 日。最后一条记录是在 2017 年 1 月 1 日，但对于我们的目的来说并不重要，因为它是该年度唯一记录的数据。我们还看到数据每 10 分钟记录一次。因此，本实验的时间步长为 10 分钟。**时间步长**是事件的单次发生。

第一个时间戳从 2009 年 1 月 1 日（01.01.2009）开始，数据记录从 00:00:00 到 00:10:00。第二个时间戳从 00:10:00 开始，到 00:20:00 结束。这种模式一整天都在继续，最后的时间步在 23:50:00。第二天（以及所有后续的日子），2009 年 1 月 2 日（02.01.2009），遵循相同的模式。通常，时间序列预测预测下一个时间步的观测值。

显示数据框的简洁摘要：

```py
df.info()
```

数据集包含 15 列：

+   Date Time – 日期时间参考

+   p (mbar) – 毫巴单位的大气压力

+   T (degC) – 摄氏度温度

+   Tpot (K) – 开尔文温度

+   Tdew (degC) – 相对湿度相关的摄氏度温度

+   rh (%) – 相对湿度

+   VPmax (mbar) – 毫巴单位的饱和蒸汽压

+   VPact (mbar) – 毫巴单位的蒸汽压

+   VPdef (mbar) – 毫巴单位的蒸汽压亏缺

+   sh (g/kg) – 每千克克的比湿

+   H2OC (mmol/mol) – 毫摩尔每摩尔的蒸汽浓度

+   rho (g/m**3) – 每立方米克的空气密度

+   wv (m/s) – 每秒米的风速

+   max. wv (m/s) – 每秒米的最大风速

+   wd (deg) – 风向度数

我们有 14 个特征，因为*日期时间*是一个参考列。没有缺失数据。所有数据都是 float64 类型，除了日期时间参考对象。数据集包含 420,551 行数据，索引范围从 0 到 420550。

显示 14 个特征的统计数据：

```py
stats = df.describe()
stats.transpose()
```

*describe*方法生成描述性统计信息。*transpose*方法转置索引和列。

我们还可以显示一个或多个特征的统计信息：

```py
stats = df.describe()
stats[['p (mbar)', 'T (degC)']].transpose()
```

显示数据框的形状：

```py
df.shape
```

数据框包含 420,551 行，每行包含 15 列。

## 绘制相对湿度随时间的变化图

由于数据在 pandas 数据框中，因此很容易将 14 个特征中的任何一个与*日期时间*时间步进行绘图。

绘制相对湿度的年度周期性

```py
import matplotlib.pyplot as plt
# create new dataframe with just relative humidity column
rh = df['rh (%)']
# plot it!
rh.plot()
```

由于我们每 10 分钟有一个观测值，所以每小时有六个观测值。每天有 144（6 个观测值 `×` 24 小时）个观测值。

绘制前 10 天：

```py
rh10 = df['rh (%)'][0:1439]
rh10.plot()
```

由于我们每天有 144 个观测值，因此我们绘制了总共 1,440（10 天 `×` 每天观察 144 次）个观测值。请注意，索引从*0*开始。

在数据的一个狭窄视角（前 10 天）中，我们可以看到每日周期性。我们还看到波动相当混乱，这意味着预测更困难。

在细粒度级别探索湿度随时间步的变化：

```py
df[['Date Time','rh (%)']].head()
```

## 预测一元时间序列

### 缩放数据

将数据框转换为 NumPy 数组：

```py
rh_np = rh.to_numpy()
```

如列表 10-2 所示，缩放 NumPy 数据以进行高效训练。

```py
br ='\n'
# original data
print ('first five unscaled observations:', rh_np, br)
# scale relative humidity data
rh_sc = tf.keras.utils.normalize(rh_np)
print ('shape after tf function:', rh_sc.shape)
# squeeze out '1' dimension
rh_sq = tf.squeeze(rh_sc)
print ('shape after squeeze:', rh_sq.shape, br)
# convert to numpy
rh_scaled = rh_sq.numpy()
print ('first five scaled observations:', rh_scaled[:5])
Listing 10-2
Scale data
```

缩放相对湿度数据。挤压掉 TensorFlow 函数添加的额外*1*维度，以便我们可以将 TensorFlow 张量转换为 NumPy 数组，以便更容易处理。显示前五个缩放观测值以验证缩放是否按预期工作。

### 建立训练分割

计算训练分割大小：

```py
import numpy as np
# train split with 75% of data
train_split = int(np.round(df.shape[0] * .75))
train_split
```

计算测试分割大小：

```py
# test split with 25% of data
test_split = df.shape[0] - train_split
test_split
```

计算训练和测试数据的天数：

```py
# calculate number of days of data
print (np.round(train_split / 144, 2))
print (np.round(test_split / 144, 2))
```

对于这个实验，使用数据的前 315,413 行进行训练，剩余的 105,138 行（420,551 - 315,413）用于测试集。训练数据大约占 2,190 天（315,413/144）的数据。测试数据大约占 730 天（105,138/144）的数据。

### 创建特征和标签

创建一个函数，将数据集分割成特征和标签，如列表 10-3 所示。

```py
def create_datasets(data, origin, end, window, target_size):
# list to hold feature set of windows
features = []
# list to hold labels
labels = []
# establish starting point that reflects window size
origin = origin + window
# enable split for test data
if end is None:
end = len(data) - target_size
# create feature set of 'window-sized' elements
for i in range(origin, end):
# create index set to identify each window
indices = range(i-window, i)
# reshape data from (window,) to (window, 1)
features.append(np.reshape(data[indices], (window, 1)))
# create labels
labels.append(data[i+target_size])
return np.array(features), np.array(labels)
Listing 10-3
Function that creates features and labels
```

该函数接受一个数据集，一个我们想要开始分割的索引，一个结束索引，每个窗口的大小和目标大小。参数*window*是过去信息窗口的大小。*target_size*是我们希望我们的模型学习预测的未来时间跨度。

该函数创建列表来保存特征和标签。然后它建立反映窗口大小的起始点。为了创建测试集，函数检查*end*值。如果*None*，它使用整个数据集的长度减去目标大小作为结束值，这样测试集就可以从训练集结束的地方开始。

一旦确定了训练和测试的起始点，该函数将创建特征窗口和标签。每个特征窗口的*索引*在迭代过程中作为下一个窗口建立。也就是说，每个后续的*索引*集从上一个集结束的地方开始。特征窗口被重塑以供 TensorFlow 使用，并添加到*features*列表中。标签作为下一个窗口中的最后一个观察结果创建，然后添加到*labels*列表中。特征和标签都以 numpy 数组的形式返回。

该函数可能看起来有些复杂，但它实际上只是创建了一个包含相对湿度观测窗口（对我们实验而言）的特征集，以及另一个包含目标的特征集。目标是基于下一个数据窗口中的最后一个相对湿度观测。这很有道理，因为下一个窗口的最后一个相对湿度是未来相对湿度的一个很好的指示。因此，特征集成为包含时间步观测的窗口集，而标签集包含每个窗口的预测。

### 创建训练和测试集

对于训练集，从数据集的索引 0 开始，继续到 315,412。对于测试集，取剩余部分。设置窗口大小为 20，目标为 0。

如列表 10-4 所示调用该函数。

```py
# create train and test sets
import numpy as np
window = 20
target = 0
x_train, y_train = create_datasets(rh_scaled, 0, train_split,
window, target)
x_test, y_test = create_datasets(rh_scaled, train_split, None,
window, target)
Listing 10-4
Create train and test sets
```

检查训练和测试数据：

```py
print ('train:', end=' ')
print (x_train.shape, y_train.shape)
print ('test:', end=' ')
print (x_test.shape, y_test.shape)
```

如预期，形状反映了每个数据集的大小、窗口大小和 1 维。*1*维表示我们正在对未来进行一次预测。训练集包含 315,393 条记录，由包含 20 个相对湿度读数的窗口组成。测试集包含 105,118 条记录，由包含 20 个相对湿度读数的窗口组成。

那么，为什么我们有 315,393 个训练观测值而不是原始的 315,413 个？原因是我们需要第一个窗口作为历史数据。所以，只需从 315,413 中减去第一个 20 个窗口。对于测试数据，从 105,138 中减去第一个 20 个窗口，得到 105,118。

我们可以创建更大的窗口，但这会显著增加我们必须处理的数据量。每个窗口只有 20 个观测值，我们已经有 6,307,860（315,393 `×` 20）个训练数据点和 2,102,360（105,118 `×` 20）个测试数据点！

### 查看过去历史窗口

检查训练集中的第一个窗口：

```py
print ('length of window:', len(x_train[0]), br)
print ('first window of past history:')
print (x_train[0], br)
print ('target relative humidity to predict:')
print (y_train[0])
```

如预期，窗口包含 20 个相对湿度读数。那么我们是如何得到目标的？

从下一个窗口中取最后一条记录：

```py
print ('target from the 1st window:', end='   ')
print (np.round(y_train[0], 8))
print ('last obs from the 2nd window:', end=' ')
print (np.round(x_train[1][19][0], 8))
```

通过检查第二个窗口进行验证：

```py
print ('second window of past history:')
print (x_train[1], br)
print ('target relative humidity to predict:')
print (y_train[1])
```

让我们看看下一个几个窗口的模式是否保持不变，如列表 10-5 所示。

```py
print ('target from the 2nd window:', end='   ')
print (np.round(y_train[1], 8))
print ('last obs from the 3rd window:', end=' ')
print (np.round(x_train[2][19][0], 8), br)
print ('target from the 3rd window:', end='   ')
print (np.round(y_train[2], 8))
print ('last obs from the 4th window:', end=' ')
print (np.round(x_train[3][19][0], 8), br)
print ('target from the 4th window:', end='   ')
print (np.round(y_train[3], 8))
print ('last obs from the 5th window:', end=' ')
print (np.round(x_train[4][19][0], 8), br)
print ('target from the 5th window:', end='   ')
print (np.round(y_train[4], 8))
print ('last obs from the 6th window:', end=' ')
print (np.round(x_train[5][19][0], 8))
Listing 10-5
Inspect the patterns for the next few windows
```

### 绘制单个示例

创建一个函数，返回从*长度*到*0*的时间步长列表：

```py
def create_time_steps(length):
return list(range(-length, 0))
```

从-length 到 0 开始，使用上一个窗口作为历史数据。

创建另一个函数，接受单个数据窗口及其目标、delta 和一个标题，如列表 10-6 所示。

```py
def plot(plot_data, delta=0, title='Data Window'):
labels = ['history', 'actual future', 'model prediction']
marker = ['r.-', 'b*', 'g>']
time_steps = create_time_steps(plot_data[0].shape[0])
if delta: future = delta
else: future = 0
plt.title(title)
for i, obs in enumerate(plot_data):
if i:
plt.plot(future, obs, marker[i], markersize=10,
label=labels[i])
else:
plt.plot(time_steps, obs.flatten(), marker[i],
label=labels[i])
plt.legend()
plt.xlim([time_steps[0], (future+5)*2])
plt.xlabel('time step')
return plt
Listing 10-6
Function that plots an example
```

参数*delta*表示变量的变化。默认值为无变化。该函数将数据窗口中的每个元素及其相关的时间步长绘制出来。

根据训练集中的第一个数据窗口和目标调用该函数：

```py
plot([x_train[0], y_train[0]])
```

每个窗口的*实际未来*是其标签，即下一个窗口的最后一条记录。

### 创建可视化性能基线

在训练之前，创建一个简单的可视化性能基线来与模型性能进行比较是个好主意。当然，有很多方法可以做到这一点，但一个非常简单的方法是使用最后 20 个观测值的平均值。

创建一个函数，返回观测值窗口的平均值：

```py
def baseline(history):
return np.mean(history)
```

使用基线预测绘制第一个数据窗口：

```py
plot([x_train[0], y_train[0], baseline(x_train[0])], 0,
'baseline prediction')
```

### 创建基线指标

创建一个基线指标来与我们的模型进行比较也是个好主意。最简单的方法是预测每个窗口的最后值。然后我们可以找到预测的平均均方误差，并使用这个值作为我们的指标。

创建一个基线指标，如列表 10-7 所示。

```py
# display shape of test set
print ('TensorFlow shape:', x_test[0].shape, br)
# remove '1' dimension for easier processing
x_test_np = tf.squeeze(x_test)
print ('numpy shape:', x_test_np[0].shape, br)
# predict last value for each window
y_pred = x_test_np[:, -1]
# compute average MSE
MSE = np.mean(tf.keras.losses.mean_squared_error(
y_test, y_pred))
print ('MSE:', MSE)
Listing 10-7
Create a baseline metric
```

均方误差（MSE）非常小，这意味着我们的基线指标可能很难超越。为什么？一般来说，机器学习有一个相当明显的局限性。除非学习算法被硬编码为寻找特定类型的简单模型，否则参数学习有时可能无法找到简单问题的简单解决方案。我们的时间序列问题是一个非常简单的问题。

### 完成输入管道

如列表 10-8 所示，对训练数据进行洗牌、分批和缓存。

```py
BATCH_SIZE = 256
BUFFER_SIZE = 10000
# prepare the train set
train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_one = (train.cache()
.shuffle(BUFFER_SIZE)
.batch(BATCH_SIZE)
.repeat())
# prepare the test set
test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_one = test.batch(BATCH_SIZE).repeat()
Listing 10-8
Finish the input pipeline
```

检查张量：

```py
train_one, test_one
```

如预期，窗口有 20 个观测值和 1 个预测值。

### 探索数据窗口

验证特征和标签是否以 256 元素窗口进行分批：

```py
for feature, label in train_one.take(1):
print (len(feature), len(label))
```

显示批次中的第一个窗口：

```py
for feature, label in train_one.take(1):
print ('feature:')
print (feature[0].numpy(), br)
print ('label:', label[0].numpy())
```

如预期，窗口包含一个包含 20 个观测值的特征集和一个包含一个预测的标签。

### 创建模型

建立输入形状：

```py
input_shape = x_train.shape[-2:]
input_shape
```

输入形状表示 20 个窗口大小和 1 个预测。

导入必要的库，清除之前的模型会话，生成一个用于可重复性的种子，并创建如列表 10-9 所示的模型。

```py
# clear any previous models
tf.keras.backend.clear_session()
#import libraries
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense
# generate seed to ensure reproducibility
tf.random.set_seed(0)
neurons = 32  # number of neurons in GRU layer
model = Sequential([
GRU(neurons, input_shape=input_shape),
Dense(1)
])
Listing 10-9
Create the model
```

RNN 非常适合时间序列数据，因为它的层可以为早期层提供反馈。具体来说，它按时间步逐个处理时间序列数据，同时记住它在训练期间看到的信息。我们的模型使用 GRU 层，这是一种专门的时间序列建模 RNN 层，能够记住长时间内的信息。因此，它非常适合时间序列建模。

### 模型摘要

检查模型：

```py
model.summary()
```

我们使用公式 *3* `×` *(n*^(*2*) `×` *mn + 2n)* 来计算 GRU 的可学习参数数量，其中 *m* 是输入维度，*n* 是输出维度。对于任何神经网络，将前一层输出乘以当前层的神经元（*m* `×` *n*）。此外，考虑到当前层的神经元。但由于反馈（*2n*），需要将它们计数两次。GRU 层有反馈，所以平方输出维度（*n*^(*2*)）。由于 GRU 有三个操作集（隐藏状态、重置门和更新门）需要权重矩阵，所以乘以*3*。

这里是计算 GRU 层参数的分解：

+   3 `×` (32² + 32 + 2 `×` 32)

+   3 `×` (1024 + 32 + 64)

+   3 `×` 1120

+   *3,360*

输出维度是 32。由于没有前一层，输入维度不存在。

密集层有 33 个可学习参数，这是通过将前一层（32）的神经元乘以当前层（1）的神经元，并加上当前层的神经元数量（1）计算得出的。

### 验证模型输出

从模型中进行一个*未训练*的预测：

```py
for x, y in test_one.take(1):
print(model.predict(x).shape)
```

预测显示批大小为 256，预测 1 个。所以模型按预期工作。

### 编译模型

编译：

```py
model.compile(optimizer='adam', loss="mse")
```

### 训练模型

训练：

```py
num_train_steps = 400
epochs = 10
history = model.fit(train_one, epochs=epochs,
steps_per_epoch=num_train_steps,
validation_data=test_one,
validation_steps=50)
```

### 在测试数据上推广

推广：

```py
test_loss = model.evaluate(test_one, steps=num_train_steps)
```

### 进行预测

如列表 10-10 所示进行三次预测。

```py
n = 3
title = 'GRU prediction'
for i, (x, y) in enumerate(test_one.take(n)):
p = model.predict(x)[0]
plot([x[0].numpy(), y[0].numpy(), p], 0,
title + ' window ' + str(i))
plt.show()
Listing 10-10
Make some predictions
```

虽然视觉检查很漂亮，但它并不能有效地衡量整体性能。

### 绘制模型性能图

列表 10-11 绘制了模型性能：

```py
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.figure()
plt.plot(epochs, loss, 'bo', label='training loss')
plt.plot(epochs, val_loss, 'b', label='validation loss')
plt.title('training and validation loss')
plt.legend()
plt.show()
Listing 10-11
Plot model performance
```

Voilà！我们的模型表现相当不错！

## 预测多元时间序列

我们刚刚展示了如何根据单个特征进行单次预测。现在，让我们对多个变量进行多次预测。我们可以选择 14 个特征中的任何一个（我们不想从日期时间参考中进行预测）。

以下是可以用的 14 个特征：

+   p (mbar) – 毫巴大气压力

+   T (degC) – 摄氏温度

+   Tpot (K) – 开尔文温度

+   Tdew (degC) – 相对湿度摄氏温度

+   rh (%) – 相对湿度

+   VPmax (mbar) – 毫巴饱和蒸汽压

+   VPact (mbar) – 毫巴压力蒸汽压

+   VPdef (mbar) – 蒸气压亏缺（毫巴）

+   sh (g/kg) – 每千克的具体湿度

+   H2OC (mmol/mol) – 每摩尔水蒸气浓度（毫摩尔/摩尔）

+   rho (g/m**3) – 空气密度（克/立方米）

+   wv (m/s) – 风速（米/秒）

+   最大 wv (m/s) – 最大风速（米/秒）

+   wd (deg) – 风向（度）

创建一个变量来存储我们希望考虑的实验特征：

```py
mv_features = ['Tdew (degC)', 'sh (g/kg)', 'H2OC (mmol/mol)', 'T (degC)']
```

我们选择了四个特征 – Tdew (℃), sh (g/kg), H2OC (mmol/mol), 和 T (℃)。*Tdew (℃)* 是相对于湿度的摄氏度温度。*sh (g/kg)* 是每千克的具体湿度。*H2OC (mmol/mol)* 是每摩尔水蒸气浓度。而 *T (℃)* 是摄氏度温度。我们仅为了演示目的选择了这些特征。特征的选择应基于问题域。

创建一个数据框来存储特征：

```py
mv_features = df[mv_features]
mv_features.index = df['Date Time']
mv_features.head()
```

可视化特征：

```py
mv_features.plot(subplots=True)
```

可视化单个特征：

```py
mv_features['T (degC)'].plot(subplots=True)
```

### 数据缩放

将数据框转换为 numpy 数组：

```py
f_np = mv_features.to_numpy()
f_np[:5]
```

检查观测值的数量：

```py
len(f_np)
```

如预期，我们有 420,551 个观测值。

按照列表 10-12 中的说明缩放数据。

```py
# scale features
f_sc = tf.keras.utils.normalize(f_np)
print ('shape after tf function:', f_sc.shape, br)
# squeeze
f_sq = tf.squeeze(f_sc)
# convert to numpy
f_scaled = f_sq.numpy()
print ('first five scaled observation:')
print (f_scaled[:5])
Listing 10-12
Scale data
```

### 多步模型

在相对湿度的情况下，我们只预测了一个未来的点。但我们可以创建一个模型来学习预测一系列未来的值，这正是我们将使用刚刚建立的多变量数据所要做的事情。

假设我们想要训练我们的多步模型来学习预测接下来的 6 小时。由于我们的数据时间步长是 10 分钟（每 10 分钟一个观测值），每小时有六个观测值。考虑到我们想要预测接下来的 6 小时，我们的模型做出 *36*（6 个观测值 `×` 6）个预测。

让我们再假设我们想要展示每个样本过去 3 天的模型数据。由于一天有 24 小时，我们每天有 144（6 `×` 24）个观测值。因此，我们总共有 *432*（3 `×` 144）个观测值。但我们每小时想要采样一次，因为我们不期望在 60 分钟内任何特征都会有剧烈变化。因此，*72*（432 个观测值/每小时 6 个观测值）个观测值代表每个数据窗口。

### 生成器

由于我们正在训练多个特征来预测一系列未来的值，我们创建了一个生成器函数来创建训练和测试分割。一个 **生成器** 是一个函数，它返回一个对象迭代器，我们一次迭代一个值。生成器定义得像正常函数一样，但它使用 *yield* 关键字而不是 *return* 生成值。所以添加 yield 关键字自动将一个正常函数转换为生成器函数。

生成器易于实现，但理解起来有点困难。我们以与正常函数相同的方式调用生成器函数。但是，当我们调用它时，会创建一个生成器对象。我们必须迭代生成器对象以查看其内容。当我们迭代生成器对象时，生成器函数中的所有过程都会被处理，直到它达到一个 yield 语句。一旦发生这种情况，生成器就会从其内容中产生一个新的值，并将执行权返回给 for 循环。因此，生成器为每个遇到的 yield 语句从其内容中产生一个元素。

### 使用生成器的优点

生成器函数允许我们声明一个表现得像迭代器的函数。因此，我们可以以快速、简单和干净的方式创建迭代器。**迭代器**是一个可以迭代的对象。它用于抽象数据容器，使其表现得像一个可迭代对象。作为程序员，我们经常使用字符串、列表和字典等可迭代对象。

生成器节省内存空间，因为它们在实例化时不会计算每个项的值。生成器仅在明确请求时计算值。这种行为被称为*惰性求值*。当我们处理大型数据集时，惰性求值很有用，因为它允许我们立即开始使用数据，而无需等待整个数据集处理完毕。它还节省内存，因为数据仅在需要时生成。

### 生成器注意事项

+   生成器创建一个单一的对象。因此，无论生成器产生多少值，生成器对象都只能分配给*单个*变量。

+   一旦迭代完毕，生成器就*耗尽*了。因此，必须重新运行以重新填充。

### 创建一个生成器函数

如列表 10-13 所示，创建一个生成器：

```py
def generator(d, t, o, e, w, ts, s):
# hold features and labels
features, labels = [], []
# initialize variables
data, target = d, t
origin, end = o, e
window, target_size = w, ts
step = s
# establish starting point that reflects window size
origin = origin + window
# enable split for test data
if end < 0:
end = len(data) - target_size
# create feature set of 'window-sized' elements
for i in range(origin, end):
# create index set to identify each window
indices = range(i-window, i, step)
# create features
features.append(data[indices])
# create labels
labels.append(target[i:i+target_size])
yield np.array(features), np.array(labels)
Listing 10-13
Generator function
```

### 生成训练和测试数据

如列表 10-14 所示，调用生成器创建训练和测试集。

```py
window = 432  # observations for 3 days
future_target = 36  # predictions for the next 6 hours
step = 6  # number of timesteps per hour
train_gen = generator(f_scaled, f_scaled[:, 1], 0,
train_split, window,
future_target, step)
test_gen = generator(f_scaled, f_scaled[:, 1],
train_split, -1, window,
future_target, step)
Listing 10-14
Invoke the generator
```

注意，我们将生成器对象分配给单个变量！

### 重新构成生成的张量

将生成数据重新制作成 NumPy 数组。由于训练和测试数据是生成器，我们必须迭代以创建特征和标签。

通过迭代生成器创建特征和标签列表，从训练数据开始：

```py
train_f, train_l = [], []
for i, row in enumerate(train_gen):
train_f.append(row[0])
train_l.append(row[1])
```

将列表数据转换为 NumPy 数组：

```py
train_features = np.asarray(train_f, dtype=np.float64)
train_labels = np.asarray(train_l, dtype=np.float64)
```

检查形状：

```py
train_features.shape, train_labels.shape
```

如预期，每个窗口有 72 个观察值和 4 个特征。标签有 36 个预测。

使用*tf.squeeze*函数移除*1*维：

```py
train_features, train_labels = tf.squeeze(train_features),\
tf.squeeze(train_labels)
train_features.shape, train_labels.shape
```

如列表 10-15 所示，重新构成测试数据。

```py
# create test data
test_f, test_l = [], []
for i, row in enumerate(test_gen):
test_f.append(row[0])
test_l.append(row[1])
# convert lists to numpy arrays
test_features = np.asarray(test_f, dtype=np.float64)
test_labels = np.asarray(test_l, dtype=np.float64)
# squeeze out the '1' dimension created by the generator
test_features, test_labels = tf.squeeze(test_features),\
tf.squeeze(test_labels)
test_features.shape, test_labels.shape
Listing 10-15
Reconstitute test data
```

### 完成输入管道

如列表 10-16 所示，完成输入管道。

```py
train_mv = tf.data.Dataset.from_tensor_slices(
(train_features, train_labels))
train_mv = train_mv.cache().shuffle(
BUFFER_SIZE).batch(BATCH_SIZE).repeat()
test_mv = tf.data.Dataset.from_tensor_slices(
(test_features, test_labels))
test_mv = test_mv.batch(BATCH_SIZE).repeat()
Listing 10-16
Finish the input pipeline
```

检查张量：

```py
train_mv, test_mv
```

检查一个批次：

```py
for feature, label in train_mv.take(1):
print (feature.shape)
print (label.shape)
```

每个批次包含 256 个窗口的特征数据和 256 个标签。每个窗口有 72 个观察值，每个观察值包含 4 个特征。每个标签有 36 个预测。

检查第一个训练示例：

```py
for feature, label in train_mv.take(1):
print ('observations:', len(feature[0]))
print (feature[0], br)
print ('predictions:', len(label[0]))
print (label[0])
```

如预期，第一个窗口有 72 个观察值和 4 个特征，第一个标签有 36 个预测。

建立输入形状：

```py
input_shape_multi = feature.shape[-2:]
input_shape_multi
```

如预期，窗口大小为 72，每个窗口有 4 个特征。

### 创建模型

如列表 10-17 所示创建模型。

```py
# clear any previous models
tf.keras.backend.clear_session()
# generate seed to ensure reproducibility
tf.random.set_seed(0)
neurons = 32  # neurons in GRU layer
outputs = 36  # predictions
gen_model = Sequential([
GRU(neurons, input_shape=input_shape_multi),
Dense(outputs)
])
Listing 10-17
Create the model
```

### 模型摘要

检查模型：

```py
gen_model.summary()
```

这里是计算 GRU 层可学习参数的计算分解：

+   3 `×` (32² + 32 `×` 4 + 2 `×` 32)

+   3 `×` (1024 + 128 + 64)

+   3 `×` 1216

+   *3,648*

唯一的不同之处在于我们有了四个特征。所以将输出（*n*）维度乘以 4。

密集层有 1,188 个可学习参数，这是通过将前一层（32 个）的神经元与当前层（36 个）的神经元相乘，并加上当前层的神经元数量（36 个）计算得出的。

### 编译模型

编译：

```py
gen_model.compile(optimizer='adam', loss="mse")
```

### 训练模型

训练：

```py
num_train_steps = 400
epochs = 10
gen_history = gen_model.fit(train_mv, epochs=epochs,
steps_per_epoch=num_train_steps,
validation_data=test_mv,
validation_steps=50)
```

### 在测试数据上一般化

一般化：

```py
test_loss = gen_model.evaluate(test_mv, steps=num_train_steps)
```

### 绘制性能图

如列表 10-18 所示绘制性能图。

```py
loss = gen_history.history['loss']
val_loss = gen_history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.figure()
plt.plot(epochs, loss, 'bo', label='training loss')
plt.plot(epochs, val_loss, 'b', label='validation loss')
plt.title('training and validation loss')
plt.legend()
plt.show()
Listing 10-18
Plot performance
```

模型有点过拟合，但性能仍然相当不错。

### 绘制数据窗口图

如列表 10-19 所示创建绘图函数。

```py
def multi_step_plot(window, true_future, pred):
plt.figure(figsize=(12, 6))
num_in = create_time_steps(len(window))
num_out = len(true_future)
plt.plot(num_in, np.array(window[:, 1]), 'm',
label='history')
plt.plot(np.arange(num_out)/step, np.array(true_future),
'bo', label='true future')
if pred.any():
plt.plot(np.arange(num_out)/step, np.array(pred), 'go',
label='predicted future')
plt.legend(loc='upper left')
plt.show()
Listing 10-19
Plotting function
```

从第一个批次中绘制第一个训练窗口：

```py
for x, y in train_mv.take(1):
multi_step_plot(x[0], y[0], np.array([0]))
```

### 进行预测

基于第一个训练窗口进行预测：

```py
for x, y in test_mv.take(1):
y_pred = gen_model.predict(x)[0]
multi_step_plot(x[0], y[0], y_pred)
```

还不错。
