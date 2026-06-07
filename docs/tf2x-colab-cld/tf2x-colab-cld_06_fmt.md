
# 6. 回归

**回归**是一种基于从数据集中获得的变量（或特征）之间的关系来预测事件连续输出的监督学习方法。连续结果是一个实数值，如整数或浮点值，通常量化为数量和大小。回归是深度学习建模中广泛流行的一种类型。

由于回归预测的是一个数量，性能是通过这些预测中的误差来衡量的。回归性能可以通过多种方式衡量。但最常见的是均方误差（MSE）、平均绝对误差（MAE）和均方根误差（RMSE）。

*MSE*是最常用的指标之一。然而，当单个不良预测会破坏整个模型的预测能力时，它最不有用。也就是说，当数据集中包含大量噪声时。当数据集中包含异常值或意外值时，它最有用。意外值是过高或过低的值。

与 MSE 相比，*MAE*对异常值不太敏感，因为它不会惩罚巨大的误差。它通常用于在连续变量数据上衡量性能时。它提供了一个线性值，平均加权个别差异。

*RMSE*误差在平均之前会被平方。因此，RMSE 会给较大的误差分配更高的权重。所以当存在大误差并且它们极大地影响模型性能时，RMSE 非常有用。RMSE 的一个好处是误差分数的单位与预测值相同。

我们详细地研究了著名的*boston*数据集。我们展示了如何加载数据、构建输入管道和建模数据。我们还展示了如何使用模型进行预测。最后，我们建模了一个不同的数据集。*Cars*数据集可能不像*boston*那样流行，但我们想让你体验另一组数据。

各章节的笔记本位于以下 URL：`[https://github.com/paperd/tensorflow](https://github.com/paperd/tensorflow)`。

启用 GPU（如果尚未启用）：

1. 在右上角菜单中点击*运行时*。
2. 从下拉菜单中选择*更改运行时类型*。
3. 从*硬件加速器*下拉菜单中选择*GPU*。
4. 点击*保存*。

测试 GPU 是否处于活动状态：

```py
import tensorflow as tf
# display tf version and test if GPU is active
tf.__version__, tf.test.gpu_device_name()
```

导入*tensorflow*库。如果显示‘/device:GPU:0’，则 GPU 处于活动状态。如果显示‘..’，则常规 CPU 处于活动状态。

## 波士顿住房数据集

**波士顿住房**是一个从美国人口普查局收集的有关马萨诸塞州波士顿地区住房信息的数据集。它来自 StatLib 档案([`http://lib.stat.cmu.edu/datasets/boston`](http://lib.stat.cmu.edu/datasets/boston))，并在机器学习文献中被广泛用作算法的基准。该数据集规模较小，只有 506 个案例。

这个数据集的名称是*boston*。它包含 12 个特征和 1 个结果（或目标）。特征如下：

1.  CRIM – 每人犯罪率
2.  ZN – 住宅用地中超过 25,000 平方英尺地块的比例
3.  INDUS – 每个城镇非零售商业地块的比例
4.  CHAS – 查尔斯河虚拟变量（如果地块边界是河流则为 1，否则为 0）
5.  NOX – 氮氧化物浓度（每 1000 万分之一部分）
6.  RM – 每个住宅的平均房间数
7.  AGE – 1940 年前建造的业主自住单元比例
8.  DIS – 五个波士顿就业中心的加权距离
9.  RAD – 到放射状高速公路的可达性指数
10. TAX – 每 $10,000 的全额财产税率
11. PTRATIO – 城镇的学生-教师比例
12. LSTAT – 人口中低阶层比例

目标是

+   MEDV – 业主自住房屋的中值（以千美元为单位）

数据是在 1970 年收集的，所以不要对房屋的低中值感到惊讶。

### 波士顿数据

您可以通过几个简单的步骤直接从 GitHub 访问本书的任何数据集：

1.  访问书籍 URL：`[https://github.com/paperd/tensorflow](https://github.com/paperd/tensorflow)`。
2.  定位数据集并点击它。
3.  点击 *原始* 按钮。
4.  将 URL 复制到 Colab 并将其分配给一个变量。
5.  使用 Pandas 的 read_csv 方法读取数据集。

为了方便，我们已找到适当的 URL 并将其分配给一个变量：

```py
url = 'https://raw.githubusercontent.com/paperd/tensorflow/\
master/chapter6/data/boston.csv'
```

将数据集读取到 pandas dataframe 中：

```py
import pandas as pd
data = pd.read_csv(url)
```

验证数据集是否正确读取：

```py
data.head()
```

### 探索数据集

获取数据类型：

```py
data.dtypes
```

所有特征的数据类型为 float64 或 int64。标签 *MEDV* 的数据类型为 float64。

获取一般信息：

```py
data.info()
```

数据集包含 506 条记录。

使用 *describe* 方法创建包含基本统计信息的 dataframe 并将其转置以便更容易查看：

```py
data_t = data.describe()
desc = data_t.T
desc
```

从转置的 dataframe 中获取特定统计信息：

```py
desc[['mean', 'std']]
```

从原始 dataframe 中描述特定特征：

```py
data.describe().LSTAT
```

获取列：

```py
cols = list(data)
cols
```

### 创建特征和目标集

创建目标：

```py
# create a copy of the DataFrame
df = data.copy()
# create the target
target = df.pop('MEDV')
print (target.head())
```

由于我们已经从副本中移除了 MEDV，因此它应该只包含以下特征：

```py
df.head()
```

### 从特征 DataFrame 获取特征名称

由于目标不再包含在数据框中，因此很容易获取特征：

```py
feature_cols = list(df)
feature_cols
```

获取特征数量：

```py
len(feature_cols)
```

### 转换特征和标签

使用 *values* 方法将 pandas dataframe 的值转换为浮点数：

```py
features = df.values
labels = target.values
type(features), type(labels)
```

### 将数据集拆分为训练集和测试集

列表 6-1 创建了训练集和测试集。

```py
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
features, labels, test_size=0.33, random_state=0)
br = '\n'
print ('X_train shape:', end=' ')
print (X_train.shape, br)
print ('X_test shape:', end=' ')
print (X_test.shape)
Listing 6-1
Train and test sets
```

### 缩放数据并创建 TensorFlow 张量

对于图像数据，我们通过将每个元素除以 255.0 来缩放，以确保每个输入参数（在我们的例子中是一个像素）具有相似的数据分布。然而，由连续值表示的特征缩放方式不同。我们将连续数据缩放到具有均值（μ）为 0 和标准差（σ）为 1。符号 μ 发音为 mu，符号 σ 发音为 sigma。标准差为 1 被称为单位方差。

列表 6-2 缩放了训练和测试数据。

```py
# scale feature data and create TensorFlow tensors
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.fit_transform(X_test)
train = tf.data.Dataset.from_tensor_slices(
(X_train_std, y_train))
test = tf.data.Dataset.from_tensor_slices(
(X_test_std, y_test))
Listing 6-2
Scale train and test sets
```

让我们查看列表 6-3 中显示的第一个张量。

```py
def see_samples(data, num):
for feat, targ in data.take(num):
print ('Features: {}'.format(feat), br)
print ('Target: {}'.format(targ))
n = 1
see_samples(train, n)
Listing 6-3
Display the first tensor
```

第一个样本看起来正如我们所期望的那样。

### 准备训练张量

打乱训练数据，批量处理，并预取训练和测试数据：

```py
BATCH_SIZE, SHUFFLE_BUFFER_SIZE = 16, 100
train_bs = train.shuffle(
SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE).prefetch(1)
test_bs = test.batch(BATCH_SIZE).prefetch(1)
```

检查张量：

```py
train_bs, test_bs
```

### 创建模型

如果我们没有大量的训练数据，一种避免过拟合的技术是创建一个具有少量隐藏层的简单网络。我们正是这样做的！

64 神经元输入层可以容纳我们的 12 个输入特征。我们有一个包含 64 个神经元的隐藏层。输出层有一个神经元，因为我们使用回归并且有一个目标。

创建一个如列表 6-4 所示的模型。

```py
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
import numpy as np
# clear any previous model
tf.keras.backend.clear_session()
# generate a seed for replication purposes
np.random.seed(0)
tf.random.set_seed(0)
# notice input shape accommodates 12 features!
model = Sequential([
Dense(64, activation="relu", input_shape=[12,]),
Dense(64, activation="relu"),
Dense(1)
])
Listing 6-4
Create a model
```

### 模型摘要

检查模型：

```py
model.summary()
```

第一层的输出形状为（None，64），因为我们在这个层有 64 个神经元。通过将这个层的 64 个神经元乘以 12 个特征并加上这个层的 64 个神经元，我们得到 832 个参数。

第二层的输出形状为（None，64），因为我们在这个层有 64 个神经元。通过将这个层的 64 个神经元乘以前一层的 64 个神经元并加上这个层的 64 个神经元，我们得到 4,160 个参数。

第三层的输出形状为（None，1），因为我们有一个目标。通过将前一层 64 个神经元加到这一层的 1 个神经元上，我们得到 65 个参数。

### 编译模型

编译：

```py
rmse = tf.keras.metrics.RootMeanSquaredError()
model.compile(loss='mse', optimizer="RMSProp",
metrics=[rmse, 'mae', 'mse'])
```

均方误差（MSE）是用于回归问题的一种常见损失函数。平均绝对误差（MAE）和 RMSE 也是常用的指标。经过一些实验，我们发现优化器*RMSProp*在这个数据集上表现相当不错。RMSProp 是一种用于全批量优化的算法。它试图通过仅使用梯度的符号来解决梯度幅度可能差异很大的问题，这保证了所有权重更新的大小相同。

### 训练模型

训练模型 50 个周期：

```py
history = model.fit(train_bs, epochs=50,
validation_data=test_bs)
```

### 可视化训练

创建变量*hist*来保存模型的记录作为一个 pandas 数据框。创建另一个变量*hist[‘epoch’]*来保存周期历史。显示最后五行以了解性能。

这里是代码：

```py
hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist.tail()
```

按照列表 6-5 所示构建图表。

```py
import matplotlib.pyplot as plt
def plot_history(history, limit1, limit2):
hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
plt.figure()
plt.xlabel('epoch')
plt.ylabel('MAE [MPG]')
plt.plot(hist['epoch'], hist['mae'],
label='Train Error')
plt.plot(hist['epoch'], hist['val_mae'],
label = 'Val Error')
plt.ylim([0, limit1])
plt.legend()
plt.title('MAE by Epoch')
plt.show()
plt.clf()
plt.figure()
plt.xlabel('Epoch')
plt.ylabel('MSE [MPG]')
plt.plot(hist['epoch'], hist['mse'],
label='Train Error')
plt.plot(hist['epoch'], hist['val_mse'],
label = 'Val Error')
plt.ylim([0, limit2])
plt.legend()
plt.title('MSE by Epoch')
plt.show()
plt.clf()
plt.figure()
plt.xlabel('Epoch')
plt.ylabel('RMSE [MPG]')
plt.plot(hist['epoch'], hist['root_mean_squared_error'],
label='Train Error')
plt.plot(hist['epoch'], hist['val_root_mean_squared_error'],
label = 'Val Error')
plt.ylim([0, limit2])
plt.legend()
plt.title('RMSE by Epoch')
plt.show()
# set limits to make plot readable
mae_limit, mse_limit = 10, 100
plot_history(history, mae_limit, mse_limit)
Listing 6-5
Visualize training performance
```

由于验证误差比训练误差更差，模型正在过拟合。我们能做什么？第一步是估计性能开始下降的时间。从可视化中，你能看到什么时候发生这种情况吗？

### 提前停止

在分类中，我们的目标是最大化准确率。当然，我们还想最小化损失。在回归中，我们的目标是最小化 MSE 或其他常见的误差指标。从可视化中，我们看到我们的模型正在过拟合，因为验证误差高于训练误差。我们还看到，一旦训练误差和验证误差交叉，性能开始下降。

我们可以运行一个简单的调整实验来使这个模型更有用。我们可以在训练和验证误差非常接近时停止模型。这种技术称为提前停止。**提前停止**是一种广泛使用的方法，它会在验证数据集上的性能开始下降时停止训练。

让我们修改我们的训练实验，以便在验证分数没有提高时自动停止训练。我们使用一个 *EarlyStopping* 回调，它为每个 epoch 测试训练条件。当性能开始下降时，训练会自动停止。

我们需要做的只是更新拟合方法，并按列表 6-6 所示重新训练。
Listing 6-6
Early stopping
```


```python
# clear the previous model
tf.keras.backend.clear_session()
# generate a seed for replication purposes
np.random.seed(0)
tf.random.set_seed(0)
# monitor 'val_loss' for early stopping
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss')
history = model.fit(train_bs, epochs=50,
                    validation_data=test_bs,
                    callbacks=[early_stop])
Listing 6-6
Early stopping
```

Instead of allowing the algorithm to stop early, we can add some control. Just include a parameter to force the model to continue to a point where it gives us the best performance. The *patience* parameter can be set to a given number of epochs, after which training will stop if there is no improvement.

Let's see what happens, as shown in List 6-7.

```python
# clear the previous model
tf.keras.backend.clear_session()
# generate a seed for replication purposes
np.random.seed(0)
tf.random.set_seed(0)
# set number of patience epochs
n = 4
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=n)
history = model.fit(train_bs, epochs=50,
                    validation_data=test_bs,
                    callbacks=[early_stop])
Listing 6-7
Early stopping with patience
```

By adjusting the patience parameter, we can find better results.

Let's visualize:

```python
hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
train_limit, test_limit = 10, 100
plot_history(history, train_limit, test_limit)
```

### Remove Bad Data

The Boston dataset has some inherent bad data. What's the problem? Because the census service reviewed the data, the upper limit for house prices is $50,000. They decided to set the maximum value for the price variable to $50,000, so no price can exceed this value.

What should we do? Although this may not be the best solution, we can remove records where the price is $50,000 or more. This is not the best solution because we might remove completely qualified data, and we cannot know this. Another reason is that the dataset is small from the beginning. Neural networks are designed to perform best on large datasets.

To further explore this topic, we recommend this website:

[Understanding Unknown Facts About the Boston Housing Dataset](https://towardsdatascience.com/things-you-didnt-know-about-the-boston-housing-dataset-2e87a6f960e8)

### Get Data

We do not want to try to clean the dataset for TensorFlow consumption. So just reload the original dataset:

```python
# get the raw data
url = 'https://raw.githubusercontent.com/paperd/tensorflow/master/chapter6/data/boston.csv'
boston = pd.read_csv(url)
```

Verify data:

```python
boston.head()
```

### Remove Noise

Remove bad data, hoping this will reduce inherent noise:

```python
print ('data set before removing noise:', boston.shape)
# remove noise
noise = boston.loc[boston['MEDV'] >= 50]
data = boston.drop(noise.index)
print ('data set without noise:', data.shape)
```

**Noise** is irrelevant information or randomness in the data set. We removed several records from the data set.

### Create Feature and Target Data

Create feature and target sets:

```python
# create a copy of the dataframe
df = data.copy()
# create feature and target sets
target, features = df.pop('MEDV'), df.values
labels = target.values
```

### Build Input Pipeline

By splitting the data into training and test sets, standardizing feature data, and slicing the data into TensorFlow consumable parts, we create an input pipeline. By shuffling the training data, batching, and prefetching training and test data, we complete the pipeline.

Listing 6-8 contains the code to build the input pipeline.

```python
X_train, X_test, y_train, y_test = train_test_split(
    features, labels, test_size=0.33, random_state=0)
# standardize feature data and create TensorFlow tensors
X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.fit_transform(X_test)
# slice data for TensorFlow consumption
train = tf.data.Dataset.from_tensor_slices(
    (X_train_std, y_train))
test = tf.data.Dataset.from_tensor_slices(
    (X_test_std, y_test))
# shuffle, batch, prefetch
BATCH_SIZE = 16
SHUFFLE_BUFFER_SIZE = 100
train_n = train.shuffle(
    SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE).prefetch(1)
test_n = test.batch(BATCH_SIZE).prefetch(1)
Listing 6-8
Build the input pipeline
```

Check tensors:

```python
train_n, test_n
```

### Compile and Train

Listing 6-9 contains the code to compile and train the model.

```python
rmse = tf.keras.metrics.RootMeanSquaredError()
model.compile(loss='mse', optimizer="RMSProp",
              metrics=[rmse, 'mae', 'mse'])
tf.keras.backend.clear_session()
# generate a seed for replication purposes
np.random.seed(0)
tf.random.set_seed(0)
n = 4
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=n)
history = model.fit(train_n, epochs=50,
                    validation_data=test_n,
                    callbacks=[early_stop])
Listing 6-9
Compile and train the model
```

### Visualization

Plot the results:

```python
hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
train_limit, test_limit = 10, 100
plot_history(history, train_limit, test_limit)
```

Our pattern is not perfect, but we have indeed improved performance.

### Generalize on Test Data

Evaluate:

```python
loss, rmse, mae, mse = model.evaluate(test_n, verbose=2)
print ()
print('"Testing set Mean Abs Error: {:5.2f} thousand dollars'.
      format(mae))
```

MAE provides a good concept of performance for linear continuous data. Using this dataset, we can expect the model's predicted mean absolute error to be in the thousands of dollars.

### Make Predictions

Use the *predict* method to make predictions from the processed test data *test_n*:

```python
predictions = model.predict(test_n)
```

Show the **first** prediction:

```python
# predicted housing price
first = predictions[0]
print ('predicted price:', first[0], 'thousand')
# actual housing price
print ('actual price:', y_test[0], 'thousand')
```

Compare the predicted value and the actual price to evaluate model performance.

Show the first five predictions and compare them with the actual target values:

```python
five = predictions[:5]
actuals = y_test[:5]
print ('pred', 'actual')
for i, p in enumerate(range(5)):
    print (np.round(five[i][0], 1), actuals[i])
```

### Visualize Predictions

Listing 6-10 shows the predicted results against actual prices.

```python
fig, ax = plt.subplots()
ax.scatter(y_test, predictions)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()
Listing 6-10
Predictions against actual prices plot
```

The diagonal is the chart of actual house prices. The closer the predicted values are to the diagonal, the larger the error.

### Load Boston Data from Scikit-Learn

Since the Boston data is included in *sklearn.datasets*, we can load it from this environment:

```python
from sklearn import datasets
dataset = datasets.load_boston()
data, target = dataset.data, dataset.target
```

Access keys:

```python
dataset.keys()
```

The key list tells us how to access feature names:

```python
feature_names = dataset.feature_names
feature_names
```

Note that the sklearn dataset has an additional feature *B*. This column may be considered controversial because it selects black (or African American) individuals in the town separately.

We want to remove noise from the entire dataset, so we build a data frame containing feature data and add target data:

```python
df_sklearn = pd.DataFrame(dataset.data, columns=feature_names)
df_sklearn['MEDV'] = dataset.target
df_sklearn.head()
```

Check information:

```python
df_sklearn.info()
```

### Remove Noise

Remove noisy data:

```python
# remove noisy data
print ('data set before removing noise:', df_sklearn.shape)
noise = df_sklearn.loc[df_sklearn['MEDV'] >= 50]
df_clean = df_sklearn.drop(noise.index)
print ('data set without noise:', df_clean.shape)
```

### Build Input Pipeline

Build the pipeline as shown in Listing 6-11.

```python
# create a copy of the dataframe
df = df_clean.copy()
# create the target
target = df.pop('MEDV')
# convert features and target data
features = df.values
labels = target.values
# create train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    features, labels, test_size=0.33, random_state=0)
X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.fit_transform(X_test)
# slice data into a TensorFlow consumable form
train = tf.data.Dataset.from_tensor_slices(
    (X_train_std, y_train))
test = tf.data.Dataset.from_tensor_slices(
    (X_test_std, y_test))
# finalize the pipeline
BATCH_SIZE = 16
SHUFFLE_BUFFER_SIZE = 100
train_sk = train.shuffle(
    SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE).prefetch(1)
test_sk = test.batch(BATCH_SIZE).prefetch(1)
Listing 6-11
Build the input pipeline
```

### Model Data

Model data as shown in Listing 6-12.

```python
# clear any previous model
tf.keras.backend.clear_session()
# generate a seed for replication purposes
np.random.seed(0)
tf.random.set_seed(0)
# new model with 13 input features
model = Sequential([
    Dense(64, activation="relu", input_shape=[13,]),
    Dense(64, activation="relu"),
    Dense(1)
])
# compile the new model
rmse = tf.keras.metrics.RootMeanSquaredError()
model.compile(loss='mse', optimizer="RMSProp",
              metrics=[rmse, 'mae', 'mse'])
# train
n = 4
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                              patience=n)
history = model.fit(train_sk, epochs=50,
                    validation_data=test_sk,
                    callbacks=[early_stop])
Listing 6-12
Model data
```

### Model Car Data

Let's practice with another dataset.

### Get Car Data from GitHub:

We have found the URL and assigned it to a variable:

```python
cars_url = 'https://raw.githubusercontent.com/paperd/tensorflow/master/chapter6/data/cars.csv'
```

Read the data into a pandas data frame:

```python
cars = pd.read_csv(cars_url)
```

Verify data:

```python
cars.head()
```

Get dataset information:

```python
cars.info()
```

### Convert Categorical Columns to Numerical

Machine learning algorithms can only train numerical data. Therefore, we must convert any non-numeric features to numeric. The *Origin* column is categorical and not numeric. To solve this problem, one solution is to encode the data as one-hot encoding. **One-hot encoding** is a process of converting categorical data into a numeric form for machine learning algorithms to consume.

We first split the "origin" feature column from the original data frame into its own data frame. Then we use this data frame as a template to build a new feature column for the original "origin" feature in each category of the original data frame.

Create a copy of the data frame:

```python
# create a copy of dataframe
df = cars.copy()
origin = df.pop('Origin')
```

Define one-hot encoded feature columns for US, European, and Japanese cars:

```python
df['US'] = (origin == 'US') * 1.0
df['Europe'] = (origin == 'Europe') * 1.0
df['Japan'] = (origin == 'Japan') * 1.0
df.tail(8)
```

For the *US* origin, we assign *1.0 0.0 0.0*. For the *Europe* origin, we assign *0.0 1.0 0.0*. For the *Japan* origin, we assign *0.0 0.0 1.0*. Therefore, we replace the "origin" feature with three one-hot encoded features.



# 使用 pandas 进行独热编码

首先创建数据框的副本：

```py
# create a copy of df
alt = cars.copy()
```

独热编码：

```py
# get one hot encoding of columns 'Origin'
one_hot = pd.get_dummies(alt['Origin'])
```

删除原始列：

```py
# drop column as it is now encoded
alt = alt.drop('Origin', axis=1)
```

将编码列添加到数据框中：

```py
# join the encoded df
alt = alt.join(one_hot)
alt.tail(8)
```

### 切片多余数据

由于每辆车的名称对我们可能想要做出的任何预测都没有影响，我们可以将其放入自己的数据框中，以便将来回顾：

```py
try:
    name = df.pop('Car')
except:
    print("An exception occurred")
```

如果发生异常，`Car` 列已经删除。你可以多次运行这段代码，而不会对结果产生影响。

验证：

```py
df.tail(8)
```

### 创建特征和标签

我们的目的是预测数据集中汽车的每加仑英里数。因此，目标是 `MPG`，特征是剩余的特征列。

按照列表 6-13 创建特征和目标集。

```py
# create data sets
features = df.copy()
target = features.pop('MPG')
# get feature names
feature_cols = list(features)
print(feature_cols)
# get number of features
num_features = len(feature_cols)
print(num_features)
# convert feature and target data to float
features = features.values
labels = target.values
(type(features), type(labels))
Listing 6-13
Create feature and target sets
```

### 构建输入管道

按照列表 6-14 所示构建输入管道。

```py
# split
X_train, X_test, y_train, y_test = train_test_split(
    features, labels, test_size=0.33, random_state=0)
print('X_train shape:', end=' ')
print(X_train.shape, br)
print('X_test shape:', end=' ')
print(X_test.shape)
# scale
X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.fit_transform(X_test)
# slice
train = tf.data.Dataset.from_tensor_slices(
    (X_train_std, y_train))
test = tf.data.Dataset.from_tensor_slices(
    (X_test_std, y_test))
# shuffle, batch, prefetch
BATCH_SIZE = 16
SHUFFLE_BUFFER_SIZE = 100
train_cars = train.shuffle(
    SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE).prefetch(1)
test_cars = test.batch(BATCH_SIZE).prefetch(1)
Listing 6-14
Build the input pipeline
```

检查张量：

```py
train_cars, test_cars
```

### 模型数据

按照列表 6-15 所示建模数据。

```py
# clear any previous model
tf.keras.backend.clear_session()
# create the model
model = Sequential([
    Dense(64, activation="relu", input_shape=[num_features]),
    Dense(64, activation="relu"),
    Dense(1)
])
# compile
rmse = tf.keras.metrics.RootMeanSquaredError()
optimizer = tf.keras.optimizers.RMSprop(0.001)
model.compile(loss='mse',
              optimizer=optimizer,
              metrics=[rmse, 'mae', 'mse'])
# train
tf.keras.backend.clear_session()
n = 4
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=n)
car_history = model.fit(train_cars, epochs=100,
                        validation_data=test_cars,
                        callbacks=[early_stop])
Listing 6-15
Model data
```

### 检查模型

摘要：

```py
model.summary()
```

第一层的输出形状是 (None, 64)，因为我们在这层有 64 个神经元。通过将这层的 64 个神经元乘以 9 个特征并加上这层的 64 个神经元，我们得到 640 个参数。

第二层的输出形状是 (None, 64)，因为我们在这层有 64 个神经元。通过将这层的 64 个神经元乘以前一层的 64 个神经元并加上这层的 64 个神经元，我们得到 4,160 个参数。

第三层的输出形状是 (None, 1)，因为我们有一个目标。通过将前一层 64 个神经元加上这层 1 个神经元，我们得到 65 个参数。

### 可视化训练

可视化：

```py
hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
train_limit, test_limit = 10, 100
plot_history(history, train_limit, test_limit)
```

### 在测试数据上泛化

泛化：

```py
loss, rmse, mae, mse = model.evaluate(test_cars, verbose=2)
print()
print('Testing set Mean Abs Error: {:5.2f} MPG'.format(mae))
```

### 进行预测

预测：

```py
predictions = model.predict(test_cars)
```

### 可视化预测

如列表 6-16 所示，可视化预测结果。

```py
fig, ax = plt.subplots()
ax.scatter(y_test, predictions)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()
Listing 6-16
Visualize predictions for cars data
```

预测值越偏离对角线真实值线，错误就越大。
