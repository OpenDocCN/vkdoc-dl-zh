# 神经网络中的回归

为了向你证明神经网络确实可以用于解决哪怕是最简单的回归问题，我将用一个简单的例子来演示。在我们的程序中，将使用来自 Kaggle 竞赛的一个数据集进行简单线性回归。你可以从这里（`www.kaggle.com/luddarell/101-simple-linear-regressioncsv`）下载该数据集。数据仅包含两列——GPA 和 SAT。GPA 列代表学生的平均绩点，SAT 列代表学生的 SAT（学术能力评估测试）分数。我们将开发一个线性回归模型，以建立学生 GPA 与 SAT 分数之间的关系。模型训练完成后，我们将用它来预测给定 GPA 的学生在 SAT 考试中可能获得的分数。

## 项目设置

创建一个新的 Colab 项目，并将其命名为 `LinearRegression`。在项目代码中添加以下导入：

```
import tensorflow as tf
from tensorflow import keras
import pandas as pd
```

数据文件可在本书的网站上获取。要在你的项目中下载该文件，我们将使用 `wget`。添加以下代码来安装 `wget`：

```
!pip install wget
import wget
```

现在，使用以下代码下载文件：

```
url = 'https://raw.githubusercontent.com/Apress/artificial-neural-networks-with-tensorflow-2/main/Ch05/student.csv'
wget.download(url,'data.csv')
```

下载的文件将存储在 `/content/` 文件夹中，文件名为 `data.csv`。你可以通过先将文件加载到 pandas 数据框中，然后使用以下代码打印前几条记录来检查其内容：

```
import pandas as pd
df=pd.read_csv('/content/data.csv')
df.head(10)
```

你将看到如图 5-1 所示的输出。

![../images/495303_1_En_5_Chapter/495303_1_En_5_Fig1_HTML.jpg](img/495303_1_En_5_Fig1_HTML.jpg)

图 5-1：加载数据集中的示例行

接下来，你将从此数据集中提取特征和标签。

### 提取特征和标签

该数据集仅包含两列——GPA 和 SAT。我们将使用 GPA 作为特征，SAT 作为标签。请注意，我们试图根据学生的 GPA 来预测其 SAT 分数。要提取特征和标签，请使用以下代码：

```
### 提取特征和标签
dataset = df.values
x = dataset[:,1]
y = dataset[:,0]
```

为了使这个程序足够简单，我不会创建训练集和验证集。此外，我们也不会保留任何数据进行测试。接下来，我们定义模型。

## 定义/训练模型

为了定义模型，我们像之前一样使用 Sequential API：

```
model = tf.keras.Sequential([tf.keras.layers.Dense
(units=1, input_shape=[1])])
```

我们的网络模型仅包含一个单层，该层只有一个神经元。该模型的输入是一个一维张量。接下来，我们编译模型：

```
model.compile(optimizer = 'sgd',
loss = 'mean_squared_error')
```

我们使用随机梯度下降作为优化器，均方误差作为损失函数。

模型使用通常的 `fit` 方法进行训练：

```
model.fit(X, y, epochs = 15)
```

请注意，在这个简单的例子中，我没有捕获用于评估模型性能的误差指标。

### 预测

现在，模型已经训练完成，我们可以进行一些预测了。假设你想知道一个 GPA 为 5.0 的学生在 SAT 中能得多少分。你可以使用模型的 `predict` 方法并打印其结果，如下所示：

```
result = model.predict([5.0])
print("Expected SAT score for GPA 5.0: {:.0f}"
.format(result[0][0]))
```

要找出一个 GPA 为 3.2 的学生在 SAT 中能得多少分，你可以使用以下代码：

```
result = model.predict([3.2])
print("Expected SAT score for GPA 3.2: {:.0f}"
.format(result[0][0]))
```

请注意，我们没有费心去验证这些预测的准确性。但这向我们证明了一点：神经网络可以用来为即使是最简单的线性回归问题创建机器学习模型。

接下来，我将讨论回归分析中一个更实际的问题——多重共线性。

