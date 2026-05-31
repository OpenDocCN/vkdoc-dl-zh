# 10. XAI 模型的反事实解释

本章将解释如何使用假设分析工具（WIT）来解释 AI 模型中的反事实定义，例如基于机器学习的回归模型、分类模型和多类分类模型。作为数据科学家，你不仅要开发机器学习模型，还要确保你的模型没有偏见，并且对于它未来预测的新观测值所做的决策是公平的。探究这些决策并验证算法公平性非常重要。谷歌开发了假设分析工具来解决机器学习模型中的模型公平性问题。你将看到 WIT 在三种机器学习模型中的实现：基于回归任务的机器学习、二项分类模型以及多项分类模型。

## 什么是 CFE？

反事实解释（CFE）是机器学习模型预测过程中的因果关系。我将通过一个例子来说明预测或分类。反事实解释是一种创建假设场景并在这些假设场景下生成预测的方法。这对于业务用户、非数据科学家以及没有预测建模背景的普通人来说是可以理解的。有时我们发现输入和输出之间的关系并非因果关系，但我们仍然希望建立因果关系来生成预测。反事实解释是针对预测的局部解释，这意味着它们会为单个预测生成相关的解释。

## CFE 的实现

反事实解释可以针对回归和分类任务生成。你将使用基于 Python 的 Alibi 库。Alibi 适用于 Python 3.6 及以上版本。可以使用以下命令进行安装：

```
!pip install alibi
!pip install alibi[ray]
import alibi
alibi.__version__
```

安装完成后，请使用以下代码行检查版本，以确保安装完整：`alibi.__version__`



# 使用 Alibi 生成反事实解释

可以使用 Alibi 生成模型预测的反事实解释。安装后，您可以按照类似的流程，使用 Alibi 函数进行初始化、拟合和预测。Alibi 库需要一个训练好的模型和一个需要生成预测的新数据集。例如，您可以使用极端梯度提升模型来获取训练好的模型对象，该对象可以进一步与 Alibi 库一起使用，以生成反事实解释。以下脚本将安装 `xgboost` 模型和 `witwidget` 库。如果 `xgboost` 模型报错，可以使用下面给出的 `conda` 命令来安装基于 Python 的 `xgboost` 库。

```
import pandas as pd
!pip install xgboost
import pandas as pd
import xgboost as xgb
import numpy as np
import collections
import witwidget
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.utils import shuffle
from witwidget.notebook.visualization import WitWidget, WitConfigBuilder
# 如果 xgboost 报错
# 运行
# conda install -c conda-forge xgboost
data = pd.read_csv('diabetes.csv')
data.head()
data.columns
from sklearn.model_selection import train_test_split
```

您正在使用 `diabetes.csv` 数据集。使用表 10-1 中所示的特征，您必须预测某人是否会患糖尿病。这些特征如表 10-1 所示。

**表 10-1** 特征数据字典

| 特征名称 | 描述 |
| --- | --- |
| **怀孕次数** | 怀孕次数 |
| **血糖** | 血糖水平 |
| **血压** | 舒张压水平 |
| **皮肤厚度** | 以毫米为单位的皮肤厚度 |
| **胰岛素** | 施用的胰岛素剂量 |
| **BMI** | 受试者的身体质量指数 |
| **糖尿病遗传函数** | 受试者的糖尿病遗传函数 |
| **年龄** | 受试者的年龄 |
| **结果** | 0-非糖尿病，1-糖尿病 |

```
# 将数据拆分为训练集/测试集
y = data['Outcome']
x = data[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']]
x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=1234)
```

总数据被拆分为训练对象和测试对象，即 `x_train` 和 `y_train`，以开发一个训练好的模型对象。在以下脚本中，您将使用极端梯度提升分类器通过逻辑回归模型进行训练。极端梯度提升分类器有许多超参数，但您将使用默认的超参数选择。超参数是帮助微调模型以获得最佳模型版本的参数。

```
# 训练模型，这需要几分钟时间运行
bst = xgb.XGBClassifier(
objective='reg:logistic'
)
bst.fit(x_train, y_train)
# 在测试集上获取预测并打印准确率分数
y_pred = bst.predict(x_test)
acc = accuracy_score(np.array(y_test), y_pred)
print(acc, '\n')
# 打印混淆矩阵
print('Confusion matrix:')
cm = confusion_matrix(y_test, y_pred)
print(cm)
```

训练好的模型对象存储为 `bst`。使用同一个对象，您可以生成模型的准确率。基础准确率为 72.9%。

最佳模型准确率为 72.9%，您可以存储或保存最佳模型，并将其上传到 Alibi 以生成预测，并用于填充反事实解释。

```
# 保存模型以便部署
bst.save_model('model.bst')
bst
x_train.head()
bst.predict(x_train.head())
y_train.head()
x_train.iloc[3] = x_train.iloc[1]
x_train.head()
bst.predict(x_train.head())
```

对于记录编号为 8 的人，模型预测为糖尿病，实际结果也是糖尿病。行号 651，即训练数据集中的第三条记录，没有糖尿病，如 `y_train` 记录编号 651 所示。

如果您稍微更改行号 651 的值，该记录的预测会发生变化（表 10-2）。在以下脚本中，您将进行更改。

**表 10-2** 特征值变化影响预测

| 特征名称 | 旧值 _651 | 新值 _651 |
| --- | --- | --- |
| **怀孕次数** | 1 | 5 |
| **血糖** | 117 | 97 |
| **血压** | 60 | 76 |
| **皮肤厚度** | 23 | 27 |
| **胰岛素** | 106 | 0 |
| **BMI** | 33.8 | 35.6 |
| **糖尿病遗传函数** | 0.446 | 0.378 |
| **年龄** | 27 | 52 |

以下脚本显示了特征在预测一个人是否患有糖尿病时的重要性。血糖水平被认为是最重要的特征，其次是 BMI 水平、年龄和胰岛素水平。

```
bst.predict(x_train.head())
x_train.iloc[3] = [1, 117, 60, 23, 106, 33.8, 0.466, 27]
x_train.head()
result = pd.DataFrame()
result['features_importance'] = bst.feature_importances_
result['feature_names'] = x_train.columns
result.sort_values(by=['features_importance'],ascending=False)
```

您可以对特征值进行小幅更改，并观察结果变量的变化。小幅更改后，预测没有改变。

```
# 我们将对值进行小幅更改
x_train.iloc[2] = [2, 146, 70, 38, 360, 28.0, 0.337, 29]
x_train.iloc[2]
x_train.columns
bst.predict(x_train.head())
y_train.head()
# 如果我们更改重要特征，例如血糖和年龄
x_train.iloc[2] = [3, 117, 76, 36, 245, 31.6, 0.851, 27]
x_train.iloc[2]
x_train.columns
bst.predict(x_train.head())
y_train.head()
```

经过另一轮输入特征值的更改，糖尿病类别的预测没有改变。当您不断更改特征值时，您会发现某个点，在该点上特征值的微小变化可能导致类别预测的变化。手动重复所有这些场景既繁琐又困难。即使更改最重要特征的值，对目标类别预测也没有影响。因此，需要通过某种框架来识别此类边界情况。Alibi 提供了这个框架。

```
# 如果我们更改重要特征，例如血糖、BMI 和年龄
x_train.iloc[2] = [3, 117, 76, 36, 245, 30.6, 0.851, 29]
x_train.iloc[2]
x_train.columns
bst.predict(x_train.head())
y_train.head()
import tensorflow as tf
tf.get_logger().setLevel(40) # 抑制弃用消息
tf.compat.v1.disable_v2_behavior() # 禁用 TF2 行为，因为 alibi 代码仍然依赖 TF1 结构
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.utils import to_categorical
import matplotlib
%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import os
from alibi.explainers import CounterFactualProto
print('TF version: ', tf.__version__)
print('Eager execution enabled: ', tf.executing_eagerly()) # False
```

您可以使用以 TensorFlow 为后端的 Keras 层来训练神经网络模型对象。Alibi 提供了解释器类，该类有一个用于生成反事实原型结果的模块。以下脚本创建了一个神经网络模型类作为用户自定义函数。这个 `nn_model()` 函数接收输入并应用修正线性单元激活函数。然后它创建另一个包含 40 个神经元的隐藏层，应用相同的激活函数，并使用随机梯度下降作为优化器，分类交叉熵作为损失函数。

```
x_train.shape, y_train.shape, x_test.shape, y_test.shape
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
np.random.seed(42)
tf.random.set_seed(42)
def nn_model():
x_in = Input(shape=(8,))
x = Dense(40, activation='relu')(x_in)
x = Dense(40, activation='relu')(x)
x_out = Dense(2, activation='softmax')(x)
nn = Model(inputs=x_in, outputs=x_out)
nn.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
return nn
```



以下脚本展示了神经网络模型的摘要。模型训练使用了`64`的批量大小和`500`个周期。模型训练完成后，会保存为`h5`格式。`h5`格式是一种可移植的模型对象，用于在不同平台间加载和传输模型。

```
nn = nn_model()
nn.summary()
nn.fit(x_train, y_train, batch_size=64, epochs=500, verbose=0)
nn.save('nn_diabetes.h5', save_format='h5')
```

以下脚本加载训练好的模型对象，使用测试数据集，并得到`71.8%`的测试准确率：

```
nn = load_model('nn_diabetes.h5')
score = nn.evaluate(x_test, y_test, verbose=0)
print('Test accuracy: ', score[1])
```

反事实的生成由最近类原型引导。你从测试集中取出一行包含八个特征的数据，并以反事实原型生成函数可接受的方式进行重塑。

```
#Generate counterfactual guided by the nearest class prototype
X = np.array(x_test)[1].reshape((1,) + np.array(x_test)[1].shape)
shape = X.shape
shape
```

以下脚本接收神经网络模型对象。通过 lambda 函数，你创建了一个预测函数。然后你使用反事实原型函数（表 10-3）。

**表 10-3** 反事实原型超参数

| 参数 | 说明 |
| --- | --- |
| `predict_fn` | Keras 或 TensorFlow 模型或任何其他返回类别概率的模型的预测函数 |
| `Shape` | 输入数据的形状，以批量大小开头 |
| `use_kdtree` | 如果没有编码器，是否使用 k-d 树来计算原型损失项 |
| `Theta` | 原型搜索损失项的常数 |
| `max_iterations` | 寻找反事实的最大迭代次数 |
| `feature_range` | 包含最小和最大范围的元组，用于允许扰动实例 |
| `c_init, c_steps` | 缩放攻击损失项的初始值，调整攻击损失项常数缩放的迭代次数 |

```
# define model
nn = load_model('nn_diabetes.h5')
predict_fn = lambda x: nn.predict(x)
# initialize explainer, fit and generate counterfactual
cf = CounterFactualProto(predict_fn, shape, use_kdtree=False, theta=10., max_iterations=1000,
feature_range=(np.array(x_train).min(axis=0), np.array(x_train).max(axis=0)),
c_init=1., c_steps=10)
cf.fit(x_train)
```

上述结果是在反事实拟合过程或训练过程中生成的。以下脚本显示了所有特征中可能的最低特征值：

```
x_train.min(axis=0)
```

`explain`方法接收输入数据并生成表 10-4 中所示的反事实解释。`explain`方法还会生成以下局部解释输出：

**表 10-4** 反事实的解释

| 输出 | 解释 |
| --- | --- |
| `Cf.X` | 反事实实例 |
| `Cf.class` | 反事实的预测类别 |
| `Cf.proba` | 反事实的预测类别概率 |
| `Cf. grads_graph` | 从 TF 计算图计算出的、关于反事实处输入特征的梯度值 |
| `Cf. grads_num` | 关于反事实处输入特征的数值梯度值 |
| `orig_class` | 原始实例的预测类别 |
| `orig_proba` | 原始实例的预测类别概率 |
| `All` | 一个字典，键为迭代次数，值为每次迭代中找到的反事实列表 |

```
explanation = cf.explain(X)
print(explanation)
```

```
print(f'Original prediction: {explanation.orig_class}')
print('Counterfactual prediction: {}'.format(explanation.cf['class']))
Original prediction: 0
Counterfactual prediction: 1
```

在上述脚本中，`all`键提供了寻找反事实值过程中的迭代结果。在迭代`3`时，你得到了反事实值。其余九次迭代结果中没有反事实值。原始预测为无糖尿病，而基于反事实信息的预测为糖尿病。

```
explanation = cf.explain(X)
explanation.all
explanation.cf
{'X': array([[2.5628092e+00, 1.7726180e+02, 7.6516624e+01, 2.4388657e+01,
7.2130630e+01, 2.6932917e+01, 7.8000002e-02, 2.2944651e+01]],
dtype=float32),
'class': 1,
'proba': array([[0.4956944, 0.5043056]], dtype=float32),
'grads_graph': array([[ -0.7179985 ,  -5.005951  ,  23.374054  ,  -0.96334076,
3.8287811 , -13.371899  ,  -0.38599998,  -5.9150696 ]],
dtype=float32),
'grads_num': array([[  18.74566078,   38.92183304, -117.42114276,   24.19948392, -35.82239151,   62.04843149,   93.54948252,   25.92801861]])}
explanation.orig_class
explanation.orig_proba
X
```



### 回归任务的反事实解释

在回归场景中，可以通过使用波士顿房价数据集查看神经网络模型来理解反事实解释。回归是指目标列为连续变量，并且可以使用连续变量和分类变量等混合变量的任务。波士顿房价数据集是学习回归的常用数据集，因此我选择它来展示反事实解释，因为用户对此数据集更为熟悉。

```
import tensorflow as tf
tf.get_logger().setLevel(40) # suppress deprecation messages
tf.compat.v1.disable_v2_behavior() # disable TF2 behaviour as alibi code still relies on TF1 constructs
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.utils import to_categorical
import matplotlib
%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.datasets import load_boston
from alibi.explainers import CounterFactualProto
print('TF version: ', tf.__version__)
print('Eager execution enabled: ', tf.executing_eagerly()) # False
boston = load_boston()
data = boston.data
target = boston.target
feature_names = boston.feature_names
y = np.zeros((target.shape[0],))
y[np.where(target > np.median(target))[0]] = 1
data = np.delete(data, 3, 1)
feature_names = np.delete(feature_names, 3)
mu = data.mean(axis=0)
sigma = data.std(axis=0)
data = (data - mu) / sigma
idx = 475
x_train,y_train = data[:idx,:], y[:idx]
x_test, y_test = data[idx:,:], y[idx:]
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
np.random.seed(42)
tf.random.set_seed(42)
def nn_model():
x_in = Input(shape=(12,))
x = Dense(40, activation='relu')(x_in)
x = Dense(40, activation='relu')(x)
x_out = Dense(2, activation='softmax')(x)
nn = Model(inputs=x_in, outputs=x_out)
nn.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
return nn
```

上述脚本是一个用于回归的神经网络模型。下面的脚本是神经网络架构摘要：

```
nn = nn_model()
nn.summary()
nn.fit(x_train, y_train, batch_size=64, epochs=500, verbose=0)
nn.save('nn_boston.h5', save_format='h5')
```

训练好的模型对象 `nn_boston.h5` 被保存下来用于生成反事实解释。

```
nn = load_model('nn_boston.h5')
score = nn.evaluate(x_test, y_test, verbose=0)
print('Test accuracy: ', score[1])
X = x_test[1].reshape((1,) + x_test[1].shape)
shape = X.shape
# define model
nn = load_model('nn_boston.h5')
# initialize explainer, fit and generate counterfactual
cf = CounterFactualProto(nn, shape, use_kdtree=True, theta=10., max_iterations=1000,
feature_range=(x_train.min(axis=0), x_train.max(axis=0)),
c_init=1., c_steps=10)
cf.fit(x_train)
explanation = cf.explain(X)
```

从反事实解释得出的结果可以如表 10-4 所述进行解读。1940 年以前建造的自有住房比例为 93.6%，低层人口比例为 18.68%。为了提高房价，1940 年以前建造的自有住房比例应减少 5.95%，低层人口比例应减少 4.85%。

```
print(f'Original prediction: {explanation.orig_class}')
print('Counterfactual prediction: {}'.format(explanation.cf['class']))
Original prediction: 0
Counterfactual prediction: 1
orig = X * sigma + mu
counterfactual = explanation.cf['X'] * sigma + mu
delta = counterfactual - orig
for i, f in enumerate(feature_names):
if np.abs(delta[0][i]) > 1e-4:
print('{}: {}'.format(f, delta[0][i]))
print('% owner-occupied units built prior to 1940: {}'.format(orig[0][5]))
print('% lower status of the population: {}'.format(orig[0][11]))
% owner-occupied units built prior to 1940: 93.6
% lower status of the population: 18.68
```

## 结论

在本章中，你探讨了与回归问题和分类问题相关的反事实解释。反事实的目标是在训练数据中找到与标记类别产生不同预测的相似数据点。反事实解释的推理逻辑是：如果两个数据点非常相似，那么两者应该具有相似的预测或相似的结果。目标类别的输出不应存在差异。在糖尿病预测用例中，如果两个人具有相似的特征，那么要么两人都没有糖尿病，要么两人都有糖尿病。不应出现一人有糖尿病而另一人没有的情况。存在不同的反事实信息实际上会从最终用户的角度造成混淆，这将导致对 AI 模型缺乏信任。为了建立对 AI 模型的信任，生成反事实解释至关重要。

