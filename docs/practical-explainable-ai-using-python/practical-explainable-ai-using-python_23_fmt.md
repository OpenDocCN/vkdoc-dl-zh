# 12. 通过识别预测不变性实现模型无关解释

从数学领域借用的不变性术语解释：如果对解释变量进行干预，机器学习模型生成的预测保持不变，前提是该模型是通过正式的因果关系生成的。因果机器学习模型与非因果机器学习模型之间存在差异。因果模型展示了响应变量与解释变量之间真实的因果关系。非因果模型展示的是偶然关系；预测结果会多次改变。理解因果模型与非因果模型之间差异的唯一途径是通过解释。在本章中，你将使用一个名为 `Alibi` 的 Python 库来生成模型无关的解释。

## 什么是模型无关？

术语 `模型无关` 意味着模型独立性。对机器学习模型的解释，无论模型类型如何都有效，被称为模型无关。当我们解释机器学习模型的内在行为时，我们通常不会假设该行为具有任何由机器学习模型本身展示的底层结构。在本章中，你将使用模型无关的解释来描述预测不变性。从人类理解的角度来看，机器学习模型如何工作一直是一个挑战。人们总是对机器学习模型的行为感到好奇并提出问题。为了理解模型无关的行为，人类总是认为他们可以对模型行为做出预测。

## 什么是锚点？

复杂模型通常被视为黑盒模型，因为很难理解预测为何做出。锚点概念无非是找出规则。这里的规则指的是高精度规则，这些规则在考虑局部和全局因素的情况下解释模型行为。换句话说，锚点是 if/then/else 条件，无论其他特征的值如何，这些条件都能锚定预测。

锚点算法建立在模型无关解释之上。它有三个核心功能：

- **覆盖率**：这表示改变框架以预测机器学习模型行为的频率。
- **精确度**：这表示人类预测模型行为的准确程度。
- **工作量**：这表示解释模型行为或理解模型行为生成的预测所需的前期工作量。这两种情况都难以解释。

### 使用 Alibi 进行锚点解释

为了对黑盒模型生成锚点解释，你将使用 Python 库 `Alibi`。以下脚本展示了解释该概念所需的导入语句。你将使用一个随机森林分类器，它本身也是一个黑盒模型，因为其中包含许多生成预测的决策树，这些预测被整合后产生最终输出。

```python
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from alibi.explainers import AnchorTabular
from alibi.datasets import fetch_adult
```

`Alibi` 库包含多个数据集，用于生成可用于解释概念的示例。这里你将使用 `adult` 数据集；它也被称为来自 `UCI` 机器学习库的人口普查收入数据集。这是一个关于二分类模型的问题。有几个特征可用于预测收入类别，即收入是高于 5 万美元还是低于 5 万美元。

```python
adult = fetch_adult()
adult.keys()
```

上述脚本将数据加载到 `Jupyter` 环境中。这些键是 `dict_keys(['data', 'target', 'feature_names', 'target_names', 'category_map'])`。

```python
data = adult.data
target = adult.target
feature_names = adult.feature_names
category_map = adult.category_map
```

`category_map` 函数的作用是映射数据中所有分类列或字符串列。

```python
from alibi.utils.data import gen_category_map
```

分类映射只是一个虚拟变量映射，因为你不能直接使用分类变量或字符串变量进行计算。

```python
np.random.seed(0)
data_perm = np.random.permutation(np.c_[data, target])
data = data_perm[:,:-1]
target = data_perm[:,-1]
```

在训练模型之前，你对数据集中的数据进行随机化处理，并使用 `33,000` 条记录进行模型训练，使用 `2,560` 条记录进行模型测试或验证。

```python
idx = 30000
X_train,Y_train = data[:idx,:], target[:idx]
X_test, Y_test = data[idx+1:,:], target[idx+1:]
X_train.shape,Y_train.shape
X_test.shape, Y_test.shape
ordinal_features = [x for x in range(len(feature_names)) if x not in list(category_map.keys())]
ordinal_features
```

选择好有序特征后，你可以通过管道触发有序特征转换器，如下所示：

```python
ordinal_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')),
('scaler', StandardScaler())])
categorical_features = list(category_map.keys())
categorical_features
```

有序特征是特征编号 `0`、`8`、`9` 和 `10`，分类特征是 `1`、`2`、`3`、`4`、`5`、`6`、`7` 和 `11`。对于有序特征，你应用标准缩放器。如果存在缺失值，则使用中位数进行填充。

```python
categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')),
('onehot', OneHotEncoder(handle_unknown='ignore'))])
categorical_transformer
preprocessor = ColumnTransformer(transformers=[('num', ordinal_transformer, ordinal_features),
('cat', categorical_transformer, categorical_features)])
```

对于分类特征，如果存在缺失值，你应用中位数填充技术，并对分类变量转换执行独热编码。如果分类变量值未知，则忽略该值。

```python
Preprocessor
ColumnTransformer(transformers=[('num',
Pipeline(steps=[('imputer',
SimpleImputer(strategy='median')),
('scaler', StandardScaler())]),
[0, 8, 9, 10]),
('cat',
Pipeline(steps=[('imputer',
SimpleImputer(strategy='median')),
('onehot',
OneHotEncoder(handle_unknown='ignore'))]),
[1, 2, 3, 4, 5, 6, 7, 11])])
```

预处理模块现在已准备好进行模型训练。`preprocessor.fit` 函数转换数据集以用于模型训练。用于分类的随机森林模型使用 `50` 棵树作为估计器数量进行初始化。初始化后的对象存储在 `clf` 中。`clf.fit` 函数训练模型。

```python
preprocessor.fit(X_train)
np.random.seed(0)
clf = RandomForestClassifier(n_estimators=50)
clf.fit(preprocessor.transform(X_train), Y_train)
predict_fn = lambda x: clf.predict(preprocessor.transform(x))
print('Train accuracy: ', accuracy_score(Y_train, predict_fn(X_train)))
Train accuracy:  0.9655333333333334
print('Test accuracy: ', accuracy_score(Y_test, predict_fn(X_test)))
Test accuracy:  0.855859375
```

下一步，你将使用锚点解释器并将其拟合到表格数据上。表格锚点具有 `表 12-1` 所示的参数要求。

**`表 12-1`** 锚点表格参数说明

| 参数 | 说明 |
| --- | --- |
| `Predictor` | 一个模型对象，具有预测函数，该函数接收输入数据并生成输出数据 |
| `Feature_names` | 特征列表 |
| `Categorical_names` | 一个字典，其中键是特征列，值是该特征的类别 |
| `OHE` | 分类变量是否经过独热编码。如果不是独热编码，则假定它们具有有序编码。 |
| `Seed` | 用于结果可复现性 |

```python
explainer = AnchorTabular(predict_fn, feature_names, categorical_names=category_map, seed=12345)
explainer
explainer.explanations
explainer.fit(X_train, disc_perc=[25, 50, 75])
AnchorTabular(meta={
'name': 'AnchorTabular',
'type': ['blackbox'],
'explanations': ['local'],
'params': {'seed': 1, 'disc_perc': [25, 50, 75]}}
)
idx = 0
class_names = adult.target_names
print('Prediction: ', class_names[explainer.predictor(X_test[idx].reshape(1, -1))[0]])
Prediction:  50K
```

`explainer.explain` 对象解释分类器对某个观测值所做的预测。`explain` 函数具有 `表 12-2` 所示的参数。

**`表 12-2`** 解释函数参数值

| 参数 | 说明 |
| --- | --- |
| `Xtest` | 需要被解释的新数据 |
| `Threshold` | 最小精度阈值 |
| `batch_size` | 用于采样的批次大小 |
| `coverage_samples` | 用于从结果搜索中估计覆盖率的样本数量 |
| `beam_size` | 在构建新锚点的每一步中扩展的锚点数量 |
| `stop_on_first` | 如果为 `True`，束搜索算法将返回第一个满足概率约束的锚点。 |
| `max_anchor_size` | 结果中特征的最大数量 |

```python
explanation = explainer.explain(X_test[idx], threshold=0.95)
print('Anchor: %s' % (' AND '.join(explanation.anchor)))
Output : [Anchor: Capital Loss > 0.00 AND Relationship = Husband AND Marital Status = Married AND Age > 37.00 AND Race = White AND Sex = Male AND Country = United-States
]
print('Precision: %.2f' % explanation.precision)
print('Coverage: %.2f' % explanation.coverage)
```

在上面的例子中，模型解释的精度仅为 `72%`，覆盖率仅为 `2%`。可以得出结论，覆盖率和精度较低是由于数据集不平衡造成的。收入低于 5 万美元的人数多于收入高于 5 万美元的人数。你可以将类似的锚点概念应用于文本分类数据集。在此示例中，你将使用情感分析数据集。

### 文本分类的锚点解释

锚点的 `text` 函数将帮助你获取文本分类数据集的锚点解释。它可用于垃圾邮件识别或情感分析。可用于生成锚点文本的可调参数如 `表 12-3` 所示。

**`表 12-3`** 锚点文本参数说明

| 参数 | 说明 |
| --- | --- |
| `Predictor` | 通常是一个能够生成预测的已训练模型对象 |
| `sampling_strategy` | 扰动分布方法，其中 `unknown` 用 `UNK` 替换单词；`similarity` 根据与语料库嵌入的相似度分数进行采样；`language_model` 根据语言模型的输出分布进行采样 |
| `NLP` | 当采样方法为 `unknown` 或 `similarity` 时的 `spaCy` 对象 |
| `language_model` | `Transformers` 掩码语言模型 |

你将使用的采样策略是 `unknown` 和 `similarity`，因为 `transformers` 方法超出了本书的讨论范围。

```python
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
#import spacy
from alibi.explainers import AnchorText
from alibi.datasets import fetch_movie_sentiment
#from alibi.utils.download import spacy_model
```

上述脚本从库中导入了必要的库和函数。

```python
movies = fetch_movie_sentiment()
movies.keys()
```

上述脚本加载了用于情感分类的带标签数据。

```python
data = movies.data
labels = movies.target
target_names = movies.target_names
```

这包括电影数据、目标列（标记为正面和负面）以及特征名称（语料库中唯一的标记集合）。

```python
train, test, train_labels, test_labels = train_test_split(data, labels, test_size=.2, random_state=42)
train, test, train_labels, test_labels
```

上述脚本将数据划分为训练集和测试集，以下为训练集和验证集。

```python
train, val, train_labels, val_labels = train_test_split(train, train_labels, test_size=.1, random_state=42)
train, val, train_labels, val_labels
train_labels = np.array(train_labels)
train_labels
test_labels = np.array(test_labels)
test_labels
val_labels = np.array(val_labels)
val_labels
vectorizer = CountVectorizer(min_df=1)
vectorizer.fit(train)
```

`CountVectorizer` 函数创建一个文档-词项矩阵，其中文档表示为行，词项表示为特征。

```python
np.random.seed(0)
clf = LogisticRegression(solver='liblinear')
clf.fit(vectorizer.transform(train), train_labels)
```

由于这是一个二分类问题，因此使用逻辑回归，求解器为 `liblinear`。以下 `predict` 函数接收已训练的模型对象，并在给定测试数据集的情况下进行预测：

```python
predict_fn = lambda x: clf.predict(vectorizer.transform(x))
preds_train = predict_fn(train)
preds_train
preds_val = predict_fn(val)
preds_val
preds_test = predict_fn(test)
preds_test
print('Train accuracy', accuracy_score(train_labels, preds_train))
print('Validation accuracy', accuracy_score(val_labels, preds_val))
print('Test accuracy', accuracy_score(test_labels, preds_test))
```

训练集、测试集和验证集的准确率保持一致。

要生成锚点解释，你需要安装 `spaCy`。你还需要安装 `en_core_web_md` 模块，可以通过在命令行中使用以下脚本来完成：

```bash
pip install spacy
python -m spacy download en_core_web_md
```

# 锚点解释器示例

## 锚点文本

```python
python -m spacy download en_core_web_md
import spacy
model = 'en_core_web_md'
#spacy_model(model=model)
nlp = spacy.load(model)
```

`AnchorText` 函数需要 `spacy` 模型对象，因此你需要从 `spacy` 初始化 `nlp` 对象。

```python
class_names = movies.target_names
# select instance to be explained
text = data[40]
print("* Text: %s" % text)
# compute class prediction
pred = class_names[predict_fn([text])[0]]
alternative =  class_names[1 - predict_fn([text])[0]]
print("* Prediction: %s" % pred)
explainer = AnchorText(
predictor=predict_fn,
nlp=nlp
)
explanation = explainer.explain(text, threshold=0.95)
explainer
explanation
print('Anchor: %s' % (' AND '.join(explanation.anchor)))
print('Precision: %.2f' % explanation.precision)
```

```
Anchor: watchable AND bleak AND honest
Precision: 0.99
```

```python
print('\nExamples where anchor applies and model predicts %s:' % pred)
print('\n'.join([x for x in explanation.raw['examples'][-1]['covered_true']]))
print('\nExamples where anchor applies and model predicts %s:' % alternative)
print('\n'.join([x for x in explanation.raw['examples'][-1]['covered_false']]))
```

锚点适用且模型预测为负面的示例：

the effort is UNK UNK the results UNK honest , but the UNK UNK so bleak UNK UNK UNK hardly watchable .

### 图像分类的锚点图像

与锚点文本类似，针对图像分类数据集，你也有锚点图像模型解释。这里将使用 Fashion MNIST 数据集。`表 12-4` 展示了需要解释的锚点图像参数。

**`表 12-4` 锚点图像参数说明**

| 参数 | 说明 |
| --- | --- |
| `Predictor` | 图像分类模型对象 |
| `image_shape` | 待解释图像的形状 |
| `segmentation_fn` | 内置分割函数字符串之一：`'felzenszwalb'`、`'slic'`、`'quickshift'`，或自定义分割函数（可调用），该函数返回一个带有每个超像素标签的图像掩码 |
| `segmentation_kwargs` | 内置分割函数的关键字参数 |
| `images_background` | 用于叠加超像素的背景图像 |
| `Seed` | 用于重现结果 |

让我们看看以下生成锚点图像的脚本。

```python
import matplotlib
%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D, Input
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from alibi.explainers import AnchorImage
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
print('x_train shape:', x_train.shape, 'y_train shape:', y_train.shape)
```

训练数据集包含 60,000 个样本图像，测试数据集包含 10,000 个样本图像。你将开发一个卷积神经网络模型（CNN）来预测图像类别并解释所涉及的锚点。目标类别有 9 个标签，参见 `表 12-5`。

**`表 12-5` 目标类别描述**

| 类别 | 描述 |
| --- | --- |
| 0 | T 恤/上衣 |
| 1 | 裤子 |
| 2 | 套头衫 |
| 3 | 连衣裙 |
| 4 | 外套 |
| 5 | 凉鞋 |
| 6 | 衬衫 |
| 7 | 运动鞋 |
| 8 | 包 |
| 9 | 踝靴 |

```python
idx = 57
plt.imshow(x_train[idx]);
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
```

特征归一化通过将每个像素值除以最大像素值（255）来完成。

```python
x_train = np.reshape(x_train, x_train.shape + (1,))
x_test = np.reshape(x_test, x_test.shape + (1,))
print('x_train shape:', x_train.shape, 'x_test shape:', x_test.shape)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print('y_train shape:', y_train.shape, 'y_test shape:', y_test.shape)
```

CNN 模型通常用于预测图像类别。它包含一个二维卷积滤波器以更好地识别像素值，随后是一个最大池化层（有时是平均池化，有时是最大池化），以准备抽象层。丢弃层旨在控制模型预测中可能出现的任何过拟合问题。

```python
def model():
    x_in = Input(shape=(28, 28, 1))
    x = Conv2D(filters=64, kernel_size=2, padding='same', activation='relu')(x_in)
    x = MaxPooling2D(pool_size=2)(x)
    x = Dropout(0.3)(x)
    x = Conv2D(filters=32, kernel_size=2, padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=2)(x)
    x = Dropout(0.3)(x)
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    x_out = Dense(10, activation='softmax')(x)
    cnn = Model(inputs=x_in, outputs=x_out)
    cnn.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return cnn
cnn = model()
cnn.summary()
cnn.fit(x_train, y_train, batch_size=64, epochs=3)
# 在测试集上评估模型
score = cnn.evaluate(x_test, y_test, verbose=0)
print('Test accuracy: ', score[1])
```

在图像分类示例中，超像素提供了锚点解释。因此，为了识别核心特征，你可以使用锚点图像。

```python
def superpixel(image, size=(4, 7)):
    segments = np.zeros([image.shape[0], image.shape[1]])
    row_idx, col_idx = np.where(segments == 0)
    for i, j in zip(row_idx, col_idx):
        segments[i, j] = int((image.shape[1]/size[1]) * (i//size[0]) + j//size[1])
    return segments
segments = superpixel(x_train[idx])
plt.imshow(segments);
predict_fn = lambda x: cnn.predict(x)
image_shape = x_train[idx].shape
explainer = AnchorImage(predict_fn, image_shape, segmentation_fn=superpixel)
i = 11
image = x_test[i]
plt.imshow(image[:,:,0]);
cnn.predict(image.reshape(1, 28, 28, 1)).argmax()
explanation = explainer.explain(image, threshold=.95, p_sample=.8, seed=0)
plt.imshow(explanation.anchor[:,:,0]);
explanation.meta
explanation.params
```

```
{'custom_segmentation': True,
'segmentation_kwargs': None,
'p_sample': 0.8,
'seed': None,
'image_shape': (28, 28, 1),
'segmentation_fn': 'custom',
'threshold': 0.95,
'delta': 0.1,
'tau': 0.15,
'batch_size': 100,
'coverage_samples': 10000,
'beam_size': 1,
'stop_on_first': False,
'max_anchor_size': None,
'min_samples_start': 100,
'n_covered_ex': 10,
'binary_cache_size': 10000,
'cache_margin': 1000,
'verbose': False,
'verbose_every': 1,
'kwargs': {'seed': 0}}
```

### 结论

在本章中，你了解了使用成人数据集的表格数据示例、用于情感分析的 IMDB 电影评论数据集，以及用于图像分类示例的 Fashion MNIST 数据集。锚点函数对于表格数据、文本数据和图像数据而言差异很大。在本章中，你通过示例学习了如何解释图像分类与预测、文本分类与预测，以及使用来自 Alibi 的新库解释器进行图像分类。