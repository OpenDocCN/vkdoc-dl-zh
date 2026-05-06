# 决策树模型的数据准备

以下脚本展示了准备决策树模型所需的库。同时，您还需要以 CSV 格式导入开发模型所需的数据。

```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn import tree, metrics, model_selection, preprocessing
from IPython.display import Image, display
from sklearn.metrics import confusion_matrix, classification_report
df_train = pd.read_csv('ChurnData.csv')
del df_train['Unnamed: 0']
df_train.shape
df_train.head()
```

数据中存在的额外列已被删除。`shape` 函数提供了数据集中行数和列数的信息。此外，`head` 函数提供了数据框中的前五条记录。

```
from sklearn.preprocessing import LabelEncoder
tras = LabelEncoder()
df_train['area_code_tr'] = tras.fit_transform(df_train['area_code'])
df_train.columns
del df_train['area_code']
df_train.columns
```

数据中存在一些字符串列，例如 `area_code`。需要使用标签编码器将这些数据转换为数字。这是模型训练所必需的，因为您不能直接使用字符串变量。

```
df_train['target_churn_dum'] = pd.get_dummies(df_train.churn,prefix='churn',drop_first=True)
df_train.columns
del df_train['international_plan']
del df_train['voice_mail_plan']
del df_train['churn']
df_train.info()
df_train.columns
```

下一步，您将把数据集拆分为训练集和测试集。训练集用于开发模型，测试集用于生成推断或预测。

```
from sklearn.model_selection import train_test_split
df_train.columns
X = df_train[['account_length', 'number_vmail_messages', 'total_day_minutes',
'total_day_calls', 'total_day_charge', 'total_eve_minutes',
'total_eve_calls', 'total_eve_charge', 'total_night_minutes',
'total_night_calls', 'total_night_charge', 'total_intl_minutes',
'total_intl_calls', 'total_intl_charge',
'number_customer_service_calls', 'area_code_tr']]
Y = df_train['target_churn_dum']
xtrain,xtest,ytrain,ytest=train_test_split(X,Y,test_size=0.20,stratify=Y)
tree.DecisionTreeClassifier() # 基础树模型
# 决策树分类器的默认超参数
class_weight=None,
criterion='gini',
max_depth=None,
max_features=None,
max_leaf_nodes=None,
min_impurity_decrease=0.0,
min_impurity_split=None,
min_samples_leaf=1,
min_samples_split=2,
min_weight_fraction_leaf=0.0,
presort=False,
random_state=None,
splitter='best'
dt1 = tree.DecisionTreeClassifier()
dt1.fit(xtrain,ytrain)
print(dt1.score(xtrain,ytrain))
print(dt1.score(xtest,ytest))
```

**表 4-1** 决策树超参数说明

| 参数 | 说明 |
| --- | --- |
| `Class_weight` | 与类别相关的权重 |
| `Criterion` | 衡量分裂质量的函数，如基尼系数和熵 |
| `max_depth` | 树的最大深度 |
| `max_features` | 寻找最佳分裂时需要考虑的特征数量 |
| `max_leaf_nodes` | 以最佳优先方式生成具有 `max_leaf_nodes` 个叶节点的树 |
| `min_samples_leaf` | 叶节点所需的最小样本数 |
| `min_samples_split` | 分裂内部节点所需的最小样本数 |

决策树模型中还有许多其他超参数，但表 4-1 中列出了重要的几个。一些超参数用于控制模型的过拟合，另一些则用于提高模型的准确性。

### 创建模型

下一步是创建模型并验证训练集和测试集之间的准确性。有两个模型，`dt1` 和 `dt2`：一个限制了最大深度，另一个对最大深度没有限制。区别在于 `dt1` 是一个过拟合模型，而 `dt2` 是一个进行了树剪枝且没有过拟合的模型。

```
dt1 = tree.DecisionTreeClassifier()
dt1.fit(xtrain,ytrain)
print(dt1.score(xtrain,ytrain))
print(dt1.score(xtest,ytest))
1.0
0.8590704647676162
dt2 = tree.DecisionTreeClassifier(max_depth=3)
dt2.fit(xtrain,ytrain)
print(dt2.score(xtrain,ytrain))
print(dt2.score(xtest,ytest))
0.9021005251312828
0.8875562218890555
```

无限制的过拟合模型的局限性在于，它会生成一个庞大的决策树，并且用于得出预测的规则会非常多。因此，并非所有规则都是重要的，因为随着树的增长，一些冗余规则可能会参与到树的构建过程中。

```
!pip install pydotplus
!pip install graphviz
import pydotplus
dot_data = tree.export_graphviz(dt1, out_file=None, filled=True, rounded=True,
feature_names=list(xtrain.columns),
class_names=['yes','no'])
graph = pydotplus.graph_from_dot_data(dot_data)
display(Image(graph.create_png()))
```

![../images/506619_1_En_4_Chapter/506619_1_En_4_Fig4_HTML.jpg](img/506619_1_En_4_Fig4_HTML.jpg)

**图 4-4** 使用 GraphViz 可视化默认模型下的决策树

使用所有默认超参数训练的决策树会生成一棵庞大的树，并导致产生多条规则，这不仅难以解释，而且在实际项目场景中也难以实施。如图 4-4 所示的最大可能决策树，是将树的最大深度参数设置为 `None` 的结果，这意味着您要求决策树不断生长分支。

```
dot_data = tree.export_graphviz(dt2, out_file=None, filled=True, rounded=True,
feature_names=list(xtrain.columns),
class_names=['yes','no'])
graph = pydotplus.graph_from_dot_data(dot_data)
display(Image(graph.create_png()))
```

```
tree.plot_tree(dt1)
tree.plot_tree(dt2)
```

![../images/506619_1_En_4_Chapter/506619_1_En_4_Fig6_HTML.jpg](img/506619_1_En_4_Fig6_HTML.jpg)

**图 4-6** 使用 `plot_tree` 方法可视化决策树

![../images/506619_1_En_4_Chapter/506619_1_En_4_Fig5_HTML.jpg](img/506619_1_En_4_Fig5_HTML.jpg)

**图 4-5** 同一决策树模型的剪枝版本

图 4-4 中展示的相同模型在图 4-5 和 4-6 中重新生成。后一张图是使用 sklearn Python 库的内置函数 `plot_tree` 以另一种方式生成的。两者没有区别，只是两个库的不同。如果在生产环境中安装 GraphViz 库或 Pydotplus 库时出现问题，您可以切换到 `plot_tree` 函数。

`dt1` 模型生成的树非常大，以至于任何人都难以浏览和解释结果。在应用树剪枝方法将最大深度参数限制为 3 后，您可以看到一棵小得多的树，只有相关特征参与了树的构建（图 4-7）。

![../images/506619_1_En_4_Chapter/506619_1_En_4_Fig7_HTML.jpg](img/506619_1_En_4_Fig7_HTML.jpg)

**图 4-7** 图 4-5 所示决策树模型的剪枝版本



剪枝后的决策树模型对象`dt2`使用最大深度参数`3`，这意味着在第三层分支之后，树应停止进一步扩展。该版本的模型生成了一棵更小的树，其中包含可作为`if/else`条件使用的稳健规则。这是一种最保守的决策树建模方法，可避免过拟合。

![../images/506619_1_En_4_Chapter/506619_1_En_4_Fig8_HTML.jpg](img/506619_1_En_4_Fig8_HTML.jpg)

**图 4-8** 使用`plot tree`函数生成的剪枝树可视化

图 4-8 所示的决策树版本使用了更少的规则，这使得理解决策如何做出变得更加容易，并且更容易集成到任何生产系统中。

如果你想解释、实现或嵌入由决策树模型生成的规则到其他外部应用程序中，可以通过导出规则文本来实现。

```python
from sklearn.tree import export_text
r = export_text(dt1, feature_names=list(xtrain.columns))
print(r)
```

从`dt1`模型生成了许多规则，这些规则难以解释。让我们看一下`dt2`模型生成的规则。类别`0`表示非流失场景，类别`1`表示可能的流失客户指标。规则文本显示，如果总日间费用小于或等于`44.96`，客户服务电话次数小于或等于`3.5`，并且总日间通话分钟数小于或等于`223.25`，那么这是一个非流失案例。类似地，相反的场景可以被解释为流失场景。你可以打印并查看以下所有规则。

```python
from sklearn.tree import export_text
r = export_text(dt2, feature_names=list(xtrain.columns))
print(r)
|--- number_customer_service_calls   38.27
|   |   |   |--- class: 0
|   |--- total_day_minutes >  254.55
|   |   |--- number_vmail_messages   6.50
|   |   |   |--- class: 0
|--- number_customer_service_calls >  3.50
|   |--- total_day_charge   22.57
|   |   |   |--- class: 0
|   |--- total_day_charge >  27.24
|   |   |--- total_eve_charge   13.22
|   |   |   |--- class: 0
```

来自`dt1`模型的特征重要性如下所示。

```python
list(zip(dt1.feature_importances_, xtrain.columns))
[(0.04051943775304626, 'account_length'),
(0.08298083105277364, 'number_vmail_messages'),
(0.0644144400251063, 'total_day_minutes'),
(0.028172622004021135, 'total_day_calls'),
(0.20486110565087778, 'total_day_charge'),
(0.10259170929879882, 'total_eve_minutes'),
(0.03586253729017199, 'total_eve_calls'),
(0.0673761405897894, 'total_eve_charge'),
(0.0613662104733965, 'total_night_minutes'),
(0.05654698887273517, 'total_night_calls'),
(0.03894924950827072, 'total_night_charge'),
(0.01615654226052593, 'total_intl_minutes'),
(0.039418913794511345, 'total_intl_calls'),
(0.02842685405881307, 'total_intl_charge'),
(0.11203068621155501, 'number_customer_service_calls'),
(0.020325731155606916, 'area_code_tr')]
# 提取定义树的数组
children_left = dt2.tree_.children_left
children_right = dt2.tree_.children_right
children_default = children_right.copy() # 因为 sklearn 不使用缺失值
features = dt2.tree_.feature
thresholds = dt2.tree_.threshold
values = dt2.tree_.value.reshape(dt2.tree_.value.shape[0], 2)
node_sample_weight = dt2.tree_.weighted_n_node_samples
print("     children_left", children_left)
# 注意：负的 children 值表示这是一个叶节点
print("    children_right", children_right)
print("  children_default", children_default)
print("          features", features)
print("        thresholds", thresholds.round(3))
print("            values", values.round(3))
print("node_sample_weight", node_sample_weight)
```

决策树分类器模型有一个名为`tree_`的属性，它允许你获取关于模型对象的详细解释。它以并行数组的形式存储整个二叉树结构。节点`0`是根节点，其余参数在表 4-2 中解释。左子节点或右子节点的 ID 在取负值（例如`-1`）时，表示它是一个叶节点，决策树在此终止。

**表 4-2** 决策树的底层属性提取

| 参数 | 解释 |
| --- | --- |
| `dt2.tree_.node_count` | 树中节点的总数 |
| `tree_.children_left[i]` | 节点`i`的左子节点 ID |
| `tree_.children_right[i]` | 节点`i`的右子节点 ID |
| `tree_.feature[i]` | 用于分割节点`i`的特征 |
| `tree_.threshold[i]` | 节点`i`处的阈值 |
| `tree_.n_node_samples[i]` | 到达节点`i`的训练样本数量 |
| `tree_.impurity[i]` | 节点`i`的不纯度 |
| `tree_.weighted_n_node_samples` | `n_node_samples`是每个节点中实际数据集样本的计数。`weighted_n_node_samples`是相同的，但按`class_weight`和/或`sample_weight`加权。 |

```python
# 定义一个自定义树模型
tree_dict = {
"children_left": children_left,
"children_right": children_right,
"children_default": children_default,
"features": features,
"thresholds": thresholds,
"values": values,
"node_sample_weight": node_sample_weight
}
model = {
"trees": [tree_dict]
}
import shap
explainer = shap.TreeExplainer(model)
# 提供概率作为输出
def model_churn_proba(x):
    return dt2.predict_proba(x)[:,1]
# 提供对数几率作为输出
def model_churn_log_odds(x):
    p = dt2.predict_log_proba(x)
    return p[:,1] - p[:,0]
# 制作一个标准的偏依赖图
sample_ind = 25
fig, ax = shap.partial_dependence_plot(
    "total_day_minutes", model_churn_proba, X, model_expected_value=True,
    feature_expected_value=True, show=False, ice=False
)
```

## 决策树 – SHAP

`SHAP` Python 库可用于解释决策树。`SHAP`库具有有用的函数，可以为模型可解释性提供额外的见解。

```python
import shap
explainer = shap.TreeExplainer(model)
# 提供概率作为输出
def model_churn_proba(x):
    return dt2.predict_proba(x)[:,1]
# 提供对数几率作为输出
def model_churn_log_odds(x):
    p = dt2.predict_log_proba(x)
    return p[:,1] - p[:,0]
# 制作一个标准的偏依赖图
sample_ind = 25
fig, ax = shap.partial_dependence_plot(
    "total_day_minutes", model_churn_proba, X, model_expected_value=True,
    feature_expected_value=True, show=False, ice=False
)
```



