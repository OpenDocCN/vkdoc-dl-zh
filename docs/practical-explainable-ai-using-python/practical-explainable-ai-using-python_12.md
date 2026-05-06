# 排版后的内容

为了解释如何应用 CatBoost 分类器，我们使用 Amazon 数据集。表 5-6 列出了该数据集中的特征。

**表 5-6** Amazon 数据集数据描述

| 特征名 | 描述 |
| --- | --- |
| `ACTION` | 如果资源被批准，`ACTION`为 1；如果资源未被批准，则为 0 |
| `RESOURCE` | 每个资源的 ID |
| `MGR_ID` | 当前员工 ID 记录的管理者的员工 ID；一个员工一次只能有一个管理者 |
| `ROLE_ROLLUP_1` | 公司角色分组类别 ID 1（例如，美国工程部） |
| `ROLE_ROLLUP_2` | 公司角色分组类别 ID 2（例如，美国零售部） |
| `ROLE_DEPTNAME` | 公司角色部门描述（例如，零售部） |
| `ROLE_TITLE` | 公司角色业务头衔描述（例如，高级工程零售经理） |
| `ROLE_FAMILY_DESC` | 公司角色系列扩展描述（例如，零售经理，软件工程） |
| `ROLE_FAMILY` | 公司角色系列描述（例如，零售经理） |
| `ROLE_CODE` | 公司角色代码；该代码对每个角色是唯一的（例如，经理） |

当一名员工开始在软件开发部门工作时，他们需要特定的访问权限和许可才能完成其工作职责。有一个手动流程来授予员工访问权限，但员工在此过程中可能会遇到障碍。没有自动化的机制来授予对必要软件系统的访问权限。此用例的目标是对历史数据进行建模，以确定员工的访问需求。目标列是`ACTION`；如果资源被批准则为 1，如果资源访问被拒绝则为 0。其余特征用于预测目标特征。

```
train_df, test_df = catboost.datasets.amazon()
train_df
set(train_df.ACTION)
y_train = train_df.ACTION
X_train = train_df.drop('ACTION',axis=1)
cat_features = list(range(0,X_train.shape[1]))
from catboost import Pool, CatBoostClassifier, cv
# Initialize CatBoostClassifier
model = CatBoostClassifier(iterations=300,
learning_rate=0.1,
random_seed=12)
# Fit model
model.fit(X_train,y_train,cat_features=cat_features,verbose=False,plot=False)
```

使用 CatBoost 分类器的模型训练过程具有一定的复杂性，这意味着存在无法通过单一函数近似的复杂关系。如果特征具有线性关系，你可以使用一条直线来近似该函数。然而，如果特征是圆形或锯齿形模式，则称为复杂模式，因为数学上没有任何函数可以近似圆形或锯齿形模式。在这些情况下，你需要多个函数或数据归一化来匹配某个数学函数。

在当前场景中，由于特征数据集的复杂性，SHAP 值是近似计算的。

```
eval_dataset = [X_train.iloc[0:1]]
eval_dataset
# Get predicted classes
preds_class = model.predict(X_train)
# Get predicted probabilities for each class
preds_proba = model.predict_proba(X_train)
# Get predicted RawFormulaVal
preds_raw = model.predict(X_train,
prediction_type='RawFormulaVal')
preds_class
preds_proba
preds_raw
```

训练好的模型存储在`model`对象中。你可以取两个样本作为示例，来生成该实例的概率和原始公式值。该样本可以是训练数据集中的任何数据点。在脚本中，我们以第一条和第 91 条记录为例。

```
import numpy as np
np.log(0.9964/(1-0.9964))
np.exp(5.62)/(1+np.exp(5.62))
cat_features
```

第一条记录显示预测为类别 1 的概率为 99.64%。对数几率值为 5.61。对数几率意味着`log(P/1-P)`，这里`P`是类别 1 的概率。如果你想从原始预测值得到概率，可以使用公式`exp(raw prediction) / (1+ exp(raw prediction))`。例如，对于第一条记录，`exp(5.61) / (1+ exp(5.61)) = 0.9964`。以类似的方式，你可以解释第 91 条记录的原始预测值`-3.4734`。CatBoost 模型提供了预测不同预测类型的选项；参见表 5-7。

**表 5-7** CatBoost 模型对象生成的预测类型

| `prediction_type` | 描述 |
| --- | --- |
| **Probability** | 对应于每个类别的概率值 |
| **Class** | 将生成类别标签 |
| **RawFormula Value** | 对数几率值 |
| **Exponent** | 预测概率的指数值 |
| **LogProbability** | 概率的对数值 |

来自 CatBoost 模型的不同类型的预测提供了更多洞察，有助于理解预测值以及模型如何生成预测。所有类型之间的关联提供了更好的理解。你可以转换原始预测值以得到类别 1 的概率。

```
from catboost.datasets import *
train_df, test_df = catboost.datasets.amazon()
y = train_df.ACTION
X = train_df.drop('ACTION', axis=1)
cat_features = list(range(0, X.shape[1]))
model = CatBoostClassifier(iterations=300, learning_rate=0.1, random_seed=12)
model.fit(X, y, cat_features=cat_features, verbose=False, plot=False)
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(Pool(X, y, cat_features=cat_features))
test_objects = [X.iloc[7:9], X.iloc[56:58]]
for obj in test_objects:
print('Probability of class 1 = {:.4f}'.format(model.predict_proba(obj)[0][1]))
print('Formula raw prediction = {:.4f}'.format(model.predict(obj, prediction_type='RawFormulaVal')[0]))
print('\n')
shap.force_plot(explainer.expected_value, shap_values[0,:], X.iloc[0,:])
```

![../images/506619_1_En_5_Chapter/506619_1_En_5_Figa_HTML.jpg](img/506619_1_En_5_Figa_HTML.jpg)

```
shap.force_plot(explainer.expected_value, shap_values[91,:], X.iloc[91,:])
```

![../images/506619_1_En_5_Chapter/506619_1_En_5_Fig27_HTML.jpg](img/506619_1_En_5_Fig27_HTML.jpg)

**图 5-27** 训练集中记录 1 的力导向图

在图 5-27 中，力导向图展示了原始预测值是如何基于拉低（蓝色，下方图形）和推高（红色，上方图形）的特征值来决定的。从 1.124 到 6.124 刻度上反映的值是原始预测值，即相对于类别 1 的预测概率的对数几率比。平均对数几率比值为 3.624。拉低值通过使用诸如`resource`、`role title`、`roll`、`roll-up 2`等特征来降低预测的几率比。推高特征（`role dept name`、`MGR id`等）会将预测的几率比提高到更高水平。最终值为 5.61，这是最终的预测原始概率值，也称为几率比。

```
shap.summary_plot(shap_values, X)
```

![../images/506619_1_En_5_Chapter/506619_1_En_5_Fig28_HTML.jpg](img/506619_1_En_5_Fig28_HTML.jpg)

**图 5-28** SHAP 值摘要图，显示对模型输出的影响

图 5-28 显示了影响模型输出的特征及其重要性。`Resource`是最具影响力的特征，其次是`MGR ID`，依此类推，如图所示。



### 使用 SHAP 多分类 CatBoost 模型

CatBoost 分类器模型也可用于训练多分类模型。训练好的模型对象可以传递给 SHAP 树解释器以生成 SHAP 值。参见图 5-29。

![../images/506619_1_En_5_Chapter/506619_1_En_5_Fig29_HTML.jpg](img/506619_1_En_5_Fig29_HTML.jpg)

图 5-29

特征值与 SHAP 值表示

```
model = CatBoostClassifier(loss_function = 'MultiClass', iterations=300, learning_rate=0.1, random_seed=123)
model.fit(X, y, cat_features=cat_features, verbose=False, plot=False)
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(Pool(X, y, cat_features=cat_features))
shap.summary_plot(shap_values[0], X)
```

CatBoost 分类器或回归器有许多超参数。表 5-8 展示了一些重要参数及其对模型结果的影响。

表 5-8

CatBoost 分类器的超参数

| 参数 | 说明 |
| --- | --- |
| `cat_features` | 训练数据集中存在的分类列索引 |
| `text_features` | 文本列索引（指定为整数）或名称（指定为字符串） |
| `learning_rate` | 用于损失函数优化，减小梯度步长 |
| `Iterations` | 解决机器学习问题时可以构建的最大树数量 |
| `Max_depth` | 树的深度 |
| `leaf_estimation_method` | 用于计算叶节点值的方法。可能的值：`Newton`、`Gradient`、`Exact`。取决于模式和所选损失函数：使用 `Quantile` 或 `MAE` 损失函数的回归——一次精确迭代；使用除 `Quantile` 或 `MAE` 之外的任何损失函数的回归——一次梯度迭代；分类模式——十次牛顿迭代；多分类模式——一次牛顿迭代 |
| `boosting_type` | 提升方案。可能的值：`Ordered`——通常在小数据集上提供更好的质量，但可能比 `Plain` 方案慢；`Plain`——经典的梯度提升方案 |

为了获得更好的结果，可以更改或修改上述参数，并使用 SHAP 库进行解释。

### 使用 SHAP 解释 Light GBM 模型

如 CatBoost 分类器部分所述，类似地，你也可以使用轻量梯度提升模型。Light GBM 模型和 CatBoost 模型之间的区别在于，在混合数据场景（即训练数据集中存在混合特征集）下，Light GBM 运行速度更快。当数据集中有大量分类特征时，CatBoost 分类器更有用。在 Light GBM 中，你需要对分类特征进行编码，但在 CatBoost 模型中，模型具有处理分类特征的内部机制。

安装 Light GBM 模型：

```
conda install -c conda-forge lightgbm
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import shap
X,y = shap.datasets.adult()
X_display,y_display = shap.datasets.adult(display=True)
# create a train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)
d_train = lgb.Dataset(X_train, label=y_train)
d_test = lgb.Dataset(X_test, label=y_test)
params = {
"max_bin": 512,
"learning_rate": 0.05,
"boosting_type": "gbdt",
"objective": "binary",
"metric": "binary_logloss",
"num_leaves": 10,
"verbose": -1,
"min_data": 100,
"boost_from_average": True
}
model = lgb.train(params, d_train, 10000, valid_sets=[d_test], early_stopping_rounds=50, verbose_eval=1000)
```

你可以使用相同的人口普查收入分类数据集，通过 Light GBM 模型训练分类器，并为预测结果生成解释。通过查阅 Light GBM 模型的官方文档页面，可以了解许多超参数。表 5-9 展示了以下代码行中使用的超参数。

表 5-9

LGB 模型的超参数

| 参数 | 说明 |
| --- | --- |
| `Max_bin` | 特征值将被分入的最大箱数。箱数较少可能会降低训练精度，但可能会提高泛化能力（处理过拟合） |
| `Learning_rate` | 收缩率 |
| `Boosting_type` | `gbdt`，传统梯度提升决策树，别名：`gbrt`；`rf`，随机森林，别名：`random_forest`；`dart`，Dropouts meet Multiple Additive Regression Trees；`goss`，基于梯度的单边采样 |
| `Objective` | 应用类型，例如二分类、多分类等 |
| `Metric` | 在评估集上评估的指标 |
| `Num_leaves` | 一棵树中最大叶节点数 |
| `Min_data` | 每个分类组的最小数据量 |

训练完 Light GBM 模型后，你可以使用 SHAP 树解释器来解释预测结果，还可以为 SHAP 值生成力图（图 5-30）。

![../images/506619_1_En_5_Chapter/506619_1_En_5_Fig30_HTML.jpg](img/506619_1_En_5_Fig30_HTML.jpg)

图 5-30

使用 SHAP 特征值预测对数几率比的力图

```
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)
shap.force_plot(explainer.expected_value[1], shap_values[1][0,:], X_display.iloc[0,:])
```

在图 5-30 中，教育年限和年龄将预测的几率比推向更高值，而资本收益和关系将预测的几率比拉向更低值。因此，数据集中第一条记录的最终预测几率比为 -8.43，远低于基准值 -2.43 所示的平均几率比。

```
shap.force_plot(explainer.expected_value[1], shap_values[1][:1000,:], X_display.iloc[:1000,:])
```

![../images/506619_1_En_5_Chapter/506619_1_En_5_Fig31_HTML.jpg](img/506619_1_En_5_Fig31_HTML.jpg)

图 5-31

LGBM 模型生成输出值时样本顺序的相似性

在图 5-31 中，输出是几率比，并且如本章所示，你可以从几率比反推出目标类别的概率值。此图显示了样本的相似性，然后据此排序。生成较高几率比的样本位于 0-50 之间。之后，其余样本生成的几率比范围在 +3 到 -9 之间，直到第 900 条记录。此后，几率比下降到非常负的范围。红线表示有助于提高预测几率比的特征值，蓝线表示降低预测几率比的特征值。平均线对应 -2.43 的值。如果你计算整个数据集的几率比然后求平均值，结果就是 -2.43。

```
shap.summary_plot(shap_values, X)
```

![../images/506619_1_En_5_Chapter/506619_1_En_5_Fig32_HTML.jpg](img/506619_1_En_5_Fig32_HTML.jpg)

图 5-32

汇总图

图 5-32 显示了汇总图。这是一个非常重要的图。它提供了目标列中每个类别对应的不同特征的平均 SHAP 值。此图有趣之处在于，平均值在两个类别之间均匀分布，与特征值无关。



## 结论

本章涵盖了用于分类和回归场景的几种基于提升的模型。类似地，装袋分类器也可以进行训练，并生成类似的输出图表。图表本身不会改变，但随着你从一个模型切换到另一个模型，对数值的解释会发生变化。以下是需要重点关注的五点：

- 在回归场景中，你预测目标列，并根据特征贡献观察预测结果是上升还是下降。因此，如果你改变一个输入特征，就能轻松理解对目标输出的影响。
- 在分类场景中，你将对数几率比作为连续变量进行预测，并根据特征输入值预测结果。你知道哪些特征重要，因此能够清晰地看到基于输入特征变化的预测结果。
- 无论是装袋、提升还是堆叠，都可以生成类似的偏依赖图（PDP）、力图和汇总图框架，以理解模型的决策过程。然而，以偏依赖图为例，其展示的模式会发生变化。
- 图表模式发生变化，是因为模型试图捕捉数据中可能存在的复杂模式。随着数据复杂性的增加，模型变得更加复杂，从视觉上看可能难以解释。然而，从数值中得出的推论将保持不变。
- 因此，基于本章的分析，你涵盖了多少个集成模型并不重要。更重要的是建立一个可以应用于其他集成模型的框架。

