# 偏依赖图

偏依赖图（PDP）用于可视化特征列与目标列（或响应列）之间的交互关系。目标列包含两个标签：0 表示未流失案例，1 表示流失案例。当你使用 `predict_proba()` 函数时，可以生成类别 0 和类别 1 的概率。

```
pd.DataFrame(dt2.predict_proba(X)) # 0 - 未流失, 1- 流失
model_churn_proba(X).max()
model_churn_proba(X).mean()
model_churn_proba(X).min()
```

上述脚本中的函数用于选择第 1 列，即流失概率。例如，第一条记录显示概率为 0.090909，这意味着流失几率低于 10%。你可以得出结论，这是一个未流失样本。在下面的脚本中，`model_churn_proba` 仅显示第二列的流失概率。

在绘制流失概率与总日通话分钟数（如图 4-10 的 PDP 图所示）的关系图之前，你应该先使用分布图查看流失概率的分布情况。这将有助于你理解如何解读偏依赖图。概率分布如图 4-9 所示。

![../images/506619_1_En_4_Chapter/506619_1_En_4_Fig10_HTML.jpg](img/506619_1_En_4_Fig10_HTML.jpg)

图 4-10

总日通话分钟数与流失概率的 PDP 图

![../images/506619_1_En_4_Chapter/506619_1_En_4_Fig9_HTML.jpg](img/506619_1_En_4_Fig9_HTML.jpg)

图 4-9

决策树模型得出的流失概率分布

```
import seaborn as sns
sns.distplot(model_churn_proba(X))
pd.DataFrame(model_churn_proba(X))
from sklearn.inspection import plot_partial_dependence
xtrain.columns
plot_partial_dependence(dt2, X, ['account_length', 'number_vmail_messages', 'total_day_minutes'])
plot_partial_dependence(dt2, X, [
'total_day_calls', 'total_day_charge', 'total_eve_minutes',
])
plot_partial_dependence(dt2, X, [
'total_eve_calls', 'total_eve_charge', 'total_night_minutes'])
```

图 4-10 中展示的总日通话分钟数及其期望值呈分段线性关系，但观察整条蓝色曲线，它并非线性。这是针对样本编号 25 的局部解释。总日通话分钟数是数值型数据，在训练数据集中被视为连续列。对于同一个样本观测值 25，语音邮件数量也会影响该特征对流失预测的边际贡献。在图 4-10 中，如果用户的总日通话分钟数不超过 225 分钟，那么流失概率将小于或等于平均流失概率（14.25）。即使总日通话分钟数超过 225 分钟，流失概率仍低于 25%。

```
# 生成标准偏依赖图
sample_ind = 25
fig,ax = shap.partial_dependence_plot(
"number_vmail_messages", model_churn_proba, X, model_expected_value=True,
feature_expected_value=True, show=False, ice=False
)
```

![../images/506619_1_En_4_Chapter/506619_1_En_4_Fig11_HTML.jpg](img/506619_1_En_4_Fig11_HTML.jpg)

图 4-11

语音邮件数量与流失概率的 PDP 图

当你观察账户时长对流失预测的边际贡献与账户时长之间的关系时，可以看到一条蓝色的平行线（图 4-11）。这表明无论账户时长如何变化，边际贡献都保持不变，意味着该特征没有预测价值，不重要，并且在流失预测模型中不起作用。

```
# 生成标准偏依赖图
sample_ind = 25
fig,ax = shap.partial_dependence_plot(
"account_length", model_churn_proba, X, model_expected_value=True,
feature_expected_value=True, show=False, ice=False
)
```

![../images/506619_1_En_4_Chapter/506619_1_En_4_Fig12_HTML.jpg](img/506619_1_En_4_Fig12_HTML.jpg)

图 4-12

账户时长特征的 PDP 图

账户时长的 PDP 图对流失概率没有影响，这一点从图 4-12 中的蓝色直线可以清晰看出。蓝色直线遵循流失概率的平均值。

```
# 生成标准偏依赖图
sample_ind = 25
fig,ax = shap.partial_dependence_plot(
"number_customer_service_calls", model_churn_proba, X, model_expected_value=True,
feature_expected_value=True, show=False, ice=False
)
```

![../images/506619_1_En_4_Chapter/506619_1_En_4_Fig13_HTML.jpg](img/506619_1_En_4_Fig13_HTML.jpg)

图 4-13

客户服务呼叫次数的 PDP 图

需要注意的是，客户服务呼叫次数越高，流失概率也越高。这是因为用户遇到了问题，导致客户服务呼叫次数增加。你可以在图 4-13 中看到相同的模式：超过四次呼叫会增加流失概率。

```
# 生成标准偏依赖图
sample_ind = 25
fig,ax = shap.partial_dependence_plot(
"total_day_charge", model_churn_proba, X, model_expected_value=True,
feature_expected_value=True, show=False, ice=False
)
```

![../images/506619_1_En_4_Chapter/506619_1_En_4_Fig14_HTML.jpg](img/506619_1_En_4_Fig14_HTML.jpg)

图 4-14

总日通话费用的 PDP 图

如果总日通话费用超过 45 美元，那么流失概率将会更高（图 4-14）。这是由于费用较高，用户可能会被迫选择其他服务提供商。

```
# 生成标准偏依赖图
sample_ind = 25
fig,ax = shap.partial_dependence_plot(
"total_eve_minutes", model_churn_proba, X, model_expected_value=True,
feature_expected_value=True, show=False, ice=False
)
```

![../images/506619_1_En_4_Chapter/506619_1_En_4_Fig15_HTML.jpg](img/506619_1_En_4_Fig15_HTML.jpg)

图 4-15

总晚间通话分钟数的 PDP 图

总晚间通话分钟数对流失概率有一定影响，但不会将用户从未流失状态转变为流失状态（图 4-15）。在总晚间通话分钟数不超过 180 分钟时，流失概率保持非常低且恒定，为 13%。当超过 180 分钟时，流失概率略微上升至 15.5%。在 250 分钟及之后，概率上升至 16.5%，但相对而言仍然很低。

```
# 生成标准偏依赖图
sample_ind = 25
fig,ax = shap.partial_dependence_plot(
"total_eve_calls", model_churn_proba, X, model_expected_value=True,
feature_expected_value=True, show=False, ice=False
)
```

![../images/506619_1_En_4_Chapter/506619_1_En_4_Fig16_HTML.jpg](img/506619_1_En_4_Fig16_HTML.jpg)

图 4-16

总晚间通话次数的 PDP 图

总晚间通话次数对流失概率没有影响，因为无论总晚间通话次数如何增加，概率都保持在平均值不变（图 4-16）。表 4-3 显示了每个特征的特征重要性得分。

表 4-3

每个特征的特征重要性得分



| 特征名称 | 得分 |
| --- | --- |
| **4** | `total_day_charge` | 0.306375 |
| **14** | `number_customer_service_calls` | 0.282235 |
| **2** | `total_day_minutes` | 0.223336 |
| **1** | `number_vmail_messages` | 0.105760 |
| **5** | `total_eve_minutes` | 0.082294 |
| **0** | `account_length` | 0.000000 |
| **3** | `total_day_calls` | 0.000000 |
| **6** | `total_eve_calls` | 0.000000 |
| **7** | `total_eve_charge` | 0.000000 |
| **8** | `total_night_minutes` | 0.000000 |
| **9** | `total_night_calls` | 0.000000 |
| **10** | `total_night_charge` | 0.000000 |
| **11** | `total_intl_minutes` | 0.000000 |
| **12** | `total_intl_calls` | 0.000000 |
| **13** | `total_intl_charge` | 0.000000 |
| **15** | `area_code_tr` | 0.000000 |

```
shap_values_churn.feature_names
temp_df = pd.DataFrame()
temp_df['Feature Name'] = pd.Series(X.columns)
temp_df['Score'] = pd.Series(dt2.feature_importances_.flatten())
temp_df.sort_values(by='Score',ascending=False)
```

在上图中，你看到了基于 SHAP 库的偏依赖图。同样的图也可以使用 `scikit-learn` 库的检查模块生成。

## 使用 Scikit-Learn 的 PDP

`scikit-learn` 模块套件中有一个新模块，可以帮助你生成偏依赖图。你可以使用 `scikit-learn` 库中的 `inspection` 模块。请参见图 4-17 至 4-22。

![../images/506619_1_En_4_Chapter/506619_1_En_4_Fig22_HTML.jpg](img/506619_1_En_4_Fig22_HTML.jpg)

图 4-22

客户服务电话次数和区号的 PDP

![../images/506619_1_En_4_Chapter/506619_1_En_4_Fig21_HTML.jpg](img/506619_1_En_4_Fig21_HTML.jpg)

图 4-21

国际电话总次数和国际话费总额的 PDP

![../images/506619_1_En_4_Chapter/506619_1_En_4_Fig20_HTML.jpg](img/506619_1_En_4_Fig20_HTML.jpg)

图 4-20

三个不影响流失概率的特征的 PDP

![../images/506619_1_En_4_Chapter/506619_1_En_4_Fig19_HTML.jpg](img/506619_1_En_4_Fig19_HTML.jpg)

图 4-19

另外三个变量的 PDP

![../images/506619_1_En_4_Chapter/506619_1_En_4_Fig18_HTML.jpg](img/506619_1_En_4_Fig18_HTML.jpg)

图 4-18

下一组三个特征的 PDP

![../images/506619_1_En_4_Chapter/506619_1_En_4_Fig17_HTML.jpg](img/506619_1_En_4_Fig17_HTML.jpg)

图 4-17

三个变量一起的 PDP

```
from sklearn.inspection import plot_partial_dependence
xtrain.columns
plot_partial_dependence(dt2, X, ['account_length', 'number_vmail_messages', 'total_day_minutes'])
```

```
plot_partial_dependence(dt2, X, [
'total_day_calls', 'total_day_charge', 'total_eve_minutes',
])
```

```
plot_partial_dependence(dt2, X, [
'total_eve_calls', 'total_eve_charge', 'total_night_minutes'])
```

```
plot_partial_dependence(dt2, X, [
'total_night_calls', 'total_night_charge', 'total_intl_minutes'])
```

```
plot_partial_dependence(dt2, X, [
'total_intl_calls', 'total_intl_charge'])
```

```
plot_partial_dependence(dt2,X, [
'number_customer_service_calls', 'area_code_tr'])
```

图 4-17 至 4-22 是通过 `scikit-learn` 库中的一个函数生成的，该函数产生的解释与 SHAP 库类似。其解读方式也类似。

## 非线性模型解释 – LIME

在流失预测过程中起重要作用的最重要特征包括 `total_day_charge`、`number_customer_service_calls`、`total_day_minutes`、`number_vmail_messages` 和 `total_eve_minutes`。其他特征不起作用，因此无关特征的偏依赖图是平行线。

你还可以利用 LIME Python 库中的一些函数来解释决策树模型所做的决策。

```
import lime
import lime.lime_tabular
explainer = lime.lime_tabular.LimeTabularExplainer(np.array(xtrain),
feature_names=list(xtrain.columns),
class_names=['target_churn_dum'],
verbose=True, mode='classification')
# this record is a no churn scenario
exp = explainer.explain_instance(xtest.iloc[0], dt2.predict_proba,
num_features=16)
exp.as_list()
pd.DataFrame(exp.as_list())
exp.show_in_notebook(show_table=True)
```

![../images/506619_1_En_4_Chapter/506619_1_En_4_Fig23_HTML.jpg](img/506619_1_En_4_Fig23_HTML.jpg)

图 4-23

记录 0 的特征重要性以及对预测概率的正负贡献

图 4-23 显示了测试集中第一条记录 `xtest[0]` 的局部解释模型说明。图 4-23 中间的图表显示了两种颜色。蓝色表示对目标流失预测概率的贡献，橙色表示对另一类别（即非流失场景）的贡献。决策树将特征表示为一系列值，例如 `total_day_minutes` 在 179.90 到 216.20 之间，该特征对流失概率的贡献为 0.08（8%）。如果从模型中移除两个特征 `number_customer_service_calls <= 1.00` 和 `total_day_minutes > 179.90` 且 `<= 216.20`，那么目标流失预测概率将降低 0.16（16%），即 (0.94 – 0.08 -0.08) = 0.78（78%）。另一方面，如果移除 `number_vmail_messages <=0.00`，则目标流失概率将增加 4%（0.04）。图 4-23 中的第三个表格显示了每个特征对预测的贡献值。可以对更多记录进行类似的分析和解读，如图 4-24 至 4-26 所示。

![../images/506619_1_En_4_Chapter/506619_1_En_4_Fig25_HTML.jpg](img/506619_1_En_4_Fig25_HTML.jpg)

图 4-25

所有记录的特征重要性以及对预测概率的正负贡献

![../images/506619_1_En_4_Chapter/506619_1_En_4_Fig24_HTML.jpg](img/506619_1_En_4_Fig24_HTML.jpg)

图 4-24

记录编号 20 的特征重要性以及对预测概率的正负贡献

```
# This is s churn scenario
exp = explainer.explain_instance(xtest.iloc[20], dt2.predict_proba, num_features=16)
exp.as_list()
exp.show_in_notebook(show_table=True)
```

```
xtest.iloc[20]
ytest.iloc[20]
dt2.predict(xtest)[20]
explainer = lime.lime_tabular.LimeTabularExplainer(np.array(xtrain),
feature_names=list(xtrain.columns),
class_names=['target_churn_dum'],
verbose=True, mode='classification')
# Code for SP-LIME
import warnings
from lime import submodular_pick
# SP-LIME returns exaplanations on a sample set to provide a non redundant global decision boundary of original model
sp_obj = submodular_pick.SubmodularPick(explainer, np.array(xtrain),
dt2.predict_proba,
num_features=14,
num_exps_desired=10)
```



## 非线性解释 – Skope-Rules

还有另一个名为 `Skope-rules` 的可解释性库，可用于从训练模型中生成规则，并且这些规则可用于对任何新数据集进行预测。

以下代码可用于安装基于 Python 的库。参数说明见表 4-4。

**表 4-4** Skope-Rules 参数说明

| 参数 | 说明 |
| --- | --- |
| `feature_names` | 每个特征的名称，用于以字符串格式返回规则 |
| `precision_min` | 规则被选中的最小精确度 |
| `recall_min` | 规则被选中的最小召回率 |
| `n_estimators` | 用于预测的基础估计器（规则）数量 |
| `max_depth_duplication` | 用于规则去重的决策树最大深度 |
| `max_depth` | 决策树的最大深度 |
| `max_samples` | 从 X 中抽取用于训练每棵决策树的样本数量，规则由此生成并筛选 |

![../images/506619_1_En_4_Chapter/506619_1_En_4_Fig26_HTML.jpg](img/506619_1_En_4_Fig26_HTML.jpg)

**图 4-26** 所有记录的特征重要性以及对预测概率的正负贡献

```
!pip install skope-rules
import six
import sys
sys.modules['sklearn.externals.six'] = six
import skrules
from skrules import SkopeRules
clf = SkopeRules(max_depth_duplication=2,
n_estimators=30,
precision_min=0.3,
recall_min=0.1,
feature_names=list(xtrain.columns))
clf
clf.fit(xtrain,ytrain)
print('以下是最精确的 5 条规则：')
for rule in clf.rules_[:5]:
print(rule[0])
```

使用 `clf.fit()` 方法可以训练决策树模型，并得出以下规则。可以使用 `precision` 和 `recall` 参数对规则进行筛选或整理。如果将阈值降低到较低水平，将会生成大量规则，这对用户毫无用处，因为误报规则会进入业务应用。较高的阈值会减少规则数量，只有最相关的规则才会被用于业务应用。

最精确的五条规则如下：

```
number_vmail_messages  249.70000457763672
number_vmail_messages  44.989999771118164 and total_eve_minutes > 145.39999389648438
number_customer_service_calls > 3.5 and total_day_minutes  245.0999984741211 and total_eve_minutes > 204.20000457763672
number_customer_service_calls > 3.5 and total_day_charge <= 30.65999984741211
clf.predict(xtest)
clf.predict_top_rules(xtest,5)
clf.score_top_rules(xtest)
```

`predict` 函数使用所有规则来预测目标流失类别：0 表示未流失，1 表示流失。

`predict_top_rules` 函数用于利用这五条规则来预测结果。5 是用于预测的规则数量。如果 `n_rules` 条性能最佳的规则中有一条被激活，则预测结果等于 1。对于每个观测值，该函数会根据所选规则判断其是否应被视为异常值（1 或 0）。

```
clf.decision_function(xtrain)
clf.decision_function(xtest)
```

`Score top rules` 表示基础分类器（规则）之间的排序。当实例被一条性能良好的规则检测到时，得分会很高。正分数代表异常值，零分数代表正常值。

`decision_function` 是输入样本的异常得分，计算方式为二元规则输出的加权和，权重为每条规则各自的精确度。对于输入样本的异常得分，得分越高，表示异常程度越高。正分数代表异常值，零分数代表正常值。

## 结论

在本章中，您学习了如何解释非线性模型，特别是决策树模型，而不是用于二分类的逻辑回归。类似地，决策树模型也可用于回归模型，并可扩展到多项分类。非线性模型是较易于解释的模型，每个人都通过简单的 if/then else 规则理解这些模型的工作原理。因此，人们对基于树的非线性模型有很高的信任度。在本章中，您从多个角度探讨了如何使用可解释 AI 库（如 LIME、SHAP 和 Skope-rules）为非线性模型创建视图。在下一章中，您将学习集成模型的模型可解释性。

