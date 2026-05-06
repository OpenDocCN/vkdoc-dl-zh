# 每个特征贡献的平均绝对 SHAP 值

```
# make a standard partial dependence plot
sample_ind = 25
fig,ax = shap.partial_dependence_plot(
"number_vmail_messages", model_churn_proba, X, model_expected_value=True,
feature_expected_value=True, show=False, ice=False
)
shap_values_churn.feature_names
# compute the SHAP values for the linear model
explainer_log_odds = shap.Explainer(log_model, background_churn,feature_names=list(X.columns))
shap_values_churn_log_odds = explainer_log_odds(X)
shap_values_churn_log_odds
shap.plots.bar(shap_values_churn_log_odds)
```

```
shap.plots.bar(shap_values_churn_log_odds.abs.max(0))
```

```
shap.plots.beeswarm(shap_values_churn_log_odds)
```

```
shap.plots.heatmap(shap_values_churn_log_odds[:1000])
```

| `特征名称` | `系数` |   |
| `14` | `number_customer_service_calls` | `0.383573` |
| `2` | `total_day_minutes` | `0.008251` |
| `4` | `total_day_charge` | `0.001378` |
| `5` | `total_eve_minutes` | `0.000947` |
| `7` | `total_eve_charge` | `0.000098` |
| `10` | `total_night_charge` | `-0.000048` |
| `13` | `total_intl_charge` | `-0.000196` |
| `11` | `total_intl_minutes` | `-0.000464` |
| `0` | `account_length` | `-0.000573` |
| `8` | `total_night_minutes` | `-0.001730` |
| `3` | `total_day_calls` | `-0.009254` |
| `9` | `total_night_calls` | `-0.010050` |
| `6` | `total_eve_calls` | `-0.012706` |
| `1` | `number_vmail_messages` | `-0.019944` |
| `15` | `area_code_tr` | `-0.033119` |
| `12` | `total_intl_calls` | `-0.097870` |

```
temp_df = pd.DataFrame()
temp_df['Feature Name'] = pd.Series(X.columns)
temp_df['Coefficients'] = pd.Series(log_model.coef_.flatten())
temp_df.sort_values(by='Coefficients',ascending=False)
```

### 解释

当你将某个特征的值增加一个单位时，模型方程会产生两个几率：一个是基准几率，另一个是特征增量值对应的几率。这里的目标是观察特征值每增加或减少一个单位时，几率比的变化。特征变化一个单位会导致几率比以相应贝塔系数的指数倍发生变化。这可以通过以下方程进一步解释，其中 `beta 0` 是截距项，`beta 1` 到 `beta k` 是模型参数，`x1` 到 `xk` 是模型的独立预测变量：

![$$ \frac{P\left(y=1\right)}{1-P\left(y=1\right)}= odds=\mathit{\exp}\left({\beta}_0+{\beta}_1{x}_1+\dots +{\beta}_p{x}_p\right) $$](img/506619_1_En_3_Chapter_TeX_Equg.png)

我们将方程右侧称为 `exp(a)`，其中 `a` 代表线性回归概念的方程。如果你增加模型的任意参数，方程将变化一个单位，因此我们称之为 `b`，那么右侧变为 `exp(b)`。相对于预测变量值变化一个单位的几率比将是 `odds_new / odd_old = exp(a - b)`。你可以用这种格式解释所有数值特征。这也适用于所有分类特征或二元特征。

### LIME 推断

为了解释逻辑回归模型，你可以使用 SHAP 值。然而，其复杂性在于时间成本。如果你有百万条记录，并且需要抽取一个相当大的样本来生成所有排列组合，以便在全局层面解释局部准确性，那么你将需要更多时间。为了避免处理大型数据集时的这一瓶颈，LIME 在生成解释方面提供了速度优势。为了解释表格矩阵数据（即结构化数据），你必须使用 `LimeTabularExplainer`。对于数值特征，通过从 `Normal(0,1)` 中采样并进行均值中心化和缩放的逆操作（根据训练数据中的均值和标准差）来扰动它们。对于分类特征，通过根据训练分布进行采样，并创建一个二元特征（当该值与正在解释的实例相同时为 1）来扰动。

在生成 LIME 解释器时，你需要将数据作为数组传递，提供列名列表，提供目标列名，并根据你计划使用的机器学习任务将模式设置为回归或分类。verbose 选项用于启用模型预测。

```
import lime
import lime.lime_tabular
explainer = lime.lime_tabular.LimeTabularExplainer(np.array(xtrain),
feature_names=list(xtrain.columns),
class_names=['target_churn_dum'],
verbose=True, mode='classification')
# this record is a no churn scenario
exp = explainer.explain_instance(xtest.iloc[0], log_model.predict_proba, num_features=16)
exp.as_list()
```

一旦生成了解释器模型对象，你就可以检查单个预测和全局预测以生成解释。在具有两个类别或多个类别的分类问题中，你可以针对每个类别相对于特征列生成单独的特征重要性。在此示例中，你考虑两条记录：记录 1（模型正确预测了结果）和记录 20（模型错误地进行了预测）。你将解释在这两种情况下模型为何做出这样的决策。与目标类别具有正相关关系的特征为正数，而具有负相关关系的类别则带有负号。你可以将结果以表格笔记本的形式展示。你也可以通过仅考虑其中最重要的特征来限制视图。

```
pd.DataFrame(exp.as_list())
```

DataFrame 视图中的特征权重如表 3-7 所示。

**表 3-7** 具有阈值的特征及其对预测值的权重

| **0** | **1** |
| **0** | `number_customer_service_calls <= 1.00` | `-0.106490` |
| **1** | `total_day_minutes <= 143.70` | `-0.082492` |
| **2** | `number_vmail_messages <= 0.00` | `0.063827` |
| **3** | `total_eve_calls > 114.00` | `-0.046997` |
| **4** | `101.00 < total_night_calls <= 114.00` | `-0.014762` |
| **5** | `total_eve_minutes > 235.07` | `0.009634` |
| **6** | `account_length > 126.00` | `-0.007626` |
| **7** | `1.00 < area_code_tr <= 2.00` | `-0.007580` |
| **8** | `2.27 < total_intl_charge <= 2.75` | `0.006753` |
| **9** | `87.00 < total_day_calls <= 101.00` | `0.006710` |
| **10** | `166.93 < total_night_minutes <= 200.40` | `0.005046` |
| **11** | `total_day_charge <= 24.43` | `-0.004913` |
| **12** | `3.00 < total_intl_calls <= 4.00` | `0.004285` |
| **13** | `7.51 < total_night_charge <= 9.02` | `-0.001845` |
| **14** | `total_eve_charge > 19.98` | `-0.001836` |
| **15** | `8.40 < total_intl_minutes <= 10.20` | `-0.000155` |



对应于记录 1，可以生成下表。截距项为 0.126，LIME 预测的客户流失本地概率为 0.18，逻辑回归模型预测的流失概率为 0.15。这基本上是截距项加上不同特征的所有权重。由于流失概率较低，因此被归类为非流失场景。因此，图 3-23 中的蓝色条形图显示非流失概率为 0.84，流失概率为 0.16。特征按其权重的总体重要性显示在表格的右侧。中间的表格显示了按特征值划分的权重。

![../images/506619_1_En_3_Chapter/506619_1_En_3_Fig23_HTML.jpg](img/506619_1_En_3_Fig23_HTML.jpg)

**图 3-23** 测试集记录 1 的摘要及其局部解释

```
exp.show_in_notebook(show_table=True)
```

对于测试集中的记录 20，模型预测了不同的结果，而实际测试结果也不同。这是一种模型预测与真实情况不符的场景，因此模型需要解释为什么会发生这种情况。你可以使用 LIME 局部实例获得更清晰的视图。

```
# 这是一个流失场景
exp = explainer.explain_instance(xtest.iloc[20], log_model.predict_proba, num_features=16)
exp.as_list()
exp.show_in_notebook(show_table=True)
xtest.iloc[20]
```

在图 3-24 中，预测概率有两个条形图：蓝色条形图显示非流失概率，橙色条形图显示流失概率。如果你查看旁边的表格，它会清晰地显示特征及其在构成整个橙色条形图时的权重。

![../images/506619_1_En_3_Chapter/506619_1_En_3_Fig24_HTML.jpg](img/506619_1_En_3_Fig24_HTML.jpg)

**图 3-24** 测试集记录 20 的局部解释

除了两个特征（总日通话时长和账户时长）之外，所有其他特征都对流失概率有贡献，因此模型正确预测了结果，并且预测得到了解释。每个特征的阈值及其权重都已给出。这为业务人员理解预测模型的行为提供了更清晰的图景。

```
explainer = lime.lime_tabular.LimeTabularExplainer(np.array(xtrain),
feature_names=list(xtrain.columns),
class_names=['target_churn_dum'],
verbose=True, mode='classification')
# SP-LIME 的代码
import warnings
from lime import submodular_pick
# SP-LIME 返回样本集上的解释，以提供原始模型的非冗余全局决策边界
sp_obj = submodular_pick.SubmodularPick(explainer, np.array(xtrain),
log_model.predict_proba,
num_features=14,
num_exps_desired=10)
```

子模块选择选项提供了原始模型的全局决策边界。你可以使用`explainer`对象、训练数据集以及从训练模型中提取的概率，然后指定描述中应包含的特征数量和所需的表达式数量。参见图 3-25 至图 3-28。

![../images/506619_1_En_3_Chapter/506619_1_En_3_Fig28_HTML.jpg](img/506619_1_En_3_Fig28_HTML.jpg)

**图 3-28** 10 条记录中第四条记录的局部解释

![../images/506619_1_En_3_Chapter/506619_1_En_3_Fig27_HTML.jpg](img/506619_1_En_3_Fig27_HTML.jpg)

**图 3-27** 10 条记录中第三条记录的局部解释

![../images/506619_1_En_3_Chapter/506619_1_En_3_Fig26_HTML.jpg](img/506619_1_En_3_Fig26_HTML.jpg)

**图 3-26** 10 条记录中第二条记录的局部解释

![../images/506619_1_En_3_Chapter/506619_1_En_3_Fig25_HTML.jpg](img/506619_1_En_3_Fig25_HTML.jpg)

**图 3-25** 10 条记录中第一条记录的局部解释

Skater 生成的结果与 LIME 库类似，因此不再赘述其描述，因为我们已使用 LIME 库进行了介绍。ELI5 主要用于文本分类用例，因此 ELI5 不适用于解释线性或逻辑回归模型。更多信息请参见表 3-8。

**表 3-8** 何时使用哪个库的摘要视图

| 库名称 | 定义 | 使用时机 | 优势 | 局限性 |
| --- | --- | --- | --- | --- |
| **SHAP** | 使用 Shapley 值解释任何机器学习模型 | 表格数据、图像数据 | 提供更多指标和统计信息，解释更优 | 不明显 |
| **LIME** | 局部可解释模型解释（LIME） | 用于局部解释、表格数据 | 适用于单个实例的可解释性 | 全局解释不直观。不适用于图像数据。 |
| **Skater** | Skater 包中的通用工作流程是创建解释、创建模型并运行解释算法。 | 表格数据，提供两个模块：内存模型和已部署模型 | 模型训练和解释只需运行一次。无需作为单独进程运行。 | 仅支持少数模型。未能涵盖所有类型的模型。 |
| **ELI5** | 一个帮助调试机器学习分类器并解释其预测的 Python 包 | Scikit-learn 模型、文本分类、Keras 模型解释 | 适用于文本分类 | 对于所有其他任务来说不是一个成熟的库。解释非常基础。 |

## 结论

在本章中，你学习了如何解释线性模型、用于预测的线性回归模型以及用于二分类的逻辑回归模型。类似地，逻辑回归模型也可以扩展到多项分类。线性模型是较易于解释的模型，每个人都很好地理解这些模型的工作原理。因此，人们对线性模型始终抱有高度的信任。然而，在本章中，你从多个角度探讨了如何使用可解释 AI 库（如 LIME 和 SHAP）为线性模型创建视图。在下一章中，你将学习非线性模型的模型可解释性。

