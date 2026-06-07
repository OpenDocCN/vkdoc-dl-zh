# 7. 案例研究

本章将通过一系列案例研究，探讨如何将决策智能整合到组织中。我们将深入研究各种方法论，并考察它们如何独立或与其他方法结合使用，为决策者提供可操作的见解。

虽然我们在前面的章节中已经介绍了很多方法论，并展示了它们如何应用于决策智能，但很少能找到单一方法论就能有效解决决策智能问题。这些问题通常非常复杂，需要结合多种方法论才能达到预期效果。

让我们通过两个案例研究（一个分类问题，一个回归问题）来探讨人工智能/机器学习以及其他先进技术如何辅助决策。

## 案例研究 1：电信客户流失管理

**问题陈述：** 一家电信公司正面临高客户流失率的问题，正在寻找创新的 AI 解决方案来减少客户流失。该公司希望利用机器学习及其他技术来实现以下目标：

-   监控客户行为并识别客户流失的早期预警信号
-   识别导致客户流失的因素
-   创建个性化的客户保留解决方案并进行模拟
-   根据模拟输出更新解决方案

**解决方案：**

AI 团队提出一个混合解决方案来解决该问题。计划是首先构建一个机器学习模型，该模型输入历史客户详细信息，如账户信息、人口统计信息、使用的服务以及流失状态。模型输出将用于找出未来可能流失的潜在客户。该团队还提出了一个反事实生成引擎，该引擎可以提供预测结果发生改变的各种场景。例如，如果某客户可能流失，且其流失概率为 90%，反事实引擎可以生成输入特征的改变，使得该客户的流失概率可能降至 40%。输入参数的这种改变可以作为其保留策略的一部分。第三个组件是假设分析解决方案，它可以帮助决策者根据某些约束条件调整反事实建议，并检查这些改变是否仍能留住客户。

AI 团队提出了使用该混合解决方案的以下好处：

-   **提高准确性：** 机器学习模型可以分析大量数据，识别出人类分析师不易察觉的模式和见解。反事实分析和假设分析可用于测试和验证机器学习模型的准确性，并获得关于如何留住可能流失客户的想法，从而提高客户留存率。
-   **个性化解决方案：** 机器学习可以分析客户行为和偏好，提供有助于留住客户的个性化解决方案。反事实分析和假设分析可用于测试不同场景，并预测客户对不同解决方案的反应，从而产生更有效的个性化解决方案。
-   **更快的决策：** 机器学习可以实时处理数据并提供见解，帮助决策者更快、更明智地做出决策。反事实分析和假设分析可用于测试不同场景，并预测不同决策对客户流失的影响，使决策者能够更快地做出更明智的决策。
-   **成本效益：** 所提出的解决方案可以自动化许多原本需要人类分析师完成的任务，从而为公司节省成本。通过自动化数据分析、报告生成等常规任务，人类分析师可以专注于需要人类直觉和专业知识的更复杂任务。
-   **可扩展性：** 该解决方案可以处理大量数据，并随着客户群的增长而扩展。这对于正在经历快速增长、需要能够跟上其不断扩大的客户群的解决方案的公司尤其有用。
-   **持续改进：** 该解决方案可以从新数据中学习，并持续改进其预测和建议。随着决策智能系统变得更加复杂和准确，随着时间的推移，这可以带来更好的结果。
-   **收入增长：** 留住现有客户有助于收入增长，因为满意的客户更有可能购买额外的产品和服务。

**竞争优势：** 成功预测并缓解客户流失，能使公司在竞争中脱颖而出。通过提供卓越的客户体验并满足客户需求，可以建立忠诚的客户群，并在市场中占据优势地位。

### 数据集详情

该客户流失数据集涉及一家假设的电信企业，该企业在第三季度于加州运营，涵盖了 7043 名客户的信息，这些客户要么已流失、要么已留存、要么是新获取的公司服务用户。该数据集还进一步包含了每位客户的各种关键人口统计变量。

#### 人口统计信息

- `CustomerID`：用于标识每位客户的唯一 ID。
- `Gender`：客户性别：男、女。
- `Senior Citizen`：表示客户是否年满 65 岁：是、否。
- `Partner`：表示客户是否有伴侣：是、否。
- `Dependents`：表示客户是否与受抚养人同住：是、否。受抚养人可能包括子女、父母、祖父母等。

#### 服务信息

- `Tenure in Months`：表示截至季度末，客户使用该公司服务的总月数。
- `Phone Service`：表示客户是否订阅了该公司的家庭电话服务：是、否。
- `Multiple Lines`：表示客户是否订阅了该公司的多条电话线路：是、否。
- `Internet Service`：表示客户是否订阅了该公司的互联网服务：否、DSL、光纤、有线电视。
- `Online Security`：表示客户是否订阅了该公司提供的额外在线安全服务：是、否。
- `Online Backup`：表示客户是否订阅了该公司提供的额外在线备份服务：是、否。
- `Device Protection Plan`：表示客户是否为其互联网设备订阅了该公司提供的额外设备保护计划：是、否。
- `Premium Tech Support`：表示客户是否订阅了该公司提供的、等待时间更短的额外技术支持计划：是、否。
- `Streaming TV`：表示客户是否使用其互联网服务从第三方提供商处流式传输电视节目：是、否。该公司不对此服务收取额外费用。
- `Streaming Movies`：表示客户是否使用其互联网服务从第三方提供商处流式传输电影：是、否。该公司不对此服务收取额外费用。
- `Contract`：表示客户当前的合同类型：按月、一年、两年。
- `Paperless Billing`：表示客户是否选择了无纸化账单：是、否。
- `Payment Method`：表示客户支付账单的方式：银行扣款、信用卡、邮寄支票。
- `Monthly Charge`：表示客户当前每月为该公司所有服务支付的总费用。
- `Total Charges`：表示截至季度末计算得出的客户总费用。

#### 流失状态

- `Churn Label`：是 = 该客户本季度离开了公司。否 = 该客户仍留在公司。这与流失值直接相关。

**来源：** [`https://community.ibm.com/community/user/businessanalytics/blogs/steven-macko/2019/07/11/telco-customer-churn-1113`](https://community.ibm.com/community/user/businessanalytics/blogs/steven-macko/2019/07/11/telco-customer-churn-1113)

### 阶段 1：AI 模型创建

本活动的目标是构建一个能够合理预测客户流失的机器学习模型。尽管存在高级数据处理技术和机器学习算法，但我们将使用那些能给出满意结果的方法，以确保我们将更多精力投入到目标上，即构建一个决策智能系统。

#### 步骤 1：导入所需库

```python
import dice_ml
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
```

#### 步骤 2：加载电信客户流失数据集

```python
churn_data_all = pd.read_csv('telco_churn.csv')
```

#### 步骤 3：检查数据特征，查看数据中是否存在任何差异（例如，缺失值、错误的数据类型等）

```python
churn_data_all.info()
```

```
[Out]:
RangeIndex: 7043 entries, 0 to 7042
Data columns (total 21 columns):
##### Column            Non-Null Count  Dtype
---  ------            --------------  -----
0   customerID        7043 non-null   object
1   gender            7043 non-null   object
2   SeniorCitizen     7043 non-null   int64
3   Partner           7043 non-null   object
4   Dependents        7043 non-null   object
5   tenure            7043 non-null   int64
6   PhoneService      7043 non-null   object
7   MultipleLines     7043 non-null   object
8   InternetService   7043 non-null   object
9   OnlineSecurity    7043 non-null   object
10  OnlineBackup      7043 non-null   object
11  DeviceProtection  7043 non-null   object
12  TechSupport       7043 non-null   object
13  StreamingTV       7043 non-null   object
14  StreamingMovies   7043 non-null   object
15  Contract          7043 non-null   object
16  PaperlessBilling  7043 non-null   object
17  PaymentMethod     7043 non-null   object
18  MonthlyCharges    7043 non-null   float64
19  TotalCharges      7032 non-null   float64
20  Churn             7043 non-null   object
dtypes: float64(2), int64(2), object(17)
memory usage: 1.1+ MB
```

#### 步骤 4：数据预处理

除 `TotalCharges` 列外，所有列均无空值。`TotalCharges` 列需要采用适当的缺失值处理策略。让我们查看该列存在空值的行。

| customerID | gender | SeniorCitizen | Partner | Dependents | tenure | PhoneService | MultipleLines | TotalCharges | Churn |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4472-LVYGI | Female | 0 | Yes | Yes | 0 | No | No phone service |   | No |
| 3115-CZMZD | Male | 0 | No | Yes | 0 | Yes | No |   | No |
| 5709-LVOEQ | Female | 0 | Yes | Yes | 0 | Yes | No |   | No |
| 4367-NUYAO | Male | 0 | Yes | Yes | 0 | Yes | Yes |   | No |
| 1371-DWPAZ | Female | 0 | Yes | Yes | 0 | No | No phone service |   | No |
| 7644-OMVMY | Male | 0 | Yes | Yes | 0 | Yes | No |   | No |
| 3213-VVOLG | Male | 0 | Yes | Yes | 0 | Yes | Yes |   | No |
| 2520-SGTTA | Female | 0 | Yes | Yes | 0 | Yes | No |   | No |
| 2923-ARZLG | Male | 0 | Yes | Yes | 0 | Yes | No |   | No |
| 4075-WKNIU | Female | 0 | Yes | Yes | 0 | Yes | Yes |   | No |
| 2775-SEFEE | Male | 0 | No | Yes | 0 | Yes | Yes |   | No |

```python
churn_data_all[churn_data_all['TotalCharges'].isnull()].head()
```

（上表中未显示部分列，以适应可视窗口。）

从上表我们发现，当使用期限（`tenure`）为 0 时，`TotalCharges` 列为空。这意味着那些刚加入、使用该电信运营商服务甚至未满 1 个月的客户，其 `TotalCharges` 将为空。让我们将空值转换为 0，因为这是最合适的值。

```python
churn_data_all['TotalCharges'] = np.where(churn_data_all['TotalCharges'].isnull(), 0, churn_data_all['TotalCharges'])
```

虽然 `SeniorCitizen` 列包含二进制值，但其编码方式与其他属性不同。因此，我们将其转换为符合所有变量标准表示的格式。

```python
churn_data_all['SeniorCitizen'] = np.where(churn_data_all.SeniorCitizen == 1,"Yes","No")
```

我们根据列的类型创建一个列名列表，并使用该列表。

```markdown
# 数据预处理与特征工程

## 数据准备

```python
[In]:
all_columns = [x for x in churn_data_all.drop('customerID', axis = 1).columns]
id_column = ['customerID']
target_column = ['Churn']
categorical_columns = [y for y in churn_data_all.drop('customerID', axis = 1).select_dtypes(include = [object]).columns]
numeric_columns = [z for z in all_columns if z not in categorical_columns]
```

## 分类列分析

```python
[In]:
get_dummies = []
label_encoding = []
for i in categorical_columns:
    print('Column Name:', i, ', Unique Value Counts:', len(churn_data_all[i].unique()), ', Values:', churn_data_all[i].unique())
    if len(churn_data_all[i].unique()) > 2:
        get_dummies.append(i)
    else:
        label_encoding.append(i)
```

[Out]:
```
Column Name: gender , Unique Value Counts: 2 , Values: ['Female' 'Male']
Column Name: SeniorCitizen , Unique Value Counts: 2 , Values: ['No' 'Yes']
Column Name: Partner , Unique Value Counts: 2 , Values: ['Yes' 'No']
Column Name: Dependents , Unique Value Counts: 2 , Values: ['No' 'Yes']
Column Name: PhoneService , Unique Value Counts: 2 , Values: ['No' 'Yes']
Column Name: MultipleLines , Unique Value Counts: 3 , Values: ['No phone service' 'No' 'Yes']
Column Name: InternetService , Unique Value Counts: 3 , Values: ['DSL' 'Fiber optic' 'No']
Column Name: OnlineSecurity , Unique Value Counts: 3 , Values: ['No' 'Yes' 'No internet service']
Column Name: OnlineBackup , Unique Value Counts: 3 , Values: ['Yes' 'No' 'No internet service']
Column Name: DeviceProtection , Unique Value Counts: 3 , Values: ['No' 'Yes' 'No internet service']
Column Name: TechSupport , Unique Value Counts: 3 , Values: ['No' 'Yes' 'No internet service']
Column Name: StreamingTV , Unique Value Counts: 3 , Values: ['No' 'Yes' 'No internet service']
Column Name: StreamingMovies , Unique Value Counts: 3 , Values: ['No' 'Yes' 'No internet service']
Column Name: Contract , Unique Value Counts: 3 , Values: ['Month-to-month' 'One year' 'Two year']
Column Name: PaperlessBilling , Unique Value Counts: 2 , Values: ['Yes' 'No']
Column Name: PaymentMethod , Unique Value Counts: 4 , Values: ['Electronic check' 'Mailed check' 'Bank transfer (automatic)'
 'Credit card (automatic)']
Column Name: Churn , Unique Value Counts: 2 , Values: ['No' 'Yes']
```

我们可以看到，某些分类列有两个唯一值，而另一些则有两个以上。为了应用合适的技术，我们对这些列进行了拆分。

## 特征编码

### 虚拟变量创建

我们对具有两个以上唯一值的列应用虚拟变量创建技术。

```python
[In]:
churn_data_all_dl = pd.get_dummies(churn_data_all, prefix=get_dummies, columns=get_dummies)
```

### 标签编码

我们对具有两个唯一值的列应用标签编码技术，并保存映射关系。

```python
mappings = {}
for col in label_encoding:
    le = LabelEncoder()
    churn_data_all_dl[col] = le.fit_transform(churn_data_all_dl[col])
    mappings[col] = dict(zip(le.classes_,range(len(le.classes_))))
mappings
```

[Out]:
```
{'gender': {'Female': 0, 'Male': 1},
 'SeniorCitizen': {'No': 0, 'Yes': 1},
 'Partner': {'No': 0, 'Yes': 1},
 'Dependents': {'No': 0, 'Yes': 1},
 'PhoneService': {'No': 0, 'Yes': 1},
 'PaperlessBilling': {'No': 0, 'Yes': 1},
 'Churn': {'No': 0, 'Yes': 1}}
```

## 第 5 步：建模

我们将数据集拆分为训练集和推理集。

```python
[In]:
X = churn_data_all_dl.drop(['customerID', 'Churn'], axis=1)
y = churn_data_all_dl['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify=y, random_state = 0)
```

### 超参数调优

超参数调优的过程涉及为学习算法找到最佳的超参数组合，这可用于优化其在任何给定数据集上的性能。通过最小化预先指定的损失函数，所选超参数可以减少误差并改善模型的结果。

运行以下代码行有助于确定我们机器学习算法的理想超参数：

```python
[In]:
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 2100, num = 6)]
feature_name = list(X_test.columns)
max_depth = [int(x) for x in np.linspace(10, 100, num = 5)]
max_depth.append(None)
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4, 6, 8, 10]
random_grid = {'n_estimators':n_estimators,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}
print(random_grid)
rf = RandomForestClassifier()
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 10, cv = 2, verbose=2, random_state=42, n_jobs = -1)
rf_random.fit(X_train[feature_name], y_train)
print(rf_random.best_params_)
```

[Out]:
```
{'n_estimators': [100, 500, 900, 1300, 1700, 2100], 'max_depth': [10, 32, 55, 77, 100, None], 'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4, 6, 8, 10]}
```

我们对 10 个候选参数中的每一个进行 2 折交叉验证，总共拟合 20 次。

```
{'n_estimators': 100, 'min_samples_split': 2, 'min_samples_leaf': 8, 'max_depth': 10}
```

我们看到，对于随机森林模型（本例中选用的算法），最佳的超参数组合如下：

*   `n_estimators` = 100
*   `min_samples_split` = 2
*   `min_samples_leaf` = 8
*   `max_depth` = 10

我们使用上述超参数来构建随机森林模型。

```python
[In]:
feature_name = list(X_test.columns)
churn_classifier=RandomForestClassifier(n_estimators=100,min_samples_split=2,min_samples_leaf=8,max_depth=10)
churn_classifier.fit(X_train[feature_name],y_train)
```

[Out]:
```
RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                       criterion='gini', max_depth=10, max_features='auto',
                       max_leaf_nodes=None, max_samples=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=8, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=100,
                       n_jobs=None, oob_score=False, random_state=None,
                       verbose=0, warm_start=False)
```

现在，我们使用训练好的模型对测试数据进行预测，并保存预测结果及其概率。

```python
[In]:
pred_df = X_test.copy()
pred_df['Churn'] = y_test
pred_df['pred'] = churn_classifier.predict(X_test[feature_name])
prediction_of_probability = churn_classifier.predict_proba(X_test[feature_name])
pred_df['prob_0'] = prediction_of_probability[:,0]
pred_df['prob_1'] = prediction_of_probability[:,1]
```

我们评估模型的性能。

```python
[In]:
print("Accuracy: ", (accuracy_score(pred_df['Churn'], pred_df['pred']))*100)
print("Precision: ", (precision_score(pred_df['Churn'], pred_df['pred']))*100)
print("Recall: ", (recall_score(pred_df['Churn'], pred_df['pred']))*100)
print("F1 Score: ", (f1_score(pred_df['Churn'], pred_df['pred']))*100)
```

[Out]:
```
Accuracy:  80.48261178140525
Precision:  67.1280276816609
Recall:  51.87165775401069
F1 Score:  58.521870286576174
```

如前所述，我们的目标是获得令人满意的模型性能以进行演示。有多种方法可以获得更好的模型性能，以下是一些主要方法：

*   **特征工程：** 这涉及创建新特征或转换现有特征，使其对模型更具信息量。
*   **正则化：** 这涉及在损失函数中添加惩罚项，以防止模型对训练数据过拟合。
*   **集成方法：** 这涉及组合多个模型以创建更准确的预测。
*   **数据增强：** 这涉及通过使用旋转、翻转或向数据添加噪声等技术生成新样本来增加训练数据集的大小。
*   **过采样/欠采样：** 这涉及通过增加少数类中的实例数量（过采样）或减少多数类中的实例数量（欠采样）来平衡不平衡数据集。
*   **迁移学习：** 这涉及利用预训练模型来解决类似问题，或使用预训练模型作为特征提取器。
*   **梯度裁剪：** 在训练过程中对梯度进行裁剪，以防止其变得过大或过小。
*   **早停法：** 当模型在验证集上的性能停止提升时，停止训练过程。

现在我们已经准备好了模型和预测结果，接下来让我们开始使用它们进行决策。

## 阶段 2：生成反事实

我们将结合使用反事实分析和假设分析来驱动决策智能。让我们看看它们的含义以及如何使用。

### 反事实分析

反事实指的是一种分析类型，涉及识别对输入实例所需的最小更改集，以使机器学习模型的输出更改为期望的结果。简单来说，反事实是假设性的输入示例，这些示例会导致机器学习模型产生不同的输出。

反事实分析可用于解释机器学习模型的行为，并提供关于如何改进模型以及可以采取哪些行动以产生有利结果的见解。它还可用于各种应用，例如公平性分析，其中实现期望结果所需的最小更改集可用于识别模型中潜在的偏差来源。此外，反事实分析可用于因果推断，其中最小更改集可用于估计处理对结果的因果效应。

我们将使用 `DiCEML` Python 包来生成反事实。

### DiCEML

`DiCEML`（通过混合整数线性规划生成多样化反事实解释）是一个框架，能够为机器学习模型生成反事实示例。`DiCEML` 采用混合整数线性规划来找到将模型输出更改为期望结果所需的对输入实例的最小更改集。这是一种生成多样化反事实解释的最先进方法，它考虑了输入数据的约束和特征，可用于各种应用，例如可解释人工智能、公平性分析和因果推断。

### 步骤 6：构建反事实解释器对象

```python
[In]:
def initialize_counterfactuals(
    train_df, model, feature_list, continuous_features, target, model_type
):
    """
    为给定的输入模型初始化反事实解释器对象，同时初始化特征范围字典
    （两者都用于计算局部可解释性中的反事实）

    参数:
    train_df (dataframe) : 训练数据框
    model (object) : 输入模型（待解释）
    feature_list (list) : 模型中使用的特征列表
    continuous_features (list) : 连续特征列表
    target (str) : 目标列名称
    model_type (str) : 分类或回归

    返回:
    explainer (object) : 反事实解释器对象
    feature_range (dict) : 包含连续特征允许范围的字典
    """
    df_model = train_df[feature_list + [target]]

    # 将小数四舍五入到 4 位
    df_model[continuous_features] = (
        df_model[continuous_features].astype(float).round(4)
    )  # .astype(str)
    print("连续特征", continuous_features)

    # dice 数据初始化
    data_dice = dice_ml.Data(
        dataframe=df_model,  # 用于扰动策略
        continuous_features=continuous_features,
        outcome_name=target,
    )

    # dice 模型初始化
    if model_type == "regression" or model_type == "time series":
        model_dice = dice_ml.Model(model=model, backend="sklearn", model_type="regressor")
    else:
        model_dice = dice_ml.Model(model=model, backend="sklearn")

    # 将模型和数据整合在一起
    explainer = dice_ml.Dice(data_dice, model_dice, method="random")
```

### 阶段 3：假设分析

#### 步骤 7：使用解释器对象生成保留客户的可行策略

现在，我们有了一个名为 `explainer` 的反事实对象，它可以用来生成反事实解释，即：对输入进行哪些更改才能获得期望的输出。在我们的案例中，它将是：“对于那些可能流失的客户，可以更改哪些方面（服务、账户相关、人口统计特征），以便他/她能被保留下来。”让我们通过一个例子来理解这一点。

以下是 `DiCEML` 期望的输入：

-   训练好的机器学习模型。
-   我们想要为其生成反事实的输入实例。我们将从被预测为可能流失的用户池中选取一个样本用户。
-   要变化的特征。对于这个例子，我们将看看提供诸如无纸化账单、在线安全、在线备份、设备保护、技术支持、用于流媒体电视或电影的互联网服务、一年或两年合同等服务，或者更改月度套餐，或者组合这些特征，是否能改变客户的想法，让他们计划继续留在电信运营商那里。
-   期望的输出或我们想要为其生成反事实的目标类别。在这种情况下，它将是非流失，即 0（从 1 变为 0）。

具体操作如下：

| PaperlessBilling | MonthlyCharges | OnlineSecurity_Yes | OnlineBackup_Yes | DeviceProtection_Yes | TechSupport_Yes | StreamingTV_Yes | StreamingMovies_Yes | Contract_One year | Contract_Two year |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 101.15 | 0 | 0 | 1 | 0 | 1 | 1 | 0 | 0 |

```python
test_df = X_test.copy()
test_df["Churn"] = y_test
test_df_churners = pred_df[pred_df['Churn'] == 1].sort_values(by=['prob_1'], ascending=False)
test_df_churners_input = test_df_churners.drop(['Churn', 'pred', 'prob_0', 'prob_1'], inplace=False, axis=1)
test_df_churners_input_record = test_df_churners_input[61:62]

features_vary = ['PaperlessBilling', 'MonthlyCharges',
                 'OnlineSecurity_Yes', 'OnlineBackup_Yes',
                 'DeviceProtection_Yes', 'TechSupport_Yes',
                 'StreamingTV_Yes', 'StreamingMovies_Yes',
                 'Contract_One year', 'Contract_Two year']

dice_exp = explainer.generate_counterfactuals(test_df_churners_input_record, total_CFs=4, desired_class="opposite", features_to_vary=features_vary)
dice_exp.visualize_as_dataframe()
```

```
100%|██████████| 1/1 [00:31<00:00, 31.02s/it]
```

```python
test_df_churners_input_record[features_vary]
```

| PaperlessBilling | MonthlyCharges | OnlineSecurity_Yes | OnlineBackup_Yes | DeviceProtection_Yes | TechSupport_Yes | StreamingTV_Yes | StreamingMovies_Yes | Contract_One year | Contract_Two year |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 55.36 | 1 | 0 | 1 | 0 | 1 | 1 | 0 | 0 |
| 0 | 35.86 | 1 | 0 | 1 | 0 | 1 | 1 | 0 | 0 |
| 0 | 22.32 | 1 | 0 | 1 | 0 | 1 | 1 | 0 | 0 |
| 0 | 67.89 | 1 | 0 | 1 | 0 | 1 | 1 | 0 | 0 |

```python
dice_exp.cf_examples_list[0].final_cfs_df[features_vary]
```

从之前的输出中，我们看到解释器模型提出了四种可能的保留客户的方法。

-   将月费从 101.15 降至 55.36，并提供在线安全服务。
-   将月费从 101.15 降至 35.86，并提供在线安全服务。
-   将月费从 101.15 降至 22.32，并提供在线安全服务。
-   将月费从 101.15 降至 67.89，并提供在线安全服务。

然而，业务方面可能需要考虑某些约束条件，不能仅仅依赖反事实分析的建议。人们可能希望获取反事实分析的输入，根据业务约束对其进行一些调整，然后检查这些调整后的输入是否能产生期望的输出。这可以通过假设分析来实现。

#### 步骤 8：使用假设分析获取正确的客户保留策略集（基于约束条件）

假设分析是一种用于理解机器学习模型在不同场景或“假设”情况下的行为的技术。它涉及用不同的输入值或场景测试模型，以了解模型在这些情况下的行为。

现在，假设对于给定的反事实建议，我们有一个约束条件：客户的月费降幅不能超过 10%，因为这可能导致巨大的收入损失。因此，对于给定的客户，我们不能将月费降至 91 以下。此外，我们对免费提供服务没有任何限制，那么让我们看看在给定的受约束月费和一系列服务组合下，我们能否留住客户。

**选项：**

-   将月费从 101.7 降至 91；提供在线安全和在线备份服务。

```python
custom_input = test_df_churners_input_record.copy()
custom_input['MonthlyCharges'] = custom_input['MonthlyCharges'].replace(101.15, 91)
custom_input['OnlineBackup_Yes'] = custom_input['OnlineBackup_Yes'].replace(0, 1)
custom_input['OnlineSecurity_Yes'] = custom_input['OnlineSecurity_Yes'].replace(0, 1)

if churn_classifier.predict(custom_input)==0:
    print("Successfully retained the customer")
else:
    print("Sorry, customer could not be retained")

print("\nProbability of Churn:",churn_classifier.predict_proba(custom_input)[0][1].round(2))
```

```
Sorry, customer could not be retained

Probability of Churn: 0.57
```

我们看到，使用给定的输入，我们无法留住客户。让我们尝试另一组输入，

-   将月费从 101.7 降至 91；提供在线安全、在线备份服务，并签订两年合同。

```python
custom_input2 = test_df_churners_input_record.copy()
custom_input2['MonthlyCharges'] = custom_input2['MonthlyCharges'].replace(101.15, 91)
custom_input2['OnlineBackup_Yes'] = custom_input2['OnlineBackup_Yes'].replace(0, 1)
custom_input2['OnlineSecurity_Yes'] = custom_input2['OnlineSecurity_Yes'].replace(0, 1)
custom_input2['Contract_Two year'] = custom_input2['Contract_Two year'].replace(0, 1)

if churn_classifier.predict(custom_input2)==0:
    print("Successfully retained the customer")
else:
    print("Sorry, customer could not be retained")

print("\nProbability of Churn:",churn_classifier.predict_proba(custom_input2)[0][1].round(2))
```

```
Successfully retained the customer

Probability of Churn: 0.47
```

我们可以看到，使用此选项中的给定输入集，我们成功留住了客户，客户的流失概率降至 47%。因此，此选项可以转化为针对该客户的个性化优惠。

反事实分析和“假设”分析可用于检测机器学习模型中的偏差。它涉及分析反事实输出或测试模型对输入数据或模型参数变化的敏感性，以识别模型可能存在偏差的区域。让我们看看如何使用反事实分析 + 假设分析来检测偏差，使用之前相同的记录，并将人口统计特征 `SeniorCitizen` 和 `gender` 添加到要变化的特征矩阵中。

```python
```

test_df_churners_bias_inputs = test_df_churners_input[(test_df_churners_input['SeniorCitizen'] == 1) | (test_df_churners_input['gender'] == 1)]

features_vary_bias = ['SeniorCitizen', 'gender']

dice_exp_bias = explainer.generate_counterfactuals(test_df_churners_bias_inputs[100:130], total_CFs=4, desired_class="opposite", features_to_vary=features_vary_bias)

dice_exp_bias.visualize_as_dataframe()

先前的反事实输出在少数情况下会建议更改性别或年龄（从老年人改为非老年人）以获取有利结果。这表明需要检查模型中是否存在偏差。需要强调的是，这些方法在检测和减轻 AI 系统偏差方面并非万无一失，必须结合多种技术、数据验证、审计以及多样性与包容性实践，才能确保 AI 系统产生公平的结果。

公平性指标，例如差异性影响、机会均等差异、假阳性率均等、均等化几率以及校准度，用于评估 AI 模型是否公平对待不同群体。指标的选择取决于具体的应用场景和目标。差异性影响是衡量公平性的一种指标，它检验 AI 模型的结果是否基于受保护特征对特定群体产生了不成比例的影响，其公式用于计算两个被比较群体之间的差异性影响比率。

让我们计算数据集中`Gender`和`SeniorCitizen`列的差异性影响值。

```python
protected_columns = ["gender", "SeniorCitizen"]

target="pred"

for pro_col in protected_columns:

categories = list(pred_df[pro_col].unique())

selection_rate_list = []

for cat in categories:

selection_rate = pred_df[(pred_df[target]==1)&(pred_df[pro_col]==cat)].shape[0]/pred_df[pred_df[pro_col]==cat].shape[0]

selection_rate_list.append(np.round(selection_rate,3))

disparate_impact = np.round(min(selection_rate_list)/max(selection_rate_list),3)

print("Disparate Impact for",pro_col,":",disparate_impact)
```

```
Disparate Impact for gender : 0.843

Disparate Impact for SeniorCitizen : 0.438
```

`SeniorCitizen`列的差异性影响得分相当低，表明存在针对该特征的偏差，而`Gender`列的得分约为 84%，表明处理非常公平，超过了 80%的阈值。然而，`SeniorCitizen`列约 44%的得分显著低于阈值，表明存在不公平的偏差。重要的是，要结合使用多个公平性指标来得出结论。

## 案例研究 2：手机定价/配置策略

**问题陈述：** 一家手机制造公司希望利用人工智能和机器学习算法的力量来改进其定价策略。该公司拥有一个包含多种手机型号的大型数据库，这些型号具有屏幕尺寸、摄像头质量、处理器速度、电池续航等不同特征。然而，目前缺乏一种基于这些特征来准确高效预测手机价格的方法。该公司正在寻求一种解决方案，能够帮助他们高精度地预测新型号手机的价格，从而优化定价策略，在市场中保持竞争力。

**解决方案：**

AI 团队提出了一种混合解决方案来解决该问题。计划是首先构建一个机器学习模型，该模型输入历史手机详细信息（如屏幕尺寸、摄像头质量、处理器速度、电池续航等）以及价格。模型输出将用于预测手机的潜在价格。团队还提出了一个反事实生成引擎，该引擎可以提供各种场景，在这些场景中，预测结果会根据预期的价格范围发生变化。例如，如果某款手机的价格预测为 X，而制造商希望以比预测价格高出 10%到 15%的价格出售，该引擎将建议更改手机配置以达到预期的价格范围。输入参数的这种变化可以作为产品策略的一部分。第三个组件是一个假设分析解决方案，可以帮助决策者根据某些约束条件调整反事实建议，并检查这些更改是否仍能帮助他们在预期价格范围内销售手机。

AI 团队建议使用所提出的混合解决方案进行手机定价预测具有以下优势：

*   **准确的价格预测：** AI 算法可以分析海量数据，并识别人类可能遗漏的模式。这使得该工具能够基于各种特征更准确地预测手机价格。

*   **实时定价：** 借助 AI 驱动的工具，可以根据市场供需变化实时做出定价决策。这使得手机制造商能够快速响应不断变化的市场状况，并相应调整其定价策略。

*   **提高利润率：** 通过准确预测价格，制造商可以优化其定价策略以最大化利润率。该工具可以识别产品的最佳价格点，同时考虑生产成本、竞争和市场需求等因素。

*   **提高客户满意度：** 通过为产品设定合适的价格，制造商可以提高客户满意度。如果客户认为产品定价公平且准确反映了其价值，他们更有可能购买。

*   **更好的决策：** AI 驱动的工具可以为制造商提供有关客户偏好和购买行为的宝贵见解。这有助于在产品开发、营销和定价策略方面做出明智的决策。

**数据集详情：**

以下是数据集详情：

*   **Product_id：** 每部手机的 ID

*   **Price：** 每部手机的价格

*   **Sale：** 销售数量

*   **weight：** 每部手机的重量

*   **resolution：** 每部手机的分辨率

*   **ppi：** 手机像素密度

*   **cpu core：** 每部手机的 CPU 核心类型

*   **cpu freq：** 每部手机的 CPU 频率

*   **internal mem：** 每部手机的内部存储

*   **ram：** 每部手机的 RAM

*   **RearCam：** 后置摄像头分辨率

*   **Front_Cam：** 前置摄像头分辨率

*   **battery：** 电池容量（单位：mA）

*   **thickness：** 手机厚度

来源：[`https://www.kaggle.com/datasets/mohannapd/mobile-price-prediction`](https://www.kaggle.com/datasets/mohannapd/mobile-price-prediction)

### 阶段 1：AI 模型创建

本活动的目标是构建一个能够合理预测手机价格的机器学习模型。尽管有先进的数据处理技术和机器学习算法可用，但我们将使用那些能给出满意结果的方法，以确保我们将更多精力投入到目标上，即构建一个决策智能系统。

#### 步骤 1：导入所需库

```python
import pandas as pd

import dice_ml

from dice_ml import Dice

from sklearn.metrics import r2_score

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split
```

#### 步骤 2：加载数据集

| Product_id | Price | Sale | weight | resolution | ppi | cpu core | cpu freq | internal mem | ram | RearCam | Front_Cam | battery | thickness |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 203 | 2357 | 10 | 135 | 5.2 | 424 | 8 | 1.35 | 16 | 3 | 13 | 8 | 2610 | 7.4 |
| 880 | 1749 | 10 | 125 | 4 | 233 | 2 | 1.3 | 4 | 1 | 3.15 | 0 | 1700 | 9.9 |
| 40 | 1916 | 10 | 110 | 4.7 | 312 | 4 | 1.2 | 8 | 1.5 | 13 | 5 | 2000 | 7.6 |
| 99 | 1315 | 11 | 118.5 | 4 | 233 | 2 | 1.3 | 4 | 0.512 | 3.15 | 0 | 1400 | 11 |
| 880 | 1749 | 11 | 125 | 4 | 233 | 2 | 1.3 | 4 | 1 | 3.15 | 0 | 1700 | 9.9 |

```python
all_data = pd.read_csv('cellphone_price_data.csv')

all_data.head()
```

#### 步骤 3：检查数据特征，查看数据中是否存在任何差异（例如，缺失值、错误的数据类型等）

```python
all_data.info()
```

```
RangeIndex: 161 entries, 0 to 160

Data columns (total 14 columns):

##### Column        Non-Null Count  Dtype

---  ------        --------------  -----

0   Product_id    161 non-null    int64

1   Price         161 non-null    int64

2   Sale          161 non-null    int64

3   weight        161 non-null    float64

4   resolution    161 non-null    float64

5   ppi           161 non-null    int64

6   cpu core      161 non-null    int64

7   cpu freq      161 non-null    float64

8   internal mem  161 non-null    float64

9   ram           161 non-null    float64

10  RearCam       161 non-null    float64

11  Front_Cam     161 non-null    float64

12  battery       161 non-null    int64

13  thickness     161 non-null    float64

dtypes: float64(8), int64(6)

memory usage: 17.7 KB
```

从之前的详细信息可以看出，数据集没有任何缺失值或与数据相关的问题。因此，这里不需要进行数据预处理，我们可以直接进入建模阶段。

#### 步骤 4：建模

我们将数据集拆分为训练集和推理集。

```python
x=all_data.drop(['Price', 'Product_id'],axis=1)

y=all_data['Price']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
```

现在，我们构建一个随机森林回归模型，并使用它对测试数据进行预测，保存并评估这些预测结果。

```python
rf = RandomForestRegressor()

rf.fit(x_train,y_train)
```

```
RandomForestRegressor()
```

```python
predictions = rf.predict(x_test)

print('R2 score for Training Data: ', rf.score(x_train, y_train))

print('R2 score for Inference Data: ', r2_score(y_test,predictions))
```

```
R2 score for Training Data:  0.9945696702046988

R2 score for Inference Data:  0.9678002631345382
```

```python
x_test['predicted_price'] = predictions

x_train.info()
```

```
Int64Index: 128 entries, 80 to 47

Data columns (total 12 columns):

##### Column        Non-Null Count  Dtype

---  ------        --------------  -----

0   Sale          128 non-null    int64

1   weight        128 non-null    float64

2   resolution    128 non-null    float64

3   ppi           128 non-null    int64

4   cpu core      128 non-null    int64

5   cpu freq      128 non-null    float64

6   internal mem  128 non-null    float64

7   ram           128 non-null    float64

8   RearCam       128 non-null    float64

9   Front_Cam     128 non-null    float64

10  battery       128 non-null    int64

11  thickness     128 non-null    float64

dtypes: float64(8), int64(4)

memory usage: 13.0 KB
```

```python
x_train['Price'] = all_data['Price']
```

### 阶段 2：生成反事实

我们将结合使用反事实分析和假设分析来驱动决策智能。与分类用例不同，由于这是一个回归用例，我们将使用手机的预期价格范围作为期望结果。

```python
d_mobile = dice_ml.Data(dataframe=x_train, continuous_features=['Sale', 'weight', 'resolution', 'ppi', 'cpu core', 'cpu freq', 'internal mem', 'ram', 'RearCam', 'Front_Cam', 'battery', 'thickness'], outcome_name='Price')

m_mobile = dice_ml.Model(model=rf, backend="sklearn", model_type='regressor')

explainer_mobile = Dice(d_mobile, m_mobile, method="random")
```

### 阶段 3：假设分析

#### 步骤 5：使用解释器对象生成可能的定价策略

现在我们有了 `explainer` 反事实对象，它可以用来生成反事实解释，即，可以对输入进行哪些更改以获得期望的输出。在我们的案例中，它将是：“对于给定的预测价格，可以更改什么（手机配置，如屏幕尺寸、分辨率等），以便手机能以更高的价格出售。” 让我们通过一个例子来理解这一点。

| Sale | weight | resolution | ppi | cpu core | cpu freq | internal mem | ram | RearCam | Front_Cam | battery | thickness |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 302 | 149 | 5.5 | 534 | 8 | 1.6 | 32 | 3 | 16 | 8 | 3000 | 7 |

```python
input_data = x_test.drop(['predicted_price'], axis=1)

input_record = input_data[0:1]

input_record.head()
```

| Sale | weight | resolution | ppi | cpu core | cpu freq | internal mem | ram | RearCam | Front_Cam | battery | thickness | predicted_price |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 302 | 149 | 5.5 | 534 | 8 | 1.6 | 32 | 3 | 16 | 8 | 3000 | 7 | 2941.57 |

```python
x_test[0:1].head()
```

对于给定的记录，我们看到预测价格约为 2.9K。假设制造商希望以比预测价格高 10% 到 15% 的价格出售手机，则可以使用解释器来查看能够实现该目标的手机配置建议。

| Sale | weight | resolution | ppi | cpu core | cpu freq | internal mem | ram | RearCam | Front_Cam | battery | thickness | Price |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 302 | 149 | 5.5 | 534 | 8 | 1.6 | 32 | 3 | 16 | 8 | 3000 | 7 | 2942 |

```python
predicted_price = x_test[0:1].predicted_price

expected_price_range = [round(float(predicted_price*1.10), 0), round(float(predicted_price*1.15), 0)]

random_mobile = explainer_mobile.generate_counterfactuals(input_record, total_CFs=2,

desired_range=expected_price_range)

random_mobile.visualize_as_dataframe(show_only_changes=True)
```

```
100%|██████████| 1/1 [00:00<00:00,  1.40it/s]

Query instance (original outcome : 2942)
```

```
Diverse Counterfactual set (new outcome: [3236.0, 3383.0])



| `Sale` | `weight` | `resolution` | `ppi` | `cpu core` | `cpu freq` | `internal mem` | `ram` | `RearCam` | `Front_Cam` | `battery` | `thickness` | `Price` |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| - | - | - | - | - | 1.6 | 84.8 | 5.7 | - | - | - | - | 3278.169922 |
| - | - | - | - | - | 1.6 | - | 5.2 | - | - | 5116 | - | 3305.25 |
```

从之前的输出中，我们看到解释模型提出了两种可能的方式，以在预期的价格范围内销售该手机。

1. 将`internal mem`从 32 GB 改为 84.8 GB，并将`ram`从 3 GB 增加到 5.7 GB。
2. 将`ram`从 3 GB 改为 5.2 GB，并将`battery`容量从 3000 mA 增加到 5116 mA。

然而，企业可能需要考虑某些限制条件，不能仅仅依赖反事实分析的建议。人们可能希望采纳反事实分析的输入，根据业务限制对其进行一些调整，然后检查这些调整后的输入是否能产生预期的输出。这可以通过假设分析来实现。

### 步骤 6：使用假设分析为给定手机找到正确的定价策略

现在，假设对于给定的反事实建议，我们有一个限制：诸如`ram`之类的昂贵部件不能增加，但诸如`internal mem`之类的较便宜部件可以增加到当前的两到四倍，而`battery`最多可以增加 50%。因此，对于给定的手机，我们不能增加`ram`。让我们检查哪种配置可以帮助我们获得期望的价格。

以下是选项：

1. 将`internal mem`增加两倍，并将`battery`增加 50%。

```python
expected_internal_memory = int(input_record['internal mem']*2)

expected_battery = int(input_record['battery']*1.5)

custom_inputs = input_record.copy()

custom_inputs['internal mem'] = custom_inputs['internal mem'].replace(int(input_record['internal mem']), expected_internal_memory)

custom_inputs['battery'] = custom_inputs['battery'].replace(int(input_record['battery']), expected_battery)

print('基于更新配置的新价格: ', rf.predict(custom_inputs))

if rf.predict(custom_inputs)>=expected_price_range[0]:

print("该配置有助于达到最低预期价格")

else:

print("该配置无助于达到最低预期价格",expected_price_range[0])
```

```
基于更新配置的新价格:  [3043.53]

该配置无助于达到最低预期价格 3236.0
```

我们看到，将`internal mem`增加 2 倍并将`battery`容量增加 50% 并不能帮助达到预期的价格范围。让我们看看是否可以通过不同的组合来实现目标。

2. 将`internal mem`增加四倍，并将`battery`增加 50%。

```python
expected_internal_memory2 = int(input_record['internal mem']*4)

expected_battery2 = int(input_record['battery']*1.5)

custom_inputs2 = input_record.copy()

custom_inputs2['internal mem'] = custom_inputs2['internal mem'].replace(int(input_record['internal mem']), expected_internal_memory2)

custom_inputs2['battery'] = custom_inputs2['battery'].replace(int(input_record['battery']), expected_battery2)

print('基于更新配置的新价格: ', rf.predict(custom_inputs2))

if rf.predict(custom_inputs2)>=expected_price_range[0]:

print("该配置有助于达到最低预期价格")

else:

print("该配置无助于达到最低预期价格",expected_price_range[0])
```

```
基于更新配置的新价格:  [3247.54]

该配置有助于达到最低预期价格 3236.0
```

我们可以看到，第二个选项提供了正确的配置，以实现将手机价格至少提高预测价格 10% 的目标。

## 结论

在本章中，我们了解了如何通过应用 AI/ML 和其他先进技术（反事实分析和假设分析）来支持决策。我们看到了组织如何不仅能够预测哪些客户会流失，还可以借助反事实分析和假设分析主动采取措施来避免流失。我们还看到了机器学习预测、反事实分析和假设分析的结合如何帮助手机制造商设计其产品配置（屏幕尺寸、`ram`等），并在预测价格的基础上获得期望的价格。通过这些用例，我们涵盖了分类和回归问题的决策智能。

# 索引

## A

- 问责制
- 人工智能驱动工具
- 人工智能/机器学习模型
- 人工智能驱动决策
- 人工智能
- 先进技术
- 挑战
- 定制化
- 数据分析
- 目标
- 集成
- 关键伦理考量
    - 问责制
    - 偏见与公平性
    - 人类价值观与权利
    - 隐私与安全
    - 透明度与可解释性
- 机器学习算法
- 风险管理
- 透明度
- 类型
- 多种应用场景
- 归因模型包
- 基于权威的决策
- 自主系统

## B

- 贝叶斯推断
- 贝叶斯网络
- 偏见检测审计
    - 数据分析
    - 专家评审
    - 测试
- 偏见检测工具
    - 数据分析技术
    - 公平性指标
    - 人工审查
    - 机器学习算法
    - 统计分析技术
- 有偏见的决策
- 偏见
    - 算法偏见
    - 自动化偏见
    - 认知偏见
    - 确认偏见
    - 群体归因偏见
    - 对组织的影响
    - 抽样偏见
    - 选择偏见
- 商业分析
- 商业工具

## C

- 客户流失
- 流失预测
- 认知偏见效应
- 认知偏见
- 协作决策
- 计算机系统
- 基于共识的决策
- 反事实分析
- 反事实
- 客户关系管理系统

## D

- 数据分析
- 数据收集
- 数据驱动决策
- 数据准备
- 决策分析
- 决策智能
    - 问责制
    - 行动
    - 采用障碍
    - 敏捷性与灵活性
    - 人工智能生命周期
    - 人工智能模型到商业工具的映射
    - 数据
    - 人工智能预测到商业工具
    - 应用
    - 偏见
    - 构建/集成应用与业务流程
    - 与业务流程集成的挑战
    - 与现有系统集成
    - 认知偏见
    - 竞争优势
    - 复杂与不确定环境
    - 复杂性
    - 持续改进
    - 成本效益
    - 定制化
    - 数据质量问题
    - 决策对数据的依赖
    - 开发、应用
    - 提升准确性
    - 伦理问题
    - 历史
    - 人类专业知识
    - 人在回路中
    - 改进决策
    - 提升决策质量
    - 改进风险管理
    - 提高效率
    - 行业
    - 信息过载
    - 集成挑战
    - 机器智能
    - 方法论
    - 组织
    - 过度依赖技术
    - 个性化
    - 隐私问题
    - 宝洁公司
    - 实时决策
    - 风险管理
    - 可扩展性
    - 步骤、业务流程
    - 技术专长
    - 技术
    - 透明度
    - 联合包裹服务公司
    - 用户友好界面
- 决策智能需求
    - 人工智能预测
    - 人工智能项目
    - 方法
    - 审批机制/组织对齐
    - 人工智能决策
    - 定义明确指标
    - 框架
    - 指南/路径
    - 误区
    - 组织挑战
    - 规划
    - 投资回报率
    - 价值
    - 每次决策的价值
- 决策智能方法论
    - 条件
    - 线性模型
    - 非线性模型
    - 非线性问题
    - 非线性解决方案
    - 生产优化
    - `PuLP` 库
- 决策
    - 备选方案
    - 授权服务中心
    - 组成部分
    - 约束条件
    - 定义
    - 评估
    - 方法论分类
    - 人类与机器
    - 概述
    - 问题定义
    - 问题陈述
    - 战略、战术与运营
    - 结构化方法
    - 类型
- 决策理论
- 义务论
- 描述性分析
- 诊断性分析
- 差异性影响

## E

- 伦理人工智能
- 基于伦理的决策类型
- 基于伦理/道德的决策
    - 优势
    - 劣势
- 基于经验的决策
    - 优势
    - 劣势
    - 基于证据
    - 基于直觉
    - 基于规则
- 解释性反事实对象

## F

- 公平性指标
- 通过人工干预的反馈
- 首个可接受匹配
- 模糊逻辑

## G

- 博弈论
- 群体决策

## H

- 人类决策史
- 人在回路系统
- 人机决策
- 仅人类决策
- 超参数调优
- 假设的电信业务

## I, J, K

- 个体决策
- Instagram
- 基于指令/规则的系统
- 基于直觉的决策

## L

- 线性数学模型
- 线性模型
- 线性规划

## M

- 机器学习算法
- 机器学习模型
- 营销渠道
- 马尔可夫链模型
- 数学与概率模型
- 数学决策系统
- 数学模型
- 移动定价预测
- 模型部署
- 模型训练
- 监控与优化
- 蒙特卡洛模拟
- 多准则决策分析
- 跨国公司
- 多准则决策
    - 优势
    - 单准则决策

## N

- 非线性数学模型
- 非线性规划

## O

- 运营决策
- 运筹学
- 最优解函数
- 优化或最大化决策

## P, Q

- 个人成就感
- 预测性分析
- 规范性分析
- 概率决策
- 概率模型
- 基于概率的决策系统
- 生产优化
- Python 库
- Python 包
- Python 编程语言

## R

- 随机决策
    - 优势
    - 劣势
- 随机森林模型
- 投资回报率
- 风险指数
- 基于规则的决策
- 基于规则的人机决策

## S

- 满意的模型性能
- 单准则决策
- 分层 K 折交叉验证
- 监督学习

## T

- 战术决策
- 基于阈值的决策
- `TotalCharges`

## U

- 无监督学习
- 用户友好界面
- 用户界面
- 功利主义

## V

- 美德伦理学
- 基于投票的决策

## W, X, Y, Z

- 假设分析
- 劳动力约束