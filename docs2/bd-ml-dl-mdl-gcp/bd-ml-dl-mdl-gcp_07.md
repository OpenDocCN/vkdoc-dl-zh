# 第七部分：Google Cloud Platform 上的高级分析/机器学习

# 38. Google BigQuery

BigQuery 是一款由 Google 管理的高度可扩展、快速且针对数据分析进行了优化的数据仓库产品，其产品提供包括基本的内置机器学习功能。它也是 Google 多款无服务器产品之一。这意味着您不需要物理管理基础设施资产和相关的责任/成本。它仅用于解决业务用例，并且以高性能的方式运行。

BigQuery 适用于存储和分析结构化数据。结构化数据的概念是它必须有一个模式来描述数据集的列或字段。CSV 或 JSON 文件是结构化数据格式的示例。BigQuery 与其他关系数据库的不同之处在于，它可以存储其他字段（或列）的集合作为记录类型，并且行中的特定字段可以具有多个值。这些特性使得 BigQuery 在存储没有关系数据库扁平行约束的数据集时更加灵活。

与关系数据库类似，BigQuery 将行组织到 *表格* 中，并使用熟悉的结构化查询语言（SQL）进行数据库访问。然而，无法通过运行 SQL 更新语句来更新表格中的单个行。表格只能追加或完全重写。同时，BigQuery 中的多个表格组织成 *数据集*。

当在 BigQuery 中执行查询时，它将在数千个核心上并行运行。这一特性极大地加速了查询执行的性能，从而加快了从数据中获取洞察的速度。这种大规模并行执行的能力是个人、公司和机构选择 BigQuery 作为其首选数据仓库的主要原因之一。

此外，BigQueryML 是一个强大的平台，可以在 BigQuery 内部构建机器学习模型。这些模型利用自动化的特征工程和超参数优化，并根据底层数据集的变化自动更新。这一特性极为强大，降低了商业智能和分析人员利用机器学习进行业务预测和决策的门槛。

## BigQuery 不是什么

尽管 BigQuery 功能强大且用途广泛，但它可能并不适合某些用例：

+   BigQuery 并不是关系数据库的替代品。某些业务用例可能涉及大量表格行的更新；在这种情况下，BigQuery 很可能不是首选的数据存储解决方案，因为关系数据库非常适合此类高度事务性任务。GCP 提供了 Cloud SQL 和 Cloud Spanner 作为其托管关系型产品的一部分。

+   BigQuery 不是一个 NoSQL 数据库。存储在 BigQuery 中的数据必须有模式。NoSQL 是一种无模式的数据库存储解决方案。GCP 还有 Cloud BigTable 和 Cloud Datastore，它们是高度可扩展和性能卓越的托管 NoSQL 产品。

## BigQuery 入门

BigQuery 可以通过多种方式访问和使用；它们包括

+   BigQuery 网页界面

+   命令行工具，**‘bq’**

+   客户端 API 库用于程序性访问

在本节中，我们将通过使用网页界面来介绍 BigQuery，因为它提供了 BigQuery 内数据集和表的图形视图，并且适合快速在查询引擎上执行查询。

要从 GCP 仪表板打开 BigQuery，请点击左上角的三条横线，并从标记为**大数据**的产品部分选择**BigQuery**，如图 38-1 所示。

![../images/463852_1_En_38_Chapter/463852_1_En_38_Fig1_HTML.jpg](img/463852_1_En_38_Fig1_HTML.jpg)

图 38-1

打开 BigQuery

BigQuery 网页界面仪表板如图 38-2 所示。

![../images/463852_1_En_38_Chapter/463852_1_En_38_Fig2_HTML.jpg](img/463852_1_En_38_Fig2_HTML.jpg)

图 38-2

BigQuery 网页界面

在图 38-2 中，BigQuery 网页界面有三个标记的部分，我们将简要解释：

1.  导航面板：此面板包含一组 BigQuery 资源，例如

    +   查询历史记录：用于查看以前的查询

    +   保存的查询：用于存储常用查询

    +   工作历史记录：用于查看数据加载、复制和导出等 BigQuery 作业

    +   转移：链接到 BigQuery 数据传输服务 UI

    +   资源：显示已固定项目及其包含的数据集列表

1.  查询编辑器：这是使用熟悉的 SQL 数据库语言编写查询的地方。

1.  详细信息面板：当在**资源**选项卡中点击时，此面板显示项目、数据集和表的详细信息。此外，此面板还显示执行查询的结果。

### 公共数据集

BigQuery 附带对一些公共数据集的访问权限；我们将使用这些数据集来探索与 BigQuery 一起工作。要查看公共数据集，请访问

[`console.cloud.google.com/bigquery?p=bigquery-public-data&page=project`](https://console.cloud.google.com/bigquery%253Fp%253Dbigquery-public-data%2526page%253Dproject) .

公共数据集现在将显示在导航面板的资源部分（见图 38-3）。

![../images/463852_1_En_38_Chapter/463852_1_En_38_Fig3_HTML.jpg](img/463852_1_En_38_Fig3_HTML.jpg)

图 38-3

公共数据集

## 运行您的第一个查询

对于我们的第一个查询，我们将使用**‘census_bureau_international’**数据集，该数据集“提供自 1950 年以来各国人口估计以及到 2050 年的预测。”在这个查询中，我们选择一个国家及其 2018 年的预期寿命（男女两性）。

```py
SELECT
country_name,
life_expectancy
FROM
`bigquery-public-data.census_bureau_international.mortality_life_expectancy`
WHERE
year = 2018
ORDER BY
life_expectancy DESC
```

查询结果的示例显示在**查询结果**下的图 38-4 中。

![../images/463852_1_En_38_Chapter/463852_1_En_38_Fig4_HTML.jpg](img/463852_1_En_38_Fig4_HTML.jpg)

图 38-4

第一次查询

在 **查询编辑器** 中输入查询后，请注意以下内容，如图 38-4 中编号所示：

1.  点击 **‘运行查询’** 按钮以执行查询。

1.  绿色的 **状态指示器** 显示查询是一个有效的 SQL 语句，并在旁边显示查询大小的估计值。

1.  查询结果可以很容易地分析和可视化使用 Data Studio。

1.  我们可以看到查询在一秒多就完成了。

## 将数据加载到 BigQuery 中

在这个简单的数据摄取示例中，我们将把存储在 Google Cloud Storage（GCS）上的 CSV 文件加载到 BigQuery 中。在 GCP 中，Google Cloud Storage 是所有文件类型的通用存储位置，并且作为数据的中转区域或存档库的首选。让我们按以下步骤进行。

### 在 GCS 中准备数据

让我们通过以下步骤在 Google Cloud Storage 中准备数据：

1.  激活 Cloud Shell，如图 38-5 所示。

    ![../images/463852_1_En_38_Chapter/463852_1_En_38_Fig5_HTML.jpg](img/463852_1_En_38_Fig5_HTML.jpg)

    图 38-5

    激活 Google Cloud Shell

1.  在 GCS 上创建一个桶（记得给桶起一个唯一名称）。

1.  将数据传输到桶。本例中使用的 CSV 数据是存储在代码仓库中的加密货币数据集。使用 ‘gsutil cp’ 命令将数据集移动到 GCS 桶。

```py
gsutil mb gs://my-test-data
```

1.  在桶中显示传输的数据。

```py
gsutil cp crypto-markets.csv gs://my-test-data
```

```py
gsutil ls gs://my-test-data/
```

### 使用 BigQuery Web UI 加载数据

让我们按以下步骤使用 Web UI 将数据加载到 BigQuery 中：

![../images/463852_1_En_38_Chapter/463852_1_En_38_Fig9_HTML.jpg](img/463852_1_En_38_Fig9_HTML.jpg)

图 38-9

创建表选项

1.  在导航面板中点击项目名称，然后在详细信息面板中点击 **创建数据集**（见图 38-6）。

    ![../images/463852_1_En_38_Chapter/463852_1_En_38_Fig6_HTML.jpg](img/463852_1_En_38_Fig6_HTML.jpg)

    图 38-6

    创建数据集

1.  将 **DatasetID** 输入为 ‘crypto_data’，并将数据位置选择为 ‘United States (US)’（见图 38-7）。

    ![../images/463852_1_En_38_Chapter/463852_1_En_38_Fig7_HTML.jpg](img/463852_1_En_38_Fig7_HTML.jpg)

    图 38-7

    创建数据集参数

1.  接下来，在导航面板中点击新创建的数据集，然后在详细信息面板中点击 **创建表**（见图 38-8）。

    ![../images/463852_1_En_38_Chapter/463852_1_En_38_Fig8_HTML.jpg](img/463852_1_En_38_Fig8_HTML.jpg)

    图 38-8

    创建表

1.  我们将从存储在 Google Cloud Storage 上的 CSV 文件创建一个表格。在创建表格页面，选择以下参数，如图 38-9 所示：

    1.  选择 **‘Google Cloud Storage’** 作为源数据。

    1.  从桶 **‘my-test-data’** 中选择文件 **‘crypto-markets.csv’**。

    1.  选择**CSV**作为文件格式。

    1.  将**‘markets’**作为目标表。

    1.  切换到“以文本编辑”并输入以下作为模式：

        ```py
        slug,symbol,name,date,ranknow,open,high,low,close,volume,market,close_ratio,spread
        ```

    1.  展开“高级选项”并将“跳过的标题行”设置为 1。

    1.  点击**创建表**。

在导航面板中点击**作业历史**以查看加载作业的状态（见图 38-10）。

![../images/463852_1_En_38_Chapter/463852_1_En_38_Fig10_HTML.jpg](img/463852_1_En_38_Fig10_HTML.jpg)

图 38-10

BigQuery 加载作业

创建的表的预览如图 38-11 所示。

![../images/463852_1_En_38_Chapter/463852_1_En_38_Fig11_HTML.jpg](img/463852_1_En_38_Fig11_HTML.jpg)

图 38-11

加载表的预览

### bq 命令行实用程序

让我们通过 Cloud Shell 终端上的‘bq’实用程序来了解一些有用的命令：

+   列出可访问的项目。

+   列出默认项目中的数据集。

```py
bq ls –p
projectId           friendlyName
----------------------- ------------------
secret-country-192905   My First Project
```

+   列出数据集中的表。

```py
bq ls
datasetId
-------------
crypto_data
```

+   列出最近执行的任务。这包括加载作业和执行的查询。

```py
bq ls crypto_data
tableId   Type    Labels   Time Partitioning
--------- ------- -------- -------------------
markets   TABLE
```

```py
bq ls –j
jobId                         Job Type  State     Start Time       Duration
----------------------------  --------  --------  ---------------  --------
bquxjob_767fb332_16625172a52  load      SUCCESS   29 Sep 07:29:27  0:00:10
bquxjob_2a33184c_16625141949  load      SUCCESS   29 Sep 07:26:06  0:00:13
bquxjob_582a116b_16624b3717a  query     SUCCESS   29 Sep 05:41:20  0:00:01
bquxjob_7b18cd73_16624a0f378  query     SUCCESS   29 Sep 05:40:32  0:00:01
```

### 使用命令行 bq 实用程序加载数据。

以下命令通过终端使用 bq 实用程序将数据集加载到 BigQuery 中：

+   创建一个新的数据集。

+   列出数据集以确认新数据集的创建。

```py
bq mk crypto_data_terminal
Dataset 'secret-country-192905:crypto_data_terminal' successfully created.
```

+   将数据作为表加载到新创建的数据集中。我们使用“bq load”命令加载数据。此命令将数据加载到新表或现有表中。在我们的示例中，我们将数据从 GCS 存储桶“gs://my-test-data/crypto-markets.csv”加载到名为“markets_terminal”的新表中，其模式为“slug,symbol,name,date,ranknow,open,high,low,close,volume,market,close_ratio,spread”。

```py
bq ls
datasetId
----------------------
crypto_data
crypto_data_terminal
```

+   列出数据集中的表。

```py
bq load crypto_data_terminal.markets_terminal gs://my-test-data/crypto-markets.csv slug,symbol,name,date,ranknow,open,high,low,close,volume,market,close_ratio,spread
```

+   检查表模式。

```py
bq ls crypto_data_terminal
tableId        Type    Labels   Time Partitioning
------------------ ------- -------- -------------------
markets_terminal   TABLE
```

+   删除一个表。

```py
bq show crypto_data_terminal.markets_terminal
Table secret-country-192905:crypto_data_terminal.markets_terminal
Last modified            Schema           Total Rows   Total Bytes   Expiration   Time Partitioning   Labels
----------------- ------------------------ ------------ ------------- ------------ ------------------- --------
29 Sep 09:12:24   |- slug: string          498381       52777964
|- symbol: string
|- name: string
|- date: string
|- ranknow: string
|- open: string
|- high: string
|- low: string
|- close: string
|- volume: string
|- market: string
|- close_ratio: string
|- spread: string
```

+   删除数据集。此命令将删除包含所有表的整个数据集。

```py
bq rm crypto_data_terminal.markets_terminal
```

```py
bq rm -r crypto_data_terminal
```

## BigQuery SQL

在本节中，我们将通过执行一些示例来概述 SQL，这些示例提供了使用 SQL 可以实现的大致视角。之前未使用过 SQL 的新用户将受益于本节。此外，SQL 非常易于使用且直观，即使是营销和销售等领域的技术人员也擅长使用，有时甚至超过程序员。它是一种表达性声明性语言。

BigQuery 与标准 SQL 和传统 SQL 语法兼容，后者是 SQL 的非标准变体。然而，标准 SQL 是 BigQuery 的首选查询语法。在实验 SQL 时，我们将使用 **census_bureau_international** 公共数据集。以下查询可在书籍存储库的章节笔记本中找到。

### 过滤

以下查询从“census_bureau_international”数据集中的“age_specific_fertility_rates”表中选择了 2018 年每个国家的生育率。结果表按降序排列。

```py
bq query --use_legacy_sql=false 'SELECT
country_name AS country,
total_fertility_rate AS fertility_rate
FROM
`bigquery-public-data.census_bureau_international.age_specific_fertility_rates`
WHERE
year = 2018
ORDER BY
fertility_rate DESC
LIMIT
10'
Waiting on bqjob_r142a3f484f713c4a_0000016626f7f063_1 ... (0s) Current status: DONE
+-------------+----------------+
|   country   | fertility_rate |
+-------------+----------------+
| Niger       |         6.3504 |
| Angola      |         6.0945 |
| Burundi     |          5.934 |
| Mali        |            5 .9 |
| Chad        |            5.9 |
| Somalia     |          5.702 |
| Uganda      |           5.62 |
| Zambia      |          5.582 |
| Malawi      |         5.4286 |
| South Sudan |           5.34 |
+-------------+----------------+
```

在前面的查询中，SQL 命令 SELECT 用于从表中选择字段或列。SELECT 关键字之后是列名列表，列名之间用逗号分隔。关键字 AS 用于为将在查询执行时显示在结果表中的列提供一个别名。关键字 FROM 用于指向数据正在从中检索的表。在 BigQuery 中，使用标准 SQL，表名前缀为数据库名，项目 ID 用一对反引号包围（即，`project_id.database_name.table_name`）。

关键字 WHERE 用于过滤查询返回的行。关键字 ORDER BY 用于根据指定的列或列集按升序或降序排列检索到的数据。关键字 LIMIT 截断从查询中检索到的结果。

### 聚合

以下查询从“census_bureau_international”数据集中的“midyear_population”表中选择 2000 年至 2018 年之间每个国家的平均人口。结果表按降序排列。

```py
bq query --use_legacy_sql=false 'SELECT
country_name AS country,
AVG(midyear_population) AS average_population
FROM
`bigquery-public-data.census_bureau_international.midyear_population`
WHERE
year >= 2000 AND year <= 2018
GROUP BY
country
ORDER BY
average_population DESC
LIMIT
20'
Waiting on bqjob_r95be3d17e726415_000001662890a68f_1 ... (1s) Current status: DONE
+------------------+----------------------+
|     country      |  average_population  |
+------------------+----------------------+
| China            | 1.3285399873157892E9 |
| India            |  1.154912377105263E9 |
| United States    | 3.0594302226315784E8 |
| Indonesia        | 2.3984691394736844E8 |
| Brazil           |  1.930978929473684E8 |
| Pakistan         | 1.8112083526315784E8 |
| Nigeria          | 1.6255564478947365E8 |
| Bangladesh       |  1.447749475789474E8 |
| Russia           | 1.4330035963157892E8 |
| Japan            | 1.2727527184210527E8 |
| Mexico           | 1.1269223210526317E8 |
| Philippines      |          9.1357295E7 |
| Vietnam          |   8.83786184736842E7 |
| Ethiopia         |  8.460339989473683E7 |
| Germany          |  8.168817173684208E7 |
| Egypt            |  8.064017099999999E7 |
| Iran             |  7.427240431578948E7 |
| Turkey           |  7.389499394736844E7 |
| Congo (Kinshasa) |   6.82958565263158E7 |
| Thailand         |  6.619103463157895E7 |
+------------------+----------------------+
```

在前面的查询中，使用 SELECT 命令检索的字段通过聚合函数传递，以给出 2000 年至 2018 年（含）之间年中人口的平均值。为了混合聚合字段和非聚合字段，我们需要 GROUP BY 命令按一个或多个列对结果进行分组，否则由于聚合函数的存在，只会返回单个结果。

### 连接

以下查询选择每个国家 2018 年的平均人口和预期寿命。数据来自“census_bureau_international”数据集中的“midyear_population”表和“mortality_life_expectancy”表。结果表按国家名称和年份分组，并按降序排列。

```py
bq query --use_legacy_sql=false 'SELECT
midyearpop.country_name AS country,
midyearpop.year AS year,
AVG(midyearpop.midyear_population) AS population,
AVG(mortality.life_expectancy) AS life_expectancy
FROM
`bigquery-public-data.census_bureau_international.midyear_population` AS midyearpop
JOIN
`bigquery-public-data.census_bureau_international.mortality_life_expectancy` AS mortality
ON
midyearpop.country_name = mortality.country_name
WHERE
midyearpop.year = 2018
GROUP BY
country, year
ORDER BY
population DESC
LIMIT
20'
Waiting on bqjob_r4ecdb3f115b3f5d3_0000016628b526ea_1 ... (0s) Current status: DONE
+------------------+------+---------------+--------------------+
|     country      | year |  population   |  life_expectancy   |
+------------------+------+---------------+--------------------+
| China            | 2018 | 1.384688986E9 |  75.58754098360653 |
| India            | 2018 | 1.296834042E9 |  69.15033333333334 |
| United States    | 2018 |  3.29256465E8 |  82.25324324324323 |
| Indonesia        | 2018 |  2.62787403E8 |  70.89647887323946 |
| Brazil           | 2018 |  2.08846892E8 |  71.26444444444446 |
| Pakistan         | 2018 |  2.07862518E8 |  66.57942857142856 |
| Nigeria          | 2018 |  2.03452505E8 | 53.483061224489774 |
| Bangladesh       | 2018 |  1.59453001E8 |  69.93685714285715 |
| Russia           | 2018 |  1.42122776E8 |  71.61112903225805 |
| Japan            | 2018 |  1.26168156E8 |   85.6562295081967 |
| Mexico           | 2018 |  1.25959205E8 |              75.22 |
| Ethiopia         | 2018 |  1.08386391E8 | 59.355633802816925 |
| Philippines      | 2018 |  1.05893381E8 |  69.13042253521127 |
| Egypt            | 2018 |   9.9413317E7 |   73.8963636363636 |
| Vietnam          | 2018 |   9.7040334E7 |   74.0014516129032 |
| Congo (Kinshasa) | 2018 |   8.5281024E7 | 56.483376623376614 |
| Iran             | 2018 |   8.3024745E7 |  72.58799999999997 |
| Turkey           | 2018 |   8.1257239E7 |  73.33577464788735 |
| Germany          | 2018 |   8.0457737E7 |  80.61900000000001 |
| Thailand         | 2018 |   6.8615858E7 |  75.35032786885246 |
+------------------+------+---------------+--------------------+
```

JOIN 命令用于通过匹配各自的行将两个或多个表中的数据合并或连接起来。该命令使用 ON 子句来确定将用于匹配的列。

### 子查询

以下查询选择每个国家 2018 年的平均人口和预期寿命。数据来自“census_bureau_international”数据集中的“midyear_population”表和“mortality_life_expectancy”表。查询在第一个 FROM 子句中使用子查询语句按年份和特定国家进行筛选。结果表按国家名称和年份分组，并按降序排列。子查询语句的一般思想是能够在不使用中间表的情况下创建更复杂的查询。

```py
bq query --use_legacy_sql=false 'SELECT
midyearpop.country_name AS country,
midyearpop.year AS year,
AVG(midyearpop.midyear_population) AS population,
AVG(mortality.life_expectancy) AS life_expectancy
FROM (
SELECT
country_name,
year,
midyear_population
FROM
`bigquery-public-data.census_bureau_international.midyear_population`
WHERE
year = 2018
AND (country_name LIKE "Nigeria"
OR country_name LIKE "Egypt")) AS midyearpop
JOIN
`bigquery-public-data.census_bureau_international.mortality_life_expectancy` AS mortality
ON
midyearpop.country_name = mortality.country_name
GROUP BY
country,
year
ORDER BY
population DESC
LIMIT
20'
Waiting on bqjob_r5d381c26fcb6480e_0000016628e220c3_1 ... (0s) Current status: DONE
+---------+------+--------------+--------------------+
| country | year |  population  |  life_expectancy   |
+---------+------+--------------+--------------------+
| Nigeria | 2018 | 2.03452505E8 | 53.483061224489774 |
| Egypt   | 2018 |  9.9413317E7 |   73.8963636363636 |
+---------+------+--------------+--------------------+
```

### 反对使用 SELECT *

在 BigQuery 中，运行 SELECT ∗ 命令是不推荐的，这个命令在 SQL 中用于从表中检索所有列。在 BigQuery 中，这个命令相当昂贵，尤其是如果你的表包含数以兆字节的数据。如果你想要了解数据集中列及其条目的感觉，你可以执行命令‘bq head [table_name]’来检索表的几行。例如，我们在以下示例列表中使用该命令检索了之前从 GCS 加载到‘crypto_data’数据集的‘market’表的几行。

```py
bq head crypto_data.markets
+------+--------+------+------------+---------+----------+----------+----------+----------+----------+-----------+-------------+--------+
| slug | symbol | name |    date    | ranknow |   open   |   high   |   low    |  close   |  volume  |  market   | close_ratio | spread |
+------+--------+------+------------+---------+----------+----------+----------+----------+----------+-----------+-------------+--------+
| 0x   | ZRX    | 0x   | 2017-08-16 | 41      | 0.111725 | 0.280031 | 0.103962 | 0.224399 | 5232600  | 67034800  | 0.684       | 0.18   |
| 0x   | ZRX    | 0x   | 2017-08-17 | 41      | 0.223022 | 0.238935 | 0.206735 | 0.206735 | 2752410  | 133813000 | 0           | 0.03   |
| 0x   | ZRX    | 0x   | 2017-08-18 | 41      | 0.205558 | 0.35026  | 0.205558 | 0.293387 | 12793800 | 123335000 | 0.607       | 0.14   |
......
......
| 0x   | ZRX    | 0x   | 2017-08-28 | 41      | 0.352459 | 0.354823 | 0.32362  | 0.343713 | 6639910  | 176230000 | 0.6439      | 0.03   |
+------+--------+------+------------+---------+----------+----------+----------+----------+----------+-----------+-------------+--------+
```

## 在 AI 云实例和 Google Colab 上使用 BigQuery 笔记本

BigQuery 与 Google Notebook AI 实例和 Google Colab 上的笔记本很好地集成。在本节中，我们将介绍从笔记本中执行 BigQuery 数据集和表的几种方法。与 BigQuery 交互有几个方法，但一种快速简单的方法是使用 BigQuery 客户端库 **‘google-cloud-bigquery’** 中的 **‘%bigquery’** 魔法命令来运行查询，以最少的语法运行查询。

**%%bigquery** 魔法运行 SQL 查询并将结果作为 pandas DataFrame 返回。在这里，我们使用 **‘%%bigquery’** 魔法命令与 BigQuery 交互。首先，在 GCP AI 笔记本实例或 Colab 中打开一个笔记本：

![../images/463852_1_En_38_Chapter/463852_1_En_38_Fig12_HTML.jpg](img/463852_1_En_38_Fig12_HTML.jpg)

图 38-12

比特币加密货币柱状图绘制

1.  如果在 Google Colab 上运行，通过运行以下代码验证笔记本

    from google.colab import auth

    auth.authenticate_user()

    print(‘Authenticated’)

1.  导入 Pandas 和 Matplotlib。

    ```py
    import pandas as pd
    import matplotlib.pyplot as plt
    ```

1.  将以下查询输出存储为名为 **‘litcoin_crypto’** 的 Pandas DataFrame。在 **‘--project’** 属性后放置你的项目 ID。确保更新 FROM 字段以包含你的数据集和表 ID。

    ```py
    %%bigquery --project ekabasandbox litcoin_crypto
    SELECT
    symbol,
    date,
    close,
    open,
    high,
    low,
    spread
    FROM
    `crypto_data.markets`
    WHERE
    symbol = 'LTC'
    LIMIT 10
    symbol  date    close   open    high    low   spread
    0   LTC 2013-04-28  4.35    4.3     4.4     4.18    0.22
    1   LTC 2013-05-07  3.33    3.37    3.41    2.94    0.47
    2   LTC 2013-05-03  3.04    3.39    3.45    2.4     1.05
    3   LTC 2013-05-04  3.48    3.03    3.64    2.9     0.74
    4   LTC 2013-05-05  3.59    3.49    3.69    3.35    0.34
    5   LTC 2013-05-06  3.37    3.59    3.78    3.12    0.66
    6   LTC 2013-05-02  3.37    3.78    4.04    3.01    1.03
    7   LTC 2013-05-01  3.8     4.29    4.36    3.52    0.84
    8   LTC 2013-04-29  4.38    4.37    4.57    4.23    0.34
    9   LTC 2013-04-30  4.3     4.4     4.57    4.17    0.4
    ```

1.  变量‘litcoin_crypto’是一个 Pandas DataFrame。现在，让我们修改数据属性并绘制一个柱状图。

    ```py
    # convert columns to numeric
    litcoin_crypto = litcoin_crypto.apply(pd.to_numeric, errors="ignore")
    # check the datatypes
    litcoin_crypto.dtypes
    symbol     object
    date       object
    close     float64
    open      float64
    high      float64
    low       float64
    spread    float64
    dtype: object
    ```

1.  使用变量‘date’在 x 轴上和收盘价在 y 轴上绘制柱状图（见图 38-12）。

    ```py
    # plot the bar chart
    litcoin_crypto.plot(kind='bar', x="date", y="close")
    plt.show()
    ```

## BigQueryML

BigQuery 机器学习通过使用简单的标准 SQL 命令，使利用 BigQuery 中的数据集进行机器学习的强大功能变得快速简单。此功能包括使用数据子集在数据集上训练和测试模型的能力，以及自动调整学习模型超参数的能力。

在撰写本文时，以下学习模型在 BigQuery 中可用：

+   线性回归

+   二元和多类逻辑回归

在本节中，我们将使用 Google AI VMs 上的 Colab 笔记本实例与 BigQuery ML 一起工作，使用我们之前导入到 BigQuery 中的‘crypto_data’数据集中的‘market’表来构建预测模型。该模型将尝试根据一组市场属性预测比特币加密货币的下一日收盘价。数据处理和机器学习建模全部使用标准 SQL 完成：

1.  打开一个新的笔记本。

1.  选择用于训练机器学习模型的特征。在 SQL 代码中，我们使用‘LEAD()’函数来返回下一行的值。偏移量 1 表示我们想要获取查询中向前一步的下一个值。有了这个，就可以轻松调整查询以预测 2 到 n 天的窗口。LEAD() 函数是一个窗口函数，它遍历行集。因此，使用 OVER() 函数在查询中定义一个窗口，而 PARTITION BY 和 ORDER BY 子句将查询结果分成分区，并定义每个分区中行的排列。

    我们使用‘params’变量来采样一半的数据并将其存储在‘TRAIN’集中。这确保了剩余的数据集在模型训练中不被使用，可以在模型评估阶段用来检查模型是否具有良好的泛化能力。

    确保更新 FROM 字段，包含您的数据集和表 ID。

    ```py
    %%bigquery --project ekabasandbox btc_market
    WITH
    params AS (
    SELECT
    1 AS TRAIN,
    2 AS EVAL ),
    btc_market AS (
    SELECT
    symbol,
    date,
    open,
    high,
    low,
    close,
    spread,
    cast(LEAD(close, 1) OVER (PARTITION BY symbol ORDER BY symbol DESC) AS NUMERIC) AS next_day_close
    FROM
    `crypto_data.markets`,
    params
    WHERE
    symbol = 'BTC'
    AND MOD(ABS(FARM_FINGERPRINT(CAST(date AS STRING))),4) = params.TRAIN )
    SELECT
    *
    FROM
    btc_market
    WHERE
    next_day_close IS NOT NULL
    ```

1.  显示查询的前十行。

    btc_market.head(10)

    ```py
    symbol  date    open    high    low close   spread  next_day_close
    0   BTC 2013-05-05  112.9   118.8   107.14  115.91  11.66 112.3
    1   BTC 2013-05-06  115.98  124.66  106.64  112.3   18.02 112.67
    2   BTC 2013-05-09  113.2   113.46  109.26  112.67  4.2   115.24
    3   BTC 2013-05-11  117.7   118.68  113.01  115.24  5.67  111.5
    4   BTC 2013-05-14  117.98  119.8   110.25  111.5   9.55  114.22
    5   BTC 2013-05-15  111.4   115.81  103.5   114.22  12.31 121.99
    6   BTC 2013-05-19  123.21  124.5   119.57  121.99  4.93  123.89
    7   BTC 2013-05-22  122.89  124     122     123.89  2     133.2
    8   BTC 2013-05-24  126.3   133.85  125.72  133.2   8.13  131.98
    9   BTC 2013-05-25  133.1   133.22  128.9   131.98  4.32  133.48
    ```

1.  训练好的模型存储在 BigQuery 数据集中。在这种情况下，我们将创建一个 BigQuery 数据集来存储模型。

    ```py
    from google.cloud import bigquery
    client = bigquery.Client(project='ekabasandbox')
    # create a BigQuery dataset to store your ML model
    dataset = client.create_dataset('btc_crypto')
    print('Dataset: `{}` created.'.format(dataset.dataset_id))
    Dataset: `btc_crypto` created.
    ```

1.  在准备我们的训练数据集之后，现在是时候训练模型了。确保更新 FROM 字段，包含您的数据集和表 ID。

    ```py
    %%bigquery --project ekabasandbox model
    CREATE OR REPLACE MODEL `btc_crypto.market_closing_model`
    OPTIONS
    (model_type='linear_reg',
    labels=['next_day_close']) AS
    WITH
    params AS (
    SELECT
    1 AS TRAIN,
    2 AS EVAL ),
    btc_market AS (
    SELECT
    CAST(open AS NUMERIC) AS open,
    CAST(high AS NUMERIC) AS high,
    CAST(low AS NUMERIC) AS low,
    CAST(close AS NUMERIC) AS close,
    CAST(spread AS NUMERIC) AS spread,
    CAST(LEAD(close, 1) OVER (PARTITION BY symbol ORDER BY symbol DESC) AS NUMERIC) AS next_day_close
    FROM
    `crypto_data.markets`,
    params
    WHERE
    symbol = 'BTC'
    AND MOD(ABS(FARM_FINGERPRINT(CAST(date AS STRING))),4) = params.TRAIN )
    SELECT
    *
    FROM
    btc_market
    WHERE
    next_day_close IS NOT NULL
    ```

1.  检查创建的模型是否存在于数据集“btc_crypto”中。我们在笔记本单元格中用感叹号（‘!’）作为前缀来执行 bash 命令。

    ```py
    !bq ls btc_crypto
    tableId          Type    Labels   Time Partitioning
    ---------------------- ------- -------- -------------------
    market_closing_model   MODEL
    ```

1.  评估模型以估计模型的性能。RMSE 指标通过在 BigQuery 中调用训练模型的‘mean_squared_error’字段并通过‘SQRT()’函数传递来评估。为了评估模型，通过函数‘ML.EVALUATE()’传递模型。这次我们选择数据集的剩余子集并将其存储在‘params.EVAL’中。

1.  确保更新 FROM 字段，包含您的数据集和表 ID。

    ```py
    %%bigquery --project ekabasandbox rmse
    SELECT
    SQRT(mean_squared_error) AS rmse
    FROM
    ML.EVALUATE(MODEL `btc_crypto.market_closing_model`,
    (
    WITH
    params AS (
    SELECT
    1 AS TRAIN,
    2 AS EVAL ),
    btc_market AS (
    SELECT
    CAST(open AS NUMERIC) AS open,
    CAST(high AS NUMERIC) AS high,
    CAST(low AS NUMERIC) AS low,
    CAST(close AS NUMERIC) AS close,
    CAST(spread AS NUMERIC) AS spread,
    CAST(LEAD(close, 1) OVER (PARTITION BY symbol ORDER BY symbol DESC) AS NUMERIC) AS next_day_close
    FROM
    `crypto_data.markets`,
    params
    WHERE
    symbol = 'BTC'
    AND MOD(ABS(FARM_FINGERPRINT(CAST(date AS STRING))),4) = params.EVAL )
    SELECT
    *
    FROM
    btc_market
    WHERE
    next_day_close IS NOT NULL ))
    rmse
    0   393.265715
    ```

1.  使用训练好的模型预测比特币加密货币的下一日收盘价。确保更新 FROM 字段，包含您的数据集和表 ID。

```py
%%bigquery --project ekabasandbox predict
SELECT
*
FROM
ml.PREDICT(MODEL `btc_crypto.market_closing_model`,
(
WITH
params AS (
SELECT
1 AS TRAIN,
2 AS EVAL ),
btc_market AS (
SELECT
CAST(close AS NUMERIC) AS close,
date,
CAST(open AS NUMERIC) AS open,
CAST(high AS NUMERIC) AS high,
CAST(low AS NUMERIC) AS low,
CAST(spread AS NUMERIC) AS spread,
CAST(LEAD(close, 1) OVER (PARTITION BY symbol ORDER BY symbol DESC) AS NUMERIC) AS next_day_close
FROM
`crypto_data.markets`,
params
WHERE
symbol = 'BTC'
AND MOD(ABS(FARM_FINGERPRINT(CAST(date AS STRING))),4) = params.EVAL )
SELECT
*
FROM
btc_market
WHERE
next_day_close IS NOT NULL ))
predict  predicted_next_day_close  close    date        open     high     low      spread  next_day_close
0        193.523361                116.99   2013-05-01  139      139.89   107.72   32.17   112.5
1        162.505189                112.5    2013-05-04  98.1     115      92.5     22.5    111.5
2        158.389055                111.5    2013-05-07  112.25   113.44   97.7     15.74   117.2
3        158.700481                117.2    2013-05-10  112.8    122      111.55   10.45   115
...      ...                       ...      ...         ...      ...      ...      ...     ...
388      4491.052680               4703.39  2017-08-31  4555.59  4736.05  4549.4   186.65  4597.12
389      4422.931411               4597.12  2017-09-06  4376.59  4617.25  4376.59  240.66  4122.94
390      4163.348876               4122.94  2017-09-10  4229.34  4245.44  3951.04  294.4   4161.27
391      4029.355833               4161.27  2017-09-11  4122.47  4261.67  4099.4   162.27  4130.81
...      ...                       ...      ...         ...      ...      ...      ...     ...
416      14723.798445              15201    2018-01-03  14978.2  15572.8  14844.5  728.3   15599.2
417      15421.170791              15599.2  2018-01-04  15270.7  15739.7  14522.2  1217.5  14595.4
```

本章提供了在 GCP 上使用 Google BigQuery 作为数据仓库和数据分析平台的概述。它涵盖了在 Google Colab 或 GCP AI 实例上托管笔记本中使用 BigQuery，以及如何使用 BigQuery ML 通过 SQL 命令构建机器学习预测模型。

下一章将介绍 Cloud Dataprep，用于在 GCP 上可视化和转换大型数据集。

# 39. Google Cloud Dataprep

Google Cloud Dataprep 是一种托管云服务，用于快速数据探索和转换。Dataprep 使得清理和转换大量数据集以进行分析变得容易。由于它利用了 Google Cloud Dataflow 的分布式处理能力，因此它是自动可扩展的。

通常，Cloud Dataprep 旨在简化数据准备过程。来自现实世界用例的数据集通常杂乱无章。在这种情况下，它不能用于下游分析或机器学习建模。因此，建模过程的大部分时间都涉及准备和清理数据。之前讨论过的编程库，如 Pandas，在执行数据准备时被集中使用。然而，Google Cloud Dataprep 提供了一个简单的可视化界面来执行数据清理。能够快速重新组织数据集以进行建模，无需编码，这为 Dataprep 提供了即时的吸引力，因为它可以大大加快作为整体建模流程一部分的数据准备时间。另一个好处是，由于 Dataprep 建立在无服务器基础设施之上，它可以处理 PB 级规模的数据。Dataprep 可用于处理结构化和非结构化数据集。

在本节中，我们将通过使用它来准备已存储在 Google Cloud Storage 上的“crypto_markets.csv”数据集，对 Google Dataprep 进行简要的浏览。

## Cloud Dataprep 入门

从 GCP 仪表板，点击左上角的三个短横线，向下滚动到**大数据**部分下的“Dataprep”，如图 39-1 所示。

![../images/463852_1_En_39_Chapter/463852_1_En_39_Fig1_HTML.jpg](img/463852_1_En_39_Fig1_HTML.jpg)

图 39-1

通过 GCP 仪表板打开 Dataprep

Dataprep 是由与 Trifacta 公司合作在 GCP 上提供的一项服务。要开始使用 Dataprep，请同意并接受所有许可协议（见图 39-2）。Dataprep 在 GCS 上创建一个存储上传到 Dataprep 的文件及其转换输出的存储桶（见图 39-3）。Dataprep 仪表板如图 39-4 所示。

![../images/463852_1_En_39_Chapter/463852_1_En_39_Fig3_HTML.png](img/463852_1_En_39_Fig3_HTML.png)

图 39-3

Dataprep GCS 位置设置

![../images/463852_1_En_39_Chapter/463852_1_En_39_Fig2_HTML.png](img/463852_1_En_39_Fig2_HTML.png)

图 39-2

Trifacta 许可协议

## 使用流程转换数据

Dataprep 流程是一个创建的对象，用于组织和管理工作在数据清理和转换过程中涉及的数据集和操作：

1.  我们首先通过点击 Dataprep 仪表板右上角的“创建流程”按钮来创建一个流程（见图 39-4）。输入用户定义的流程名称，然后点击“创建”，如图 39-5 所示。流程页面如图 39-6 所示。

    ![../images/463852_1_En_39_Chapter/463852_1_En_39_Fig6_HTML.png](img/463852_1_En_39_Fig6_HTML.png)

    图 39-6

    流程页面

    ![../images/463852_1_En_39_Chapter/463852_1_En_39_Fig5_HTML.png](img/463852_1_En_39_Fig5_HTML.png)

    图 39-5

    创建流程

    ![../images/463852_1_En_39_Chapter/463852_1_En_39_Fig4_HTML.jpg](img/463852_1_En_39_Fig4_HTML.jpg)

    图 39-4

    Dataprep 仪表板

1.  让我们先在我们的数据集放在一个 GCS 存储桶中。我们将在终端上运行以下命令来完成此操作。

创建一个新的存储桶。

1.  从 GitHub 转移数据到存储桶。

```py
gsutil mb gs://my-dataprep-data
```

1.  接下来，我们将我们的“crypto-market”数据集从“my-dataprep-data”存储桶转移到 Dataprep 预处理存储桶。我们可以在终端上执行以下代码来快速完成此操作。

```py
gsutil cp crypto-markets.csv gs://my-dataprep-data
```

1.  接下来，我们将导入并添加数据集到流程中。数据集可以直接上传到 Dataprep，然后存储到 Dataprep 在启动时生成的存储桶中。此外，Dataprep 还可以导入已存储在 BigQuery 或 GCS 中的数据集。在这种情况下，我们将导入之前转移到 Dataprep 预处理存储桶中的“crypto-market”数据集，该存储桶位于“my-dataprep-data”文件夹中（见图 39-7）。图 39-8 显示了数据集加载到 Dataprep 的过程。

    ![../images/463852_1_En_39_Chapter/463852_1_En_39_Fig8_HTML.png](img/463852_1_En_39_Fig8_HTML.png)

    图 39-8

    将数据集加载到 Dataprep 中

    ![../images/463852_1_En_39_Chapter/463852_1_En_39_Fig7_HTML.png](img/463852_1_En_39_Fig7_HTML.png)

    图 39-7

    从 GCS 导入数据集到 Dataprep

1.  接下来，我们将创建一个配方。一个 Dataprep 配方包含用于清理和加工数据集所采取的转换步骤。这个配方随后作为 Dataflow 作业执行，以操作数据集并得出结果。点击“添加新配方”按钮来创建一个配方。该配方在图 39-9 中的红色边界框内。

    ![../images/463852_1_En_39_Chapter/463852_1_En_39_Fig9_HTML.jpg](img/463852_1_En_39_Fig9_HTML.jpg)

    图 39-9

    数据集配方

1.  然后点击“编辑配方”按钮打开“转换网格”，在那里我们可以在数据集上执行各种清理和加工步骤。

1.  在本节示例中，我们将通过删除一些未使用的列然后删除数据集中除比特币加密货币之外的所有行来执行一个简单的转换过程：

    1.  删除“slug”列。在红色框内点击“添加”以删除该列（见图 39-10）。

        ![../images/463852_1_En_39_Chapter/463852_1_En_39_Fig10_HTML.jpg](img/463852_1_En_39_Fig10_HTML.jpg)

        图 39-10

        删除“slug”列

    1.  删除“名称”列（见图 39-11）。

        ![../images/463852_1_En_39_Chapter/463852_1_En_39_Fig11_HTML.jpg](img/463852_1_En_39_Fig11_HTML.jpg)

        图 39-11

        删除“名称”列

    1.  接下来，我们将过滤数据集中的行，仅保留比特币记录（见图 39-12 和 39-13）。

        ![../images/463852_1_En_39_Chapter/463852_1_En_39_Fig13_HTML.png](img/463852_1_En_39_Fig13_HTML.png)

        图 39-13

        移除所有行（除了比特币记录）

        ![../images/463852_1_En_39_Chapter/463852_1_En_39_Fig12_HTML.png](img/463852_1_En_39_Fig12_HTML.png)

        图 39-12

        使用 Dataprep 过滤行

1.  图 39-14 展示了数据集转换食谱。在图 39-14 和图 39-15 中点击“运行作业”以在 Cloud Dataflow 上运行作业。

    ![../images/463852_1_En_39_Chapter/463852_1_En_39_Fig15_HTML.jpg](img/463852_1_En_39_Fig15_HTML.jpg)

    图 39-15

    在 Dataflow 上运行作业

    ![../images/463852_1_En_39_Chapter/463852_1_En_39_Fig14_HTML.jpg](img/463852_1_En_39_Fig14_HTML.jpg)

    图 39-14

    查看转换食谱

1.  图 39-16 展示了正在运行的作业，图 39-17 展示了经过几分钟后的完成作业。

    ![../images/463852_1_En_39_Chapter/463852_1_En_39_Fig17_HTML.png](img/463852_1_En_39_Fig17_HTML.png)

    图 39-17

    完成作业

    ![../images/463852_1_En_39_Chapter/463852_1_En_39_Fig16_HTML.png](img/463852_1_En_39_Fig16_HTML.png)

    图 39-16

    在 Dataflow 上运行的作业

1.  查看作业结果（见图 39-18）。

    ![../images/463852_1_En_39_Chapter/463852_1_En_39_Fig18_HTML.jpg](img/463852_1_En_39_Fig18_HTML.jpg)

    图 39-18

    查看作业结果

1.  从图 39-19 所示的结果页面中，我们可以将结果导回 GCS（见图 39-20）。

    ![../images/463852_1_En_39_Chapter/463852_1_En_39_Fig20_HTML.png](img/463852_1_En_39_Fig20_HTML.png)

    图 39-20

    导出完成的工作

    ![../images/463852_1_En_39_Chapter/463852_1_En_39_Fig19_HTML.png](img/463852_1_En_39_Fig19_HTML.png)

    图 39-19

    作业结果页面

```py
gsutil cp -r gs://my-dataprep-data gs://dataprep-staging-7fc4d500-8b76-48a1-9562-83675643ca4b
Copying gs://my-dataprep-data/crypto-markets.csv [Content-Type=application/octet-stream]...
/ [1 files][ 47.0 MiB/ 47.0 MiB]
Operation completed over 1 objects/47.0 MiB.
```

本章节提供了一个使用 Google Cloud Dataflow 分布式处理基础设施通过 Dataprep 进行可视化和转换大型数据集的示例概述。在下一章中，我们将介绍如何使用 Cloud Dataflow 构建自定义数据转换管道。

# 40. Google Cloud Dataflow

Google Cloud Dataflow 提供了一种无服务器、并行和分布式的基础设施，用于运行批量和流数据处理的作业。Dataflow 的核心优势之一是它几乎无缝地处理从批量历史数据处理到流数据集的转换，同时优雅地考虑了流处理的优点，如窗口功能。Dataflow 是 GCP 上数据/ML 管道的一个主要组件。通常，Dataflow 用于将来自各种来源的庞大数据集（如 Cloud Pub/Sub 或 Apache Kafka）转换为像 BigQuery 或 Google Cloud Storage 这样的目标。

对于 Dataflow 来说，使用 Apache Beam 编程模型来构建批量和流操作的并行数据处理管道至关重要。使用 Beam SDKs 构建的数据处理管道可以在 Apache Apex、Apache Spark、Apache Flink 和当然还有 Google Cloud Dataflow 等各种处理后端上执行。在本节中，我们将使用 Beam Python SDK 构建数据转换管道。截至撰写本文时，Beam 还支持使用 Java、Go 和 Scala 语言构建数据管道。

## Beam 编程

Apache Beam 提供了一套广泛的概念，以简化构建分布式批量和流作业的转换管道的过程。我们将通过提供简单的代码示例来介绍这些概念：

+   管道：管道对象封装了整个操作，并通过定义管道的输入数据源、数据如何转换以及数据将写入的位置来规定转换过程。此外，管道对象还指示要执行的分布式处理后端。实际上，管道是 Beam 执行的中心组件。创建管道的代码如下所示：

    ```py
    import apache_beam as beam
    from apache_beam.options.pipeline_options import PipelineOptions
    p = beam.Pipeline(options=PipelineOptions())
    ```

    在前述代码片段中，使用‘PipelineOptions’配置管道对象以设置所需的字段。这可以通过程序化和从命令行两种方式完成。

+   PCollection：PCollection 用于定义数据源。数据源可以是 *有界* 或 *无界* 的。有界数据源指的是批量或历史数据，而无界数据源指的是流数据。Beam 使用一种称为 *窗口* 的技术，通过使用数据的某些属性（如时间戳）将无界 PCollections 分割成有限逻辑段。PCollections 还可以从内存中的数据创建，在这种情况下，PCollections 是管道中特定步骤的输入和输出。让我们看看从外部源读取 csv 数据的示例：

    ```py
    lines = p | 'ReadMyFile' >> beam.io.ReadFromText('gs://gcs_bucket/my_data.csv')
    ```

    前述代码中的管道操作符‘|’也称为应用方法，用于将 PCollection 应用到作为‘p’实例化的管道中。

+   PTransform：PTransform 指的是在管道中的一个或多个 PCollections 上执行的特殊转换任务。PTransform 可以按以下方式应用于 PCollections。

    ```py
    [Output PCollection] = [Input PCollection] | [Transform]
    ```

    注意，虽然 PTransform 创建了一个新的 PCollection，但它不会修改或更改输入集合。许多核心 Beam 转换包括

    +   ParDo：用于并行处理

    +   GroupByKey：用于处理键/值对集合

    +   CoGroupByKey：用于具有相同键类型的两个或多个键/值 PCollection 的关系连接

    +   Combine：用于合并数据中的元素或值集合

    +   Flatten：用于合并多个 PCollection 对象

    +   Partition：将单个 PCollection 分割成更小的集合

+   I/O 转换：这些是读取或写入不同外部存储系统的 PTransforms。目前与 Beam Python SDK 一起工作的某些可用 I/O 转换包括

    +   avroio：用于从 Avro 文件中读取和写入

    +   textio：用于从文本文件中读取和写入

对于一个简单的线性管道，其处理图看起来如图 40-1 所示。

![../images/463852_1_En_40_Chapter/463852_1_En_40_Fig1_HTML.jpg](img/463852_1_En_40_Fig1_HTML.jpg)

图 40-1

一个简单的线性管道，具有顺序转换

## 构建简单的数据处理管道

在这个简单的 Beam 应用程序中，我们将构建一个 Dataflow 管道，从 GCS 存储桶中预处理 CSV 文件，并将输出写回 GCS。此示例选择对下游建模任务感兴趣的特征和行。在这里，我们考虑了‘crypto-markets.csv’数据集。在数据预处理管道中，我们删除了可能对分析/建模不相关的数据属性，并且我们还过滤了与‘bitcoin’相关的记录。以下步骤创建了一个简单的 Beam 管道，并在 Google Dataflow 上执行：

1.  从 API & 服务仪表板启用 GCP Cloud Dataflow API 和 Cloud Resource Manager API。

1.  打开一个新的笔记本。

1.  注意，在撰写本文时，Apache Beam 只与 Python 版本 2.7 兼容，因此请确保切换 Python 解释器的内核。在笔记本单元中添加以下代码块。

1.  如果在 Google Colab 上运行，首先使用 GCP 验证笔记本。

1.  安装 Apache beam 库和其他重要设置包。

```py
from google.colab import auth
auth.authenticate_user()
print('Authenticated')
# configure GCP project. Change to your project ID
project_id = 'ekabasandbox'
!gcloud config set project {project_id}
```

1.  安装后，将笔记本运行时类型更改为 Python 2。

1.  然后，在运行代码之前重置笔记本内核以导入相关库。

```py
%%bash
pip install apache-beam[gcp]
```

1.  为管道分配参数。用您的条目替换相关参数。

```py
import apache_beam as beam
from apache_beam.io import ReadFromText
from apache_beam.io import WriteToText
```

1.  构建和运行管道的方法。

```py
# parameters
staging_location = 'gs://enter_bucket_name/staging' # change this
temp_location = 'gs://enter_bucket_name/temp' # change this
job_name = 'dataflow-crypto'
project_id = enter_project_id' # change this
source_bucket = 'enter_bucket_name' # change this
target_bucket = 'enter_bucket_name' # change this
```

1.  运行管道。

```py
def run(project, source_bucket, target_bucket):
import csv
options = {
'staging_location': staging_location,
'temp_location': temp_location,
'job_name': job_name,
'project': project,
'max_num_workers': 24,
'teardown_policy': 'TEARDOWN_ALWAYS',
'no_save_main_session': True,
'runner': 'DataflowRunner'
}
options = beam.pipeline.PipelineOptions(flags=[], **options)
crypto_dataset = 'gs://{}/crypto-markets.csv'.format(source_bucket)
processed_ds = 'gs://{}/transformed-crypto-bitcoin'.format(target_bucket)
pipeline = beam.Pipeline(options=options)
# 0:slug, 3:date, 5:open, 6:high, 7:low, 8:close
rows = (
pipeline |
'Read from bucket' >> ReadFromText(crypto_dataset) |
'Tokenize as csv columns' >> beam.Map(lambda line: next(csv.reader([line]))) |
'Select columns' >> beam.Map(lambda fields: (fields[0], fields[3], fields[5], fields[6], fields[7], fields[8])) |
'Filter bitcoin rows' >> beam.Filter(lambda row: row[0] == 'bitcoin')
)
combined = (
rows |
'Write to bucket' >> beam.Map(lambda (slug, date, open, high, low, close): '{},{},{},{},{},{}'.format(
slug, date, open, high, low, close)) |
WriteToText(
file_path_prefix=processed_ds,
file_name_suffix=".csv", num_shards=2,
shard_name_template="-SS-of-NN",
header='slug, date, open, high, low, close')
)
pipeline.run()
```

```py
if __name__ == '__main__':
print 'Run pipeline on the cloud'
run(project=project_id, source_bucket=source_bucket, target_bucket=target_bucket)
```

图 40-2 中的图像显示了此作业创建的数据流管道。

![../images/463852_1_En_40_Chapter/463852_1_En_40_Fig2_HTML.jpg](img/463852_1_En_40_Fig2_HTML.jpg)

图 40-2

Google Cloud Dataflow 上的预处理管道

本书不涉及 Google Cloud Dataflow 的更复杂和高级用法，因为它们更多地属于构建大规模数据转换的大数据管道领域。然而，这一节被包含在内，因为在大规模解决特定业务用例时，大数据转换是机器学习模型设计和生产化的一个重要组成部分。对于读者来说，了解这些技术的操作方式是很重要的。

本章介绍了使用运行在 Google Dataflow 计算基础设施上的 Python Apache Beam 编程模型构建大规模大数据转换管道的入门知识。下一章将介绍如何使用 Google Cloud Machine Learning Engine 训练和部署大规模模型。

# 41. Google Cloud Machine Learning Engine（Cloud MLE）

Google Cloud Machine Learning Engine，简称 Cloud MLE，是一个托管在 Google 基础设施上的服务，用于训练和部署“大规模”机器学习模型。Cloud ML Engine 是 GCP AI 平台的一部分。这个托管基础设施可以训练使用 TensorFlow、Keras、Scikit-learn 或 XGBoost 构建的大规模机器学习模型。它还提供了在线或批量预测服务来提供或消费训练好的模型。使用在线预测，基础设施可以根据请求进行扩展，而使用批量模式，Cloud MLE 可以为 TB 级别的数据提供推理。

Cloud MLE 的两个重要特性是在训练模型时能够执行分布式训练和自动超参数调整。自动超参数调整的大优势是能够找到最佳参数集，以最小化模型成本或损失函数。这节省了迭代实验中的开发时间。

## Cloud MLE 的训练/部署过程

Cloud MLE 上的训练/部署过程的高级概述如图 41-1 所示：

![../images/463852_1_En_41_Chapter/463852_1_En_41_Fig1_HTML.jpg](img/463852_1_En_41_Fig1_HTML.jpg)

图 41-1

Cloud MLE 上的训练/部署过程

1.  训练/推理数据存储在 GCS 上。

1.  执行脚本使用应用程序逻辑在 Cloud MLE 上使用训练数据训练模型。

1.  训练好的模型存储在 GCS 上。

1.  在 Cloud MLE 上使用训练好的模型创建了一个预测服务。

1.  外部应用程序将数据发送到已部署的模型进行推理。

## 在 Cloud MLE 上准备训练和部署

在这个虚构的例子中，我们将使用著名的 Iris 数据集，在 Cloud MLE 上使用 Estimator API 训练并部署 TensorFlow 模型。首先，让我们回顾以下步骤：

![../images/463852_1_En_41_Chapter/463852_1_En_41_Fig2_HTML.jpg](img/463852_1_En_41_Fig2_HTML.jpg)

图 41-2

启用 Cloud Machine Learning API

1.  通过在云终端上运行 gsutil mb 命令在 GCS 上创建一个存储桶。用唯一的存储桶名称替换它。

    ```py
    export bucket_name=iris-dataset'
    gsutil mb gs://$bucket_name
    ```

1.  将训练和测试数据从代码仓库传输到 GCP 存储桶。

1.  移动训练数据。

    ```py
    gsutil cp train_data.csv gs://$bucket_name
    ```

1.  移动训练数据。

    ```py
    gsutil cp test_data.csv gs://$bucket_name
    ```

1.  移动用于批量预测的保留数据。

    ```py
    gsutil cp hold_out_test.csv gs://$bucket_name
    ```

1.  启用 Cloud Machine Learning API 以能够在 GCP Cloud MLE 上创建和使用机器学习模型：

    1.  前往 API & 服务。

    1.  点击 “Enable APIs & Services”。

    1.  搜索 “Cloud Machine Learning Engine”。

    1.  如图 41-2 所示，点击 ENABLE API。

## 为在 Cloud MLE 上训练打包代码

在 Cloud MLE 上进行训练的代码必须打包成一个 Python 包。推荐的项目结构如下所述：

IrisCloudML: [作为父文件夹的项目名称]

+   Trainer: [包含模型和执行代码的文件夹]

    +   __init__.py: [一个空的特殊 Python 文件，表示包含的文件夹是一个 Python 包]

    +   model.py: [包含使用 TensorFlow、Keras 等编写的模型逻辑的脚本]

    +   task.py: [包含管理训练作业的应用程序的脚本]

+   scripts: [包含在 Cloud MLE 上执行作业的脚本的文件夹]

    +   distributed-training.sh: [在 Cloud MLE 上运行分布式训练作业的脚本]

    +   hyper-tune.sh: [在 Cloud MLE 上运行带有超参数调优的训练作业的脚本]

    +   single-instance-training.sh: [在 Cloud MLE 上运行单个实例训练作业的脚本]

    +   online-prediction.sh: [在 Cloud MLE 上执行在线预测作业的脚本]

    +   create-prediction-service.sh: [在 Cloud MLE 上创建预测服务的脚本]

+   hptuning_config: [在 Cloud MLE 上进行超参数调优的配置文件]

+   gpu_hptuning_config.yaml: [在 Cloud MLE 上使用 GPU 训练进行超参数调优的配置文件]

### 注意：按照以下说明在 Cloud Machine Learning Engine 上运行训练示例

1.  在 GCP AI 平台上启动 Notebook 实例。

1.  拉取代码仓库。

1.  导航到书籍文件夹。在子文件夹 `tensorflow’ 中运行脚本。

1.  如果您选择使用 Google Colab，通过运行代码进行用户认证

```py
from google.colab import auth
auth.authenticate_user()
```

## TensorFlow 模型

现在，让我们简要检查文件 ‘**model.py**’ 中的 TF 模型代码。

```py
import six
import tensorflow as tf
from tensorflow.python.estimator.model_fn import ModeKeys as Modes
# Define the format of your input data including unused columns.
CSV_COLUMNS = [
'sepal_length', 'sepal_width', 'petal_length',
'petal_width', 'class'
]
CSV_COLUMN_DEFAULTS = [[0.0], [0.0], [0.0], [0.0], [“]]
LABEL_COLUMN = 'class'
LABELS = ['setosa', 'versicolor', 'virginica']
# Define the initial ingestion of each feature used by your model.
# Additionally, provide metadata about the feature.
INPUT_COLUMNS = [
# Continuous base columns.
tf.feature_column.numeric_column('sepal_length'),
tf.feature_column.numeric_column('sepal_width'),
tf.feature_column.numeric_column('petal_length'),
tf.feature_column.numeric_column('petal_width')
]
UNUSED_COLUMNS = set(CSV_COLUMNS) - {col.name for col in INPUT_COLUMNS} - \
{LABEL_COLUMN}
def build_estimator(config, hidden_units=None, learning_rate=None):
"""Deep NN Classification model for predicting flower class.
Args:
config: (tf.contrib.learn.RunConfig) defining the runtime environment for
the estimator (including model_dir).
hidden_units: [int], the layer sizes of the DNN (input layer first)
learning_rate: (int), the learning rate for the optimizer.
Returns:
A DNNClassifier
"""
(sepal_length, sepal_width, petal_length, petal_width) = INPUT_COLUMNS
columns = [
sepal_length,
sepal_width,
petal_length,
petal_width,
]
return tf.estimator.DNNClassifier(
config=config,
feature_columns=columns,
hidden_units=hidden_units or [256, 128, 64],
n_classes = 3,
optimizer=tf.train.AdamOptimizer(learning_rate)
)
def parse_label_column(label_string_tensor):
"""Parses a string tensor into the label tensor.
Args:
label_string_tensor: Tensor of dtype string. Result of parsing the CSV
column specified by LABEL_COLUMN.
Returns:
A Tensor of the same shape as label_string_tensor, should return
an int64 Tensor representing the label index for classification tasks,
and a float32 Tensor representing the value for a regression task.
"""
# Build a Hash Table inside the graph
table = tf.contrib.lookup.index_table_from_tensor(tf.constant(LABELS))
# Use the hash table to convert string labels to ints and one-hot encode
return table.lookup(label_string_tensor)
# [START serving-function]
def csv_serving_input_fn():
"""Build the serving inputs."""
csv_row = tf.placeholder(shape=[None], dtype=tf.string)
features = _decode_csv(csv_row)
# Ignore label column
features.pop(LABEL_COLUMN)
return tf.estimator.export.ServingInputReceiver(features,
{'csv_row': csv_row})
def json_serving_input_fn():
"""Build the serving inputs."""
inputs = {}
for feat in INPUT_COLUMNS:
inputs[feat.name] = tf.placeholder(shape=[None], dtype=feat.dtype)
return tf.estimator.export.ServingInputReceiver(inputs, inputs)
# [END serving-function]
SERVING_FUNCTIONS = {
'JSON': json_serving_input_fn,
'CSV': csv_serving_input_fn
}
def _decode_csv(line):
"""Takes the string input tensor and returns a dict of rank-2 tensors."""
# Takes a rank-1 tensor and converts it into rank-2 tensor
row_columns = tf.expand_dims(line, -1)
columns = tf.decode_csv(row_columns, record_defaults=CSV_COLUMN_DEFAULTS)
features = dict(zip(CSV_COLUMNS, columns))
# Remove unused columns
for col in UNUSED_COLUMNS:
features.pop(col)
return features
def input_fn(filenames,
num_epochs=None,
shuffle=True,
skip_header_lines=1,
batch_size=200):
"""Generates features and labels for training or evaluation.
This uses the input pipeline based approach using file name queue
to read data so that entire data is not loaded in memory.
"""
dataset = tf.data.TextLineDataset(filenames).skip(skip_header_lines).map(
_decode_csv)
if shuffle:
dataset = dataset.shuffle(buffer_size=batch_size * 10)
iterator = dataset.repeat(num_epochs).batch(
batch_size).make_one_shot_iterator()
features = iterator.get_next()
return features, parse_label_column(features.pop(LABEL_COLUMN))
```

代码大部分是自解释的；然而，读者应注意以下要点：

+   函数 ‘build_estimator’ 使用预制的 Estimator API 在 Cloud MLE 上训练一个 ‘DNNClassifier’ 模型。模型的学习率和隐藏单元可以在训练期间作为超参数进行调整和调优。

+   方法 ‘csv_serving_input_fn’ 和 ‘json_serving_input_fn’ 定义了 CSV 和 JSON 服务输入格式。

+   方法 ‘input_fn’ 使用 TensorFlow Dataset API 构建在 Cloud MLE 上进行训练和评估的输入管道。此方法调用私有方法 _decode_csv() 将 CSV 列转换为张量。

## 应用程序逻辑

让我们看看文件 ‘**task.py**’ 中的应用程序逻辑。

```py
import argparse
import json
import os
import tensorflow as tf
from tensorflow.contrib.training.python.training import hparam
import trainer.model as model
def _get_session_config_from_env_var():
"""Returns a tf.ConfigProto instance that has appropriate device_filters set.
"""
tf_config = json.loads(os.environ.get('TF_CONFIG', '{}'))
if (tf_config and 'task' in tf_config and 'type' in tf_config['task'] and
'index' in tf_config['task']):
# Master should only communicate with itself and ps
if tf_config['task']['type'] == 'master':
return tf.ConfigProto(device_filters=['/job:ps', '/job:master'])
# Worker should only communicate with itself and ps
elif tf_config['task']['type'] == 'worker':
return tf.ConfigProto(device_filters=[
'/job:ps',
'/job:worker/task:%d' % tf_config['task']['index']
])
return None
def train_and_evaluate(hparams):
"""Run the training and evaluate using the high level API."""
train_input = lambda: model.input_fn(
hparams.train_files,
num_epochs=hparams.num_epochs,
batch_size=hparams.train_batch_size
)
# Don't shuffle evaluation data
eval_input = lambda: model.input_fn(
hparams.eval_files,
batch_size=hparams.eval_batch_size,
shuffle=False
)
train_spec = tf.estimator.TrainSpec(
train_input, max_steps=hparams.train_steps)
exporter = tf.estimator.FinalExporter(
'iris', model.SERVING_FUNCTIONS[hparams.export_format])
eval_spec = tf.estimator.EvalSpec(
eval_input,
steps=hparams.eval_steps,
exporters=[exporter],
name='iris-eval')
run_config = tf.estimator.RunConfig(
session_config=_get_session_config_from_env_var())
run_config = run_config.replace(model_dir=hparams.job_dir)
print('Model dir %s' % run_config.model_dir)
estimator = model.build_estimator(
learning_rate=hparams.learning_rate,
# Construct layers sizes with exponential decay
hidden_units=[
max(2, int(hparams.first_layer_size * hparams.scale_factor**i))
for i in range(hparams.num_layers)
],
config=run_config)
tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
if __name__ == '__main__':
parser = argparse.ArgumentParser()
# Input Arguments
parser.add_argument(
'--train-files',
help='GCS file or local paths to training data',
nargs='+',
default='gs://iris-dataset/train_data.csv')
parser.add_argument(
'--eval-files',
help='GCS file or local paths to evaluation data',
nargs='+',
default='gs://iris-dataset/test_data.csv')
parser.add_argument(
'--job-dir',
help='GCS location to write checkpoints and export models',
default='/tmp/iris-estimator')
parser.add_argument(
'--num-epochs',
help="""\
Maximum number of training data epochs on which to train.
If both --max-steps and --num-epochs are specified,
the training job will run for --max-steps or --num-epochs,
whichever occurs first. If unspecified will run for --max-steps.\
""",
type=int)
parser.add_argument(
'--train-batch-size',
help='Batch size for training steps',
type=int,
default=20)
parser.add_argument(
'--eval-batch-size',
help='Batch size for evaluation steps',
type=int,
default=20)
parser.add_argument(
'--learning_rate',
help='The training learning rate',
default=1e-4,
type=int)
parser.add_argument(
'--first-layer-size',
help='Number of nodes in the first layer of the DNN',
default=256,
type=int)
parser.add_argument(
'--num-layers', help='Number of layers in the DNN', default=3, type=int)
parser.add_argument(
'--scale-factor',
help='How quickly should the size of the layers in the DNN decay',
default=0.7,
type=float)
parser.add_argument(
'--train-steps',
help="""\
Steps to run the training job for. If --num-epochs is not specified,
this must be. Otherwise the training job will run indefinitely.\
""",
default=100,
type=int)
parser.add_argument(
'--eval-steps',
help='Number of steps to run evalution for at each checkpoint',
default=100,
type=int)
parser.add_argument(
'--export-format',
help='The input format of the exported SavedModel binary',
choices=['JSON', 'CSV'],
default='CSV')
parser.add_argument(
'--verbosity',
choices=['DEBUG', 'ERROR', 'FATAL', 'INFO', 'WARN'],
default='INFO')
args, _ = parser.parse_known_args()
# Set python level verbosity
tf.logging.set_verbosity(args.verbosity)
# Set C++ Graph Execution level verbosity
os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(
tf.logging.__dict__[args.verbosity] / 10)
# Run the training job
hparams = hparam.HParams(**args.__dict__)
train_and_evaluate(hparams)
```

注意以下代码：

+   方法‘_get_session_config_from_env_var()’定义了 Estimator 在 Cloud MLE 上的运行时环境配置。

+   方法‘train_and_evaluate()’执行了多个编排事件，包括

    +   将训练和评估数据集路由到‘model.py’中的模型函数

    +   设置 Estimator 的运行时环境

    +   将超参数传递给 Estimator 模型

+   代码行“if __name__ == ‘__main__’:”通过终端会话定义了 Python 脚本的入口点。在此脚本中，代码将通过‘argparse.ArgumentParser()’方法从终端接收输入。

## 在 Cloud MLE 上进行训练

训练执行代码是存储在 shell 脚本中的 bash 命令。Shell 脚本以‘.sh’后缀结尾。

### 运行单个实例训练作业

下面的 bash 代码展示了在 Cloud MLE 上执行单个实例训练的代码。相应地更改存储桶名称。

```py
DATE=`date '+%Y%m%d_%H%M%S'`
export JOB_NAME=iris_$DATE
export GCS_JOB_DIR=gs://iris-dataset/jobs/$JOB_NAME
export TRAIN_FILE=gs://iris-dataset/train_data.csv
export EVAL_FILE=gs://iris-dataset/test_data.csv
echo $GCS_JOB_DIR
gcloud ai-platform jobs submit training $JOB_NAME \
--stream-logs \
--runtime-version 1.8 \
--job-dir $GCS_JOB_DIR \
--module-name trainer.task \
--package-path trainer/ \
--region us-central1 \
-- \
--train-files $TRAIN_FILE \
--eval-files $EVAL_FILE \
--train-steps 5000 \
--eval-steps 100
```

此代码存储在文件‘single-instance-training.sh’中，并通过在终端上运行命令来执行。

```py
source ./scripts/single-instance-training.sh
'Output:'
gs://iris-dataset/jobs/iris_20181112_010123
Job [iris_20181112_010123] submitted successfully.
INFO    2018-11-12 01:01:25 -0500   service     Validating job requirements...
INFO    2018-11-12 01:01:26 -0500   service     Job creation request has been successfully validated.
INFO    2018-11-12 01:01:26 -0500   service     Job iris_20181112_010123 is queued.
INFO    2018-11-12 01:01:26 -0500   service     Waiting for job to be provisioned.
INFO    2018-11-12 01:05:32 -0500   service     Waiting for training program to start.
...
INFO    2018-11-12 01:09:05 -0500   ps-replica-2        Module completed; cleaning up.
INFO    2018-11-12 01:09:05 -0500   ps-replica-2        Clean up finished.
INFO    2018-11-12 01:09:55 -0500   service             Finished tearing down training program.
INFO    2018-11-12 01:10:53 -0500   service             Job completed successfully.
endTime: '2018-11-12T01:08:35'
jobId: iris_20181112_010123
startTime: '2018-11-12T01:07:34'
state: SUCCEEDED
```

### 运行分布式训练作业

下面的代码展示了在 Cloud MLE 上启动分布式训练的代码，该代码存储在文件‘distributed-training.sh’中。对于分布式作业，属性‘- -scale-tier’设置为高于基本机器类型的级别。相应地更改存储桶名称。

```py
export SCALE_TIER=STANDARD_1 # BASIC | BASIC_GPU | STANDARD_1 | PREMIUM_1 | BASIC_TPU
DATE=`date '+%Y%m%d_%H%M%S'`
export JOB_NAME=iris_$DATE
export GCS_JOB_DIR=gs://iris-dataset/jobs/$JOB_NAME
export TRAIN_FILE=gs://iris-dataset/train_data.csv
export EVAL_FILE=gs://iris-dataset/test_data.csv
echo $GCS_JOB_DIR
gcloud ai-platform jobs submit training $JOB_NAME \
--stream-logs \
--scale-tier $SCALE_TIER \
--runtime-version 1.8 \
--job-dir $GCS_JOB_DIR \
--module-name trainer.task \
--package-path trainer/ \
--region us-central1 \
-- \
--train-files $TRAIN_FILE \
--eval-files $EVAL_FILE \
--train-steps 5000 \
--eval-steps 100
```

下面的代码执行分布式训练作业。

```py
source ./scripts/distributed-training.sh
```

### 运行具有超参数调整的分布式训练作业

要运行具有超参数调整的训练作业，请添加‘- -config’属性并链接到‘.yaml’超参数配置文件。运行作业的代码相同，但添加了属性‘- -config’。相应地更改存储桶名称。

```py
export SCALE_TIER=STANDARD_1 # BASIC | BASIC_GPU | STANDARD_1 | PREMIUM_1 | BASIC_TPU
DATE=`date '+%Y%m%d_%H%M%S'`
export JOB_NAME=iris_$DATE
export HPTUNING_CONFIG=hptuning_config.yaml
export GCS_JOB_DIR=gs://iris-dataset/jobs/$JOB_NAME
export TRAIN_FILE=gs://iris-dataset/train_data.csv
export EVAL_FILE=gs://iris-dataset/test_data.csv
echo $GCS_JOB_DIR
gcloud ai-platform jobs submit training $JOB_NAME \
--stream-logs \
--scale-tier $SCALE_TIER \
--runtime-version 1.8 \
--config $HPTUNING_CONFIG \
--job-dir $GCS_JOB_DIR \
--module-name trainer.task \
--package-path trainer/ \
--region us-central1 \
-- \
--train-files $TRAIN_FILE \
--eval-files $EVAL_FILE \
--train-steps 5000 \
--eval-steps 100
```

### hptuning_config.yaml 文件

此文件包含我们希望在 Cloud MLE 上调整训练作业时探索的超参数和范围。调整作业的目标是“最大化”‘accuracy’指标。

```py
trainingInput:
hyperparameters:
goal: MAXIMIZE
hyperparameterMetricTag: accuracy
maxTrials: 4
maxParallelTrials: 2
params:
- parameterName: learning-rate
type: DOUBLE
minValue: 0.00001
maxValue: 0.005
scaleType: UNIT_LOG_SCALE
- parameterName: first-layer-size
type: INTEGER
minValue: 50
maxValue: 500
scaleType: UNIT_LINEAR_SCALE
- parameterName: num-layers
type: INTEGER
minValue: 1
maxValue: 15
scaleType: UNIT_LINEAR_SCALE
- parameterName: scale-factor
type: DOUBLE
minValue: 0.1
maxValue: 1.0
scaleType: UNIT_REVERSE_LOG_SCALE
```

## 执行具有超参数调整的训练作业

在终端上运行以下代码以启动分布式训练作业。

```py
source ./scripts/hyper-tune.sh
gs://iris-dataset/jobs/iris_20181114_190121
Job [iris_20181114_190121] submitted successfully.
INFO    2018-11-14 12:41:07 -0500   service     Validating job requirements...
INFO    2018-11-14 12:41:07 -0500   service     Job creation request has been successfully validated.
INFO    2018-11-14 12:41:08 -0500   service     Job iris_20181114_190121 is queued.
INFO    2018-11-14 12:41:18 -0500   service     Waiting for job to be provisioned.
INFO    2018-11-14 12:41:18 -0500   service     Waiting for job to be provisioned.
...
INFO    2018-11-14 12:56:38 -0500   service     Finished tearing down training program.
INFO    2018-11-14 12:56:45 -0500   service     Finished tearing down training program.
INFO    2018-11-14 12:57:37 -0500   service     Job completed successfully.
INFO    2018-11-14 12:57:43 -0500   service     Job completed successfully.
endTime: '2018-11-14T13:04:34'
jobId: iris_20181114_190121
startTime: '2018-11-14T12:41:12'
state: SUCCEEDED
```

图 41-3 展示了超参数训练作业的详细信息。

![../images/463852_1_En_41_Chapter/463852_1_En_41_Fig3_HTML.jpg](img/463852_1_En_41_Fig3_HTML.jpg)

图 41-3

作业详情：Cloud MLE 上的超参数分布式训练作业

在**“训练输出”**下，第一个**“trialID”**包含最小化成本函数并在评估指标上表现最佳的超参数集。观察红色框内的试验运行，其**“objectiveValue”**属性具有最高的准确度值。这如图 41-4 所示。

![../images/463852_1_En_41_Chapter/463852_1_En_41_Fig4_HTML.jpg](img/463852_1_En_41_Fig4_HTML.jpg)

图 41-4

选择最佳的超参数集

## 在 Cloud MLE 上进行预测

要在 Cloud MLE 上进行预测，我们首先创建一个预测实例。为此，运行以下‘create-prediction-service.sh’中的代码。变量‘MODEL_BINARIES’指向 GCS 上的文件夹位置，该位置存储了具有‘**trialID** = 2’超参数设置的训练模型。

```py
export MODEL_VERSION=v1
export MODEL_NAME=iris
export MODEL_BINARIES=$GCS_JOB_DIR/3/export/iris/1542241126
# Create a Cloud ML Engine model
gcloud ai-platform models create $MODEL_NAME
# Create a model version
gcloud ai-platform versions create $MODEL_VERSION \
--model $MODEL_NAME \
--origin $MODEL_BINARIES \
--runtime-version 1.8
```

运行以下代码以创建预测服务。

```py
source ./scripts/create-prediction-service.sh
Creating model...
Created ml engine model [projects/quantum-ally-219323/models/iris].
Creating model version...
Creating version (this might take a few minutes)......done.
```

创建的模型版本详情如图 41-5 所示。

![../images/463852_1_En_41_Chapter/463852_1_En_41_Fig5_HTML.jpg](img/463852_1_En_41_Fig5_HTML.jpg)

图 41-5

在 Cloud MLE 上创建用于服务的模型

## 运行批量预测

现在，让我们在 Cloud MLE 上运行一个批量预测作业。以下提供了在 Cloud MLE 上执行批量预测调用的代码，并存储在‘run-batch-predictions.sh’中。

```py
export JOB_NAME=iris_prediction
export MODEL_NAME=iris
export MODEL_VERSION=v1
export TEST_FILE=gs://iris-dataset/hold_out_test.csv
# submit a batched job
gcloud ai-platform jobs submit prediction $JOB_NAME \
--model $MODEL_NAME \
--version $MODEL_VERSION \
--data-format TEXT \
--region $REGION \
--input-paths $TEST_FILE \
--output-path $GCS_JOB_DIR/predictions
# stream job logs
echo "Job logs..."
gcloud ai-platform jobs stream-logs $JOB_NAME
# read output summary
echo "Job output summary:"
gsutil cat $GCS_JOB_DIR/predictions/prediction.results-00000-of-00001
```

使用以下命令执行代码

```py
source ./scripts/run-batch-prediction.sh
Job [iris_prediction] submitted successfully.
jobId: iris_prediction
state: QUEUED
Job logs...
INFO    2018-11-12 14:48:18 -0500   service     Validating job requirements...
INFO    2018-11-12 14:48:18 -0500   service     Job creation request has been successfully validated.
INFO    2018-11-12 14:48:19 -0500   service     Job iris_prediction is queued.
Job output summary:
Job output summary:
{"classes": ["0", "1", "2"], "scores": [8.242315743700601e-06, 0.9921771883964539, 0.007814492098987103]}
{"classes": ["0", "1", "2"], "scores": [2.7296309657032225e-09, 0.015436310321092606, 0.9845637083053589]}
{"classes": ["0", "1", "2"], "scores": [5.207379217608832e-06, 0.9999237060546875, 7.100913353497162e-05]}
........          ........          ........          ........
{"classes": ["0", "1", "2"], "scores": [0.999919056892395, 8.089694165391847e-05, 9.295699552171275e-16]}
{"classes": ["0", "1", "2"], "scores": [0.9999765157699585, 2.3535780201200396e-05, 1.2826575252518792e-17]}
{"classes": ["0", "1", "2"], "scores": [1.8082465658153524e-06, 0.7016969919204712, 0.29830116033554077]}
```

Cloud MLE 上的预测作业详情如图 41-6 所示。

![../images/463852_1_En_41_Chapter/463852_1_En_41_Fig6_HTML.jpg](img/463852_1_En_41_Fig6_HTML.jpg)

图 41-6

批量预测作业详情

## 在 Cloud MLE 上使用 GPU 进行训练

在 GPU 上训练模型可以大大减少处理时间。为了在 Cloud MLE 上使用 GPU，我们在代码示例中做了以下修改：

1.  将规模层更改为**‘CUSTOM’**。CUSTOM 层提供多个 GPU 加速器，具体如下：

    1.  standard_gpu: 单个 NVIDIA Tesla K80 GPU

    1.  complex_model_m_gpu: 四个 NVIDIA Tesla K80 GPU

    1.  complex_model_l_gpu: 八个 NVIDIA Tesla K80 GPU

    1.  standard_p100: 单个 NVIDIA Tesla P100 GPU

    1.  complex_model_m_p100: 四个 NVIDIA Tesla P100 GPU

    1.  standard_v100: 单个 NVIDIA Tesla V100 GPU

    1.  large_model_v100: 单个 NVIDIA Tesla V100 GPU

    1.  complex_model_m_v100: 四个 NVIDIA Tesla V100 GPU

    1.  complex_model_l_v100: 八个 NVIDIA Tesla V100 GPU

1.  将以下参数添加到‘.yaml’文件中，以配置 GPU 实例。

    ```py
    trainingInput:
    scaleTier: CUSTOM
    masterType: complex_model_m_gpu
    workerType: complex_model_m_gpu
    parameterServerType: large_model
    workerCount: 2
    parameterServerCount: 3
    ```

1.  ‘gpu_hptuning_config.yaml’中的完整配置文件现在如下所示：

    ```py
    trainingInput:
    scaleTier: CUSTOM
    masterType: complex_model_m_gpu
    workerType: complex_model_m_gpu
    parameterServerType: large_model
    workerCount: 2
    parameterServerCount: 3
    hyperparameters:
    goal: MAXIMIZE
    hyperparameterMetricTag: accuracy
    maxTrials: 4
    maxParallelTrials: 2
    params:
    - parameterName: learning-rate
    type: DOUBLE
    minValue: 0.00001
    maxValue: 0.005
    scaleType: UNIT_LOG_SCALE
    - parameterName: first-layer-size
    type: INTEGER
    minValue: 50
    maxValue: 500
    scaleType: UNIT_LINEAR_SCALE
    - parameterName: num-layers
    type: INTEGER
    minValue: 1
    maxValue: 15
    scaleType: UNIT_LINEAR_SCALE
    - parameterName: scale-factor
    type: DOUBLE
    minValue: 0.1
    maxValue: 1.0
    scaleType: UNIT_REVERSE_LOG_SCALE
    ```

注意，在以下区域中才可使用 Cloud MLE 上的 GPU：

+   us-east1

+   us-central1

+   us-west1

+   asia-east1

+   europe-west1

+   europe-west4

在 Cloud MLE 上使用 GPU 的更新执行代码已保存为‘gpu-hyper-tune.sh’（以下显示代码）。

```py
export SCALE_TIER=CUSTOM
DATE=`date '+%Y%m%d_%H%M%S'`
export JOB_NAME=iris_$DATE
export HPTUNING_CONFIG=gpu_hptuning_config.yaml
export GCS_JOB_DIR=gs://iris-dataset/jobs/$JOB_NAME
export TRAIN_FILE=gs://iris-dataset/train_data.csv
export EVAL_FILE=gs://iris-dataset/test_data.csv
echo $GCS_JOB_DIR
gcloud ai-platform jobs submit training $JOB_NAME \
--stream-logs \
--scale-tier $SCALE_TIER \
--runtime-version 1.8 \
--config $HPTUNING_CONFIG \
--job-dir $GCS_JOB_DIR \
--module-name trainer.task \
--package-path trainer/ \
--region us-central1 \
-- \
--train-files $TRAIN_FILE \
--eval-files $EVAL_FILE \
--train-steps 5000 \
--eval-steps 100
```

要执行代码，请运行

```py
source ./scripts/gpu-hyper-tune.sh
gs://iris-dataset/jobs/iris_20181112_211040
Job [iris_20181112_211040] submitted successfully.
...
INFO    2018-11-12 21:35:36 -0500   ps-replica-2    4   Module completed; cleaning up.
INFO    2018-11-12 21:35:36 -0500   ps-replica-2    4   Clean up finished.
INFO    2018-11-12 21:36:18 -0500   service     Finished tearing down training program.
INFO    2018-11-12 21:36:25 -0500   service     Finished tearing down training program.
INFO    2018-11-12 21:37:11 -0500   service     Job completed successfully.
INFO    2018-11-12 21:37:11 -0500   service     Job completed successfully.
endTime: '2018-11-12T21:38:26'
jobId: iris_20181112_211040
startTime: '2018-11-12T21:10:47'
state: SUCCEEDED
```

## Scikit-learn on Cloud MLE

本节将提供使用相同的 Iris 数据集示例在 Google Cloud MLE 上训练 Scikit-learn 模型的操作步骤。我们首先将适当的数据文件从本书的 GitHub 仓库移动到 GCS。

## 将数据文件移动到 GCS

按照以下步骤将数据文件移动到 GCS：

1.  创建存储数据集的存储桶。

    ```py
    gsutil mb gs://iris-sklearn
    ```

1.  在终端上运行以下命令将训练和测试数据集移动到存储桶：

**训练集特征。**

```py
gsutil cp X_train.csv gs://iris-sklearn
```

**训练集目标。**

```py
gsutil cp y_train.csv gs://iris-sklearn
```

**在线预测的测试样本。**

```py
gsutil cp test-sample.json gs://iris-sklearn
```

## 准备训练脚本

在 Cloud MLE 上训练 Scikit-learn 模型的代码也已作为 Python 包准备。项目结构如下：

Iris_SklearnCloudML: [作为父文件夹的项目名称]

+   Trainer: [包含模型和执行代码的文件夹]

    +   __init__.py: [一个空的特殊 Python 文件，表示包含的文件夹是一个 Python 包]

    +   model.py: [包含使用 Scikit-learn 编写的模型逻辑的文件]

+   scripts: [包含在 Cloud MLE 上执行作业的脚本的文件夹]

    +   single-instance-training.sh: [在 Cloud MLE 上运行单个实例训练作业的脚本]

    +   online-prediction.sh: [在 Cloud MLE 上执行在线预测作业的脚本]

    +   create-prediction-service.sh: [在 Cloud MLE 上创建预测服务的脚本]

+   config.yaml: [指定模型版本的配置文件]

在 Cloud MLE 上使用 Scikit-learn 进行训练的模型代码（如下所示）存储在文件‘model.py’中。此模型中使用的机器学习算法是随机森林分类器。

```py
# [START setup]
import datetime
import os
import subprocess
import sys
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from tensorflow.python.lib.io import file_io
# Fill in your Cloud Storage bucket name
BUCKET_ID = 'iris-sklearn'
# [END setup]
# [START download-and-load-into-pandas]
iris_data_filename = 'gs://iris-sklearn/X_train.csv'
iris_target_filename = 'gs://iris-sklearn/y_train.csv'
# Load data into pandas
with file_io.FileIO(iris_data_filename, 'r') as iris_data_f:
iris_data = pd.read_csv(filepath_or_buffer=iris_data_f,
header=None, sep=',').values
with file_io.FileIO(iris_target_filename, 'r') as iris_target_f:
iris_target = pd.read_csv(filepath_or_buffer=iris_target_f,
header=None, sep=',').values
iris_target = iris_target.reshape((iris_target.size,))
# [END download-and-load-into-pandas]
# [START train-and-save-model]
# Train the model
classifier = RandomForestClassifier()
classifier.fit(iris_data, iris_target)
# Export the classifier to a file
model = 'model.joblib'
joblib.dump(classifier, model)
# [END train-and-save-model]
# [START upload-model]
# Upload the saved model file to Cloud Storage
model_path = os.path.join('gs://', BUCKET_ID, 'model', datetime.datetime.now().strftime(
'iris_%Y%m%d_%H%M%S'), model)
subprocess.check_call(['gsutil', 'cp', model, model_path], stderr=sys.stdout)
# [END upload-model]
```

注意以下代码块中的以下要点：

+   代码使用来自包‘tensorflow.python.lib.io’的‘file.io’模块从 Cloud Storage 流式传输存储的文件。

+   以下代码块中的其余部分运行分类器以构建模型并将模型导出到 GCS 上的一个存储桶位置。当构建在线预测的预测服务时，Cloud MLE 将从该存储桶中读取。

## 在 Cloud MLE 上执行 Scikit-learn 训练作业

以下展示了用于执行 Scikit-learn 模型训练作业的 bash 代码，并保存在文件‘single-instance-training.sh’中。

```py
export SCALE_TIER=BASIC # BASIC | BASIC_GPU | STANDARD_1 | PREMIUM_1 | BASIC_TPU
DATE=`date '+%Y%m%d_%H%M%S'`
export JOB_NAME=iris_sklearn_$DATE
export GCS_JOB_DIR=gs://iris-sklearn/jobs/$JOB_NAME
echo $GCS_JOB_DIR
gcloud ml-engine jobs submit training $JOB_NAME \
--stream-logs \
--scale-tier $SCALE_TIER \
--runtime-version 1.8 \
--job-dir $GCS_JOB_DIR \
--module-name trainer.model \
--package-path trainer/ \
--region us-central1 \
--python-version 3.5
```

以下代码运行一个训练作业以构建 Scikit-learn 随机森林模型。

```py
source ./scripts/single-instance-training.sh
gs://iris-sklearn/jobs/iris_sklearn_20181119_000349
Job [iris_sklearn_20181119_000349] submitted successfully.
INFO    2018-11-19 00:03:51 -0500   service     Validating job requirements...
INFO    2018-11-19 00:03:52 -0500   service     Job creation request has been successfully validated.
INFO    2018-11-19 00:03:52 -0500   service     Job iris_sklearn_20181119_000349 is queued.
INFO    2018-11-19 00:03:52 -0500   service     Waiting for job to be provisioned.
INFO    2018-11-19 00:03:54 -0500   service     Waiting for training program to start.
...
INFO    2018-11-19 00:05:19 -0500   master-replica-0        Module completed; cleaning up.
INFO    2018-11-19 00:05:19 -0500   master-replica-0        Clean up finished.
INFO    2018-11-19 00:05:19 -0500   master-replica-0        Task completed successfully.
endTime: '2018-11-19T00:09:38'
jobId: iris_sklearn_20181119_000349
startTime: '2018-11-19T00:04:29'
state: SUCCEEDED
```

## 在 Cloud MLE 上创建 Scikit-learn 预测服务

创建预测服务的代码如下，并保存在文件‘create-prediction-service.sh’中。

```py
export MODEL_VERSION=v1
export MODEL_NAME=iris_sklearn
export REGION=us-central1
# Create a Cloud ML Engine model
echo "Creating model..."
gcloud ml-engine models create $MODEL_NAME --regions=$REGION
# Create a model version
echo "Creating model version..."
gcloud ml-engine versions create $MODEL_VERSION \
--model $MODEL_NAME \
--config config.yaml
```

上述代码引用了一个配置文件‘config.yaml’。此文件（如下所示）包含 Scikit-learn 模型的配置。让我们简要地浏览一下列出的属性：

+   deploymentUri: 此指向 Scikit-learn 模型的存储桶位置。

+   runtime version: 此属性指定 Cloud MLE 运行时版本。

+   framework: 此属性特别重要，因为它指定了正在使用的模型框架；这可以是 SCIKIT_LEARN、XGBOOST 或 TENSORFLOW。对于此示例，它设置为 SCIKIT_LEARN。

+   pythonVersion: 此属性指定正在使用的 Python 版本。

‘config.yaml’的定义如下：

```py
deploymentUri: "gs://iris-sklearn/iris_20181119_050517"
runtimeVersion: '1.8'
framework: "SCIKIT_LEARN"
pythonVersion: "3.5"
```

运行以下命令以创建预测服务。

```py
source ./scripts/create-prediction-service.sh
Creating model...
Created ml engine model [projects/quantum-ally-219323/models/iris_sklearn].
Creating model version...
Creating version (this might take a few minutes)......done.
```

## 从 Scikit-learn 模型进行在线预测

从 Scikit-learn 模型进行在线预测的代码如下，并存储在文件‘online-prediction.sh’中。在线预测中，输入数据直接作为 JSON 字符串传递。

```py
export JOB_NAME=iris_sklearn_prediction
export MODEL_NAME=iris_sklearn
export MODEL_VERSION=v1
export TEST_FILE_GCS=gs://iris-sklearn/test-sample.json
export TEST_FILE=./test-sample.json
# download file
gsutil cp $TEST_FILE_GCS .
# submit an online job
gcloud ml-engine predict --model $MODEL_NAME \
--version $MODEL_VERSION \
--json-instances $TEST_FILE
echo "0 -> setosa, 1 -> versicolor, 2 -> virginica"
```

以下展示了存储为 JSON 字符串的输入数据。

```py
[5.1, 3.5, 1.4, 0.2]
```

运行以下命令以执行对云 MLE 上托管模型的在线预测请求。

```py
source ./scripts/online-prediction.sh
Copying gs://iris-sklearn/test-sample.json...
/ [1 files][   20.0 B/   20.0 B]
Operation completed over 1 objects/20.0 B.
[0]
0 -> setosa, 1 -> versicolor, 2 -> virginica
```

在本章中，我们讨论了使用 Google Cloud Machine Learning Engine 训练大规模模型，它是 Google AI 平台的一部分。在本章的示例中，我们使用 Estimator 高级 API 和 Scikit-learn 训练了模型。重要的是要提到，Keras 高级 API 也可以用于在 Cloud MLE 上训练大规模模型。

在下一章中，我们将介绍使用 Google Cloud AutoML 训练自定义图像识别模型。

# 42. Google AutoML：云视觉

Google Cloud AutoML Vision 简化了为图像识别用例创建自定义视觉模型的过程。这项托管服务在底层使用迁移学习和神经架构搜索的概念，以找到最佳网络架构以及该架构的最优超参数配置，从而最小化模型的损失函数。本章将介绍一个使用 Google Cloud AutoML Vision 构建自定义图像识别模型的示例项目。在本章中，我们将构建一个图像模型以识别精选的谷物箱。

## 在 GCP 上启用 AutoML Cloud Vision

按以下步骤操作以在 GCP 上启用 AutoML Cloud Vision：

1.  通过点击 GCP 仪表板左上角的三个横线打开云视觉。如图 42-1 所示，在**人工智能**产品部分下选择**视觉**。

    ![../images/463852_1_En_42_Chapter/463852_1_En_42_Fig1_HTML.jpg](img/463852_1_En_42_Fig1_HTML.jpg)

    图 42-1

    打开 Google AutoML：云视觉

1.  选择如图 42-2 和 42-3 所示的 Google 用户账户以激活 AutoML。

    ![../images/463852_1_En_42_Chapter/463852_1_En_42_Fig3_HTML.jpg](img/463852_1_En_42_Fig3_HTML.jpg)

    图 42-3

    验证 AutoML

    ![../images/463852_1_En_42_Chapter/463852_1_En_42_Fig2_HTML.jpg](img/463852_1_En_42_Fig2_HTML.jpg)

    图 42-2

    选择用于验证 AutoML 的账户

1.  验证后，将打开 Google Cloud Vision 欢迎页面（见图 42-4）。

    ![../images/463852_1_En_42_Chapter/463852_1_En_42_Fig4_HTML.jpg](img/463852_1_En_42_Fig4_HTML.jpg)

    图 42-4

    云视觉欢迎页面

1.  从下拉菜单中选择将用于设置 AutoML 的**项目 ID**（已启用计费）（见图 42-5）。

    ![../images/463852_1_En_42_Chapter/463852_1_En_42_Fig5_HTML.jpg](img/463852_1_En_42_Fig5_HTML.jpg)

    图 42-5

    选择用于配置 AutoML 的项目 ID

1.  最后一步配置是在 GCP 项目上启用 AutoML API 并创建一个 GCS 存储桶以存储输出模型。点击**“立即设置”**以自动完成配置，如图 42-6 所示。

    ![../images/463852_1_En_42_Chapter/463852_1_En_42_Fig6_HTML.jpg](img/463852_1_En_42_Fig6_HTML.jpg)

    图 42-6

    自动完成 AutoML 配置

1.  当配置完成后，AutoML Vision 仪表板被激活（见图 42-7）。

    ![../images/463852_1_En_42_Chapter/463852_1_En_42_Fig7_HTML.jpg](img/463852_1_En_42_Fig7_HTML.jpg)

    图 42-7

    自动完成 AutoML 配置

## 准备训练数据集

在使用 AutoML Cloud Vision 构建自定义图像识别模型之前，数据集必须以特定格式准备；它们包括

1.  对于训练，支持 JPEG、PNG、WEBP、GIF、BMP、TIFF 和 ICO 图像格式，每个图像的最大大小为 30mb。

1.  对于推理，支持的图像格式为 JPEG、PNG 和 GIF，每个图像的最大大小为 1.5mb。

1.  最佳做法是将每个图像类别放置在图像文件夹内的包含子文件夹中。例如：

    +   [image-directory]

        +   [image-class-1-dir]

        +   [image-class-2-dir]

        +   …

        +   [image-class-n-dir]

1.  接下来，必须创建一个 CSV 文件，该文件指向图像及其对应标签的路径。AutoML 使用 CSV 文件来指向训练图像及其标签的位置。CSV 文件放置在与图像文件相同的 GCS 桶中。使用在配置 AutoML Vision 时自动创建的桶。在我们的例子中，这个桶被命名为‘gs://quantum-ally-219323-vcm’。我们使用以下代码段来创建用于谷物分类器示例的 CSV 文件。

    ```py
    import os
    import numpy as np
    import pandas as pd
    directory = 'cereal_photos/
    data = []
    # go through sub-directories in the image directory and get the image paths
    for subdir, dirs, files in os.walk(directory):
    for file in files:
    filepath = subdir + os.sep + file
    if filepath.endswith(".jpg"):
    entry = ['{}/{}'.format('gs://quantum-ally-219323-vcm',filepath), os.path.basename(subdir)]
    data.append(entry)
    # convert to Pandas DataFrame
    data_pd = pd.DataFrame(np.array(data))
    # export CSV
    data_pd.to_csv("data.csv", header=None, index=None)
    ```

1.  上述代码将生成如下所示的 CSV 样本：

    ```py
    gs://quantum-ally-219323-vcm/cereal_photos/apple_cinnamon_cheerios/001.jpg,apple_cinnamon_cheerios
    gs://quantum-ally-219323-vcm/cereal_photos/apple_cinnamon_cheerios/002.jpg,apple_cinnamon_cheerios
    gs://quantum-ally-219323-vcm/cereal_photos/apple_cinnamon_cheerios/003.jpg,apple_cinnamon_cheerios
    ...
    gs://quantum-ally-219323-vcm/cereal_photos/none_of_the_above/images_(97).jpg,none_of_the_above
    gs://quantum-ally-219323-vcm/cereal_photos/none_of_the_above/images_(98).jpg,none_of_the_above
    gs://quantum-ally-219323-vcm/cereal_photos/none_of_the_above/images_(99).jpg,none_of_the_above
    ...
    gs://quantum-ally-219323-vcm/cereal_photos/sugar_crisp/001.jpg,sugar_crisp
    gs://quantum-ally-219323-vcm/cereal_photos/sugar_crisp/002.jpg,sugar_crisp
    gs://quantum-ally-219323-vcm/cereal_photos/sugar_crisp/003.jpg,sugar_crisp
    ```

    第一部分是图像路径或 URI，而另一部分是图像标签。

1.  在准备图像数据集时，拥有一个“**None_of_the_above**”图像类别是有用的。这个类别将包含不属于任何预测类别的随机图像。添加此类别可以对模型的整体准确性产生作用。

1.  将 GitHub 书籍仓库克隆到 Notebook 实例。

1.  导航到文件夹章节，并将图像文件复制到 GCS 桶。

    ```py
    gsutil cp -r cereal_photos gs://quantum-ally-219323-vcm
    ```

1.  将包含图像路径及其标签的 CSV 数据文件复制到 GCS 桶。

```py
gsutil cp data.csv gs://quantum-ally-219323-vcm/cereal_photos/
```

## 在 Cloud AutoML Vision 上构建自定义图像模型

在 Cloud Vision 的 AutoML 中，一个数据集包含用于构建分类器的图像及其相应的标签。本节将介绍创建数据集并在 AutoML Vision 上构建自定义图像模型的过程。

1.  从 Cloud AutoML Vision 仪表板，如图 42-8 所示，点击**新建数据集**。

    ![../images/463852_1_En_42_Chapter/463852_1_En_42_Fig8_HTML.jpg](img/463852_1_En_42_Fig8_HTML.jpg)

    图 42-8

    在 AutoML Vision 上创建新数据集

1.  要在 Cloud AutoML Vision 上创建数据集，设置以下参数，如图 42-9 所示：

    1.  数据集名称：cereal_classifier。

    1.  在云存储中选择一个 CSV 文件（这是在配置 Cloud AutoML 时创建的存储桶中放置的 CSV 文件，其中包含图像的路径）：gs://quantum-ally-219323-vcm/cereal_photos/data.csv。

    1.  点击**创建数据集**开始导入图像（见图 42-10）。

        ![../images/463852_1_En_42_Chapter/463852_1_En_42_Fig10_HTML.jpg](img/463852_1_En_42_Fig10_HTML.jpg)

        图 42-10

        Cloud AutoML Vision：导入图像

    ![../images/463852_1_En_42_Chapter/463852_1_En_42_Fig9_HTML.jpg](img/463852_1_En_42_Fig9_HTML.jpg)

    图 42-9

    在 Cloud AutoML Vision 上创建数据集

1.  导入数据集后，点击**训练**（见图 42-11）以启动构建自定义图像识别模型的过程。

    ![../images/463852_1_En_42_Chapter/463852_1_En_42_Fig11_HTML.jpg](img/463852_1_En_42_Fig11_HTML.jpg)

    图 42-11

    Cloud AutoML Vision：导入的图像及其标签

1.  在机器学习中，更多的标记训练示例可以提高模型的性能。同样，在使用 AutoML 时，每个图像类至少应有 100 个训练示例。在本节使用的示例中，某些类别的示例不足 100 个，因此 AutoML 会发出警告，如图 42-12 所示。然而，为了完成这个练习，我们将继续进行训练。点击**开始训练**。

    ![../images/463852_1_En_42_Chapter/463852_1_En_42_Fig12_HTML.jpg](img/463852_1_En_42_Fig12_HTML.jpg)

    图 42-12

    Cloud AutoML Vision 请求每个图像类更多的训练示例

1.  选择模型训练的时长。更长的训练时间可能会影响模型精度，但这可能会在 Cloud AutoML 的机器上运行时增加成本（见图 42-13）。再次点击**开始训练**以开始构建模型（见图 42-14）。

    ![../images/463852_1_En_42_Chapter/463852_1_En_42_Fig14_HTML.jpg](img/463852_1_En_42_Fig14_HTML.jpg)

    图 42-14

    在 Cloud AutoML Vision 上训练视觉模型

    ![../images/463852_1_En_42_Chapter/463852_1_En_42_Fig13_HTML.jpg](img/463852_1_En_42_Fig13_HTML.jpg)

    图 42-13

    选择训练预算

1.  训练摘要显示在图 42-15 中。

    ![../images/463852_1_En_42_Chapter/463852_1_En_42_Fig15_HTML.jpg](img/463852_1_En_42_Fig15_HTML.jpg)

    图 42-15

    Cloud AutoML Vision：训练摘要

1.  AutoML Vision 使用预留的测试图像来评估训练后的模型质量，如图 42-16 所示。显示精确度和召回率之间权衡的 F1 图如图 42-17 所示。此外，还提供了一个可视化的混淆矩阵以进一步评估模型质量（见图 42-18）。

    ![../images/463852_1_En_42_Chapter/463852_1_En_42_Fig18_HTML.png](img/463852_1_En_42_Fig18_HTML.png)

    图 42-18

    Cloud AutoML Vision 上模型评估的混淆矩阵

    ![../images/463852_1_En_42_Chapter/463852_1_En_42_Fig17_HTML.jpg](img/463852_1_En_42_Fig17_HTML.jpg)

    图 42-17

    Cloud AutoML Vision 上的 F1 评估矩阵

    ![../images/463852_1_En_42_Chapter/463852_1_En_42_Fig16_HTML.jpg](img/463852_1_En_42_Fig16_HTML.jpg)

    图 42-16

    Cloud AutoML Vision：模型评估

1.  定制的图像识别模型作为 REST 或 Python API 公开，以便作为预测服务集成到软件应用程序中（见图 42-19）。我们可以通过上传一个用于分类的样本图像来测试我们的模型，如图 42-20 所示。

    ![../images/463852_1_En_42_Chapter/463852_1_En_42_Fig20_HTML.jpg](img/463852_1_En_42_Fig20_HTML.jpg)

    图 42-20

    在 Cloud AutoML Vision 上测试预测服务

    ![../images/463852_1_En_42_Chapter/463852_1_En_42_Fig19_HTML.png](img/463852_1_En_42_Fig19_HTML.png)

    图 42-19

    Cloud AutoML Vision：作为预测服务的模型

1.  要删除一个模型，点击三个短横线并选择“模型”以导航到模型仪表板（见图 42-21）。在模型旁边点击三个点，然后选择“删除模型”（见图 42-22）。如图 42-23 所示确认删除。然而，请注意，与已删除模型相关的 API 调用将停止运行。

    ![../images/463852_1_En_42_Chapter/463852_1_En_42_Fig23_HTML.jpg](img/463852_1_En_42_Fig23_HTML.jpg)

    图 42-23

    在 Cloud AutoML Vision 上删除一个模型

    ![../images/463852_1_En_42_Chapter/463852_1_En_42_Fig22_HTML.jpg](img/463852_1_En_42_Fig22_HTML.jpg)

    图 42-22

    选择要删除的模型

    ![../images/463852_1_En_42_Chapter/463852_1_En_42_Fig21_HTML.jpg](img/463852_1_En_42_Fig21_HTML.jpg)

    图 42-21

    返回模型仪表板

本章介绍了使用 Google AutoML Cloud Vision 构建和部署自定义图像分类模型。在下一章中，我们将探讨如何使用 Google Cloud AutoML 构建和部署自定义文本分类模型，用于自然语言处理。

# 43. Google AutoML：云自然语言处理

本章将构建一个语言毒性分类模型，使用 Google Cloud AutoML 进行自然语言处理（NLP），以对有毒和无毒或干净的短语进行分类和识别。本项目使用的数据来自 Kaggle 上 Jigsaw 和 Google 举办的毒性评论分类挑战赛。数据经过修改，以包含 16,000 个有毒和 16,000 个无毒单词作为输入，以在 AutoML NLP 上构建模型。

## 启用 GCP 上的 AutoML NLP

以下步骤将启用 GCP 上的 AutoML NLP：

1.  点击界面左上角的三个短横线，然后在类别人工智能下选择**自然语言**，如图 43-1 所示。

    ![../images/463852_1_En_43_Chapter/463852_1_En_43_Fig1_HTML.jpg](img/463852_1_En_43_Fig1_HTML.jpg)

    图 43-1

    打开 Cloud AutoML 自然语言

1.  在接下来的屏幕上，点击**开始使用 AutoML**（见图 43-2）。

    ![../images/463852_1_En_43_Chapter/463852_1_En_43_Fig2_HTML.jpg](img/463852_1_En_43_Fig2_HTML.jpg)

    图 43-2

    点击“开始使用 Cloud AutoML NLP”

1.  点击**立即设置**来自动设置用于与 Cloud AutoML NLP 一起工作的 GCP 项目（见图 43-3）。此过程涉及激活 AutoML API 并在 GCP 上创建一个存储数据输入和输出模型的存储桶。我们将在下一节中使用此存储桶。

    ![../images/463852_1_En_43_Chapter/463852_1_En_43_Fig3_HTML.jpg](img/463852_1_En_43_Fig3_HTML.jpg)

    图 43-3

    在 Cloud AutoML NLP 中自动配置云自动机器学习 NLP

1.  配置后，Cloud AutoML NLP 仪表板被激活（见图 43-4）。

    ![../images/463852_1_En_43_Chapter/463852_1_En_43_Fig4_HTML.png](img/463852_1_En_43_Fig4_HTML.png)

    图 43-4

    AutoML NLP 仪表板

## 准备训练数据集

让我们逐步准备使用 Cloud AutoML NLP 构建自定义语言分类模型的训练数据集：

1.  训练输入可以是 (.txt) 格式的文档或 (.csv) 文件中的内联文本。多个文本可以组合成一个压缩的 (.zip) 文件。

1.  对于此项目，文本文件放置在子文件夹中，其分组输出标签作为文件夹名称。这后来用于创建包含数据文件路径及其标签的 CSV 文件。例如：

    +   [files]

        +   [toxic]

        +   [clean]

1.  接下来，必须生成一个 CSV 文件，该文件指向图像的路径及其相应的标签。就像 Cloud Vision 一样，Cloud NLP 使用 CSV 文件来指向训练文档或单词及其对应标签的位置。CSV 文件放置在创建 AutoML NLP 时创建的同一 GCS 存储桶中。在我们的例子中，这个存储桶命名为‘gs://quantum-ally-219323-lcm’。以下代码段准备数据并生成 CSV 文件。

    ```py
    import numpy as np
    import pandas as pd
    import re
    import pathlib
    import os
    # read the Toxic Comment Classification training dataset
    data = pd.read_csv('./data/train.csv')
    # add clean column label
    data['clean'] = (1 - data.iloc[:, 2:].sum(axis=1) >= 1).astype(int)
    # merge all other non-clean comments to toxic
    data.loc[data['clean'] == 0, ['toxic']] = 1
    # select dataframe of clean examples
    data_clean = data[data['clean'] == 1].sample(n=20000)
    # select dataframe of toxic examples
    data_toxic = data[data['toxic'] == 1].sample(n=16000)
    # join into one dataframe
    data = pd.concat([data_clean, data_toxic])
    # remove unused columns
    data.drop(['severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate'], axis=1, inplace=True)
    # create text documents and place them in their folder classes.
    for index, row in data.iterrows():
    comment_text = re.sub(r'[^\w\s]',",row['comment_text']).rstrip().lstrip().strip()
    classes = "
    if (row['toxic'] == 1):
    classes = 'toxic'
    else:
    classes = 'clean'
    pathlib.Path("./file/{}".format(classes)).mkdir(parents=True, exist_ok=True)
    with open("./file/{}/text_{}.txt".format(classes,index), "w") as text_file:
    text_file.write(comment_text)
    data_path = []
    directory = 'file/'
    # create data csv
    for subdir, dirs, files in os.walk(directory):
    for file in files:
    filepath = subdir + os.sep + file
    if filepath.endswith(".txt"):
    entry = ['{}/{}'.format('gs://quantum-ally-219323-lcm',filepath), os.path.basename(subdir)]
    data_path.append(entry)
    # convert to Pandas DataFrame
    data_pd = pd.DataFrame(np.array(data_path))
    # export data to csv
    data_pd.to_csv("data.csv", header=None, index=None)
    ```

1.  上述代码将生成如下所示的 CSV 示例：

    ```py
    gs://quantum-ally-219323-lcm/file/clean/text_100055.txt,clean
    gs://quantum-ally-219323-lcm/file/clean/text_100059.txt,clean
    gs://quantum-ally-219323-lcm/file/clean/text_100077.txt,clean
    ...
    gs://quantum-ally-219323-lcm/file/toxic/text_141122.txt,toxic
    gs://quantum-ally-219323-lcm/file/toxic/text_141138.txt,toxic
    gs://quantum-ally-219323-lcm/file/toxic/text_141143.txt,toxic
    ```

    第一部分是图像路径或 URI，而另一部分是文档标签。

1.  在准备文本数据集时，拥有一个“**None_of_the_above**”类别很有用。这个类别将包含不属于任何预测类别的文档。添加此类别可以对模型精度产生整体影响。

1.  导航到“chapter”文件夹，并将图像文件复制到 GCS 存储桶。标志 **-m** 启动并行上传以加快大型文档上传到 GCP 的时间。

    ```py
    gsutil -m cp -r file gs://quantum-ally-219323-lcm
    ```

1.  将包含文档路径及其标签的 CSV 数据文件复制到 GCS 存储桶。

    ```py
    gsutil cp data.csv gs://quantum-ally-219323-lcm/file/
    ```

## 在 Cloud AutoML NLP 上构建自定义语言分类模型

本节将指导如何创建文档数据集并在 AutoML 视觉上构建自定义语言分类模型。

1.  从 Cloud AutoML NLP 仪表板，如图 43-5 所示，点击 **NEW DATASET**。

    ![../images/463852_1_En_43_Chapter/463852_1_En_43_Fig5_HTML.jpg](img/463852_1_En_43_Fig5_HTML.jpg)

    图 43-5

    AutoML NLP 上的新数据集

1.  要在 Cloud AutoML NLP 上创建数据集，请设置以下参数，如图 43-6 所示：

    1.  数据集名称：toxicity_dataset。

    1.  在云存储中选择一个 CSV 文件（这是在配置 Cloud AutoML 时创建的存储桶中放置的 CSV 文件，其中包含文本文档的路径）：gs://quantum-ally-219323-lcm/file/data.csv。

    1.  点击 **CREATE DATASET** 开始导入图像（见图 43-7）。

        ![../images/463852_1_En_43_Chapter/463852_1_En_43_Fig8_HTML.jpg](img/463852_1_En_43_Fig8_HTML.jpg)

        图 43-8

        Cloud AutoML NLP：导入的文本文档及其标签

        ![../images/463852_1_En_43_Chapter/463852_1_En_43_Fig7_HTML.jpg](img/463852_1_En_43_Fig7_HTML.jpg)

        图 43-7

        Cloud AutoML NLP：导入文本项

    ![../images/463852_1_En_43_Chapter/463852_1_En_43_Fig6_HTML.jpg](img/463852_1_En_43_Fig6_HTML.jpg)

    图 43-6

    在 Cloud AutoML NLP 上创建数据集

1.  在导入数据集后，点击 **TRAIN**（见图 43-8）以启动构建自定义语言分类模型的过程。

1.  在本例中，我们拥有足够的训练示例，如图 43-9 所示，因此有理由期待一个良好的语言分类模型。点击 **START TRAINING** 开始训练任务。

    ![../images/463852_1_En_43_Chapter/463852_1_En_43_Fig9_HTML.jpg](img/463852_1_En_43_Fig9_HTML.jpg)

    图 43-9

    Cloud AutoML NLP 检查训练示例的充分性

1.  接受默认模型名称，并点击 **START TRAINING**（见图 43-10）以开始构建模型，如图 43-11 所示。请注意，这次训练可能需要大约一个小时才能完成。完成后，用户将收到完成通知的电子邮件。

    ![../images/463852_1_En_43_Chapter/463852_1_En_43_Fig11_HTML.jpg](img/463852_1_En_43_Fig11_HTML.jpg)

    图 43-11

    在 Cloud AutoML NLP 上训练文本分类模型

    ![../images/463852_1_En_43_Chapter/463852_1_En_43_Fig10_HTML.jpg](img/463852_1_En_43_Fig10_HTML.jpg)

    图 43-10

    接受模型名称，并点击“开始训练”

1.  训练摘要如图 43-12 所示。训练阶段持续了大约 4 小时 45 分钟。

    ![../images/463852_1_En_43_Chapter/463852_1_En_43_Fig12_HTML.jpg](img/463852_1_En_43_Fig12_HTML.jpg)

    图 43-12

    Cloud AutoML NLP：训练摘要

1.  AutoML NLP 在训练后为了评估模型的质量，将一部分文档留作测试集（见图 43-13）。F1 图显示了精确度和召回率之间的权衡。此外，混淆矩阵提供了对模型质量的进一步洞察（见图 43-14))。

    ![../images/463852_1_En_43_Chapter/463852_1_En_43_Fig14_HTML.jpg](img/463852_1_En_43_Fig14_HTML.jpg)

    图 43-14

    Cloud AutoML NLP 上的 F1 评估图和混淆矩阵

    ![../images/463852_1_En_43_Chapter/463852_1_En_43_Fig13_HTML.jpg](img/463852_1_En_43_Fig13_HTML.jpg)

    图 43-13

    Cloud AutoML NLP：模型评估

1.  自定义文本分类模型作为 REST 或 Python API 暴露出来，以便将其集成到软件应用程序中作为预测服务（见图 43-15）。我们可以通过上传一个用于分类的样本图像来测试我们的模型。图 43-16 将一个清洁文本示例传递给模型，并正确预测，概率为 98%，而图 43-17 将一个有毒文本示例传递给模型。此示例也被正确分类，概率得分为 99%。

    ![../images/463852_1_En_43_Chapter/463852_1_En_43_Fig17_HTML.jpg](img/463852_1_En_43_Fig17_HTML.jpg)

    图 43-17

    有毒词汇示例：AutoML NLP

    ![../images/463852_1_En_43_Chapter/463852_1_En_43_Fig16_HTML.jpg](img/463852_1_En_43_Fig16_HTML.jpg)

    图 43-16

    清洁词汇示例：AutoML NLP

    ![../images/463852_1_En_43_Chapter/463852_1_En_43_Fig15_HTML.jpg](img/463852_1_En_43_Fig15_HTML.jpg)

    图 43-15

    云 AutoML NLP 模型作为预测服务

本章介绍了使用 Google AutoML Cloud Vision 构建和部署自定义文本分类模型。在下一章中，我们将在 GCP 上构建一个端到端的数据科学产品。

# 44. 预测超导体临界温度的模型

本章构建了一个回归机器学习模型来预测超导体的临界温度。该数据集的特征是基于以下超导体属性推导出来的：

+   原子质量

+   首电离能

+   原子半径

+   密度

+   电子亲和力

+   融合热

+   热导率

+   电荷

对于每个属性，提取了平均值、加权平均值、几何平均值、加权几何平均值、熵、加权熵、范围、加权范围、标准差和加权标准差。因此，这导致了总共 8 x 10 = 80 个特征。除此之外，还向设计矩阵中添加了一个包含超导体中元素数量的特征。预测变量是超导体的临界温度。因此，数据集总共有 81 个特征和 21,263 行。

该数据集由宾夕法尼亚大学的 Kam Hamidieh 提供，并提交到 UCI 机器学习库。本节的目标是展示在 Google Cloud Platform 上交付端到端机器学习建模管道。

## GCP 上的建模架构

本端到端项目的目标是展示在 GCP 上使用本书中已讨论的组件构建大规模学习模型。建模架构如图 44-1 所示。让我们简要解释一下这些连接：

![../images/463852_1_En_44_Chapter/463852_1_En_44_Fig1_HTML.jpg](img/463852_1_En_44_Fig1_HTML.jpg)

图 44-1

GCP 上的建模架构

1.  将原始数据阶段在 GCS 上。

1.  将数据加载到 BigQuery 中进行分析。

1.  探索性数据分析。

1.  使用 Dataflow 进行大规模数据处理。

1.  将转换后的训练和评估数据放置在 GCS 上。

1.  在 Cloud MLE 上训练模型。

1.  将训练好的模型输出放置在 GCS 上。

1.  在 Cloud MLE 上部署模型进行推理。

## 在 GCS 上阶段原始数据

从书籍代码库中检索原始数据用于建模：

+   创建一个 GCS 存储桶。

+   导航到章节文件夹并将原始数据传输到 GCS。

```py
gsutil mb gs://superconductor
```

```py
gsutil cp train.csv gs://superconductor/raw-data/
```

## 将数据加载到 BigQuery 中进行分析。

将数据集从 Google Cloud Storage 移动到 BigQuery：

+   在 BigQuery 中创建一个数据集。

+   将 GCS 中的原始数据作为表加载到新创建的 BigQuery 数据集中。

```py
bq mk superconductor
```

+   在 BigQuery 中查看创建的表架构。

```py
bq --location=US load --autodetect --source_format=CSV superconductor.superconductor gs://superconductor/raw-data/train.csv
```

```py
bq show superconductor.superconductor
Last modified        Schema         Total Rows   Total Bytes   Expiration   Time Partitioning   Labels
------------- --------------------- ---------- ------------- ---------- ------------------- --------
08 Dec 01:16:51   |- number_of_elements: string                21264        25582000
|- mean_atomic_mass: string
|- wtd_mean_atomic_mass: string
|- wtd_mean_atomic_radius: string
|- gmean_atomic_radius: string
|- wtd_gmean_atomic_radius: string
|- entropy_atomic_radius: string
|- wtd_entropy_atomic_radius: string
...
|- range_ThermalConductivity: string
|- wtd_range_ThermalConductivity: string
|- std_ThermalConductivity: string
|- wtd_std_ThermalConductivity: string
|- mean_Valence: string
|- wtd_std_Valence: string
|- critical_temp: string
```

## 探索性数据分析

BigQuery 中的表包含 21,264 行。为了速度和快速迭代，我们不会操作这个数据集的所有行，而是选择一千行进行数据探索、转换和机器学习抽查。

```py
import pandas as pd
%%bigquery --project ekabasandbox super_cond_df
WITH super_df AS (
SELECT
number_of_elements, mean_atomic_mass, wtd_mean_atomic_mass,
gmean_atomic_mass, wtd_gmean_atomic_mass, entropy_atomic_mass,
wtd_entropy_atomic_mass, range_atomic_mass, wtd_range_atomic_mass,
std_atomic_mass, wtd_std_atomic_mass, mean_fie, wtd_mean_fie,
gmean_fie, wtd_gmean_fie, entropy_fie, wtd_entropy_fie, range_fie,
wtd_range_fie, std_fie, wtd_std_fie, mean_atomic_radius, wtd_mean_atomic_radius,
gmean_atomic_radius, wtd_gmean_atomic_radius, entropy_atomic_radius,
wtd_entropy_atomic_radius, range_atomic_radius, wtd_range_atomic_radius,
std_atomic_radius, wtd_std_atomic_radius, mean_Density, wtd_mean_Density,
gmean_Density, wtd_gmean_Density, entropy_Density, wtd_entropy_Density,
range_Density, wtd_range_Density, std_Density, wtd_std_Density, mean_ElectronAffinity,
wtd_mean_ElectronAffinity, gmean_ElectronAffinity, wtd_gmean_ElectronAffinity
entropy_ElectronAffinity, wtd_entropy_ElectronAffinity, range_ElectronAffinity,
wtd_range_ElectronAffinity, std_ElectronAffinity, wtd_std_ElectronAffinity,
mean_FusionHeat, wtd_mean_FusionHeat, gmean_FusionHeat, wtd_gmean_FusionHeat,
entropy_FusionHeat, wtd_entropy_FusionHeat, range_FusionHeat,
wtd_range_FusionHeat, std_FusionHeat, wtd_std_FusionHeat, mean_ThermalConductivity,
wtd_mean_ThermalConductivity, gmean_ThermalConductivity, wtd_gmean_ThermalConductivity,
entropy_ThermalConductivity, wtd_entropy_ThermalConductivity, range_ThermalConductivity,
wtd_range_ThermalConductivity, std_ThermalConductivity, wtd_std_ThermalConductivity,
mean_Valence, wtd_mean_Valence, gmean_Valence, wtd_gmean_Valence,
entropy_Valence, wtd_entropy_Valence, range_Valence, wtd_range_Valence,
std_Valence, wtd_std_Valence, critical_temp, ROW_NUMBER() OVER (PARTITION BY number_of_elements) row_num
FROM
`superconductor.superconductor` )
SELECT
*
FROM
super_df
LIMIT
1000
# Dataframe shape
super_cond_df.shape
```

接下来，我们将探索数据集以更深入地了解特征及其关系。这个过程被称为探索性数据分析（EDA）。

+   检查列数据类型。

```py
# check column datatypes
super_cond_df.dtypes
number_of_elements                   int64
mean_atomic_mass                   float64
wtd_mean_atomic_mass               float64
gmean_atomic_mass                  float64
wtd_gmean_atomic_mass              float64
entropy_atomic_mass                float64
wtd_entropy_atomic_mass            float64
...
range_Valence                        int64
wtd_range_Valence                  float64
std_Valence                        float64
wtd_std_Valence                    float64
critical_temp                      float64
row_num                              int64
Length: 82, dtype: object
```

从结果来看，所有数据属性都是数值类型：

+   接下来，我们将使用一个名为**pandas profiling**的工具。这个包为 Pandas DataFrame 对象生成一系列探索性数据分析。结果包括数据集的摘要统计，如变量数量、数据观测数量和缺失值数量（如果有）。它还包括每个属性的直方图可视化、描述性统计（如平均值、众数、标准差、总和、中位数绝对偏差、变异系数、峰度和偏度），以及分位数统计（如最小值、Q1、中位数、Q3、最大值、范围和四分位距）。此外，该配置文件还生成多元相关图，并列出高度相关的变量列表。

导入 pandas profiling 库。

```py
# pandas profiling
import pandas_profiling
```

运行配置文件并保存输出。

```py
# run report
profile_result = pandas_profiling.ProfileReport(super_cond_df)
```

要查看完整的报告，请运行保存的输出变量：

+   检索被拒绝的变量（即高度相关的属性）。

```py
profile_result
```

+   通过移除高度相关的变量来过滤数据集列。

```py
# get rejected variables (i.e, attributes with high correlation)
rejected_vars = profile_result.get_rejected_variables
```

+   接下来，标准化数据集的值，使它们落在相同的尺度范围内（我们将使用 Scikit-learn minmax_scale 函数）。标准化值可以提高模型的预测性能，因为优化算法可以更好地最小化成本函数。

```py
# filter from attributes set
super_cond_df.drop(rejected_vars(), axis=1, inplace=True)
```

+   此外，属性值也被归一化，以便分布更接近正态或高斯分布。这项技术也被认为对模型性能有积极影响。

```py
# scale the dataframe values
from sklearn.preprocessing import minmax_scale
dataset = pd.DataFrame(minmax_scale(super_cond_df), columns= super_cond_df.columns)
```

+   绘制变量的直方图分布（见图 44-2）。

```py
# normalize the dataframe
from sklearn.preprocessing import Normalizer
dataset = pd.DataFrame(Normalizer().fit(dataset).transform(dataset), columns=dataset.columns)
```

![../images/463852_1_En_44_Chapter/463852_1_En_44_Fig2_HTML.jpg](img/463852_1_En_44_Fig2_HTML.jpg)

图 44-2

展示变量分布的直方图

```py
# plot the histogram distribution of the variables
import matplotlib.pyplot as plt
%matplotlib inline
dataset.hist(figsize=(18, 18))
plt.show()
```

## 断点检查机器学习算法

使用我们的缩减数据集，让我们采样一些候选算法，以了解它们的性能以及哪个更有可能在这个问题域中表现最佳。让我们采取以下步骤：

+   数据集被分为设计矩阵及其相应的标签向量。

+   随机将数据集分为训练集和测试集。

```py
# split features and labels
dataset_y = dataset['critical_temp']
dataset_X = dataset.drop(['critical_temp', 'row_num'], axis=1)
```

+   概述候选算法以创建模型。

```py
# train-test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(dataset_X, dataset_y, shuffle=True)
```

+   创建候选算法的字典。

```py
# spot-check ML algorithms
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt
```

+   对于每个候选算法，使用训练集进行训练，并在保留的测试集上进行评估。

```py
ml_models = {
'Linear Reg.': LinearRegression(),
'Dec. Trees': DecisionTreeRegressor(),
'Rand. Forest': RandomForestRegressor(),
'SVM': SVR(),
'XGBoost': XGBRegressor(),
'NNets': MLPRegressor(warm_start=True, early_stopping=True, learning_rate="adaptive")
}
```

+   模型性能的图表显示在图 44-3 中。

```py
ml_results = {}
for name, model in ml_models.items():
# fit model on training data
model.fit(X_train, y_train)
# make predictions for test data
prediction = model.predict(X_test)
# evaluate predictions
rmse = sqrt(mean_squared_error(y_test, prediction))
# append accuracy results to dictionary
ml_results[name] = rmse
print('RMSE: {} -> {}'.format(name, rmse))
'Output':
RMSE: SVM -> 0.0748587427887
RMSE: XGBoost -> 0.0222440358318
RMSE: Rand. Forest -> 0.0227742725953
RMSE: Linear Reg. -> 0.025615918858
RMSE: Dec. Trees -> 0.0269103025639
RMSE: NNets -> 0.0289585489638
```

![../images/463852_1_En_44_Chapter/463852_1_En_44_Fig3_HTML.jpg](img/463852_1_En_44_Fig3_HTML.jpg)

图 44-3

机器学习算法的 RMSE 估计

```py
plt.plot(ml_results.keys(), ml_results.values(), 'o')
plt.title("RMSE estimates for ML algorithms")
plt.xlabel('Algorithms')
plt.ylabel('RMSE')
```

## 大规模数据处理中的 Dataflow 和 TensorFlow Transform

在本节中，我们使用 Google Cloud Dataflow 对庞大的数据集进行大规模数据处理。如前所述，Google Dataflow 是一个无服务器、并行和分布式的基础设施，用于运行批处理和流数据处理作业。Dataflow 是构建和部署大规模机器学习产品生产管道的关键组件。与 Cloud Dataflow 结合使用时，我们使用 TensorFlow Transform（TFT），这是一个用于预处理的 Tensorflow 库。使用 TFT 的目标是在模型训练和模型用于消费或部署时，对数据集应用一致的转换操作。在以下步骤中，每个代码块都在 Notebook 单元格中执行：

+   导入相关库。记住，Apache Beam（目前）仅支持 Python 2。更重要的是，TFT 仅与 Tensorflow 和 Apache Beam 包的特定组合一起工作。在这种情况下，TFT 0.8.0 与 TF 1.8 和 Apache Beam [GCP] 2.5.0 兼容。导入库后，务必**重启 Notebook 内核**。

在这一点上，将 Notebook 运行时类型更改为 Python 2.0。

```py
%%bash
source activate py2env
pip install --upgrade tensorflow
pip install --upgrade apache-beam[gcp]
pip install --upgrade tensorflow_transform==0.8.0
apt-get install libsnappy-dev
pip install --upgrade python-snappy==0.5.1
```

在执行 pip install 后重启内核。

+   连接到 GCP。

+   创建查询方法，从 BigQuery 检索训练和测试数据集。

```py
from google.colab import auth
auth.authenticate_user()
print('Authenticated')
# configure GCP project - update with your parameters
project_id = 'ekabasandbox'
bucket_name = 'superconductor'
region = 'us-central1'
tf_version = '1.8'
# configure gcloud
!gcloud config set project {project_id}
!gcloud config set compute/region {region}
```

+   创建 requirements.txt 文件以在 Dataflow 工作机上安装依赖项（在本例中为 tensorflow_transform）。

```py
def create_query(phase, EVERY_N=None):
"""
EVERY_N: Integer. Sample one out of every N rows from the full dataset. Larger values will yield smaller sample
phase: 1=train 2=valid
"""
base_query = """
WITH super_df AS (
SELECT
number_of_elements, mean_atomic_mass, wtd_mean_atomic_mass,
gmean_atomic_mass, wtd_gmean_atomic_mass, entropy_atomic_mass,
wtd_entropy_atomic_mass, range_atomic_mass, wtd_range_atomic_mass,
std_atomic_mass, wtd_std_atomic_mass, mean_fie, wtd_mean_fie,
gmean_fie, wtd_gmean_fie, entropy_fie, wtd_entropy_fie, range_fie,
wtd_range_fie, std_fie, wtd_std_fie, mean_atomic_radius, wtd_mean_atomic_radius,
gmean_atomic_radius, wtd_gmean_atomic_radius, entropy_atomic_radius,
wtd_entropy_atomic_radius, range_atomic_radius, wtd_range_atomic_radius,
std_atomic_radius, wtd_std_atomic_radius, mean_Density, wtd_mean_Density,
gmean_Density, wtd_gmean_Density, entropy_Density, wtd_entropy_Density,
range_Density, wtd_range_Density, std_Density, wtd_std_Density, mean_ElectronAffinity,
wtd_mean_ElectronAffinity, gmean_ElectronAffinity, wtd_gmean_ElectronAffinity
entropy_ElectronAffinity, wtd_entropy_ElectronAffinity, range_ElectronAffinity,
wtd_range_ElectronAffinity, std_ElectronAffinity, wtd_std_ElectronAffinity,
mean_FusionHeat, wtd_mean_FusionHeat, gmean_FusionHeat, wtd_gmean_FusionHeat,
entropy_FusionHeat, wtd_entropy_FusionHeat, range_FusionHeat,
wtd_range_FusionHeat, std_FusionHeat, wtd_std_FusionHeat, mean_ThermalConductivity,
wtd_mean_ThermalConductivity, gmean_ThermalConductivity, wtd_gmean_ThermalConductivity,
entropy_ThermalConductivity, wtd_entropy_ThermalConductivity, range_ThermalConductivity,
wtd_range_ThermalConductivity, std_ThermalConductivity, wtd_std_ThermalConductivity,
mean_Valence, wtd_mean_Valence, gmean_Valence, wtd_gmean_Valence,
entropy_Valence, wtd_entropy_Valence, range_Valence, wtd_range_Valence,
std_Valence, wtd_std_Valence, critical_temp, ROW_NUMBER() OVER (PARTITION BY number_of_elements) row_num
FROM
`superconductor.superconductor`)
SELECT
*
FROM
super_df
"""
if EVERY_N == None:
if phase < 2:
# training
query = "{0} WHERE MOD(row_num,4) < 2".format(base_query)
else:
query = "{0} WHERE MOD(row_num,4) = {1}".format(base_query, phase)
else:
query = "{0} WHERE MOD(row_num,{1}) = {2}".format(base_query, EVERY_N, phase)
return query
```

+   以下代码块使用 Apache Beam 构建数据预处理管道，将原始数据集转换为适合构建预测模型的形式。转换与之前对减少的数据集所执行的过程相同，包括删除具有高相关性的列并将数据集的数值缩放到相同的范围。预处理管道的输出产生一个训练集和一个评估集。Beam 管道还使用 TensorFlow Transform 保存数据转换的元数据（包括原始和处理的），以及可以后来用作部署模型服务功能的转换图。我们制作了这个示例以包含 TensorFlow Transform 的参考用途。

```py
%%writefile requirements.txt
tensorflow-transform==0.8.0
```

![../images/463852_1_En_44_Chapter/463852_1_En_44_Fig4_HTML.jpg](img/463852_1_En_44_Fig4_HTML.jpg)

图 44-4

数据流管道图

+   数据流管道图如图 44-4 所示。

```py
import datetime
import snappy
import tensorflow as tf
import apache_beam as beam
import tensorflow_transform as tft
from tensorflow_transform.beam import impl as beam_impl
def get_table_header(projection_fields):
header = "
for cnt, val in enumerate(projection_fields):
if cnt > 0:
header+=','+val
else:
header+=val
return header
def preprocess_tft(inputs):
result = {}
for attr, value in inputs.items():
result[attr] = tft.scale_to_0_1(value)
return result
def cleanup(rowdict):
# pull columns from BQ and create a line
CSV_COLUMNS = 'number_of_elements,mean_atomic_mass,entropy_atomic_mass,wtd_entropy_atomic_mass,range_atomic_mass,wtd_range_atomic_mass,mean_fie,wtd_mean_fie,wtd_entropy_fie,range_fie,wtd_range_fie,mean_atomic_radius,wtd_mean_atomic_radius,range_atomic_radius,wtd_range_atomic_radius,mean_Density,entropy_Density,wtd_entropy_Density,range_Density,wtd_range_Density,mean_ElectronAffinity,wtd_entropy_ElectronAffinity,range_ElectronAffinity,wtd_range_ElectronAffinity,mean_FusionHeat,gmean_FusionHeat,entropy_FusionHeat,wtd_entropy_FusionHeat,range_FusionHeat,wtd_range_FusionHeat,mean_ThermalConductivity,wtd_mean_ThermalConductivity,gmean_ThermalConductivity,entropy_ThermalConductivity,wtd_entropy_ThermalConductivity,range_ThermalConductivity,wtd_range_ThermalConductivity,mean_Valence,wtd_mean_Valence,range_Valence,wtd_range_Valence,wtd_std_Valence,critical_temp'.split(',')
def tofloat(value, ifnot):
try:
return float(value)
except (ValueError, TypeError):
return ifnot
result = {
k : tofloat(rowdict[k], -99) if k in rowdict else -99 for k in CSV_COLUMNS
}
row = ('{}'+',{}'*(len(result)-1)).format(result['number_of_elements'],result['mean_atomic_mass'],
result['entropy_atomic_mass'], result['wtd_entropy_atomic_mass'],result['range_atomic_mass'],
result['wtd_range_atomic_mass'],result['mean_fie'],result['wtd_mean_fie'],
result['wtd_entropy_fie'],result['range_fie'],result['wtd_range_fie'],
result['mean_atomic_radius'],result['wtd_mean_atomic_radius'],
result['range_atomic_radius'],result['wtd_range_atomic_radius'],result['mean_Density'],
result['entropy_Density'],result['wtd_entropy_Density'],result['range_Density'],
result['wtd_range_Density'],result['mean_ElectronAffinity'],
result['wtd_entropy_ElectronAffinity'],result['range_ElectronAffinity'],
result['wtd_range_ElectronAffinity'],result['mean_FusionHeat'],result['gmean_FusionHeat'],
result['entropy_FusionHeat'],result['wtd_entropy_FusionHeat'],result['range_FusionHeat'],
result['wtd_range_FusionHeat'],result['mean_ThermalConductivity'],
result['wtd_mean_ThermalConductivity'],result['gmean_ThermalConductivity'],
result['entropy_ThermalConductivity'],result['wtd_entropy_ThermalConductivity'],
result['range_ThermalConductivity'],result['wtd_range_ThermalConductivity'],
result['mean_Valence'],result['wtd_mean_Valence'],result['range_Valence'],
result['wtd_range_Valence'],result['wtd_std_Valence'],result['critical_temp'])
yield row
def preprocess():
import os
import os.path
import datetime
from apache_beam.io import WriteToText
from apache_beam.io import tfrecordio
from tensorflow_transform.coders import example_proto_coder
from tensorflow_transform.tf_metadata import dataset_metadata
from tensorflow_transform.tf_metadata import dataset_schema
from tensorflow_transform.beam import tft_beam_io
from tensorflow_transform.beam.tft_beam_io import transform_fn_io
job_name = 'preprocess-features' + '-' + datetime.datetime.now().strftime('%y%m%d-%H%M%S')
print 'Launching Dataflow job {} ... hang on'.format(job_name)
OUTPUT_DIR = 'gs://{0}/preproc_csv/'.format(bucket_name)
import subprocess
subprocess.call('gsutil rm -r {}'.format(OUTPUT_DIR).split())
EVERY_N = 3
options = {
'staging_location': os.path.join(OUTPUT_DIR, 'tmp', 'staging'),
'temp_location': os.path.join(OUTPUT_DIR, 'tmp'),
'job_name': job_name,
'project': project_id,
'max_num_workers': 24,
'teardown_policy': 'TEARDOWN_ALWAYS',
'no_save_main_session': True,
'requirements_file': 'requirements.txt'
}
opts = beam.pipeline.PipelineOptions(flags=[], **options)
RUNNER = 'DataflowRunner'
# set up metadata
raw_data_schema = {
colname : dataset_schema.ColumnSchema(tf.float32, [], dataset_schema.FixedColumnRepresentation())
for colname in 'number_of_elements,mean_atomic_mass,entropy_atomic_mass,wtd_entropy_atomic_mass,range_atomic_mass,wtd_range_atomic_mass,mean_fie,wtd_mean_fie,wtd_entropy_fie,range_fie,wtd_range_fie,mean_atomic_radius,wtd_mean_atomic_radius,range_atomic_radius,wtd_range_atomic_radius,mean_Density,entropy_Density,wtd_entropy_Density,range_Density,wtd_range_Density,mean_ElectronAffinity,wtd_entropy_ElectronAffinity,range_ElectronAffinity,wtd_range_ElectronAffinity,mean_FusionHeat,gmean_FusionHeat,entropy_FusionHeat,wtd_entropy_FusionHeat,range_FusionHeat,wtd_range_FusionHeat,mean_ThermalConductivity,wtd_mean_ThermalConductivity,gmean_ThermalConductivity,entropy_ThermalConductivity,wtd_entropy_ThermalConductivity,range_ThermalConductivity,wtd_range_ThermalConductivity,mean_Valence,wtd_mean_Valence,range_Valence,wtd_range_Valence,wtd_std_Valence,critical_temp'.split(',')
}
raw_data_metadata = dataset_metadata.DatasetMetadata(dataset_schema.Schema(raw_data_schema))
# run Beam
with beam.Pipeline(RUNNER, options=opts) as p:
with beam_impl.Context(temp_dir=os.path.join(OUTPUT_DIR, 'tmp')):
# save the raw data metadata
_ = (raw_data_metadata
| 'WriteInputMetadata' >> tft_beam_io.WriteMetadata(
os.path.join(OUTPUT_DIR, 'metadata/rawdata_metadata'),
pipeline=p))
projection_fields = ['number_of_elements', 'mean_atomic_mass', 'entropy_atomic_mass',
'wtd_entropy_atomic_mass', 'range_atomic_mass',
'wtd_range_atomic_mass', 'mean_fie', 'wtd_mean_fie',
'wtd_entropy_fie', 'range_fie', 'wtd_range_fie',
'mean_atomic_radius', 'wtd_mean_atomic_radius',
'range_atomic_radius', 'wtd_range_atomic_radius', 'mean_Density',
'entropy_Density', 'wtd_entropy_Density', 'range_Density',
'wtd_range_Density', 'mean_ElectronAffinity',
'wtd_entropy_ElectronAffinity', 'range_ElectronAffinity',
'wtd_range_ElectronAffinity', 'mean_FusionHeat', 'gmean_FusionHeat',
'entropy_FusionHeat', 'wtd_entropy_FusionHeat', 'range_FusionHeat',
'wtd_range_FusionHeat', 'mean_ThermalConductivity',
'wtd_mean_ThermalConductivity', 'gmean_ThermalConductivity',
'entropy_ThermalConductivity', 'wtd_entropy_ThermalConductivity',
'range_ThermalConductivity', 'wtd_range_ThermalConductivity',
'mean_Valence', 'wtd_mean_Valence', 'range_Valence',
'wtd_range_Valence', 'wtd_std_Valence', 'critical_temp']
header = get_table_header(projection_fields)
# analyze and transform training
raw_data = (p
| 'train_read' >> beam.io.Read(beam.io.BigQuerySource(query=create_query(1, EVERY_N), use_standard_sql=True)))
raw_dataset = (raw_data, raw_data_metadata)
transformed_dataset, transform_fn = (
raw_dataset | beam_impl.AnalyzeAndTransformDataset(preprocess_tft))
transformed_data, transformed_metadata = transformed_dataset
_ = (transformed_data
| 'train_filter' >> beam.FlatMap(cleanup)
| 'WriteTrainData' >> beam.io.Write(beam.io.WriteToText(
file_path_prefix=os.path.join(OUTPUT_DIR, 'data', 'train'),
file_name_suffix=".csv",
shard_name_template="-SS-of-NN",
header=header,
num_shards=1)))
# transform eval data
raw_test_data = (p
| 'eval_read' >> beam.io.Read(beam.io.BigQuerySource(query=create_query(2, EVERY_N), use_standard_sql=True)))
raw_test_dataset = (raw_test_data, raw_data_metadata)
transformed_test_dataset = (
(raw_test_dataset, transform_fn) | beam_impl.TransformDataset())
transformed_test_data, _ = transformed_test_dataset
_ = (transformed_test_data
| 'eval_filter' >> beam.FlatMap(cleanup)
| 'WriteTestData' >> beam.io.Write(beam.io.WriteToText(
file_path_prefix=os.path.join(OUTPUT_DIR, 'data', 'eval'),
file_name_suffix=".csv",
shard_name_template="-SS-of-NN",
num_shards=1)))
_ = (transform_fn
| 'WriteTransformFn' >>
transform_fn_io.WriteTransformFn(os.path.join(OUTPUT_DIR, 'metadata')))
preprocess()
```

## 在 Cloud MLE 上训练

以下代码示例将在 Google Cloud MLE 上训练处理后的数据集。此时，将 Notebook 运行时类型更改为 Python 3.0。

+   配置 GCP 项目。

+   创建“trainer”目录。

```py
# configure GCP project - update with your parameters
project_id = 'ekabasandbox'
bucket_name = 'superconductor'
region = 'us-central1'
tf_version = '1.8'
import os
os.environ['bucket_name'] = bucket_name
os.environ['tf_version'] = tf_version
os.environ['project_id'] = project_id
os.environ['region'] = region
```

+   创建文件 __init__.py。

```py
# create directory trainer
import os
try:
os.makedirs('./trainer')
print('directory created')
except OSError:
print('could not create directory')
```

+   创建 trainer 文件 task.py。将存储桶名称替换为您的值。

```py
%%writefile trainer/__init__.py
```

+   创建包含模型代码的文件 model.py。

```py
%%writefile trainer/task.py
import argparse
import json
import os
import tensorflow as tf
from tensorflow.contrib.training.python.training import hparam
import trainer.model as model
def _get_session_config_from_env_var():
"""Returns a tf.ConfigProto instance that has appropriate device_filters set.
"""
tf_config = json.loads(os.environ.get('TF_CONFIG', '{}'))
if (tf_config and 'task' in tf_config and 'type' in tf_config['task'] and
'index' in tf_config['task']):
# Master should only communicate with itself and ps
if tf_config['task']['type'] == 'master':
return tf.ConfigProto(device_filters=['/job:ps', '/job:master'])
# Worker should only communicate with itself and ps
elif tf_config['task']['type'] == 'worker':
return tf.ConfigProto(device_filters=[
'/job:ps',
'/job:worker/task:%d' % tf_config['task']['index']
])
return None
def train_and_evaluate(hparams):
"""Run the training and evaluate using the high level API."""
train_input = lambda: model.input_fn(
tf.gfile.Glob(hparams.train_files),
num_epochs=hparams.num_epochs,
batch_size=hparams.train_batch_size
)
# Don't shuffle evaluation data
eval_input = lambda: model.input_fn(
tf.gfile.Glob(hparams.eval_files),
batch_size=hparams.eval_batch_size,
shuffle=False
)
train_spec = tf.estimator.TrainSpec(
train_input, max_steps=hparams.train_steps)
exporter = tf.estimator.FinalExporter(
'superconductor', model.SERVING_FUNCTIONS[hparams.export_format])
eval_spec = tf.estimator.EvalSpec(
eval_input,
steps=hparams.eval_steps,
exporters=[exporter],
name='superconductor-eval')
run_config = tf.estimator.RunConfig(
session_config=_get_session_config_from_env_var())
run_config = run_config.replace(model_dir=hparams.job_dir)
print('Model dir %s' % run_config.model_dir)
estimator = model.build_estimator(
learning_rate=hparams.learning_rate,
# Construct layers sizes with exponential decay
hidden_units=[
max(2, int(hparams.first_layer_size * hparams.scale_factor**i))
for i in range(hparams.num_layers)
],
config=run_config,
output_dir=hparams.output_dir)
tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
if __name__ == '__main__':
parser = argparse.ArgumentParser()
# Input Arguments
parser.add_argument(
'--train-files',
help='GCS file or local paths to training data',
nargs='+',
# update the bucket name
default='gs://{}/preproc_csv/data/{}*{}*'.format('superconductor', tf.estimator.ModeKeys.TRAIN, 'of'))
parser.add_argument(
'--eval-files',
help='GCS file or local paths to evaluation data',
nargs='+',
# update the bucket name
default='gs://{}/preproc_csv/data/{}*{}*'.format('superconductor', tf.estimator.ModeKeys.EVAL, 'of'))
parser.add_argument(
'--job-dir',
help='GCS location to write checkpoints and export models',
default='/tmp/superconductor-estimator')
parser.add_argument(
'--num-epochs',
help="""\
Maximum number of training data epochs on which to train .
If both --max-steps and --num-epochs are specified,
the training job will run for --max-steps or --num-epochs,
whichever occurs first. If unspecified will run for --max-steps.\
""",
type=int)
parser.add_argument(
'--train-batch-size',
help='Batch size for training steps',
type=int,
default=20)
parser.add_argument(
'--eval-batch-size',
help='Batch size for evaluation steps',
type=int,
default=20)
parser.add_argument(
'--learning-rate',
help='The training learning rate',
default=1e-4,
type=float)
parser.add_argument(
'--first-layer-size',
help='Number of nodes in the first layer of the DNN',
default=256,
type=int)
parser.add_argument(
'--num-layers', help='Number of layers in the DNN', default=3, type=int)
parser.add_argument(
'--scale-factor',
help='How quickly should the size of the layers in the DNN decay',
default=0.7,
type=float)
parser.add_argument(
'--train-steps',
help="""\
Steps to run the training job for. If --num-epochs is not specified,
this must be. Otherwise the training job will run indefinitely.\
""",
default=100,
type=int)
parser.add_argument(
'--eval-steps',
help='Number of steps to run evalution for at each checkpoint',
default=100,
type=int)
parser.add_argument(
'--export-format',
help='The input format of the exported SavedModel binary',
choices=['JSON', 'CSV', 'EXAMPLE'],
default='CSV')
parser.add_argument(
'--output-dir',
help='Location of the exported model',
nargs='+')
parser.add_argument(
'--verbosity',
choices=['DEBUG', 'ERROR', 'FATAL', 'INFO', 'WARN'],
default='INFO')
args, _ = parser.parse_known_args()
# Set python level verbosity
tf.logging.set_verbosity(args.verbosity)
# Set C++ Graph Execution level verbosity
os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(
tf.logging.__dict__[args.verbosity] / 10)
# Run the training job
hparams = hparam.HParams(**args.__dict__)
train_and_evaluate(hparams)
```

+   创建超参数配置文件。

```py
%%writefile trainer/model.py
import six
import tensorflow as tf
from tensorflow.python.estimator.model_fn import ModeKeys as Modes
# Define the format of your input data including unused columns.
CSV_COLUMNS = [
'number_of_elements', 'mean_atomic_mass', 'entropy_atomic_mass',
'wtd_entropy_atomic_mass', 'range_atomic_mass',
'wtd_range_atomic_mass', 'mean_fie', 'wtd_mean_fie',
'wtd_entropy_fie', 'range_fie', 'wtd_range_fie',
'mean_atomic_radius', 'wtd_mean_atomic_radius',
'range_atomic_radius', 'wtd_range_atomic_radius', 'mean_Density',
'entropy_Density', 'wtd_entropy_Density', 'range_Density',
'wtd_range_Density', 'mean_ElectronAffinity',
'wtd_entropy_ElectronAffinity', 'range_ElectronAffinity',
'wtd_range_ElectronAffinity', 'mean_FusionHeat', 'gmean_FusionHeat',
'entropy_FusionHeat', 'wtd_entropy_FusionHeat', 'range_FusionHeat',
'wtd_range_FusionHeat', 'mean_ThermalConductivity',
'wtd_mean_ThermalConductivity', 'gmean_ThermalConductivity',
'entropy_ThermalConductivity', 'wtd_entropy_ThermalConductivity',
'range_ThermalConductivity', 'wtd_range_ThermalConductivity',
'mean_Valence', 'wtd_mean_Valence', 'range_Valence',
'wtd_range_Valence', 'wtd_std_Valence', 'critical_temp'
]
CSV_COLUMN_DEFAULTS = [[0.0] for i in range(0, len(CSV_COLUMNS))]
LABEL_COLUMN = 'critical_temp'
# Define the initial ingestion of each feature used by your model.
# Additionally, provide metadata about the feature.
INPUT_COLUMNS = [tf.feature_column.numeric_column(i) for i in CSV_COLUMNS[:-1]]
UNUSED_COLUMNS = set(CSV_COLUMNS) - {col.name for col in INPUT_COLUMNS} - \
{LABEL_COLUMN}
def build_estimator(config, output_dir, hidden_units=None, learning_rate=None):
"""
Deep NN Regression model.
Args:
config: (tf.contrib.learn.RunConfig) defining the runtime environment for
the estimator (including model_dir).
hidden_units: [int], the layer sizes of the DNN (input layer first)
learning_rate: (int), the learning rate for the optimizer.
Returns:
A DNNRegressor
"""
(number_of_elements,mean_atomic_mass,entropy_atomic_mass,wtd_entropy_atomic_mass, \
range_atomic_mass,wtd_range_atomic_mass,mean_fie,wtd_mean_fie,wtd_entropy_fie,range_fie,\
wtd_range_fie,mean_atomic_radius,wtd_mean_atomic_radius,range_atomic_radius,wtd_range_atomic_radius,\
mean_Density,entropy_Density,wtd_entropy_Density,range_Density,wtd_range_Density,mean_ElectronAffinity,\
wtd_entropy_ElectronAffinity,range_ElectronAffinity,wtd_range_ElectronAffinity,mean_FusionHeat,\
gmean_FusionHeat,entropy_FusionHeat,wtd_entropy_FusionHeat,range_FusionHeat,wtd_range_FusionHeat,\
mean_ThermalConductivity,wtd_mean_ThermalConductivity,gmean_ThermalConductivity,entropy_ThermalConductivity,\
wtd_entropy_ThermalConductivity,range_ThermalConductivity,wtd_range_ThermalConductivity,mean_Valence,\
wtd_mean_Valence,range_Valence,wtd_range_Valence,wtd_std_Valence) = INPUT_COLUMNS
columns = [number_of_elements,mean_atomic_mass,entropy_atomic_mass,wtd_entropy_atomic_mass, \
range_atomic_mass,wtd_range_atomic_mass,mean_fie,wtd_mean_fie,wtd_entropy_fie,range_fie,\
wtd_range_fie,mean_atomic_radius,wtd_mean_atomic_radius,range_atomic_radius,wtd_range_atomic_radius,\
mean_Density,entropy_Density,wtd_entropy_Density,range_Density,wtd_range_Density,mean_ElectronAffinity,\
wtd_entropy_ElectronAffinity,range_ElectronAffinity,wtd_range_ElectronAffinity,mean_FusionHeat,\
gmean_FusionHeat,entropy_FusionHeat,wtd_entropy_FusionHeat,range_FusionHeat,wtd_range_FusionHeat,\
mean_ThermalConductivity,wtd_mean_ThermalConductivity,gmean_ThermalConductivity,entropy_ThermalConductivity,\
wtd_entropy_ThermalConductivity,range_ThermalConductivity,wtd_range_ThermalConductivity,mean_Valence,\
wtd_mean_Valence,range_Valence,wtd_range_Valence,wtd_std_Valence]
estimator = tf.estimator.DNNRegressor(
model_dir=output_dir,
config=config,
feature_columns=columns,
hidden_units=hidden_units or [256, 128, 64],
optimizer=tf.train.AdamOptimizer(learning_rate)
)
# add extra evaluation metric for hyperparameter tuning
estimator = tf.contrib.estimator.add_metrics(estimator, add_eval_metrics)
return estimator
def add_eval_metrics(labels, predictions):
pred_values = predictions['predictions']
return {
'rmse': tf.metrics.root_mean_squared_error(labels, pred_values)
}
# [START serving-function]
def csv_serving_input_fn():
"""Build the serving inputs."""
csv_row = tf.placeholder(shape=[None], dtype=tf.string)
features = _decode_csv(csv_row)
# Ignore label column
features.pop(LABEL_COLUMN)
return tf.estimator.export.ServingInputReceiver(features,
{'csv_row': csv_row})
def example_serving_input_fn():
"""Build the serving inputs."""
example_bytestring = tf.placeholder(
shape=[None],
dtype=tf.string,
)
features = tf.parse_example(
example_bytestring,
tf.feature_column.make_parse_example_spec(INPUT_COLUMNS))
return tf.estimator.export.ServingInputReceiver(
features, {'example_proto': example_bytestring})
def json_serving_input_fn():
"""Build the serving inputs."""
inputs = {}
for feat in INPUT_COLUMNS:
inputs[feat.name] = tf.placeholder(shape=[None], dtype=feat.dtype)
return tf.estimator.export.ServingInputReceiver(inputs, inputs)
# [END serving-function]
SERVING_FUNCTIONS = {
'JSON': json_serving_input_fn,
'EXAMPLE': example_serving_input_fn,
'CSV': csv_serving_input_fn
}
def _decode_csv(line):
"""Takes the string input tensor and returns a dict of rank-2 tensors."""
# Takes a rank-1 tensor and converts it into rank-2 tensor
row_columns = tf.expand_dims(line, -1)
columns = tf.decode_csv(row_columns, record_defaults=CSV_COLUMN_DEFAULTS)
features = dict(zip(CSV_COLUMNS, columns))
# Remove unused columns
for col in UNUSED_COLUMNS:
features.pop(col)
return features
def input_fn(filenames, num_epochs=None, shuffle=True, skip_header_lines=1, batch_size=200):
"""Generates features and labels for training or evaluation.
This uses the input pipeline based approach using file name queue
to read data so that entire data is not loaded in memory.
Args:
filenames: [str] A List of CSV file(s) to read data from.
num_epochs: (int) how many times through to read the data. If None will loop through data indefinitely
shuffle: (bool) whether or not to randomize the order of data. Controls randomization of both file order and line order within files.
skip_header_lines: (int) set to non-zero in order to skip header lines in CSV files.
batch_size: (int) First dimension size of the Tensors returned by input_fn
Returns:
A (features, indices) tuple where features is a dictionary of
Tensors, and indices is a single Tensor of label indices.
"""
dataset = tf.data.TextLineDataset(filenames).skip(skip_header_lines).map(
_decode_csv)
if shuffle:
dataset = dataset.shuffle(buffer_size=batch_size * 10)
iterator = dataset.repeat(num_epochs).batch(
batch_size).make_one_shot_iterator()
features = iterator.get_next()
return features, features.pop(LABEL_COLUMN)
```

+   以下代码在 Cloud MLE 上执行训练作业。

```py
%%writefile hptuning_config.yaml
trainingInput:
hyperparameters:
hyperparameterMetricTag: rmse
goal: MINIMIZE
maxTrials: 4 #20
maxParallelTrials: 2 #5
enableTrialEarlyStopping: True
algorithm: RANDOM_SEARCH
params:
- parameterName: learning-rate
type: DOUBLE
minValue: 0.00001
maxValue: 0.005
scaleType: UNIT_LOG_SCALE
- parameterName: first-layer-size
type: INTEGER
minValue: 50
maxValue: 500
scaleType: UNIT_LINEAR_SCALE
- parameterName: num-layers
type: INTEGER
minValue: 1
maxValue: 15
scaleType: UNIT_LINEAR_SCALE
- parameterName: scale-factor
type: DOUBLE
minValue: 0.1
maxValue: 1.0
scaleType: UNIT_REVERSE_LOG_SCALE
```

![../images/463852_1_En_44_Chapter/463852_1_En_44_Fig5_HTML.jpg](img/463852_1_En_44_Fig5_HTML.jpg)

图 44-5

Cloud MLE 训练输出

+   Cloud MLE 训练输出如图 44-5 所示。

```py
%%bash
JOB_NAME=superconductor_$(date -u +%y%m%d_%H%M%S)
HPTUNING_CONFIG=hptuning_config.yaml
GCS_JOB_DIR=gs://$bucket_name/jobs/$JOB_NAME
echo $GCS_JOB_DIR
gcloud ai-platform jobs submit training $JOB_NAME \
--stream-logs \
--runtime-version $tf_version \
--job-dir $GCS_JOB_DIR \
--module-name trainer.task \
--package-path trainer/ \
--region us-central1 \
--scale-tier=STANDARD_1 \
--config $HPTUNING_CONFIG \
-- \
--train-steps 5000 \
--eval-steps 100
gs://superconductor/jobs/superconductor_181222_040429
endTime: '2018-12-22T04:24:50'
jobId: superconductor_181222_040429
startTime: '2018-12-22T04:04:35'
state: SUCCEEDED
```

## 部署训练模型

部署具有最低 **objectiveValue** 的最佳模型试验以在 Cloud MLE 上进行推理：

+   显示所选训练模型目录的内容。

+   部署模型。

```py
%%bash
gsutil ls gs://${BUCKET}/jobs/superconductor_181222_040429/4/export/superconductor/1545452450
'Output':
gs://superconductor/jobs/superconductor_181222_040429/4/export/superconductor/1545452450/
gs://superconductor/jobs/superconductor_181222_040429/4/export/superconductor/1545452450/saved_model.pb
gs://superconductor/jobs/superconductor_181222_040429/4/export/superconductor/1545452450/variables/
```

```py
%%bash
MODEL_NAME="superconductor"
MODEL_VERSION="v1"
MODEL_LOCATION=gs://$bucket_name/jobs/superconductor_181222_040429/4/export/superconductor/1545452450
echo "Deploying model $MODEL_NAME $MODEL_VERSION"
gcloud ai-platform models create ${MODEL_NAME} --regions us-central1
gcloud ai-platform versions create ${MODEL_VERSION} --model ${MODEL_NAME} --origin ${MODEL_LOCATION} --runtime-version ${tf_version}
```

## 批量预测

以下代码对部署的模型进行推理：

+   提交批量预测作业。

+   列出 GCS 中预测输出目录的内容。

```py
%%bash
JOB_NAME=superconductor_prediction
MODEL_NAME=superconductor
MODEL_VERSION=v1
TEST_FILE=gs://$bucket_name/preproc_csv/data/eval-00-of-01.csv
OUTPUT_DIR=gs://$bucket_name/jobs/$JOB_NAME/predictions
echo $OUTPUT_DIR
# submit a batched job
gcloud ai-platform jobs submit prediction $JOB_NAME \
--model $MODEL_NAME \
--version $MODEL_VERSION \
--data-format TEXT \
--region $region \
--input-paths $TEST_FILE \
--output-path $OUTPUT_DIR
# stream job logs
echo "Job logs..."
gcloud ml-engine jobs stream-logs $JOB_NAME
'Output':
gs://superconductor/jobs/superconductor_prediction/predictions
Job logs...
INFO    2018-12-22 22:04:22 +0000   service     Validating job requirements...
INFO    2018-12-22 22:04:22 +0000   service     Job creation request has been successfully validated.
INFO    2018-12-22 22:04:22 +0000   service     Job superconductor_prediction is queued.
INFO    2018-12-22 22:09:09 +0000   service     Job completed successfully.
```

+   显示预测 RMSE 输出。

```py
%%bash
gsutil ls gs://superconductor/jobs/superconductor_prediction/predictions/
'Output':
gs://superconductor/jobs/superconductor_prediction/predictions/prediction.errors_stats-00000-of-00001
gs://superconductor/jobs/superconductor_prediction/predictions/prediction.results-00000-of-00002
gs://superconductor/jobs/superconductor_prediction/predictions/prediction.results-00001-of-00002
```

```py
%bash
# read output summary
echo "Job output summary:"
gsutil cat 'gs://superconductor/jobs/superconductor_prediction/predictions/prediction.results-00000-of-00002'
'Output':
{"outputs": [0.02159707620739937]}
{"outputs": [0.13300871849060059]}
{"outputs": [0.02054387889802456]}
{"outputs": [0.09370037913322449]}
...
{"outputs": [0.41005855798721313]}
{"outputs": [0.39907798171043396]}
{"outputs": [0.4040292799472809]}
{"outputs": [0.43743470311164856]}
```

本章提供了一个端到端流程的概述，用于在 Google Cloud Platform 上建模和部署机器学习解决方案。下一章将介绍微服务架构的概念。它提供了在 GCP 上使用 Docker 容器及其与 Kubernetes 的编排的概述。
