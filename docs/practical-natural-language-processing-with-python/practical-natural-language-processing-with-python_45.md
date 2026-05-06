# 第 3 章 在线评论中的自然语言处理

***代码清单 3-53.***

```
def filter_pos(fltr,sent_list):
    str1 = ""
    for i in sent_list:
        if(i[1].find(fltr)>=0):
            str1 = str1 + i[0].lower() + " "
    return str1
```

***代码清单 3-54.***

```
get_pos_tags = nltk.pos_tag_sents(t3["Reviews1"].str.split())
str_sel_list = []
for i in get_pos_tags:
    str_sel = filter_pos("NN",i)
    str_sel_list.append(str_sel)
```

包含名词形式属性的变量 `str_sel_list` 与 `words_coll` 合并。这是为了确保即使某些替换后的单词被词性标注器遗漏，它们仍能成为分析的一部分。见代码清单 3-55。

***代码清单 3-55.***

```
t3["pos_tags"] = str_sel_list
t3["all_attrs_ext"] = t3["pos_tags"] + t3["words_Coll"]
```

列 `all_attrs_ext` 将用于将属性映射到类别。

### 步骤 3：创建映射文件

你需要创建一个映射文件。你可以抓取手机评论网站，然后开始生成映射文件。例如，请参阅 [www.gsmarena.com/xiaomi_redmi_k30-review-2055.php](http://www.gsmarena.com/xiaomi_redmi_k30-review-2055.php) 上的规格。



## 设备规格与自然语言处理

## 设备规格

- **机身：** 金属框架，前后 Gorilla Glass 5 玻璃，重 208g。
- **显示屏：** 6.67 英寸 IPS LCD，120Hz 刷新率，HDR10，分辨率 1080 x 2400px，20:9 屏幕比例，395ppi。
- **后置摄像头：** 主摄：64MP，`f/1.89` 光圈，1/1.7 英寸传感器尺寸，0.8μm 像素尺寸，PDAF。超广角：8MP，`f/2.2`，1/4 英寸，1.12μm 像素。微距摄像头：2MP，`f/2.4`，1/5 英寸，1.75μm。深度传感器：2MP；支持 2160p@30fps、1080p@30/60/120fps、720p@960fps 视频录制。
- **前置摄像头：** 主摄：20MP，`f/2.2` 光圈，0.9μm 像素。深度传感器：2MP；支持 1080p/30fps 视频录制。
- **操作系统：** Android 10；MIUI 11。
- **芯片组：** Snapdragon 730G（8nm）：八核（2x2.2 GHz Kryo 470 Gold 和 6x1.8 GHz Kryo 470 Silver），Adreno 618 GPU。
- **内存：** 6/8GB RAM；64/128/256GB UFS 2.1 存储；共享 microSD 卡槽。
- **电池：** 4,500mAh；支持 27W 快充。
- **连接性：** 双卡双待；LTE-A，4 频段载波聚合，LTE Cat-12/Cat-13；USB-C；Wi-Fi a/b/g/n/ac；双频 GPS；蓝牙 5.0；FM 收音机；NFC；红外发射器。
- **其他：** 侧面指纹识别；3.5mm 音频插孔。

## 第三章：在线评论中的自然语言处理

例如，后置和前置摄像头可以映射到“摄像头”类别。LTE、WiFi 等可以映射到“网络”类别。通过查看多个评论网站，我将属性类别标准化为以下类别：

1. 通信（Comm）
2. 尺寸
3. 重量
4. 构造
5. 音质
6. SIM 卡
7. 显示屏
8. 操作系统
9. 性能
10. 价格

![Image 38](img/index-121_1.jpg)

11. 电池
12. 收音机
13. 摄像头
14. 键盘
15. 应用
16. 保修
17. 保证
18. 通话质量
19. 存储

`attr_cat.csv` 是映射文件。映射文件的快速预览如表 3-6 所示。

***表 3-6.** 映射文件快速预览*

你还可以从 `str_sel_list` 获取名词的顶部列表。这可用于调整类别属性的映射列表。任何出现在顶部词汇但不在映射文件中的名词都应被包含。请参见列表 3-56 和图 3-19。

![Image 39](img/index-122_1.jpg)

***列表 3-56.***

### 获取顶部词汇

```python
from collections import Counter

str_sel_list_all = ' '.join(str_sel_list)
str_sel_list_all = str_sel_list_all.replace('.','')
str_sel_list_all_list = Counter(str_sel_list_all.split())
str_sel_list_all_list.most_common()
```

***图 3-19.***

### 步骤 4：将每条评论映射到属性

设置好映射文件后，你将迭代地将每条评论映射到一个属性。请参见列表 3-57。

***列表 3-57.***

```python
attr_cats = pd.read_csv("attr_cat.csv")
attrs_all = []

for index,row in t3.iterrows():
    ext1 = row["all_attrs_ext"]
    attr_list = []
    for index1,row1 in attr_cats.iterrows():
        wrd = row1["words"]
        cat = row1["cat"]
        if(ext1.find(wrd)>=0):
            attr_list.append(cat)
    attr_list = list(set(attr_list))
    attr_str = ' '.join(attr_list)
    attrs_all.append(attr_str)
```

数据框中的 `attrs_all` 列包含提取出的属性。

### 步骤 5：分析品牌

现在你将分析不同品牌在不同类别中的表现。为此，你选择了顶级品牌。你还会将评分分类为 `pol_tag`。4 分和 5 分的高评分将被分类为正面，1 分和 2 分的低评分为负面，3 分为中性。请参见列表 3-58。

***列表 3-58.*** 评分分类

```python
t3["attrs"] = attrs_all
t3["pol_tag"] = "neu"
t3.loc[t3.Rating>=4,"pol_tag"] = "pos"
t3.loc[t3.Rating<=2,"pol_tag"] = "neg"
```

顶级品牌是指在样本数据集中至少有 100 条评论的品牌。请参见列表 3-59。

***列表 3-59.*** 顶级品牌

```python
brand_df = pd.DataFrame(t3["Brand Name"].value_counts()).reset_index()
brand_df.columns = ["brand","count"]
brand_df1 = brand_df[brand_df["count"]>=100]
brand_list = list(brand_df1["brand"])
```

```python
['Samsung',
 'BLU',
 'Apple',
 'LG',
 'Nokia',
 'Motorola',
 'BlackBerry',
 'CNPGD',
 'HTC']
```



Listing 3-59 的输出是数据集中顶级品牌的列表。分析将针对这些品牌进行。通过以下函数，您可以获得正面评价中最常见的属性以及负面评价中最常见的属性及其计数。您创建一个参数化变量`col3`，其中包含每个属性正面得分与负面得分的比率。此函数将在整体层面以及每个品牌层面被调用。

现在，您可以比较不同品牌的比率，并了解每个品牌的感知主张。请参见 Listing 3-60。

***Listing 3-60.***

```
def get_attrs_df(df1,col1,col2,col3):
    list1 = ' '.join(list(df1.loc[(df1.pol_tag=='pos'),"attrs"]))
    list2 = Counter(list1.split())
    df_pos = pd.DataFrame(list2.most_common())
    list1 = ' '.join(list(df1.loc[df1.pol_tag=='neg',"attrs"]))
    list2 = Counter(list1.split())
    df_neg = pd.DataFrame(list2.most_common())
    df_pos.columns = ["attrs",col1]
    df_neg.columns = ["attrs",col2]
    df_all = pd.merge(df_pos,df_neg,on="attrs")
    df_all[col3] = df_all[col1]/df_all[col2]
    return df_all
```

第一次调用`df_gen`函数会获取最常见的正面属性和最常见的负面属性及其在数据框中的出现次数。然后，它会迭代地为品牌列表中的每个品牌调用该函数。请参见 Listing 3-61 和 Figure 3-20。

***Listing 3-61.***

```
df_gen = get_attrs_df(t3,"pos_count","neg_count","ratio_all")
for num,i in enumerate(brand_list):
    brand_only_df = t3.loc[t3["Brand Name"]==i]
    col1 = 'pos_count_' + i
    col2 = "neg_count_" + i
    col3 = "ratio_" + i
    df_brand = get_attrs_df(brand_only_df,col1,col2,col3)
    df_gen = pd.merge(df_gen,df_brand[["attrs",col3]],how='left',on="attrs")
df_gen
```

| attrs | ratio_all | ratio_Samsung | ratio_BLU | ratio_Apple | ratio_LG | ratio_Nokia | ratio_Motorola | ratio_BlackBerry | ratio_CNPGD | ratio_HTC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| battery | 1.11 | 1.64 | 2.14 | 1.97 | 1.15 | 1.47 | 1.89 | 1.67 | 0.60 | 1.60 |
| camera | 3.00 | 3.35 | 3.78 | 3.50 | 4.00 | 3.40 | 3.80 | 4.00 | 2.00 | 1.50 |
| display | 0.71 | 1.67 | 2.67 | 2.15 | 0.65 | 2.57 | 1.75 | 2.67 | 0.25 | 0.10 |
| price | 0.11 | 1.06 | 1.13 | 2.21 | 0.64 | 3.00 | 0.86 | 0.67 | 0.67 | 0.36 |
| comm | 2.00 | 1.84 | 4.20 | 1.73 | 1.00 | 3.50 | 3.67 | 4.00 | 1.33 | 0.60 |

***Figure 3-20.***

第一列`ratio_all`是属性的总体得分。总体而言，人们对相机的评价更积极，对价格的评价最不积极。在价格方面，您可以看到`Blu`品牌似乎比其他品牌更受青睐。在相机方面，`Apple`和`Blackberry`表现突出。您将绘制顶级品牌与属性相对得分（正面与负面得分）之间的图表。但首先，您需要将数据准备成`matplotlib`所需的格式。您首先收集顶级品牌的所有比率列的名称。请参见 Listing 3-62 和 Figure 3-21。

***Listing 3-62.***

```
ratio_list = []
for i in brand_list:
    ratio_list.append("ratio_" + i)
ratio_list
```

![Image 40](img/index-126_1.png)

***Figure 3-21.***

接下来，您将每个列中的所有值制作成一个列表的列表。`list_all`中的每组列表对应一个品牌，每个品牌列表中的值集对应感兴趣的属性。这里您获取了数据框开头的属性：`battery`、`display`、`camera`、`price`、`comm`和`os`。请参见 Listing 3-63 和 Figure 3-22。

***Listing 3-63.***

```
import matplotlib.pyplot as plt
cols = ["battery","display","camera","price","comm","os"]
list_all = []
for j in ratio_list:
    list1 = df_gen.loc[df_gen.attrs.isin(cols),j].round(2).fillna(0).values.flatten().tolist()
    list_all.append(list1)
list_all
```

![Image 41](img/index-127_1.jpg)

***Figure 3-22.***

现在数据已准备就绪，您可以使用`matplotlib`来绘制图表。您绘制每个



