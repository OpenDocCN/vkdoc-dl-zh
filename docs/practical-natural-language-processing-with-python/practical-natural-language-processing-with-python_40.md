# 第 3 章 在线评论中的自然语言处理

***列表 3-39.*** 最小-最大缩放器

```
from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()
x_train_scaled = min_max_scaler.fit_transform(x_train3)
x_test_scaled = min_max_scaler.transform(x_test3)
```

你使用特征选择算法减少特征数量，就像上一章所做的那样。请参见列表 3-40。

***列表 3-40.***

```
from sklearn.feature_selection import SelectPercentile, f_classif
selector = SelectPercentile(f_classif, percentile=40)
selector.fit(x_train3, y_train)
x_train4 = selector.fit_transform(x_train_scaled, y_train)
x_test4 = selector.transform(x_test_scaled)
```

你还需要将分类目标变量转换为独热编码形式。独热编码是一种形式，其中每个级别由其他级别的缺失和该级别的存在来表示，因此正面可以表示为 `100`，负面表示为 `010`，中性表示为 `001`。

***列表 3-41.*** 独热编码

```
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
le = LabelEncoder()
y_train1 = le.fit_transform(y_train)
y_train2 = np_utils.to_categorical(y_train1)
y_test1 = le.transform(y_test)
print(y_train2.shape)
(11368, 3)
```

