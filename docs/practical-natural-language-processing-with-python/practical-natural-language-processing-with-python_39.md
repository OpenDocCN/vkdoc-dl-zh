# 第 3 章 在线评论中的自然语言处理

***图 3-14.***

列表 3-36 展示了数据集中的数值特征。现在，你对包含形容词、停用词以及正面和负面匹配词的列进行向量化处理并创建矩阵。请参见列表 3-37。

***列表 3-37.***

```
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(min_df=0.001, analyzer=u'word', ngram_range=(1,1))
tfidf_matrix_tr = tfidf_vectorizer.fit_transform(x_train["pos_neg_comb_adj_st"])
tfidf_matrix_te = tfidf_vectorizer.transform(x_test["pos_neg_comb_adj_st"])
x_train2 = tfidf_matrix_tr.todense()
x_test2 = tfidf_matrix_te.todense()
```

`x_train2` 和 `x_test2` 是处理了一组文本特征的矩阵。这些矩阵随后将与数值矩阵 `x_train1` 和 `x_test2` 拼接。请参见列表 3-38。

***列表 3-38.***

```
import numpy as np
x_train3 = np.concatenate([x_train1, x_train2], axis=1)
x_test3 = np.concatenate([x_test1, x_test2], axis=1)
```

由于你将使用 `score_bkt` 作为目标变量构建神经网络分类器，因此对特征进行标准化非常重要。你将使用最小-最大缩放器进行标准化。列表 3-39 展示了使用 sklearn [函数 `sklearn`](https://paperpile.com/c/y3boi7/LUXA) 的最小-最大缩放器公式。

```
X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
X_scaled = X_std * (max - min) + min
```

