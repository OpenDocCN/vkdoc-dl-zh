# 第 3 章 在线评论中的自然语言处理

***清单 3-24.***

```
gword = "good"
bword = "bad"
hits_good = get_total(gword)
hits_bad = get_total(bword)
hits_total = get_total(gword + " OR " + bword)
list1 = ["delight", "pathetic", "average", "awesome", "tiresome", "angry", "furious"]
```

现在你遍历这个列表并确定每个词的极性。最终输出可以是正面、负面或不确定。参见清单 3-25。

***清单 3-25.*** 确定极性

```
for i in list1:
    str1 = i + "+reviews"
    str1_tot = get_total(str1)
    query_word_pos, query_word_neg = get_query_word(str1, gword, bword)
    sr_results_pos_int = get_total(query_word_pos)
    sr_results_neg_int = get_total(query_word_neg)
    pmi_score = get_pmi(hits_good, hits_bad, hits_total, sr_results_pos_int, sr_results_neg_int, str1_tot)
    jc_score = get_jaccard(hits_good, hits_bad, sr_results_pos_int, sr_results_neg_int, str1_tot)

    if (pmi_score[0] == "na" or pmi_score[1] == "na"):
        print(i, "pmi indeterminate")
    elif (pmi_score[0] > pmi_score[1]):
        print(i, "pmi pos")
    elif (pmi_score[0] < pmi_score[1]):
        print(i, "pmi neg")
    else:
        print(i, "pmi neutral")

    if (jc_score[0] == "na") or (jc_score[1] == "na"):
        print(i, "jc indeterminate")
    elif (jc_score[0] > jc_score[1]):
        print(i, "jc pos")
    elif (jc_score[0] < pmi_score[1]):
        print(i, "jc neg")
    else:
        print(i, "jc neutral")
```

输出如清单 3-26 所示。部分输出，例如“average”，显示为正面，因为你没有定义中性状态。你可以添加一个条件：如果正面和负面之间的差异在某个阈值内，则该词将被视为中性。

***清单 3-26.*** 输出结果

```
delight pmi pos
delight jc pos
pathetic pmi neg
pathetic jc neg
average pmi pos
average jc pos
awesome pmi indeterminate
awesome jc indeterminate
tiresome pmi indeterminate
tiresome jc indeterminate
angry pmi neg
angry jc neg
furious pmi neg
furious jc neg
```

你也可以使用其他语料库，例如维基百科或新闻语料库。你可以尝试不同的锚点词组合以及其他一些范围限定词（在你的例子中是单词“reviews”）以获得更好的结果。

