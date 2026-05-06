# 第 3 章 在线评论中的自然语言处理

接下来，你使用一个函数，将句子中提取的正则表达式替换为相应的单词。变量 `words_Coll` 是提取并映射后的单词集合。如果 `Reviews1` 中被替换的单词未被“词性”标注器识别为名词形式，我们仍通过使用步骤 3 中 `words_Coll` 提取的单词，确保它们可用于最终的映射过程。见代码清单 3-51。

***代码清单 3-51.***

```
def repl_text(t3,to_repl,word_to_repl):
    t3["Reviews1"] = t3["Reviews1"].str.replace(to_repl,word_to_repl)
    ind_list = t3[t3["Reviews"].str.contains(to_repl)].index
    t3.loc[ind_list,"words_Coll"] = t3.loc[ind_list,"words_Coll"] + " " + word_to_repl
    return t3
```

在代码清单 3-52 中，你迭代运行行，替换模式，并将替换后的单词保存在单独的列中。

***代码清单 3-52.***

```
t3["Reviews1"] = t3["Reviews"].str.lower()
t3["words_Coll"] = ""
for index,row in rrpl.iterrows():
    repl = row["repl"]
    word = row["word"]
    t3 = repl_text(t3,repl,word)
```

在此步骤结束时，你得到两列：`Reviews1`（替换模式后的文本）和 `words_Coll`（从模式中提取的单词）。

### 步骤 2：提取名词形式

现在，你将在以下函数中使用 `filter_pos` 提取名词形式。这里的 `fltr` 是你传递给函数的变量。如果你想尝试其他词性标签，可以重复使用它。见代码清单 3-53 和 3-54。

