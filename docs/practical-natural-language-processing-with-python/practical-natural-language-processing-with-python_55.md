# 第 5 章 虚拟助手中的自然语言处理

## 实现架构

作为第一步，你对客户文本和 `bot_text` 进行预处理。参见代码清单 5-16。

**代码清单 5-16.**

```python
import pandas as pd
t2 = pd.read_csv("letsgo_op.csv")
```

你替换 `bot_text` 中的一些常见句子。参见代码清单 5-17。

**代码清单 5-17.**

```python
string_repl = "welcome to the cmu let's go bus information system. to get help at any time, just say help or press zero. what can i do for you? i am an automated spoken dialogue system that can give you schedule information for bus routes in pittsburgh's east end. you can ask me about the following buses: 28x, 54c, 56u, 59u, 61a, 61b, 61c, 61d, 61f, 64a, 69a, and 501. what bus schedule information are you looking for? for example, you can say, when is the next 28x from downtown to the airport? or i'd like to go from mckeesport to homestead tomorrow at 10 a.m."

t2["bot_text"] = t2["bot_text"].str.replace("welcome.*mckeesport to homestead.*automated spoken dialogue.*","greeting_templ")
t2["trim_bot_txt"] = t2["bot_text"].str.replace('goodbye.*','goodbye')
```

鉴于这些数据是关于公交时刻表的，客户经常会提到公交站和地名。我收集了匹兹堡的公交站名，并向列表中添加了更多名称，如市中心、购物中心、大学等。你可以使用这个文件（`bus_stops.npz`）将部分公交站名替换为一些通用名称，以帮助模型泛化。参见代码清单 5-18。

**代码清单 5-18.**

```python
import numpy as np
bus_sch = np.load("bus_stops.npz",allow_pickle=True)
bus_sch1 = list(bus_sch['arr_0'])
bus_sch1
```

```python
['freeporroad',
 'springarden',
 'bellevue',
 'shadeland',
 'coraopolis',
 'fairywood',
 'banksville',
 'bowehill',
 'carrick',
 'homesteaparlimited',
 'hazelwood',
 'nortbraddock',
 'lawrenceville-waterfront',
 'edgewootowcenter',
 'hamilton']
```

现在，你使用代码清单 5-19 中的函数 `repl_place_bus`，通过计算句子中单词与 `bus_sch1` 之间的 `fuzz.ratio`（模糊匹配比率），来替换 `cust_text` 和 `bot_text` 中的地名。`fuzz.ratio` 使用莱文斯坦距离（将一个字符串转换为另一个字符串所需的最小单字符编辑次数）来计算字符串之间的相似度。



## 文本排版

`fuzzywuzzy`是用于计算`fuzz.ratio`的包。

你可以使用`pip install fuzzywuzzy`安装。另请参见清单 5-20。

**清单 5-19.**

```
!pip install fuzzywuzzy
```

**清单 5-20.**

```
from fuzzywuzzy import fuzz

str_list = []

def repl_place_bus(row,req_col):
    ctext = row[req_col]
    fg=0
    str1=ctext
    try:
        for j in ctext.split():
            for k in bus_sch1:
                score = fuzz.ratio(j,k)
                if(score>=70):
                    fg=1
                    break
            if(fg==1):
                fg=0
                str1 = str1.replace(j," place_name ")
    except:
        print (j,i)
    return str1
```

你有 7000 多行数据。在每一行中，你都在遍历句子中的单词。这会显著增加运行时间。为了加快运行速度，你可以在 Python 中并行化应用这些函数。你使用名为`joblib`的库，跨多个进程同时运行函数。`n_jobs`参数提供了将要并行运行的进程数量。

你可以使用`pip install`安装`joblib`库。参见清单 5-21 和 5-22。

**清单 5-21.**

```
!pip install joblib
```

**清单 5-22.**

```
from joblib import Parallel, delayed

com0 = Parallel(n_jobs=12, backend="threading",verbose=1)(delayed(repl_place_bus)(row,"cust_text") for i,row in t2.iterrows())
bot_list = Parallel(n_jobs=12, backend="threading",verbose=1)(delayed(repl_place_bus)(row,"trim_bot_txt") for i,row in t2.iterrows())
t2["corrected_cust"] = com0
t2["corrected_bot"] = bot_list
```

列`corrected_cust`和`corrected_bot`包含替换后的句子。

现在你执行以下预处理：规范化方向（因为是巴士预订）、规范化 A.M./P.M.、将数字单词替换为数字、规范化巴士名称（如`8a`、`11c`等）、时间、替换数字。参见清单 5-23 和 5-24。

**清单 5-23.**

```
direct_list = ['east','west','north','south']

def am_pm_direct_repl(t2,col):
    t2[col] = t2[col].str.replace('p.m','pm')
    t2[col] = t2[col].str.replace('p m','pm')
    t2[col] = t2[col].str.replace('a.m','am')
    t2[col] = t2[col].str.replace('a m','am')
    for i in direct_list:
        t2[col] = t2[col].str.replace(i,' direction ')
    return t2
```

**清单 5-24.**

```
t3 = am_pm_direct_repl(t2,"corrected_cust")
t3 = am_pm_direct_repl(t2,"corrected_bot")
```

以下函数使用`num2words`包，为数字单词（如`one`、`two`、`three`等）获取数字形式。你使用它来用正确的数字替换句子。你将替换“th”单词，例如，`Fifth Avenue`会被替换为`num_th`。你在清单 5-25 中安装`num2words`包，并在清单 5-26 和 5-27 中使用它。

**清单 5-25.** 安装`num2words`

```
!pip install num2words
```

**清单 5-26.** 使用`num2words`

```
from num2words import num2words

word_num_list = []
word_num_th_list = []

for i in range(0,100):
    word_num_list.append(num2words(i))
    word_num_th_list.append(num2words(i,to='ordinal'))

print (word_num_list[0:5])
print (word_num_th_list[0:5])
```

输出：
```
['zero', 'one', 'two', 'three', 'four']
['zeroth', 'first', 'second', 'third', 'fourth']
```

**清单 5-27.**

```
def repl_num_words(t2,col):
    for num,i in enumerate(word_num_th_list):
        t2[col] = t2[col].str.replace(i,"num_th")
        t2[col] = t2[col].str.replace(word_num_list[num],str(num))
    return t2

t4 = repl_num_words(t3,"corrected_cust")
t4 = repl_num_words(t3,"corrected_bot")
```

清单 5-28 中的代码替换巴士名称和时间。

**清单 5-28.**

```
t4["corrected_cust"] = t4["corrected_cust"].str.replace('[0-9]{1,2}[a-z]{1,2}','bus_name')
t4["corrected_cust"] = t4["corrected_cust"].str.replace('[0-9]{1,2}[\.\s]*[0-9]{1,2}[\s]*pm','time_name')
t4["corrected_cust"] = t4["corrected_cust"].str.replace('[0-9]{1,2}[\.\s]*[0-9]{1,2}[\s]*am','time_name')
t4["corrected_bot"] = t4["corrected_bot"].str.replace('[0-9]{1,2}[a-z]{1,2}','bus_name')
```



