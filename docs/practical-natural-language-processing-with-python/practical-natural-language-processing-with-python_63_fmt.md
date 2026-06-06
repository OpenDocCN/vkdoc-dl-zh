# 第 5 章 虚拟助手中的自然语言处理

## 预处理

预处理步骤在 `清单 5-55` 中突出显示。（你将 `pandas` 替换转换为正则表达式替换）。这是训练时遵循过程的重复：替换地名、方向、数字名称、公交名称、时间和特殊字符。这应用于客户句子，然后输入到模型中。另请参见 `清单 5-56`。

***清单 5-55.***

```python
from fuzzywuzzy import fuzz

def repl_place_bus(ctext):
    # for i,row in t2.iterrows():
    # ctext = row[req_col]
    fg = 0
    str1 = ctext
    for j in ctext.split():
        # print (j)
        for k in bus_sch1:
            score = fuzz.ratio(j, k)
            if (score >= 70):
                # print (i,j,k)
                fg = 1
                break
        # if(fg==1):
        # break;
        if (fg == 1):
            fg = 0
            str1 = str1.replace(j, " place_name ")
    return str1
```

***清单 5-56.***

```python
direct_list = ['east', 'west', 'north', 'south']

def am_pm_direct_repl(ctext):
    ctext = re.sub('p.m', 'pm', ctext)
    ctext = re.sub('p m', 'pm', ctext)
    ctext = re.sub('a.m', 'am', ctext)
    ctext = re.sub('a m', 'am', ctext)
    for i in direct_list:
        ctext = re.sub(i, ' direction ', ctext)
    return ctext

from num2words import num2words

word_num_list = []
word_num_th_list = []
for i in range(0, 100):
    word_num_list.append(num2words(i))
    word_num_th_list.append(num2words(i, to='ordinal'))

def repl_num_words(ctext):
    for num, i in enumerate(word_num_th_list):
        ctext = re.sub(i, "num_th", ctext)
        ctext = re.sub(word_num_list[num], str(num), ctext)
    return ctext

def preprocces(ctext):
    ctext = ctext.lower()
    ctext1 = repl_place_bus(ctext)
    ctext1 = am_pm_direct_repl(ctext1)
    ctext1 = repl_num_words(ctext1)
    ctext1 = re.sub('[0-9]{1,2}[a-z]{1,2}', 'bus_name', ctext1)
    ctext1 = re.sub('[0-9]{1,2}[\.\s]*[0-9]{1,2}[\s]*pm', 'time_name', ctext1)
    ctext1 = re.sub('[0-9]{1,2}[\.\s]*[0-9]{1,2}[\s]*am', 'time_name', ctext1)
    ctext1 = re.sub('(\.\s){1,3}', ' ', ctext1)
    ctext1 = re.sub('(\.){1,3}', ' ', ctext1)
    return ctext1
```

现在，使用 `清单 5-57` 中的代码测试预处理。