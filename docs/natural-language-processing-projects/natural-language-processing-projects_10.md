# 提取实体

现在我们已经得到了每份简历相对于各职位描述的相似度得分，接下来提取所需实体，例如候选人姓名、电话号码、电子邮件地址、技能、工作年限以及前雇主信息。

我们编写正则表达式来提取电话号码：

```
# 原始简历的 DataFrame
t = pd.DataFrame({'原始简历':resumeTxt})
dt = pd.concat([Data,t],axis=1)
# 查找电话号码的函数
def number(text):
# compile 帮助我们定义在文本中匹配的模式
pattern = re.compile(r'([+(]?\d+[)\-]?[ \t\r\f\v]*[(]?\d{2,}[()\-]?[ \t\r\f\v]*\d{2,}[()\-]?[ \t\r\f\v]*\d*[ \t\r\f\v]*\d*[ \t\r\f\v]*)')
#  findall 查找 compile 中定义的模式
pt = pattern.findall(text)
#  sub 替换文本中匹配的模式
pt = [re.sub(r'[,.]', '', ah) for ah in pt if len(re.sub(r'[()\-.,\s+]', '', ah))>9]
pt = [re.sub(r'\D$', '', ah).strip() for ah in pt]
pt = [ah for ah in pt if len(re.sub(r'\D','',ah))  3: continue
for x in ah.split("-"):
try:
#  isdigit 检查文本是否为数字
if x.strip()[-4:].isdigit():
if int(x.strip()[-4:]) in range(1900, 2100):
#  移除提到的文本
pt.remove(ah)
except: pass
number = None
number = list(set(pt))
return number
```

`dt['电话号码']` 包含候选人的所有电话号码。

```
# 调用 number 函数获取候选人号码列表
dt['电话号码']=dt['原始简历'].apply(lambda x: number(x))
print("从数据框列中提取号码：")
dt['电话号码'][0:5]
```

图 5-9 显示了输出结果。

![../images/517793_1_En_5_Chapter/517793_1_En_5_Fig9_HTML.jpg](img/517793_1_En_5_Fig9_HTML.jpg)

图 5-9

显示从简历中提取的前五个号码。

现在，我们使用正则表达式提取每位候选人的电子邮件地址。

```
# 这里我们从简历中提取电子邮件
def email_ID(text):
# compile 帮助我们定义在文本中匹配的模式
r = re.compile(r'[A-Za-z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+')
return r.findall(str(text))
```

`dt['电子邮件 ID']` 包含候选人的电子邮件地址。

```
#  调用 email_ID 函数获取候选人电子邮件列表
dt['电子邮件 ID']=dt['原始简历'].apply(lambda x: email_ID(x))
print("从数据框列中提取电子邮件：")
dt['电子邮件 ID'][0:5]
```

图 5-10 显示了输出结果。

![../images/517793_1_En_5_Chapter/517793_1_En_5_Fig10_HTML.jpg](img/517793_1_En_5_Fig10_HTML.jpg)

图 5-10

显示从简历中提取的前五个电子邮件。

接下来，我们从简历语料库中移除候选人的电话号码和电子邮件地址，因为我们要提取工作年限和候选人姓名，而电话号码中的整数可能会被误认为是工作年限，或者干扰识别候选人姓名的函数。

```
# 移除电话号码以提取工作年限和候选人姓名的函数
def rm_number(text):
try:
# compile 帮助我们定义在文本中匹配的模式
pattern = re.compile(r'([+(]?\d+[)\-]?[ \t\r\f\v]*[(]?\d{2,}[()\-]?[ \t\r\f\v]*\d{2,}[()\-]?[ \t\r\f\v]*\d*[ \t\r\f\v]*\d*[ \t\r\f\v]*)')
#  findall 查找 compile 中定义的模式
pt = pattern.findall(text)
#  sub 替换文本中匹配的模式
pt = [re.sub(r'[,.]', '', ah) for ah in pt if len(re.sub(r'[()\-.,\s+]', '', ah))>9]
pt = [re.sub(r'\D$', '', ah).strip() for ah in pt]
pt = [ah for ah in pt if len(re.sub(r'\D','',ah))  3: continue
for x in ah.split("-"):
try:
#  isdigit 检查文本是否为数字
if x.strip()[-4:].isdigit():
if int(x.strip()[-4:]) in range(1900, 2100):
#  移除提到的文本
pt.remove(ah)
except: pass
number = None
number = pt
number = set(number)
number = list(number)
for i in number:
text = text.replace(i," ")
return text
except:
pass
```

`dt['原始']` 包含移除候选人电话号码后的所有简历。



