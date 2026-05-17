# 使用 Python 通过 Open AI、TensorFlow 和 Keras 进行强化学习

Abhishek Nandy 和 Manisha Biswas

**Abhishek Nandy**  
印度西孟加拉邦加尔各答 Swaranika Co-Opt HSG 大楼 HIG L-2/4 室

**Manisha Biswas**  
印度西孟加拉邦北 24 帕尔加纳区

本书作者引用的任何源代码或其他补充材料，读者均可通过本书在 GitHub 上的产品页面获取，网址为 [www.apress.com/978-1-4842-3110-4](http://www.apress.com/978-1-4842-3110-4)。如需更详细信息，请访问 [`www.apress.com/source-code`](http://www.apress.com/source-code)。

ISBN 978-1-4842-3284-2  
电子书 ISBN 978-1-4842-3285-9  
[`doi.org/10.1007/978-1-4842-3285-9`](https://doi.org/10.1007/978-1-4842-3285-9)

美国国会图书馆控制号：2017962867

© Abhishek Nandy 和 Manisha Biswas 2018  
本作品受版权保护。出版商保留所有权利，无论是全部还是部分材料，特别是翻译、重印、重用插图、朗诵、广播、微缩胶片复制或以任何其他物理方式复制，以及信息存储与检索的传输、电子改编、计算机软件，或现在已知或以后开发的类似或不同方法的权利。

本书中可能出现商标名称、标识和图像。我们并非在每次出现商标名称、标识或图像时都使用商标符号，而是仅以编辑方式使用这些名称、标识和图像，以维护商标所有者的利益，且无意侵犯商标权。本出版物中对商品名称、商标、服务标记及类似术语的使用，即使未明确标识，也不应被视为对其是否受专有权利保护的看法。

尽管本书中的建议和信息在出版时被认为是真实准确的，但作者、编辑和出版商均不对可能出现的任何错误或遗漏承担法律责任。出版商对本书所含内容不作任何明示或暗示的保证。

印刷于无酸纸上

本书通过 Springer Science+Business Media New York 向全球图书贸易发行，地址：233 Spring Street, 6th Floor, New York, NY 10013。电话：1-800-SPRINGER，传真：(201) 348-4505，电子邮件：`orders-ny@springer-sbm.com`，或访问 `www.springeronline.com`。Apress Media, LLC 是加利福尼亚州的有限责任公司，其唯一成员（所有者）是 Springer Science + Business Media Finance Inc (SSBM Finance Inc)。SSBM Finance Inc 是特拉华州的一家公司。

## 关于作者与技术审校者

## 关于作者

**Abhishek Nandy** ![A454310_1_En_BookFrontmatter_Figb_HTML.jpg](img/A454310_1_En_BookFrontmatter_Figb_HTML.jpg) 拥有信息技术学士学位，并自认为是一名终身学习者。他是 Windows 平台的微软 MVP、英特尔黑带开发者，同时也是英特尔软件创新者。Abhishek 对人工智能、物联网和游戏开发有着浓厚的兴趣。他目前在一家 IT 公司担任应用架构师，并在人工智能和物联网领域提供咨询，同时从事人工智能、机器学习和深度学习方面的项目。他还是一名人工智能培训师，并负责英特尔人工智能学生开发者计划的技术部分。他曾参与首届“印度制造”计划，跻身 50 强创新者之列，并在印度管理学院艾哈迈德巴德分校接受过培训。

**Manisha Biswas** ![A454310_1_En_BookFrontmatter_Figc_HTML.jpg](img/A454310_1_En_BookFrontmatter_Figc_HTML.jpg) 拥有信息技术学士学位，目前在印度加尔各答的 InSync Tech-Fin Solutions Ltd 担任软件开发人员。她涉足多个技术领域，包括 Web 开发、物联网、软计算和人工智能。她是英特尔软件创新者，并曾获得 NASSCOM 颁发的 2016 年 Shri Dewang Mehta IT 奖（学术成绩优异证书）。她最近在印度加尔各答创立了一个“科技女性”社区，旨在赋能女性学习并探索新技术。她喜欢发明创造，推陈出新。不在终端前时，她是一个探险家、美食家、涂鸦爱好者和梦想家。她总是充满热情地与他人分享自己的知识和想法。她目前正追随自己的热情，通过分享经验来帮助社区学习，这也促使她成为了谷歌女性科技创客加尔各答分会的负责人。

## 关于技术审校者

**Avirup Basu** ![A454310_1_En_BookFrontmatter_Figd_HTML.jpg](img/A454310_1_En_BookFrontmatter_Figd_HTML.jpg) 是 Prescriber360 Solutions 的物联网应用开发者。他是一名机器人学研究员，并通过 IEEE 发表过论文。

© Abhishek Nandy 和 Manisha Biswas 2018

Abhishek Nandy 和 Manisha Biswas，《强化学习》`doi.org/10.1007/978-1-4842-3285-9_1`



