# Unity 中的神经网络

Windows 10 的 C# 编程

Abhishek Nandy  

Manisha Biswas

---

**Unity 中的神经网络**  

**Windows 10 的 C# 编程**

Abhishek Nandy  

Manisha Biswas

---

*Unity 中的神经网络*

Abhishek Nandy  

Manisha Biswas

印度，西孟加拉邦，加尔各答  

印度，西孟加拉邦，北 24 帕尔加纳区

平装本 ISBN-13: 978-1-4842-3672-7  

电子版 ISBN-13: 978-1-4842-3673-4

[`doi.org/10.1007/978-1-4842-3673-4`](https://doi.org/10.1007/978-1-4842-3673-4)

美国国会图书馆控制号：2018951222

版权 © 2018 归 Abhishek Nandy, Manisha Biswas 所有

本作品受版权保护。出版商保留所有权利，无论是全部还是部分材料，特别是翻译、重印、重用插图、朗诵、广播、微缩胶片复制或以任何其他物理方式复制、传输或信息存储与检索、电子改编、计算机软件，或现在已知或以后开发的类似或不同方法的权利。

本书中可能出现商标名称、标识和图像。我们不会在每次出现商标名称、标识或图像时都使用商标符号，而仅以编辑方式使用这些名称、标识和图像，以维护商标所有者的利益，无意侵犯商标。

在本出版物中使用商品名称、商标、服务标志和类似术语，即使这些术语未被标识为此类，也不应被视为对它们是否受所有权保护的看法。

尽管本书中的建议和信息在出版时被认为是真实和准确的，但作者、编辑和出版商均不对可能出现的任何错误或遗漏承担法律责任。出版商对本书所含材料不作任何明示或暗示的保证。

Apress Media LLC 董事总经理：Welmoed Spahr  

采编编辑：Celestin Suresh John  

开发编辑：Matthew Moodie  

协调编辑：Aditee Mirashi  

封面设计：eStudioCalamar  

封面图片设计：Freepik (www.freepik.com)

本书通过 Springer Science+Business Media New York 向全球图书贸易发行，地址：233 Spring Street, 6th Floor, New York, NY 10013。电话：1-800-SPRINGER，传真：(201) 348-4505，电子邮件：orders-ny@springer-sbm.com，或访问 www.springeronline.com。Apress Media, LLC 是一家加利福尼亚有限责任公司，其唯一成员（所有者）是 Springer Science + Business Media Finance Inc (SSBM Finance Inc)。SSBM Finance Inc 是一家特拉华州公司。

有关翻译信息，请发送电子邮件至 rights@apress.com，或访问 www.apress.com/rights-permissions。

Apress 图书可批量购买用于学术、企业或促销用途。大多数图书也提供电子版和许可证。有关更多信息，请参考我们的印刷版和电子版批量销售网页：www.apress.com/bulk-sales。

作者在本书中引用的任何源代码或其他补充材料，读者均可通过本书的产品页面在 GitHub 上获取，产品页面位于 www.apress.com/978-1-4842-3672-7。

如需更详细信息，请访问 www.apress.com/source-code。

印刷于无酸纸上

*谨以此书献给我的父母。*  

—Abhishek Nandy

*谨以此书献给我的父母和 Women Techmakers 的精神。*  

—Manisha Biswas

## 目录

关于作者  

关于技术审校  

引言  

第 1 章：神经网络基础  

第 2 章：Unity ML-Agents  

第 3 章：Unity 中的机器学习代理与神经网络  

第 4 章：Unity C# 中的反向传播  

第 5 章：Unity 中的数据可视化  

索引

## 关于作者

**Abhishek Nandy** 拥有信息技术学士学位，是一位不断学习的人。他是 Windows 平台的微软 MVP、英特尔黑带开发者以及英特尔软件创新者。他对人工智能、物联网和游戏开发有着浓厚的兴趣。

他目前在一家 IT 公司担任应用架构师，同时也在人工智能、物联网领域提供咨询，并从事人工智能、机器学习和深度学习方面的项目。他还是一名人工智能培训师，负责推动英特尔人工智能学生开发者计划的技术部分。他参与了首届“印度制造”计划，跻身前 50 名创新者之列，并在印度管理学院艾哈迈德巴德分校接受了培训。

**Manisha Biswas** 拥有信息技术学士学位，目前在印度加尔各答的 Prescriber360 担任数据科学家。她涉足多个技术领域，包括 Web 开发、物联网、软计算和人工智能。她是英特尔软件创新者，并曾因优异的学业成绩获得 NASSCOM 颁发的 SHRI DEWANG MEHTA IT AWARDS 2016 卓越证书。她是加尔各答“科技女性”组织的创始人，这是一个旨在赋能女性学习和探索新技术的科技社区。她总是喜欢发明创造，或者为旧事物赋予新面貌。当不在电脑前时，她是一个探索者、旅行者、美食家、涂鸦者和梦想家。她总是充满热情地与其他人分享自己的知识和想法。她正追随自己的热情，通过分享自己的经验来帮助社区，让他人能够学习并以新的方式实现她的想法。这使她成为了谷歌 Women Techmakers 加尔各答分会的负责人。

## 关于技术审校

**Ali Asad** 是一位充满热情的程序员，在多个领域拥有丰富经验，包括游戏玩法编程、自定义插件/工具/插件开发、计算编程、人工智能、咨询和策略制定。他的职业生涯涵盖了不同领域的应用生命周期，例如 AEC 行业和教育领域。

他撰写了《C# 程序员学习指南 (MCSD)》一书。他还是一名 *Microsoft Specialist: Programming in C#*。您可以在以下网址了解更多关于他的其他活动：[www.linkedin.com/in/imaliasad/](http://www.linkedin.com/in/imaliasad/)

## 引言

本书旨在尝试将 Unity 与机器学习和神经网络相结合。

我们首先简要介绍了有用的神经网络术语。

我们尝试使用新的 Unity-ML-Agents 0.3 版本，并清晰地构建了整个过程。

您需要什么？对 Unity 引擎在机器学习和神经网络方面有基本的了解或全新的认识。我们力求内容简单易懂。

---