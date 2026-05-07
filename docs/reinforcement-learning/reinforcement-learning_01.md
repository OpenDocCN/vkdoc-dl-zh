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

## 引言

本书主要基于机器学习的子集——强化学习。我们借助 Python 编程语言介绍强化学习的基础知识，并涉及多个方面，例如 Q 学习、马尔可夫决策过程、使用 Keras 进行强化学习、OpenAI Gym 和 OpenAI 环境，同时还涵盖了与强化学习相关的算法。读者需要具备 Python 编程的基本知识才能从本书中获益。本书面向希望进入机器学习领域并进一步了解强化学习的人群。

## 致谢

谨以此书献给我的父母。

> —Abhishek Nandy

谨以此书献给我的父母。感谢我的老师和我的合著者 Abhishek Nandy。还要感谢 Abhishek Sur，他在工作中指导我，帮助我适应新技术。同时，谨以此书献给我的公司 InSync Tech-Fin Solutions Ltd.，我在这里开启职业生涯并获得了专业成长。

> —Manisha Biswas



## 目录

